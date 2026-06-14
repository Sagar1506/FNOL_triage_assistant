"""
ingest.py
=========
ONE-TIME script — run once to populate Pinecone with FNOL GUIDE and SOP chunks.

What it does:
  1. Reads FNOL-GUIDE-001-RAG.pdf  → chunks → embeds → upserts to namespace: fnol_guide
  2. Reads FNOL-SOP-001-v2-RAG.pdf → chunks → embeds → upserts to namespace: fnol_sop

Embedding model : Google Gemini text-embedding-004 (768 dimensions)
Vector DB       : Pinecone (single index, two namespaces)

Required .env variables:
  GOOGLE_API_KEY       — Google AI Studio API key
  PINECONE_API_KEY     — Pinecone API key
  PINECONE_INDEX_NAME  — e.g. fnol-triage-poc

Run:
  python ingest.py

Do NOT re-run unless the documents have changed or the index has been cleared.
"""

import os
import time
import hashlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Validate env vars before importing heavy deps ─────────────────────────
REQUIRED = ["GOOGLE_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
missing  = [v for v in REQUIRED if not os.getenv(v)]
if missing:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing)}\n"
        "Please add them to your .env file."
    )

from pypdf import PdfReader
import requests
from pinecone import Pinecone


# ── Configuration ──────────────────────────────────────────────────────────
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBED_MODEL         = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
# REST API endpoint adds "models/" prefix automatically in the URL
EMBED_DIM           = int(os.getenv("GEMINI_EMBED_DIM", "768"))

CHUNK_SIZE          = 400   # target words per chunk
CHUNK_OVERLAP       = 50    # words of overlap between consecutive chunks

# Documents to ingest: (pdf_path, namespace, source_label)
DOCUMENTS = [
    (
        os.getenv("FNOL_GUIDE_PATH", "./FNOL-GUIDE-001-RAG.pdf"),
        "fnol_guide",
        "FNOL-GUIDE-001",
    ),
    (
        os.getenv("FNOL_SOP_PATH", "./FNOL-SOP-001-v2-RAG.pdf"),
        "fnol_sop",
        "FNOL-SOP-001-v2",
    ),
]

# ── Setup ──────────────────────────────────────────────────────────────────
# Google Embedding — uses REST API directly (no SDK version issues)
GOOGLE_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:batchEmbedContents?key={api_key}"
)
pc    = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — PDF TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════

def extract_pages(pdf_path: str) -> list[dict]:
    """
    Extract text from each page of the PDF.
    Returns list of {page_num, text}.
    """
    reader = PdfReader(pdf_path)
    pages  = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page_num": i, "text": text})
    print(f"    Extracted {len(pages)} pages with text")
    return pages


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — CHUNKING
# ══════════════════════════════════════════════════════════════════════════

def chunk_pages(pages: list[dict], source: str,
                chunk_size: int = CHUNK_SIZE,
                overlap: int    = CHUNK_OVERLAP) -> list[dict]:
    """
    Sliding-window word-level chunking across all pages.

    Each chunk carries metadata:
      source       — document identifier (e.g. FNOL-GUIDE-001)
      chunk_id     — deterministic ID based on content hash
      chunk_index  — sequential index within this document
      page_start   — first page this chunk draws from
      page_end     — last page this chunk draws from
      text         — the chunk text
    """
    # Combine all page text with page markers for metadata tracking
    all_tokens = []   # list of (word, page_num)
    for page in pages:
        words = page["text"].split()
        for w in words:
            all_tokens.append((w, page["page_num"]))

    chunks      = []
    chunk_index = 0
    start       = 0

    while start < len(all_tokens):
        end    = min(start + chunk_size, len(all_tokens))
        tokens = all_tokens[start:end]

        text       = " ".join(t[0] for t in tokens)
        page_start = tokens[0][1]
        page_end   = tokens[-1][1]

        # Deterministic ID — hash of source + chunk content
        chunk_hash = hashlib.md5(f"{source}:{text}".encode()).hexdigest()[:12]
        chunk_id   = f"{source}-chunk-{chunk_index:03d}-{chunk_hash}"

        chunks.append({
            "chunk_id":    chunk_id,
            "chunk_index": chunk_index,
            "source":      source,
            "page_start":  page_start,
            "page_end":    page_end,
            "text":        text,
            "word_count":  len(tokens),
        })

        chunk_index += 1
        # Slide forward by (chunk_size - overlap)
        step = max(1, chunk_size - overlap)
        start += step

        if end == len(all_tokens):
            break

    print(f"    Created {len(chunks)} chunks "
          f"(~{sum(c['word_count'] for c in chunks) // len(chunks)} words avg)")
    return chunks


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — EMBEDDING
# ══════════════════════════════════════════════════════════════════════════

def embed_chunks(chunks: list[dict],
                 batch_size: int = 10) -> list[dict]:
    """
    Embed each chunk using Gemini text-embedding-004 via REST API.
    Returns chunks with 'embedding' key added.
    """
    embedded = []
    total    = len(chunks)
    url      = GOOGLE_EMBED_URL.format(
        model   = EMBED_MODEL,
        api_key = GOOGLE_API_KEY,
    )

    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start: batch_start + batch_size]

        print(f"    Embedding chunks {batch_start + 1}–"
              f"{min(batch_start + batch_size, total)} of {total}...")

        # Build batch request
        # output_dimensionality truncates to EMBED_DIM — keeps Pinecone index
        # compatible regardless of the model's native output size
        requests_payload = {
            "requests": [
                {
                    "model":   f"models/{EMBED_MODEL}",
                    "content": {"parts": [{"text": c["text"]}]},
                    "taskType": "RETRIEVAL_DOCUMENT",
                    "outputDimensionality": EMBED_DIM,
                }
                for c in batch
            ]
        }

        response = requests.post(
            url,
            json    = requests_payload,
            headers = {"Content-Type": "application/json"},
            timeout = 60,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Gemini embedding API error {response.status_code}: "
                f"{response.text}"
            )

        data = response.json()
        embeddings = [e["values"] for e in data["embeddings"]]

        for chunk, vector in zip(batch, embeddings):
            assert len(vector) == EMBED_DIM, (
                f"Expected {EMBED_DIM}-dim vector, got {len(vector)}"
            )
            embedded.append({**chunk, "embedding": vector})

        # Small delay to respect rate limits
        if batch_start + batch_size < total:
            time.sleep(0.5)

    return embedded


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — UPSERT TO PINECONE
# ══════════════════════════════════════════════════════════════════════════

def upsert_to_pinecone(chunks: list[dict], namespace: str,
                       batch_size: int = 50):
    """
    Upsert embedded chunks to Pinecone under the given namespace.
    Metadata stored per vector:
      source, chunk_index, page_start, page_end, text (for retrieval)
    """
    total   = len(chunks)
    upserted = 0

    for batch_start in range(0, total, batch_size):
        batch   = chunks[batch_start: batch_start + batch_size]
        vectors = []
        for c in batch:
            vectors.append({
                "id":     c["chunk_id"],
                "values": c["embedding"],
                "metadata": {
                    "source":      c["source"],
                    "chunk_index": c["chunk_index"],
                    "page_start":  c["page_start"],
                    "page_end":    c["page_end"],
                    "text":        c["text"],          # stored for retrieval
                    "word_count":  c["word_count"],
                },
            })

        index.upsert(vectors=vectors, namespace=namespace)
        upserted += len(vectors)
        print(f"    Upserted {upserted}/{total} vectors to namespace '{namespace}'")

    # Verify
    time.sleep(1)   # allow index to update
    stats = index.describe_index_stats()
    ns_count = (stats.namespaces.get(namespace, {})
                .get("vector_count", "unknown"))
    print(f"    Namespace '{namespace}' now contains {ns_count} vectors")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  FNOL RAG INGEST — one-time setup")
    print("=" * 60)
    print(f"  Pinecone index : {PINECONE_INDEX_NAME}")
    print(f"  Embed model    : {EMBED_MODEL}  (GEMINI_EMBED_MODEL)")
    print(f"  Chunk size     : {CHUNK_SIZE} words  |  Overlap: {CHUNK_OVERLAP} words")
    print()

    # Verify Pinecone index exists and has correct dimension
    try:
        stats = index.describe_index_stats()
        print(f"  ✓ Pinecone index found")
        print(f"    Total vectors before ingest: "
              f"{stats.total_vector_count}")
        print()
    except Exception as e:
        raise RuntimeError(
            f"Cannot connect to Pinecone index '{PINECONE_INDEX_NAME}': {e}\n"
            "Check PINECONE_API_KEY and PINECONE_INDEX_NAME in your .env file."
        )

    for pdf_path, namespace, source_label in DOCUMENTS:
        print(f"{'─' * 60}")
        print(f"  Processing: {source_label}")
        print(f"  File      : {pdf_path}")
        print(f"  Namespace : {namespace}")
        print()

        # Check file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}\n"
                f"Set FNOL_GUIDE_PATH / FNOL_SOP_PATH in .env, or place the "
                f"PDF in the same folder as ingest.py."
            )

        # Step 1: Extract
        print("  [Step 1] Extracting text from PDF...")
        pages = extract_pages(pdf_path)

        # Step 2: Chunk
        print("  [Step 2] Chunking text...")
        chunks = chunk_pages(pages, source_label)

        # Step 3: Embed
        print("  [Step 3] Embedding chunks with Gemini text-embedding-004...")
        chunks = embed_chunks(chunks)

        # Step 4: Upsert
        print("  [Step 4] Upserting to Pinecone...")
        upsert_to_pinecone(chunks, namespace)

        print(f"  ✓ {source_label} complete — {len(chunks)} chunks ingested")
        print()

    # Final stats
    print("=" * 60)
    stats = index.describe_index_stats()
    print("  INGEST COMPLETE")
    print(f"  Total vectors in index: {stats.total_vector_count}")
    for ns, ns_data in stats.namespaces.items():
        print(f"    namespace '{ns}': {ns_data.get('vector_count', 0)} vectors")
    print("=" * 60)
    print()
    print("  Next step: build the RAG Q&A agent (rag_agent.py)")
    print("  The index is now ready to query.")


if __name__ == "__main__":
    main()
