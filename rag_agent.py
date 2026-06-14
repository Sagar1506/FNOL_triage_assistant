# -*- coding: utf-8 -*-
"""
rag_agent.py
============
RAG Q&A Agent for FNOL Guidelines and SOP.

Workflow:
  1. Embed the adjuster's question using Gemini text-embedding (REST API)
  2. Query Pinecone across both namespaces (fnol_guide + fnol_sop)
  3. Retrieve top-K chunks with metadata
  4. Pass question + retrieved chunks to GPT-4o mini
  5. Return grounded answer with source citations

Public API:
    ask_rag(question: str, history: list[dict]) -> dict
        Returns {answer: str, sources: list[dict], context_used: str}

All config from .env:
  GOOGLE_API_KEY       - for Gemini embedding
  GEMINI_EMBED_MODEL   - e.g. embedding-001
  GEMINI_EMBED_DIM     - e.g. 768
  PINECONE_API_KEY     - Pinecone API key
  PINECONE_INDEX_NAME  - e.g. fnol-triage-poc
  OPENAI_API_KEY       - for GPT-4o mini answer generation
  LLM_MODEL            - e.g. gpt-4o-mini
"""

import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# ── LangSmith tracing (optional — only active when LANGCHAIN_TRACING_V2=true)
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn): return fn
        return decorator if args and callable(args[0]) else decorator

# Config from .env
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY", "")
EMBED_MODEL         = os.getenv("GEMINI_EMBED_MODEL", "embedding-001")
EMBED_DIM           = int(os.getenv("GEMINI_EMBED_DIM", "768"))
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fnol-triage-poc")
LLM_MODEL           = os.getenv("LLM_MODEL", "gpt-4o-mini")

TOP_K           = 3    # chunks to retrieve per namespace
NAMESPACES      = ["fnol_guide", "fnol_sop"]

# Source labels for citations
SOURCE_LABELS = {
    "fnol_guide": "FNOL-GUIDE-001",
    "fnol_sop":   "FNOL-SOP-001-v2",
}

# Lazy singletons
_pc_index  = None
_oai_client = None


def _get_index():
    global _pc_index
    if _pc_index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pc_index = pc.Index(PINECONE_INDEX_NAME)
    return _pc_index


def _get_oai():
    global _oai_client
    if _oai_client is None:
        _oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _oai_client


# Embed question using Gemini REST API
GOOGLE_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:embedContent?key={api_key}"
)


@traceable(name="rag._embed_question", run_type="embedding")
def _embed_question(question: str) -> list[float]:
    """Embed a single question for retrieval using Gemini REST API."""
    url = GOOGLE_EMBED_URL.format(
        model   = EMBED_MODEL,
        api_key = GOOGLE_API_KEY,
    )
    payload = {
        "model":   f"models/{EMBED_MODEL}",
        "content": {"parts": [{"text": question}]},
        "taskType": "RETRIEVAL_QUERY",
        "outputDimensionality": EMBED_DIM,
    }
    response = requests.post(
        url,
        json    = payload,
        headers = {"Content-Type": "application/json"},
        timeout = 30,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini embedding error {response.status_code}: {response.text}"
        )
    return response.json()["embedding"]["values"]


# Query Pinecone across both namespaces
@traceable(name="rag._retrieve_chunks", run_type="retriever")
def _retrieve_chunks(vector: list[float], top_k: int = TOP_K) -> list[dict]:
    """
    Query both namespaces and return merged, deduplicated chunks.
    Each chunk dict has: text, source, source_label, page_start, page_end, score.
    """
    index  = _get_index()
    chunks = []

    for ns in NAMESPACES:
        try:
            results = index.query(
                vector          = vector,
                top_k           = top_k,
                namespace       = ns,
                include_metadata= True,
            )
            for match in results.matches:
                meta = match.metadata or {}
                chunks.append({
                    "text":         meta.get("text", ""),
                    "source":       meta.get("source", ns),
                    "source_label": SOURCE_LABELS.get(ns, ns),
                    "namespace":    ns,
                    "page_start":   meta.get("page_start", ""),
                    "page_end":     meta.get("page_end", ""),
                    "chunk_index":  meta.get("chunk_index", 0),
                    "score":        round(match.score, 4),
                })
        except Exception as e:
            # Namespace may be empty during testing - skip gracefully
            print(f"[RAG] Warning: namespace '{ns}' query failed: {e}")

    # Sort by score descending, keep top results
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks[:top_k * 2]


# RAG system prompt
RAG_SYSTEM = """You are an expert AI assistant for an Indian motor insurance company.
You help claims adjusters with questions about FNOL (First Notice of Loss) guidelines
and standard operating procedures.

STRICT RULES:

1. SCOPE — Answer ONLY using the context provided. Do not use general insurance
   knowledge or any information outside the retrieved context.

2. UNCERTAINTY — If the context does not contain enough information to answer
   fully or clearly, say exactly: "I am not certain based on the available
   guidelines. Please verify with your supervisor or refer to the full document."
   Never attempt to fill gaps by guessing or inferring.

3. NO FABRICATION — Never invent, paraphrase, or reconstruct policy clauses,
   section numbers, coverage rules, or procedural steps. If a specific clause
   or rule is not present in the context, do not mention it.

4. REFUSE DATA MODIFICATION REQUESTS — If asked to update, modify, approve,
   reject, delete, or trigger any action on a claim or system, refuse clearly:
   "I am not able to perform system actions or modify claim data. Please use
   the appropriate system or contact your administrator."

5. REFUSE UNSAFE OR POLICY-VIOLATING REQUESTS — If asked anything that could
   facilitate fraud, circumvent controls, violate regulations, or cause harm,
   refuse clearly and do not provide any partial guidance:
   "This request falls outside what I am able to assist with. If you have
   concerns about a claim, please escalate to a senior claims manager."

6. ESCALATION — For sensitive, unresolved, or ambiguous cases where the
   guidelines do not provide a clear answer, always end your response with:
   "If this case remains unresolved, please escalate to a senior claims
   manager or human analyst for review."

7. CITATIONS — Always cite the source document and page number at the end
   of your answer in this format: [Source: DOCUMENT-NAME, Page X]
   If multiple sources are used, cite each one separately.

8. LENGTH — Keep answers under 200 words. Use bullet points for procedures.
"""


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as numbered context blocks for the LLM."""
    if not chunks:
        return "No relevant content found in the guidelines."

    lines = []
    for i, c in enumerate(chunks, 1):
        label   = c["source_label"]
        p_start = c["page_start"]
        p_end   = c["page_end"]
        page    = f"Page {p_start}" if p_start == p_end else f"Pages {p_start}-{p_end}"
        lines.append(
            f"[Context {i} | {label} | {page} | relevance: {c['score']}]\n"
            f"{c['text']}"
        )
    return "\n\n".join(lines)


def _build_messages(question: str, context: str,
                     history: list[dict]) -> list[dict]:
    """Build the messages list for the OpenAI API call."""
    messages = [{"role": "system", "content": RAG_SYSTEM}]

    # Add conversation history (last 6 turns to stay within context)
    for turn in history[-6:]:
        messages.append({"role": "user",      "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    # Current question with context
    messages.append({
        "role": "user",
        "content": (
            f"CONTEXT FROM FNOL GUIDELINES AND SOP:\n\n{context}\n\n"
            f"QUESTION: {question}"
        ),
    })
    return messages


# Public API
@traceable(name="ask_rag", run_type="chain")
def ask_rag(question: str, history: list[dict] = None) -> dict:
    """
    Answer an adjuster's question using RAG over FNOL GUIDE and SOP.

    Args:
        question : The adjuster's question string
        history  : List of {question, answer} dicts from prior turns

    Returns:
        {
          answer       : str   - LLM-generated grounded answer with citations
          sources      : list  - retrieved chunk metadata for display
          context_used : str   - the context string sent to the LLM
          error        : str   - error message if something failed (or None)
        }
    """
    if history is None:
        history = []

    try:
        # Step 1: embed question
        vector = _embed_question(question)

        # Step 2: retrieve chunks
        chunks = _retrieve_chunks(vector)

        # Step 3: build context
        context = _build_context(chunks)

        # Step 4: generate answer
        messages = _build_messages(question, context, history)
        response  = _get_oai().chat.completions.create(
            model       = LLM_MODEL,
            temperature = 0.0,
            max_tokens  = 400,
            messages    = messages,
        )
        answer = response.choices[0].message.content.strip()

        return {
            "answer":       answer,
            "sources":      chunks,
            "context_used": context,
            "error":        None,
        }

    except Exception as e:
        return {
            "answer":       None,
            "sources":      [],
            "context_used": "",
            "error":        str(e),
        }
