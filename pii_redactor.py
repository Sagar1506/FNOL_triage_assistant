"""
pii_redactor.py
===============
Non-LLM PII Anonymisation Module for FNOL Triage Assistant.

Pipeline:
    PDF → Google Vision OCR (in-memory) → Ensemble Anonymiser → Confidence Evaluator
          → Field Splitter → RedactionResult

Ensemble — three engines, zero spaCy, zero presidio-analyzer:
    1. BERT (dslim/bert-base-NER) — PERSON, ORG, LOC via HuggingFace directly
    2. GLiNER                      — Indian person names, zero-shot
    3. Pure Python regex           — Aadhaar, PAN, mobile, VRN, DL, email, phone

presidio-anonymizer is used ONLY for text replacement (no NLP, no spaCy).

Dependencies:
    transformers, torch, gliner, presidio-anonymizer, pypdfium2, google-cloud-vision,
    python-dotenv
    Optional (only needed if calling export_side_by_side_pdf): reportlab

Environment (.env):
    GOOGLE_VISION_CREDENTIALS_PATH=./fnol-pii-detection-500718-d6868fd2609e.json
    (or set GOOGLE_APPLICATION_CREDENTIALS directly to skip this module's resolution)

No spaCy. No presidio-analyzer. Works on Python 3.14 on Windows.
"""

from __future__ import annotations

import io
import os
import re
import logging
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════
# GOOGLE VISION CREDENTIALS
# ══════════════════════════════════════════════════════════════════════════
# Justification: this module is imported directly by fnol_intake_agent.py
# (and any test notebook) — it must set up its own OCR credentials rather
# than depending on the caller to configure the environment first.
# ImageAnnotatorClient() reads GOOGLE_APPLICATION_CREDENTIALS at the moment
# it's constructed (inside _ocr_page(), below), so this must run at import
# time, before any redact_document() call.

load_dotenv()  # picks up a .env file in the caller's working directory, if present

# Resolve credentials path with this precedence:
#   1. GOOGLE_APPLICATION_CREDENTIALS already set in the environment (e.g. by
#      a deployment platform / Docker secret) — leave it untouched.
#   2. GOOGLE_VISION_CREDENTIALS_PATH from .env / environment.
#   3. Default relative path (fallback for local dev).
GOOGLE_VISION_CREDENTIALS_PATH = os.getenv(
    "GOOGLE_VISION_CREDENTIALS_PATH",
    "./fnol-pii-detection-500718-d6868fd2609e.json",
)

if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    if os.path.exists(GOOGLE_VISION_CREDENTIALS_PATH):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(
            GOOGLE_VISION_CREDENTIALS_PATH
        )
        logger.info(
            f"Google Vision credentials loaded from: {GOOGLE_VISION_CREDENTIALS_PATH}"
        )
    else:
        # Don't raise here — importing the module shouldn't crash the whole
        # FNOL pipeline. OCR calls will fail individually (per-page try/except
        # in _run_ocr already catches this) and the reason will be logged.
        logger.warning(
            "Google Vision credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS "
            "or GOOGLE_VISION_CREDENTIALS_PATH (env var or .env file), or place the "
            f"service-account JSON at: {GOOGLE_VISION_CREDENTIALS_PATH}. "
            "OCR will fail until this is configured."
        )

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

SCORE_THRESHOLD    = 0.4
GLINER_SCORE_THRESHOLD = 0.55  # GLiNER is zero-shot and noisier than BERT/regex —
                                # a higher bar reduces false positives on
                                # out-of-vocabulary text (e.g. drug names).
CONFIDENCE_PROCEED = 0.85
CONFIDENCE_WARN    = 0.65
PDF_RENDER_DPI     = 300

PII_ENTITIES = [
    "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "ADDRESS", "DATE_TIME",
    "IN_PAN", "IN_AADHAAR", "IN_VEHICLE_REGISTRATION", "IN_PASSPORT",
    "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS", "URL",
    "IN_MOBILE", "IN_DRIVING_LICENCE",
]

SPII_ENTITIES = {
    "IN_AADHAAR", "IN_PAN", "CREDIT_CARD", "IBAN_CODE",
    "PHONE_NUMBER", "IN_MOBILE",
}

CUSTOM_ENTITIES = {
    "IN_VEHICLE_REGISTRATION", "IN_MOBILE", "IN_DRIVING_LICENCE", "ADDRESS"
}

# Structural rule (NOT a drug-name list): a PERSON/ADDRESS detection
# immediately followed by a dosage unit ("40Mg", "650 Mg", "5 Tab", ...) is
# almost certainly a medicine name, not a person or place. Treatment/
# medicine details must reach the triage LLM per the field-splitting
# design, so these should never be redacted. This generalises to ANY drug
# name + dosage format — it is not tied to the specific medicines in any
# one test document.
DOSAGE_CONTEXT_PATTERN = re.compile(
    r"^\s*\d+(?:\.\d+)?\s*(?:Mg|Mcg|Ml|Gm|Gms|G|IU|Tab|Tabs|Cap|Caps|Units?)\b",
    re.IGNORECASE,
)

# BERT CoNLL-2003 label → PII entity type
# Note: ORG and MISC are intentionally NOT mapped — organisation names
# (hospitals, departments) are not PII and must reach the triage LLM.
# Mapping them to PERSON was a bug that redacted hospital/department names.
BERT_LABEL_MAP = {
    "PER":  "PERSON",
    "LOC":  "ADDRESS",
}

# Fields routed to pii_store — never passed to triage LLM
PII_FIELDS_BY_DOCTYPE: dict[str, set[str]] = {
    "REGISTRATION_CERTIFICATE": {"owner_name", "engine_number", "chassis_number"},
    "DRIVING_LICENCE":          {"name_masked", "date_of_birth_masked", "blood_group", "licence_number"},
    "FIR":                      {"officer_in_charge", "complainant_vehicle_reg"},
    "FNOL_REGISTRATION_FORM":   {"member_id", "adjuster_id", "hospital_name", "claimant_name"},
}

# ══════════════════════════════════════════════════════════════════════════
# PURE PYTHON REGEX PATTERNS
# Replaces presidio-analyzer entirely — no spaCy dependency
# ══════════════════════════════════════════════════════════════════════════

REGEX_PATTERNS: list[dict] = [
    # ── Indian SPII ───────────────────────────────────────────────────────
    {
        "entity_type": "IN_AADHAAR",
        "pattern":     re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b"),
        "score":       0.90,
    },
    {
        "entity_type": "IN_PAN",
        "pattern":     re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
        "score":       0.95,
    },
    {
        "entity_type": "IN_PASSPORT",
        "pattern":     re.compile(r"\b[A-Z]\d{7}\b"),
        "score":       0.85,
    },
    # ── Indian contact ────────────────────────────────────────────────────
    {
        "entity_type": "IN_MOBILE",
        "pattern":     re.compile(r"\b[6-9]\d{9}\b"),
        "score":       0.80,
    },
    {
        "entity_type": "IN_VEHICLE_REGISTRATION",
        "pattern":     re.compile(
            r"\b[A-Z]{2}[\s-]?\d{2}[\s-]?[A-Z]{1,2}[\s-]?\d{4}\b"
            r"|\b\d{2}BH\d{4}[A-Z]{1,2}\b"
        ),
        "score":       0.90,
    },
    {
        "entity_type": "IN_DRIVING_LICENCE",
        "pattern":     re.compile(r"\b[A-Z]{2}[\s-]\d{2}[\s-]\d{4}\d{7}\b"),
        "score":       0.85,
    },
    # ── Indian address blocks ─────────────────────────────────────────────
    {
        "entity_type": "ADDRESS",
        "pattern":     re.compile(
            r"[A-Za-z0-9\s\-\/,\.#]+"
            r"(?:Nagar|City|Colony|Society|Chowk|Road|Marg|Layout|Peth|Wadi|Galli)"
            r"[A-Za-z0-9\s\-\/,\.]*"
            r",\s*[A-Za-z\s]+"
            r",\s*[A-Za-z\s]+"
            r"(?:,\s*India)?"
            r"(?:,\s*\d{6})?"
        ),
        "score":       0.75,
    },
    # ── Indian names with salutation ──────────────────────────────────────
    {
        "entity_type": "PERSON",
        "pattern":     re.compile(
            r"\b(Mr|Mrs|Ms|Dr|Shri|Smt)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b"
        ),
        "score":       0.70,
    },
    # ── Universal patterns ────────────────────────────────────────────────
    {
        "entity_type": "EMAIL_ADDRESS",
        "pattern":     re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "score":       0.95,
    },
    {
        "entity_type": "PHONE_NUMBER",
        "pattern":     re.compile(
            r"\b(?:\+91[\s-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b"
            r"|\b(?:\+91[\s-]?)?\d{10}\b"
        ),
        "score":       0.75,
    },
    {
        "entity_type": "CREDIT_CARD",
        "pattern":     re.compile(
            r"\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})"
            r"[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
        ),
        "score":       0.90,
    },
    {
        "entity_type": "IBAN_CODE",
        "pattern":     re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]{0,16})?\b"),
        "score":       0.85,
    },
    {
        "entity_type": "IP_ADDRESS",
        "pattern":     re.compile(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
            r"|\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
        ),
        "score":       0.90,
    },
    {
        "entity_type": "URL",
        "pattern":     re.compile(
            r"https?://[^\s]+"
            r"|www\.[^\s]+"
        ),
        "score":       0.85,
    },
]


# ══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class PageRedactionResult:
    page_num:        int
    raw_text:        str
    anonymised_text: str
    detections:      list[dict]
    spii_found:      list[str]
    char_count_raw:  int
    char_count_anon: int


@dataclass
class RedactionResult:
    pages:               list[PageRedactionResult]
    anonymised_pages:    list[str]
    pii_store:           dict[int, list[dict]]
    confidence_score:    float
    confidence_label:    str
    proceed:             bool
    proceed_status:      str
    total_entities:      int
    spii_entities_found: list[str]
    entities_by_type:    dict[str, int]
    engines_used:        list[str]
    # confidence_components: the 3 sub-scores that do NOT depend on LLM
    # field extraction (residual_score, density_score, engine_score),
    # stored so finalize_redaction_confidence() can recombine them with a
    # real spii_score later without recomputing OCR/detection work.
    confidence_components: dict = None
    # spii_leakage: doc_type.field_name entries where a real SPII-shaped
    # value survived anonymisation and leaked into LLM-safe extracted
    # fields. Empty until finalize_redaction_confidence() runs.
    spii_leakage:        list = None
    # entities_to_redact: the entity types that were ACTUALLY anonymised
    # for this run — recorded on the result itself (not just passed as a
    # function argument) so build_redaction_report() and any audit trail
    # can show the caller exactly what scope was used, even for a caller
    # who didn't specify one explicitly (defaults to all of PII_ENTITIES).
    entities_to_redact:  list = None
    error:               Optional[str] = None

    def __post_init__(self):
        if self.confidence_components is None:
            self.confidence_components = {}
        if self.spii_leakage is None:
            self.spii_leakage = []
        if self.entities_to_redact is None:
            self.entities_to_redact = list(PII_ENTITIES)


# ══════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETONS
# ══════════════════════════════════════════════════════════════════════════

_bert_pipeline = None
_anonymizer    = None
_gliner        = None
_vision_client = None
_initialised   = False


def initialise() -> None:
    """
    Load all models into memory. Call once at application startup.

    Three engines — no spaCy, no presidio-analyzer, works on Python 3.14:
      1. BERT (dslim/bert-base-NER) — direct HuggingFace pipeline
      2. GLiNER                     — Indian person names, zero-shot
      3. Pure Python regex          — Aadhaar, PAN, VRN, mobile, DL, email etc.

    presidio-anonymizer used only for text replacement — no NLP dependency.
    """
    global _bert_pipeline, _anonymizer, _gliner, _vision_client, _initialised

    if _initialised:
        return

    logger.info("🔄 Initialising PII redactor — loading models...")

    # ── BERT NER — direct HuggingFace, no spaCy ───────────────────────────
    from transformers import pipeline as hf_pipeline
    _bert_pipeline = hf_pipeline(
        task="ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
        device=-1,  # CPU
    )
    logger.info("✅ BERT NER pipeline loaded (dslim/bert-base-NER)")

    # ── Presidio Anonymizer only — no analyzer, no spaCy ──────────────────
    from presidio_anonymizer import AnonymizerEngine
    _anonymizer = AnonymizerEngine()
    logger.info("✅ Presidio anonymizer loaded (replacement engine only)")

    # ── GLiNER — zero-shot NER for Indian person names ─────────────────────
    try:
        from gliner import GLiNER
        _gliner = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")
        logger.info("✅ GLiNER loaded — Indian name detection enhanced")
    except Exception as e:
        logger.warning(f"⚠️  GLiNER not available ({e}). Falling back to BERT + regex.")
        _gliner = None

    # ── Google Vision client — created ONCE, reused across every page/call ──
    # Justification: the client construction (auth handshake, connection
    # setup) was previously repeated inside _ocr_page() on every single
    # page, which is pure overhead with zero benefit since the client
    # itself is stateless and safe to reuse. Wrapped in try/except like
    # GLiNER above: missing credentials shouldn't crash the whole module
    # import — OCR calls will fail individually and log the reason, exactly
    # as before this change.
    try:
        from google.cloud import vision as gv
        _vision_client = gv.ImageAnnotatorClient()
        logger.info("✅ Google Vision client initialised (reused across pages)")
    except Exception as e:
        logger.warning(f"⚠️  Google Vision client init failed ({e}). OCR will fail until fixed.")
        _vision_client = None

    _initialised = True
    logger.info("✅ PII redactor initialised — all models ready")


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — OCR
# ══════════════════════════════════════════════════════════════════════════

def _pdf_to_vision_bytes(pdf_path: str) -> list[bytes]:
    import pypdfium2 as pdfium
    pdf   = pdfium.PdfDocument(pdf_path)
    pages = []
    scale = PDF_RENDER_DPI / 72
    for page in pdf:
        bitmap = page.render(scale=scale, fill_color=(255, 255, 255, 255))
        image  = bitmap.to_pil()
        buf    = io.BytesIO()
        image.save(buf, format="JPEG")
        pages.append(buf.getvalue())
    pdf.close()
    return pages


def _ocr_page(image_bytes: bytes) -> str:
    global _vision_client
    from google.cloud import vision as gv

    # Lazy fallback: normally initialise() sets this up once at startup, but
    # guard here too in case redact_document() is called without it (mirrors
    # the existing _initialised check at the top of redact_document()).
    if _vision_client is None:
        _vision_client = gv.ImageAnnotatorClient()

    image    = gv.Image(content=image_bytes)
    response = _vision_client.document_text_detection(image=image)
    if response.error.message:
        raise RuntimeError(f"Google Vision error: {response.error.message}")
    return response.full_text_annotation.text.strip()


def _run_ocr(pdf_path: str) -> dict[int, str]:
    """
    OCR every page of the PDF, running pages CONCURRENTLY rather than one
    after another.

    Justification: each page is an independent network call to Google
    Vision — nothing about page 2's OCR depends on page 1 finishing first.
    Running them sequentially only adds up wait time for no benefit; a
    thread pool is enough here since these are I/O-bound network calls
    (Python's GIL doesn't block this kind of concurrency). This changes
    WHEN results come back, not how many API calls are made — same total
    billed OCR units as before, just less wall-clock time waiting for them.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pages_bytes = _pdf_to_vision_bytes(pdf_path)
    ocr_results: dict[int, str] = {}

    def _ocr_one(i: int, page_bytes: bytes) -> tuple[int, str]:
        try:
            return i, _ocr_page(page_bytes)
        except Exception as e:
            logger.warning(f"   Page {i} OCR failed: {e}")
            return i, ""

    with ThreadPoolExecutor(max_workers=min(len(pages_bytes), 8) or 1) as pool:
        futures = [
            pool.submit(_ocr_one, i, page_bytes)
            for i, page_bytes in enumerate(pages_bytes, start=1)
        ]
        for future in as_completed(futures):
            i, text = future.result()
            ocr_results[i] = text

    return ocr_results


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — NORMALISE
# ══════════════════════════════════════════════════════════════════════════

def _normalise(ocr_results: dict[int, str]) -> dict[int, str]:
    """
    Title-case for BERT NER accuracy.
    'MS. JADHAV PRAYUJA' → 'Ms. Jadhav Prayuja'
    """
    return {page_num: text.title() for page_num, text in ocr_results.items()}


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — ENSEMBLE DETECTION
# ══════════════════════════════════════════════════════════════════════════

def _run_bert(text: str) -> list[dict]:
    """BERT NER via direct HuggingFace pipeline — no spaCy."""
    if _bert_pipeline is None:
        return []
    try:
        entities = _bert_pipeline(text)
        results  = []
        for e in entities:
            raw_label   = e.get("entity_group", e.get("entity", ""))
            label       = raw_label.replace("B-", "").replace("I-", "")
            entity_type = BERT_LABEL_MAP.get(label)
            if not entity_type:
                continue
            score = float(e.get("score", 0.0))
            if score < SCORE_THRESHOLD:
                continue
            results.append({
                "entity_type": entity_type,
                "value":       e["word"],
                "start":       e["start"],
                "end":         e["end"],
                "score":       round(score, 3),
                "engine":      "bert",
            })
        return results
    except Exception as ex:
        logger.warning(f"BERT prediction failed: {ex}")
        return []


def _run_gliner(text: str) -> list[dict]:
    """GLiNER zero-shot NER — best for Indian surname-first names."""
    if _gliner is None:
        return []
    try:
        labels   = ["person name", "Indian person name", "patient name", "claimant name"]
        entities = _gliner.predict_entities(text, labels, threshold=GLINER_SCORE_THRESHOLD)
        return [
            {
                "entity_type": "PERSON",
                "value":       e["text"],
                "start":       e["start"],
                "end":         e["end"],
                "score":       round(e["score"], 3),
                "engine":      "gliner",
            }
            for e in entities
        ]
    except Exception as ex:
        logger.warning(f"GLiNER prediction failed: {ex}")
        return []


def _run_regex(text: str) -> list[dict]:
    """
    Pure Python regex — no external dependencies.
    Covers Aadhaar, PAN, VRN, DL, mobile, email, phone, credit card,
    IBAN, IP, URL, Indian addresses, salutation-prefixed names.
    """
    results = []
    for rule in REGEX_PATTERNS:
        for match in rule["pattern"].finditer(text):
            results.append({
                "entity_type": rule["entity_type"],
                "value":       match.group(),
                "start":       match.start(),
                "end":         match.end(),
                "score":       rule["score"],
                "engine":      "regex",
            })
    return results


def _merge_detections(
    bert_results:   list[dict],
    gliner_results: list[dict],
    regex_results:  list[dict],
) -> list[dict]:
    """
    Merge all three engines, deduplicating overlapping spans.

    Resolution order on overlap:
      1. Containment: whichever span is WIDER wins outright, regardless of
         score. This is deliberate — a wide regex match like a full address
         ("K 301 Iris Magarpatta City Hadapsar, Pune, Maharashtra, India")
         is structurally more complete than a small nested fragment inside
         it (e.g. a BERT LOC token for just "Pune"), even if the fragment
         scores slightly higher. Without this rule, a nested high-score
         fragment can evict a correct wide span, and the OTHER fragments
         that were also nested inside it then stop overlapping anything in
         `merged` and get added as separate entries — fragmenting one
         correct entity into several disconnected small ones.
      2. Tie on width: higher score wins.
      3. Tie on width and score: engine priority — regex > bert > gliner.
    """
    all_detections = bert_results + gliner_results + regex_results
    all_detections.sort(key=lambda x: (x["start"], -x["score"]))

    engine_priority = {"regex": 3, "bert": 2, "gliner": 1}
    merged: list[dict] = []

    for det in all_detections:
        overlapping = False
        for i, existing in enumerate(merged):
            if det["start"] < existing["end"] and det["end"] > existing["start"]:
                overlapping = True

                det_width      = det["end"] - det["start"]
                existing_width = existing["end"] - existing["start"]
                det_contains_existing = det["start"] <= existing["start"] and det["end"] >= existing["end"]
                existing_contains_det = existing["start"] <= det["start"] and existing["end"] >= det["end"]

                if det_contains_existing and det_width > existing_width:
                    merged[i] = det
                elif existing_contains_det and existing_width > det_width:
                    pass  # keep the wider existing span — do not let a nested fragment evict it
                elif det["score"] > existing["score"]:
                    merged[i] = det
                elif det["score"] == existing["score"]:
                    if engine_priority.get(det["engine"], 0) > \
                       engine_priority.get(existing["engine"], 0):
                        merged[i] = det
                break
        if not overlapping:
            merged.append(det)

    merged.sort(key=lambda x: x["start"])
    return merged


def _snap_to_word_boundaries(text: str, detections: list[dict]) -> list[dict]:
    """
    Expand each detection's span outward to the nearest word boundary.

    NER models (BERT/GLiNER) can return spans that cut mid-word — e.g.
    tagging only "Pan" inside "Pantoprazole", or "Prayuja" partially as
    "uja" — because of subword tokenisation. Anonymising the raw span then
    leaves garbled text like '<PERSON>lo' or '<ADDRESS>uja'. This snaps
    every span so it always starts/ends on whitespace or punctuation,
    so the replacement always covers a whole word — regardless of which
    engine produced the span or what document it came from.
    """
    def is_word_char(c: str) -> bool:
        return c.isalnum()

    snapped = []
    for d in detections:
        start, end = d["start"], d["end"]
        while start > 0 and is_word_char(text[start - 1]) and is_word_char(text[start]):
            start -= 1
        while end < len(text) and end > 0 and is_word_char(text[end - 1]) and is_word_char(text[end]):
            end += 1
        new_d = dict(d)
        new_d["start"] = start
        new_d["end"]   = end
        new_d["value"] = text[start:end]
        snapped.append(new_d)
    return snapped


def _drop_dosage_context_false_positives(text: str, detections: list[dict]) -> list[dict]:
    """
    Drop PERSON/ADDRESS detections that are immediately followed by a
    dosage unit (e.g. "Pantoprazole 40Mg", "Dolo 650Mg") — these are
    medicine names caught by NER models misfiring on out-of-vocabulary
    drug names, not actual PII. Structural, not a hardcoded drug list.
    """
    kept = []
    for d in detections:
        if d["entity_type"] in ("PERSON", "ADDRESS"):
            following = text[d["end"]: d["end"] + 15]
            if DOSAGE_CONTEXT_PATTERN.match(following):
                logger.info(f"Dropped false positive (dosage context): '{d['value']}'")
                continue
        kept.append(d)
    return kept


# Institutional/organisational keyword CATEGORIES (not specific org names) —
# these generalise across every FNOL document type, not just hospital forms:
# hospitals & clinics (medical claims), police stations (FIRs), garages/
# workshops (motor claims), insurers, government offices (RTO, courts).
# A PERSON detection sitting right next to one of these is almost always
# part of an organisation/department name, not an actual person.
ORG_CONTEXT_WORDS = {
    # medical
    "hospital", "hospitals", "clinic", "clinics", "centre", "center",
    "centres", "centers", "research", "healthcare", "diagnostics",
    "pharmacy", "pharmacies", "chemist", "chemists", "laboratories", "labs",
    "department", "departments", "ward", "wards",
    # institutional / legal / government
    "police", "station", "stations", "court", "courts", "rto",
    "government", "govt", "office", "authority",
    # commercial / corporate
    "ltd", "pvt", "limited", "corporation", "corp", "company", "co",
    "garage", "workshop", "workshops", "motors", "insurance", "insurer",
    "bank", "society", "trust", "university", "college", "institute",
    "institutes",
}

# Justification for a SEPARATE, narrower list for ADDRESS false positives:
# ORG_CONTEXT_WORDS above was built for PERSON detections, where broad
# single-word matching is safe — a name sitting next to "hospital" or
# "society" is almost never itself a hospital/society name. For ADDRESS
# detections the opposite is true for several of those same words: "ABC
# Housing Society", "near XYZ Bank", "opposite MG College" are genuine
# personal/residential addresses where masking the location IS correct.
# So this list is restricted to institution types where (a) FNOL business
# logic genuinely needs the real location — cross-document consistency
# checks (FNOL-vs-FIR police station, RC-vs-claim RTO, hospital/injury
# cross-checks) — and (b) the keyword rarely doubles as ordinary
# residential-address vocabulary the way "society"/"bank"/"college" do.
ADDRESS_INSTITUTION_CONTEXT_WORDS = {
    # law enforcement / legal — needed for the FNOL-vs-FIR police station
    # consistency check
    "police", "station", "stations", "thana", "court", "courts", "tribunal",
    # vehicle registration authority — needed for RC cross-checks
    "rto",
    # medical — needed for hospital/injury claim cross-checks
    "hospital", "hospitals", "clinic", "clinics",
}


def _drop_org_context_false_positives(text: str, detections: list[dict]) -> list[dict]:
    """
    Drop PERSON/ADDRESS detections that are actually part of an
    institution's own name/location, not personal PII — identified
    structurally by proximity to (or containing) a common institutional
    keyword ("Hospital", "Department", "Police Station", "Garage",
    "Insurance", ...). Category-based, not a list of specific
    organisation names, so it generalises across document types.

    ADDRESS is included alongside PERSON because BERT's LOC tag (mapped
    to ADDRESS) fires on location words inside institutional names too —
    e.g. "Dwarka Sector 21 Police Station" gets tagged as ADDRESS purely
    for containing "Dwarka"/"Sector 21", even though a police station's
    location is not personal PII and is needed downstream (e.g. the
    FNOL-vs-FIR police station consistency check) — redacting it there
    silently breaks that check instead of protecting anyone's privacy.
    ADDRESS uses its OWN narrower word list (ADDRESS_INSTITUTION_CONTEXT_WORDS),
    not the PERSON one — see that set's docstring for why.

    The context window is clipped to the current LINE only. Structured
    forms put organisation names and person names on different lines/
    fields (e.g. "Doctor: Dr Cmo, Noble Hospital" then "Patient: Jane
    Doe" right below it) — a plain character-distance window would bleed
    across that line break and wrongly protect the patient's real name
    just because an unrelated org name sits close by in raw text.
    """
    kept = []
    for d in detections:
        if d["entity_type"] not in ("PERSON", "ADDRESS"):
            kept.append(d)
            continue

        # PERSON uses the broad list (safe there); ADDRESS uses the
        # narrower institution-only list (see ADDRESS_INSTITUTION_CONTEXT_WORDS
        # docstring above for why the two must differ).
        context_words = (
            ORG_CONTEXT_WORDS if d["entity_type"] == "PERSON"
            else ADDRESS_INSTITUTION_CONTEXT_WORDS
        )

        line_start = text.rfind("\n", 0, d["start"]) + 1
        line_end   = text.find("\n", d["end"])
        if line_end == -1:
            line_end = len(text)

        window_before = text[line_start: d["start"]].lower()
        window_after  = text[d["end"]: line_end].lower()
        span_words    = set(re.findall(r"[a-z]+", d["value"].lower()))
        nearby_words  = set(re.findall(r"[a-z]+", window_before + " " + window_after))

        if (span_words | nearby_words) & context_words:
            logger.info(
                f"Dropped false positive (organisation context, "
                f"{d['entity_type']}): '{d['value']}'"
            )
            continue
        kept.append(d)
    return kept


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — ANONYMISE
# ══════════════════════════════════════════════════════════════════════════

def _anonymise_page(text: str, detections: list[dict]) -> str:
    """
    Replace PII spans with <ENTITY_TYPE> tokens.
    Uses presidio-anonymizer for robust span replacement.
    presidio-anonymizer has NO spaCy dependency.
    RecognizerResult imported from presidio_anonymizer.entities — not presidio_analyzer.
    """
    if not detections:
        return text

    from presidio_anonymizer.entities import RecognizerResult, OperatorConfig

    analyzer_results = [
        RecognizerResult(
            entity_type=d["entity_type"],
            start=d["start"],
            end=d["end"],
            score=d["score"],
        )
        for d in detections
    ]

    operators = {
        entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
        for entity in PII_ENTITIES
    }

    result = _anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators=operators,
    )
    return result.text


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — CONFIDENCE EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def _score_to_status(overall: float) -> tuple[float, str, bool, str]:
    """Map a weighted overall score to (score, label, proceed, status)."""
    if overall >= CONFIDENCE_PROCEED:
        return overall, "HIGH",     True,  "PROCEED"
    elif overall >= CONFIDENCE_WARN:
        return overall, "MODERATE", True,  "WARN"
    else:
        return overall, "LOW",      False, "HOLD"


def _evaluate_confidence(
    pages: list[PageRedactionResult],
) -> tuple[float, str, bool, str, dict]:
    """
    Four-dimension confidence score — pure Python, no LLM.
    1. SPII Coverage  (40%) — PRELIMINARY here; see finalize_redaction_confidence()
    2. Residual Scan  (30%)
    3. Density        (20%)
    4. Engine Agreement (10%)

    Justification for the split: at this point (right after OCR + detection,
    before any LLM field extraction has happened), there is no way yet to
    verify whether detection actually caught every real SPII value — that
    can only be checked once the LLM's own extracted field values are
    available to scan for leakage. So spii_score here is a PRELIMINARY
    placeholder (1.0), and this function also returns the other 3
    components so finalize_redaction_confidence() can recombine them with
    a real spii_score later without redoing this work.
    """
    # 1. SPII coverage — PRELIMINARY (see docstring). Real value computed
    # later by finalize_redaction_confidence() using LLM extraction output.
    spii_score_preliminary = 1.0

    # 2. Residual scan — check anonymised text for missed PII patterns
    residual_patterns = [
        re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),   # Aadhaar-like
        re.compile(r"\b\d{12}\b"),                  # 12-digit
        re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),    # PAN-like
        re.compile(r"\b[6-9]\d{9}\b"),             # mobile-like
    ]
    residual_hits = sum(
        len(p.findall(page.anonymised_text))
        for page in pages
        for p in residual_patterns
    )
    residual_score = max(0.0, 1.0 - (residual_hits * 0.15))

    # 3. Detection density
    text_pages = [p for p in pages if len(p.raw_text) > 100]
    pages_with_detections = [p for p in text_pages if len(p.detections) > 0]
    density_score = (
        len(pages_with_detections) / len(text_pages) if text_pages else 1.0
    )

    # 4. Engine agreement
    engines_used = {d.get("engine") for p in pages for d in p.detections}
    engine_score = 1.0 if len(engines_used) > 1 else 0.6

    components = {
        "residual_score": residual_score,
        "density_score":  density_score,
        "engine_score":   engine_score,
    }

    # Weighted overall (using preliminary spii_score)
    overall = round(
        spii_score_preliminary * 0.40 +
        residual_score         * 0.30 +
        density_score          * 0.20 +
        engine_score           * 0.10,
        3
    )

    score, label, proceed, status = _score_to_status(overall)
    return score, label, proceed, status, components


# ══════════════════════════════════════════════════════════════════════════
# STEP 5b — SPII COVERAGE FINALIZATION (real check, not a placeholder)
# ══════════════════════════════════════════════════════════════════════════

def check_residual_spii_in_fields(
    extracted_fields_by_doctype: dict[str, dict],
    entities_to_redact: Optional[list] = None,
) -> tuple[float, list[str]]:
    """
    Real SPII coverage check.

    Justification: the only way to know whether the anonymiser genuinely
    missed a real SPII value is to look at the LLM's own extracted field
    output AFTER redaction — if a raw Aadhaar/PAN/mobile/credit-card/IBAN
    -shaped value survived and shows up there, that is direct proof of a
    detection miss, not an assumption. This scans exactly the fields that
    are meant to reach the triage LLM (llm_safe fields) for each doc type,
    using the SAME regex patterns already used for detection — one source
    of truth, no duplicated pattern list to drift out of sync.

    Args:
        extracted_fields_by_doctype: {doc_type: {field_name: value}} —
            the LLM-safe fields (post split_fields) for every classified
            document in this submission.
        entities_to_redact: the entity types the caller actually asked
            redact_document() to mask (see its docstring). Only SPII types
            IN this set are checked for leakage here — an entity type the
            caller deliberately chose not to redact showing up in plain
            text is expected behaviour, not a detection failure, and must
            not be flagged or penalise the confidence score. Defaults to
            None, meaning "assume everything in PII_ENTITIES was meant to
            be redacted" — the original, fully backward-compatible check.

    Returns:
        (spii_score, leaked_fields)
        spii_score:    1.0 if no leaks found, penalised per leak otherwise
        leaked_fields: ["doc_type.field_name", ...] for audit/display
    """
    redact_set = set(entities_to_redact) if entities_to_redact is not None else set(PII_ENTITIES)
    spii_patterns = [
        p for p in REGEX_PATTERNS
        if p["entity_type"] in SPII_ENTITIES and p["entity_type"] in redact_set
    ]

    leaked_fields: list[str] = []
    for doc_type, fields in (extracted_fields_by_doctype or {}).items():
        for field_name, value in (fields or {}).items():
            if value is None:
                continue
            val_str = str(value)
            for pattern_def in spii_patterns:
                if pattern_def["pattern"].search(val_str):
                    leaked_fields.append(f"{doc_type}.{field_name}")
                    break  # one match is enough to flag this field

    spii_score = max(0.0, 1.0 - (len(leaked_fields) * 0.25))
    return spii_score, leaked_fields


def finalize_redaction_confidence(
    result: RedactionResult,
    extracted_fields_by_doctype: dict[str, dict],
) -> RedactionResult:
    """
    Recompute confidence_score/label/proceed/proceed_status using a REAL
    spii_score derived from check_residual_spii_in_fields(), recombined
    with the residual/density/engine components already stored on `result`
    from _evaluate_confidence(). No OCR or detection work is redone —
    this only re-weighs the same pipeline output.

    Uses result.entities_to_redact (recorded by redact_document()) so the
    leakage check only flags entity types that were actually supposed to
    be redacted for this run — a type the caller deliberately excluded
    isn't treated as a detection failure.

    Call this after document classification + field extraction have
    produced extracted_fields_by_doctype, and before displaying
    build_redaction_report() output to the user.
    """
    spii_score, leaked_fields = check_residual_spii_in_fields(
        extracted_fields_by_doctype, result.entities_to_redact
    )
    components = result.confidence_components or {
        "residual_score": 1.0, "density_score": 1.0, "engine_score": 1.0,
    }

    overall = round(
        spii_score                       * 0.40 +
        components["residual_score"]     * 0.30 +
        components["density_score"]      * 0.20 +
        components["engine_score"]       * 0.10,
        3
    )
    score, label, proceed, status = _score_to_status(overall)

    result.confidence_score = score
    result.confidence_label = label
    result.proceed          = proceed
    result.proceed_status   = status
    result.spii_leakage      = leaked_fields
    return result


# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — FIELD SPLITTER
# ══════════════════════════════════════════════════════════════════════════

def split_fields(
    extracted: dict,
    doc_type: str,
    pii_fields_by_doctype: Optional[dict] = None,
) -> tuple[dict, dict]:
    """
    Split extracted fields into llm_safe and pii_store_fields.
    Pure Python — deterministic, no LLM, auditable.

    Args:
        extracted: {field_name: value} for one document.
        doc_type: the document type extracted was extracted for.
        pii_fields_by_doctype: optional override for which fields are PII
            per doc type — {doc_type: {field_name, ...}}. Defaults to None,
            meaning "use the module-level PII_FIELDS_BY_DOCTYPE" (the
            FNOL-specific config this module ships with). A different
            consumer of this module — e.g. a claims team with a different
            document set entirely — can supply their own mapping here
            instead of being locked into FNOL's four document types.
    """
    fields_map       = pii_fields_by_doctype if pii_fields_by_doctype is not None else PII_FIELDS_BY_DOCTYPE
    pii_field_keys   = fields_map.get(doc_type, set())
    llm_safe         = {}
    pii_store_fields = {}
    for key, value in extracted.items():
        if key in pii_field_keys:
            pii_store_fields[key] = value
        else:
            llm_safe[key] = value
    return llm_safe, pii_store_fields


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════

def redact_document(
    pdf_path: str,
    entities_to_redact: Optional[set] = None,
) -> RedactionResult:
    """
    Full PII redaction pipeline for a multi-page PDF.
    Steps: OCR → Normalise → Detect → Merge → Anonymise → Evaluate → Result

    Args:
        pdf_path: path to the PDF to process.
        entities_to_redact: which entity types to actually anonymise in
            anonymised_pages. Defaults to None, meaning "redact everything
            in PII_ENTITIES" — the original behaviour, fully backward
            compatible with existing callers.

            Entities NOT in this set are still detected and fully reported
            (pii_store, entities_by_type, entity_lines, page.detections)
            for complete audit visibility — they just aren't masked in the
            output text. E.g. a caller processing internal documents where
            addresses don't need to be hidden but Aadhaar/PAN numbers do
            could pass entities_to_redact={"IN_AADHAAR", "IN_PAN", ...}.

            This is the main lever for using this module outside the FNOL
            pipeline it was built for — different consumers can choose a
            narrower or wider redaction scope without forking the code.
    """
    if not _initialised:
        initialise()

    redact_set = set(entities_to_redact) if entities_to_redact is not None else set(PII_ENTITIES)

    try:
        ocr_results = _run_ocr(pdf_path)
        # NOTE: _normalise() (title-casing) is used ONLY to improve BERT's
        # NER accuracy on a throwaway copy of the text. It must NEVER
        # become the canonical page_text — .title() mangles all-caps
        # codes/acronyms that appear throughout these documents (e.g.
        # "LMV" -> "Lmv", "MCWG" -> "Mcwg", "NH-1" -> "Nh-1"), and
        # page_text here is what raw_text/anonymised_text are built from,
        # which is what every downstream classification/extraction LLM
        # call actually reads. Because .title() never changes string
        # length, offsets computed against the title-cased copy still
        # align character-for-character with the original OCR text, so
        # we can safely use bert_input only for _run_bert().
        bert_input  = _normalise(ocr_results)

        page_results: list[PageRedactionResult] = []
        pii_store:    dict[int, list[dict]]     = {}

        for page_num in sorted(ocr_results.keys()):
            page_text = ocr_results[page_num]

            if not page_text.strip():
                page_results.append(PageRedactionResult(
                    page_num=page_num, raw_text="", anonymised_text="",
                    detections=[], spii_found=[],
                    char_count_raw=0, char_count_anon=0,
                ))
                pii_store[page_num] = []
                continue

            bert_dets   = _run_bert(bert_input[page_num])
            gliner_dets = _run_gliner(page_text)
            regex_dets  = _run_regex(page_text)
            merged      = _merge_detections(bert_dets, gliner_dets, regex_dets)
            merged      = _snap_to_word_boundaries(page_text, merged)
            # Snapping can expand previously-disjoint fragments into the same
            # or overlapping word (e.g. two sub-word fragments both expanding
            # to the full word "Pantoprazole") — dedup again post-snap.
            merged      = _merge_detections(merged, [], [])
            merged      = _drop_dosage_context_false_positives(page_text, merged)
            merged      = _drop_org_context_false_positives(page_text, merged)

            # merged (ALL detections) is kept in full for pii_store/audit;
            # only the caller-selected subset is actually masked in the
            # text the LLM (or anyone else) will see.
            to_anonymise    = [d for d in merged if d["entity_type"] in redact_set]
            anonymised_text = _anonymise_page(page_text, to_anonymise)
            spii_on_page    = [
                d["entity_type"] for d in merged
                if d["entity_type"] in SPII_ENTITIES
            ]

            page_results.append(PageRedactionResult(
                page_num=page_num,
                raw_text=page_text,
                anonymised_text=anonymised_text,
                detections=merged,
                spii_found=spii_on_page,
                char_count_raw=len(page_text),
                char_count_anon=len(anonymised_text),
            ))
            pii_store[page_num] = merged

        confidence_score, confidence_label, proceed, proceed_status, \
            confidence_components = _evaluate_confidence(page_results)

        all_detections   = [d for p in page_results for d in p.detections]
        spii_types_found = list({
            d["entity_type"] for d in all_detections
            if d["entity_type"] in SPII_ENTITIES
        })
        entities_by_type: dict[str, int] = {}
        for d in all_detections:
            entities_by_type[d["entity_type"]] = \
                entities_by_type.get(d["entity_type"], 0) + 1

        engines_used = list({d.get("engine", "unknown") for d in all_detections})

        return RedactionResult(
            pages=page_results,
            anonymised_pages=[p.anonymised_text for p in page_results],
            pii_store=pii_store,
            confidence_score=confidence_score,
            confidence_label=confidence_label,
            proceed=proceed,
            proceed_status=proceed_status,
            total_entities=len(all_detections),
            spii_entities_found=spii_types_found,
            entities_by_type=entities_by_type,
            engines_used=engines_used,
            confidence_components=confidence_components,
            entities_to_redact=sorted(redact_set),
        )

    except Exception as e:
        logger.error(f"❌ redact_document failed: {e}", exc_info=True)
        return RedactionResult(
            pages=[], anonymised_pages=[], pii_store={},
            confidence_score=0.0, confidence_label="LOW",
            proceed=False, proceed_status="HOLD",
            total_entities=0, spii_entities_found=[],
            entities_by_type={}, engines_used=[],
            error=str(e),
        )


# ══════════════════════════════════════════════════════════════════════════
# SIDE-BY-SIDE REDACTION REPORT (downloadable audit trail, PDF)
# ══════════════════════════════════════════════════════════════════════════

def export_side_by_side_pdf(
    result: RedactionResult,
    report: dict,
    output_path: str,
) -> str:
    """
    Build a PDF with:
      1. Privacy Protection Report summary (confidence, engines, entity breakdown)
      2. Per-page side-by-side comparison: original text vs anonymised text
      3. Per-page entity detection table (engine, type, score, value, SPII flag)

    Justification: a properly laid-out table is far more readable than a
    plain-text dump for a page-by-page BEFORE/AFTER comparison — colour-
    coded columns and grid lines make it obvious what changed at a glance.

    Note on icons: this PDF deliberately uses plain-text labels ("SPII"/
    "PII", "PROCEED"/"WARN"/"HOLD") instead of the emoji already present in
    `report` (✅⚠️🔴🟡) — ReportLab's core Helvetica/Courier fonts don't
    include emoji glyphs, so they would render as blank boxes in the PDF.
    The Streamlit UI can safely use the emoji versions since browsers
    render them natively.

    reportlab is imported here, not at module level, so importing
    pii_redactor.py doesn't hard-fail if reportlab isn't installed —
    only calling this specific function requires it.

    Args:
        result:      RedactionResult from redact_document() (post
                     finalize_redaction_confidence(), for accurate scoring)
        report:      dict from build_redaction_report()
        output_path: where to write the PDF

    Returns:
        output_path, for convenience chaining.
    """
    from xml.sax.saxutils import escape
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    styles = getSampleStyleSheet()
    mono = ParagraphStyle(
        "mono", parent=styles["Normal"], fontName="Courier", fontSize=8, leading=10,
    )
    cell_header = ParagraphStyle(
        "cell_header", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=9, textColor=colors.white,
    )

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
    )
    story = []

    # ── 1. Summary ───────────────────────────────────────────────────────
    story.append(Paragraph("PII Redaction — Side-by-Side Comparison Report", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"<b>Status:</b> {report['proceed_status']} &nbsp;&nbsp; "
        f"<b>Confidence:</b> {int(report['confidence_score'] * 100)}% "
        f"({report['confidence_label']})",
        styles["Normal"],
    ))
    story.append(Paragraph(
        f"<b>Engines used:</b> {', '.join(report['engines_used']) or 'N/A'}",
        styles["Normal"],
    ))
    if report.get("spii_leakage"):
        story.append(Paragraph(
            f"<b>SPII leakage detected in:</b> "
            f"{escape(', '.join(report['spii_leakage']))}",
            styles["Normal"],
        ))
    story.append(Paragraph(escape(report["status_message"]), styles["Normal"]))
    story.append(Spacer(1, 10))

    summary_rows = [[
        Paragraph("Risk", cell_header),
        Paragraph("Entity Type", cell_header),
        Paragraph("Count", cell_header),
    ]]
    for line in report["entity_lines"]:
        summary_rows.append([
            "SPII" if line["is_spii"] else "PII",
            line["label"],
            str(line["count"]),
        ])
    if len(summary_rows) == 1:
        summary_rows.append(["-", "No entities detected", "0"])

    summary_table = Table(summary_rows, colWidths=[3 * cm, 8 * cm, 3 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(summary_table)
    story.append(PageBreak())

    # ── 2. Per-page side-by-side + detections ───────────────────────────
    for page in result.pages:
        story.append(Paragraph(f"Page {page.page_num}", styles["Heading2"]))
        story.append(Paragraph(
            f"Characters: {page.char_count_raw} \u2192 {page.char_count_anon} "
            f"(anonymised) &nbsp;&nbsp; "
            f"SPII found: {', '.join(page.spii_found) if page.spii_found else 'None'}",
            styles["Normal"],
        ))
        story.append(Spacer(1, 6))

        # Chunk into fixed-size line blocks — a single table ROW can't be
        # split across pages if its cell is taller than one page, so long
        # OCR pages (80-100+ lines) must be broken into several smaller
        # rows instead of one giant cell. Each block becomes its own row;
        # ReportLab paginates normally BETWEEN rows.
        LINES_PER_BLOCK = 25
        orig_lines = (page.raw_text or "").split("\n")
        anon_lines = (page.anonymised_text or "").split("\n")
        max_lines  = max(len(orig_lines), len(anon_lines))
        orig_lines += [""] * (max_lines - len(orig_lines))
        anon_lines += [""] * (max_lines - len(anon_lines))

        header_row = [
            Paragraph("ORIGINAL (with PII)", cell_header),
            Paragraph("ANONYMISED (safe for LLM)", cell_header),
        ]
        body_rows = []
        for start in range(0, max_lines, LINES_PER_BLOCK):
            block_orig = "\n".join(orig_lines[start:start + LINES_PER_BLOCK])
            block_anon = "\n".join(anon_lines[start:start + LINES_PER_BLOCK])
            body_rows.append([
                Paragraph(escape(block_orig).replace("\n", "<br/>"), mono),
                Paragraph(escape(block_anon).replace("\n", "<br/>"), mono),
            ])
        if not body_rows:  # empty page — still show an empty row
            body_rows = [[Paragraph("", mono), Paragraph("", mono)]]

        side_by_side = Table([header_row] + body_rows, colWidths=[8.5 * cm, 8.5 * cm])
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
        for r in range(1, len(body_rows) + 1):
            style_cmds.append(("BACKGROUND", (0, r), (0, r), colors.HexColor("#fdecea")))  # original tint
            style_cmds.append(("BACKGROUND", (1, r), (1, r), colors.HexColor("#eafaf1")))  # anonymised tint
        side_by_side.setStyle(TableStyle(style_cmds))
        story.append(side_by_side)
        story.append(Spacer(1, 10))

        # Entity detection table for this page
        if page.detections:
            det_rows = [[
                Paragraph("Risk", cell_header), Paragraph("Type", cell_header),
                Paragraph("Engine", cell_header), Paragraph("Score", cell_header),
                Paragraph("Value", cell_header),
            ]]
            for d in page.detections:
                is_spii = d["entity_type"] in SPII_ENTITIES
                det_rows.append([
                    "SPII" if is_spii else "PII",
                    d["entity_type"],
                    d.get("engine", "unknown"),
                    f"{d.get('score', 0):.2f}",
                    escape(str(d["value"])[:40]),
                ])
            det_table = Table(det_rows, colWidths=[2 * cm, 4 * cm, 2 * cm, 2 * cm, 7 * cm])
            style_cmds = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
            for i, d in enumerate(page.detections, start=1):
                if d["entity_type"] in SPII_ENTITIES:
                    style_cmds.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fdecea")))
            det_table.setStyle(TableStyle(style_cmds))
            story.append(det_table)
        else:
            story.append(Paragraph("No entities detected on this page.", styles["Normal"]))

        story.append(PageBreak())

    doc.build(story)
    return output_path


def build_side_by_side_report(result: RedactionResult) -> str:
    """
    Plain-text fallback BEFORE/AFTER report — kept for programmatic/log use
    (e.g. quick inspection without opening a PDF viewer). The Streamlit UI
    uses export_side_by_side_pdf() instead for the user-facing download,
    since a formatted PDF table is far more readable than this ASCII dump.
    """
    lines = ["PII REDACTION REPORT — BEFORE / AFTER", "=" * 60, ""]

    if result.error:
        lines.append(f"Redaction failed: {result.error}")
        return "\n".join(lines)

    lines.append(f"Overall confidence: {int(result.confidence_score * 100)}% "
                 f"({result.confidence_label}) — {result.proceed_status}")
    lines.append(f"Engines used: {', '.join(result.engines_used) or 'N/A'}")
    if result.spii_leakage:
        lines.append(f"⚠ SPII leakage detected in: {', '.join(result.spii_leakage)}")
    lines.append("")

    for page in result.pages:
        lines.append(f"PAGE {page.page_num}")
        lines.append("-" * 60)
        if page.detections:
            lines.append("Entities detected on this page:")
            for d in page.detections:
                spii_tag = " [SPII]" if d["entity_type"] in SPII_ENTITIES else ""
                lines.append(
                    f"  - {d['entity_type']}{spii_tag} "
                    f"(engine: {d.get('engine', 'unknown')}, "
                    f"score: {d.get('score', 0):.2f})"
                )
        else:
            lines.append("No entities detected on this page.")
        lines.append("")
        lines.append("BEFORE (original):")
        lines.append(page.raw_text or "(no text)")
        lines.append("")
        lines.append("AFTER (sent to AI):")
        lines.append(page.anonymised_text or "(no text)")
        lines.append("")
        lines.append("=" * 60)
        lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# REDACTION REPORT
# ══════════════════════════════════════════════════════════════════════════

def build_redaction_report(
    result: RedactionResult,
    protected_fields: Optional[list] = None,
) -> dict:
    """
    Structured report for Streamlit UI and triage PDF.

    Justification: `protected_fields` must be the ACTUAL fields protected
    for THIS submission, not a static list of every field pii_redactor is
    *capable* of protecting across all possible doc types. At the point
    redact_document() runs, document classification hasn't happened yet,
    so the real doc-type mix for this submission is unknown — passing
    None here returns an empty list rather than speculating. Call this
    again after document classification + field splitting (in
    fnol_intake_agent.py) with the real per-submission field list.
    """
    entity_lines = [
        {
            "entity_type": et,
            "label":       et.replace("_", " ").title(),
            "count":       count,
            "is_spii":     et in SPII_ENTITIES,
            "flag":        "🔴 SPII" if et in SPII_ENTITIES else "🟡 PII ",
        }
        for et, count in sorted(result.entities_by_type.items())
    ]

    return {
        "confidence_score": result.confidence_score,
        "confidence_label": result.confidence_label,
        "proceed_status":   result.proceed_status,
        "proceed":          result.proceed,
        "total_pages":      len(result.pages),
        "total_entities":   result.total_entities,
        "spii_found":       result.spii_entities_found,
        "spii_leakage":     result.spii_leakage or [],
        "entities_to_redact": result.entities_to_redact or [],
        "entity_lines":     entity_lines,
        "engines_used":     result.engines_used,
        "fields_protected": sorted(protected_fields) if protected_fields else [],
        "error":            result.error,
        "status_icon": (
            "✅" if result.proceed_status == "PROCEED"
            else "⚠️" if result.proceed_status == "WARN"
            else "🔴"
        ),
        "status_message": (
            "Anonymisation confidence is high. Safe to proceed."
            if result.proceed_status == "PROCEED"
            else "Anonymisation confidence is moderate. Proceeding with caution — flagged for review."
            if result.proceed_status == "WARN"
            else "Anonymisation confidence is low. Manual review required before proceeding."
        ),
    }
