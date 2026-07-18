"""
fnol_intake_agent.py
====================
FNOL Document Intake Agent — LangGraph node.
Converted from fnol_intake_agent.ipynb.

Public API:
    run_intake_agent(pdf_path: str) -> FNOLIntakeState
    build_intake_graph()            -> CompiledGraph

Called by main.py (Streamlit UI).
"""

import os
import json
import base64
import re
import logging
import contextvars
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypedDict, Optional

import openpyxl
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

# ── PII Redaction — non-LLM ensemble anonymiser ───────────────────────────
# Justification: All documents are anonymised before any page text reaches
# the LLM. pii_store keeps real values locally for re-enrichment later.
from pii_redactor import (
    redact_document, split_fields, build_redaction_report,
    finalize_redaction_confidence, export_side_by_side_pdf,
    initialise as init_redactor,
)

# ── Load environment variables ─────────────────────────────────────────────
load_dotenv()

# ── LangSmith tracing (optional — only active when LANGCHAIN_TRACING_V2=true)
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn): return fn
        return decorator if args and callable(args[0]) else decorator

# ── OpenAI client ──────────────────────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL  = os.getenv("LLM_MODEL", "gpt-4o-mini")


# ══════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════

class FNOLIntakeState(TypedDict):
    """LangGraph state for the FNOL intake pipeline."""
    # Input
    pdf_path: str

    # Document identification
    pages_text: list[str]
    documents_identified: list[dict]

    # Extracted fields
    fnol_fields: dict
    extracted_summary: dict

    # Mandatory field completeness
    mandatory_fields_map: dict
    completeness_checks: list[dict]
    completeness_passed: bool

    # Cross-document consistency
    consistency_checks: list[dict]
    consistency_passed: bool
    inconsistencies: list[str]

    # Document gap check (expected vs received by incident type)
    document_gap_checks: list[dict]
    document_gap_passed: bool

    # LLM-generated plain-language summary for claimant
    claimant_summary: Optional[str]

    # ── PII Redaction ─────────────────────────────────────────────────────
    # pii_store: all detected PII values, stored locally, never sent to LLM
    # {
    #   "by_page": {page_num: [{entity_type, value, start, end, score, engine}]},
    #   "extracted_pii_fields": {doc_type: {field_name: value}}
    # }
    # Justification: keys are namespaced explicitly (by_page vs
    # extracted_pii_fields) rather than mixed int/string keys in one flat
    # dict, so the shape stays unambiguous once serialised to JSON
    # (checkpoints, API responses, logs) — JSON always coerces dict keys to
    # strings, so a flat dict with both int page numbers and a string key
    # like "extracted_pii_fields" would become impossible to tell apart.
    pii_store: dict

    # redaction_report: structured report for UI display and triage PDF
    # includes confidence_score, proceed_status, entity_lines, engines_used
    redaction_report: dict

    # redaction_side_by_side_pdf_path: path to a downloadable PDF with
    # page-by-page raw vs anonymised text, plus entities detected. Contains
    # real PII, so only ever offered as a download to the document owner.
    redaction_side_by_side_pdf_path: str

    # timing_report: real, measured per-step durations for THIS run —
    # {"total_duration_s": float, "steps": [{"step", "duration_s",
    # "start_offset_s"}, ...]}. Replaces the standalone latency report's
    # estimates with actual numbers, one per submission.
    timing_report: dict

    # timing_report_path: where timing_report was saved as JSON (same
    # temp dir as the uploaded PDF), for download / later aggregation.
    timing_report_path: str

    # Agent metadata
    agent_status: str
    error_message: Optional[str]


# ══════════════════════════════════════════════════════════════════════════
# PDF UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def extract_pages_text(pdf_path: str) -> list[str]:
    """
    Fallback text extractor using pypdf.
    Used only if pii_redactor is unavailable.
    Justification: pypdf is retained as a safety net — it does not anonymise,
    so this path should never be reached in normal operation.
    """
    reader = PdfReader(pdf_path)
    pages  = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text.strip())
    return pages


# ══════════════════════════════════════════════════════════════════════════
# MANDATORY FIELDS LOADER
# ══════════════════════════════════════════════════════════════════════════

def load_mandatory_fields(excel_path: str = None) -> dict:
    """
    Reads the 'All Fields' sheet from FNOL_Mandatory_Fields_by_DocType.xlsx.
    Path priority: argument → MANDATORY_FIELDS_EXCEL env var → default.
    Returns {doc_type: [{display_name, json_key, tier, purpose, condition}]}
    """
    path = (
        excel_path
        or os.getenv("MANDATORY_FIELDS_EXCEL")
        or "./guidelines_sop/FNOL_Mandatory_Fields_by_DocType.xlsx"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Mandatory fields Excel not found at: {path}\n"
            "Set MANDATORY_FIELDS_EXCEL in your .env file."
        )

    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb["All Fields"]

    fields_map = {}
    for row in ws.iter_rows(min_row=4, values_only=True):
        if not row[1]:
            continue
        doc_type  = str(row[1]).strip()
        display   = str(row[2]).strip() if row[2] else ""
        json_key  = str(row[3]).strip() if row[3] else ""
        tier      = str(row[4]).strip() if row[4] else ""
        purpose   = str(row[5]).strip() if row[5] else ""
        condition = str(row[6]).strip() if row[6] else ""

        fields_map.setdefault(doc_type, []).append({
            "display_name": display,
            "json_key":     json_key,
            "tier":         tier,
            "purpose":      purpose,
            "condition":    condition,
        })

    wb.close()
    return fields_map


# ══════════════════════════════════════════════════════════════════════════
# GPT-4o MINI HELPERS
# ══════════════════════════════════════════════════════════════════════════

@traceable(name="intake.call_gpt_json", run_type="llm")
def call_gpt_json(system_prompt: str, user_prompt: str) -> dict:
    """GPT-4o mini call with enforced JSON output. Returns parsed dict."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return json.loads(response.choices[0].message.content.strip())


# ══════════════════════════════════════════════════════════════════════════
# DOCUMENT IDENTIFICATION
# ══════════════════════════════════════════════════════════════════════════

DOC_ID_SYSTEM = """
You are a document classification expert for an Indian motor insurance company.
You will receive the text extracted from a single page of a PDF.
Your job is to identify what type of document this page contains.

Possible document types:
  FNOL_REGISTRATION_FORM   - First Notice of Loss registration form filled by the adjuster
  REGISTRATION_CERTIFICATE - Vehicle RC (Registration Certificate) from Parivahan / RTO
  DRIVING_LICENCE          - Driving Licence issued by RTO
  FIR                      - First Information Report from police
  VEHICLE_DAMAGE_PHOTO_LOG - Vehicle damage photograph log with table of photos/observations
  UNKNOWN                  - Cannot be determined from the text

Return a JSON object with exactly these keys:
{
  "doc_type": "<one of the types above>",
  "confidence": "HIGH" | "MODERATE" | "LOW",
  "reasoning": "<one sentence explaining why>",
  "key_signals": ["<signal 1>", "<signal 2>"]
}
"""


@traceable(name="intake.identify_document", run_type="chain")
def identify_document(page_text: str, page_num: int) -> dict:
    """Identify the document type on a given page using GPT-4o mini."""
    result = call_gpt_json(
        DOC_ID_SYSTEM,
        f"Page {page_num} text:\n\n{page_text[:3000]}"
    )
    result["page"] = page_num
    return result


# ══════════════════════════════════════════════════════════════════════════
# FIELD EXTRACTION PROMPTS
# ══════════════════════════════════════════════════════════════════════════

EXTRACT_FNOL_SYSTEM = """
You are extracting fields from an Indian motor insurance FNOL Registration Form.
Extract every field present. Use null for any field not found.
Return a JSON object with these keys (use exact key names):
fnol_date, fnol_registration_number, member_id, policy_number,
vehicle_registration_number, vehicle_make, vehicle_model, vehicle_year,
incident_date_time, incident_type, incident_location, damage_description,
third_party_involved, third_party_vehicle_reg, third_party_injury_reported,
fir_filed, fir_number, police_station, legal_notice_present,
claim_type_preferred, preferred_garage, adjuster_id,
hospital_name, mlc_number, claimant_name

Field name mapping — the form may use these labels; map them to the keys above:
  "Legal Notice or Tribunal Ref." or "Legal Notice or Tribunal Reference" -> legal_notice_present
  "Claimant Name" or "Policyholder Name" or "Proposer Name" -> claimant_name

IMPORTANT — PDF text extraction pattern:
The PDF text extraction produces every field as a single line with the label and value
concatenated together with no separator. This is how the entire form is structured.
Examples of how to parse these lines:
  "FNOL Date 02-May-2026"                    -> fnol_date = "02-May-2026"
  "Member ID MEM-1002"                        -> member_id = "MEM-1002"
  "Vehicle Make Honda"                        -> vehicle_make = "Honda"
  "Third-Party Involved Yes"                  -> third_party_involved = "Yes"
  "FIR Filed Yes"                             -> fir_filed = "Yes"
  "Legal Notice or Tribunal Ref. No"          -> legal_notice_present = "No"
  "Claim Type Preferred Cashless"             -> claim_type_preferred = "Cashless"
For every line, the value is everything that comes after the field label.
Apply this pattern consistently to extract all fields correctly.
"""

EXTRACT_RC_SYSTEM = """
You are extracting fields from an Indian vehicle Registration Certificate (RC).
Extract every field present. Use null for any field not found.
Return a JSON object with these keys:
registration_number, registration_date, owner_name,
vehicle_class, make, model, body_type, fuel_type,
engine_displacement_cc, engine_number, chassis_number, colour,
rto, registration_valid_upto, fitness_valid_upto,
insurance_policy_number, insurance_valid_upto,
hypothecation, puc_valid_upto
"""

EXTRACT_DL_SYSTEM = """
You are extracting fields from an Indian Driving Licence.
Extract every field present. Use null for any field not found.
Return a JSON object with these keys:
licence_number, name_masked, date_of_birth_masked, blood_group,
issue_date, valid_non_transport_upto, valid_transport_upto,
authorised_classes, issuing_rto, licence_status,
dl_valid_on_incident_date, endorsements

Field name mapping — the document may use these labels; map them to the keys above:
  "Endorsements / Disqualifications" or "Endorsements/Disqualifications" -> endorsements
"""

EXTRACT_FIR_SYSTEM = """
You are extracting fields from an Indian First Information Report (FIR).
Extract every field present. Use null for any field not found.
Return a JSON object with these keys:
fir_number, police_station, district, state,
date_of_fir, time_of_fir, date_of_incident, time_of_incident,
place_of_incident, complainant_vehicle_reg,
incident_description_summary, third_party_vehicle_reg,
injuries_reported, injuries_detail, mlc_number,
sections_invoked, fir_status, officer_in_charge
"""

EXTRACT_PHOTO_SYSTEM = """
You are extracting fields from a Vehicle Damage Photograph Log.
Extract every field present. Use null for any field not found.
Return a JSON object with these keys:
fnol_id, vehicle_reg, vehicle_description, date_of_photos,
adjuster_id, total_photographs, photo_quality,
vehicle_position_at_photo, adjuster_observation,
photos: [{number, filename, timestamp, gps, view, damage_observed}]
"""

EXTRACT_SYSTEM_MAP = {
    "FNOL_REGISTRATION_FORM":   EXTRACT_FNOL_SYSTEM,
    "REGISTRATION_CERTIFICATE": EXTRACT_RC_SYSTEM,
    "DRIVING_LICENCE":          EXTRACT_DL_SYSTEM,
    "FIR":                      EXTRACT_FIR_SYSTEM,
    "VEHICLE_DAMAGE_PHOTO_LOG": EXTRACT_PHOTO_SYSTEM,
}


@traceable(name="intake.extract_document_fields", run_type="chain")
def extract_document_fields(page_text: str, doc_type: str) -> dict:
    """Extract structured fields from a page. Returns {} for UNKNOWN doc_type."""
    system = EXTRACT_SYSTEM_MAP.get(doc_type)
    if not system:
        return {}
    return call_gpt_json(system, f"Extract all fields from this document text:\n\n{page_text[:4000]}")


# ══════════════════════════════════════════════════════════════════════════
# COMPLETENESS CHECK
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# DOCUMENT MATRIX LOADER
# ══════════════════════════════════════════════════════════════════════════

def load_document_matrix(json_path: str = None) -> dict:
    """
    Loads document_matrix.json.
    Path priority: argument → DOCUMENT_MATRIX_JSON env var → default.
    Returns {incident_type: {blocking:[...], mandatory:[...], recommended:[...]}}
    """
    path = (
        json_path
        or os.getenv("DOCUMENT_MATRIX_JSON")
        or "./guidelines_sop/document_matrix.json"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"document_matrix.json not found at: {path}\n"
            "Set DOCUMENT_MATRIX_JSON in your .env file."
        )
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════
# DOCUMENT GAP CHECK
# ══════════════════════════════════════════════════════════════════════════

def evaluate_doc_condition(condition: str, fnol_fields: dict) -> bool:
    """
    Evaluate whether a conditional document requirement is triggered.
    Conditions defined in document_matrix.json.
    """
    if not condition:
        return True   # no condition → always required
    c = condition.lower()
    if "hypothecated = true" in c:
        # Check FNOL form damage description or any field that mentions hypothecation
        # Note: at intake stage we read from fnol_fields only.
        # Full hypothecation check uses RC hypothecation field (available in extracted_summary).
        # We default to True (require the document) to be safe — better to flag than to miss.
        return True
    if "reimbursement = true" in c:
        pref = str(fnol_fields.get("claim_type_preferred", "")).lower()
        return "reimbursement" in pref
    if "tp_involved = true" in c:
        tp = str(fnol_fields.get("third_party_involved", "")).lower()
        return "yes" in tp or "true" in tp
    return True   # unknown condition → require by default


def run_document_gap_checks(
    documents_identified: list[dict],
    fnol_fields: dict,
    document_matrix: dict,
) -> tuple[list[dict], bool]:
    """
    Checks expected documents vs received documents for the FNOL incident type.

    Logic:
      1. Read incident_type from fnol_fields.
      2. Look up expected documents from document_matrix.
      3. Compare against doc_types present in documents_identified.
      4. Flag BLOCKING documents as MISSING_BLOCKING,
         MANDATORY documents as MISSING_MANDATORY.
      5. Conditional documents evaluated via evaluate_doc_condition().

    Returns:
      gap_checks    — list of per-document result dicts
      gap_passed    — True if no BLOCKING documents are missing
    """
    gap_checks  = []
    gap_passed  = True

    incident_type = str(fnol_fields.get("incident_type", "")).strip().upper()

    if not incident_type:
        gap_checks.append({
            "status":        "SKIPPED",
            "detail":        "incident_type not found in FNOL form — cannot determine expected documents.",
            "incident_type": None,
            "documents":     [],
        })
        return gap_checks, True   # cannot evaluate — do not fail

    # Normalise incident type — replace spaces with underscores, try partial match
    incident_type = incident_type.replace(" ", "_")
    matrix_key    = incident_type
    if matrix_key not in document_matrix:
        # Partial match — LLM may return slight variations
        for key in document_matrix:
            if key.replace("_","").upper() == incident_type.replace("_","").upper():
                matrix_key = key
                break
    if matrix_key not in document_matrix:
        gap_checks.append({
            "status":        "SKIPPED",
            "detail":        f"incident_type '{incident_type}' not found in document_matrix.json.",
            "incident_type": incident_type,
            "documents":     [],
        })
        return gap_checks, True

    expected = document_matrix[matrix_key]

    # Build set of received doc_types — normalised to uppercase, stripped
    # Excludes FNOL form (always present) and UNKNOWN pages
    received_types = {
        str(d["doc_type"]).strip().upper()
        for d in documents_identified
        if d.get("doc_type") not in ("UNKNOWN", None, "FNOL_REGISTRATION_FORM")
        and d.get("doc_type")
    }

    # Also normalise the doc_type we compare against
    def _is_received(dt):
        return str(dt).strip().upper() in received_types

    results = []
    has_blocking_missing = False

    for criticality in ("blocking", "mandatory", "recommended"):
        for doc_def in expected.get(criticality, []):
            doc_type       = doc_def["doc_type"]
            document_name  = doc_def["document"]
            condition      = doc_def["condition"]
            collection_by  = doc_def["collection_by"]

            # Evaluate condition
            required = evaluate_doc_condition(condition, fnol_fields)

            received = _is_received(doc_type)

            if received:
                status = "RECEIVED"
            elif not required:
                status = "NOT_REQUIRED"
            elif criticality == "blocking":
                status = "MISSING_BLOCKING"
                has_blocking_missing = True
            elif criticality == "mandatory":
                status = "MISSING_MANDATORY"
            else:
                status = "MISSING_RECOMMENDED"

            results.append({
                "document":      document_name,
                "doc_type":      doc_type,
                "criticality":   criticality.upper(),
                "condition":     condition,
                "required":      required,
                "received":      received,
                "status":        status,
                "collection_by": collection_by,
            })

    if has_blocking_missing:
        gap_passed = False

    gap_checks.append({
        "status":        "INCOMPLETE" if has_blocking_missing else "COMPLETE",
        "incident_type": incident_type,
        "documents":     results,
    })

    return gap_checks, gap_passed


# ── Fields where "None / No / Not applicable / Nil" IS a valid answer ─────
# These fields represent a state (no loan, no endorsements, no legal notice)
# rather than an absence of data.  The LLM correctly extracts "None" or "No"
# for these — that should count as PRESENT, not missing.
NONE_IS_VALID_FIELDS = {
    "hypothecation",          # RC   — "None" means no loan, which is valid
    "endorsements",           # DL   — "None on record" means clean licence
    "legal_notice_present",   # FNOL — "No" is a complete and valid answer
}


def is_value_present(val, json_key: str = "") -> bool:
    """
    Return True if a field value is non-null and non-empty.

    For fields in NONE_IS_VALID_FIELDS, "None / No / Nil / Not applicable"
    are treated as PRESENT because they represent a meaningful state,
    not a missing value.
    """
    if val is None:
        return False
    s = str(val).strip().lower()

    # Always absent — truly empty or unparsed placeholder
    if s in ("", "null"):
        return False

    # For fields where None/No/Not-applicable is a valid answer, accept them
    if json_key in NONE_IS_VALID_FIELDS:
        return True   # any non-null, non-empty string is valid

    # For all other fields, these strings indicate the LLM could not extract
    # a real value and should be treated as missing
    return s not in ("none", "n/a", "not applicable", "not available", "nil")


def evaluate_conditional(condition: str, extracted: dict) -> bool:
    """
    Evaluate whether a CONDITIONAL field's condition is met.
    Returns True if field IS required, False if it can be absent.

    Condition strings come directly from the Excel file display_name column,
    e.g. "Required when third_party_involved = Yes".
    Each branch checks the normalised condition string against extracted values.
    """
    if not condition:
        return False
    c = condition.lower()

    # Third party involved = Yes
    if "third_party_involved = yes" in c or "third party involved = yes" in c:
        val = str(extracted.get("third_party_involved", "")).lower().strip()
        return val.startswith("yes") or val == "true"

    # Third party injury reported = Yes
    if "third_party_injury_reported = yes" in c:
        val = str(extracted.get("third_party_injury_reported", "")).lower().strip()
        return val.startswith("yes") or val == "true"

    # FIR filed = Yes
    if "fir_filed = yes" in c or "fir filed = yes" in c:
        val = str(extracted.get("fir_filed", "")).lower().strip()
        return val.startswith("yes") or val == "true"

    # TP incident types
    if "tp incident type" in c or "third_party_claim_only" in c:
        it = str(extracted.get("incident_type", "")).upper()
        return any(t in it for t in {"ACCIDENT_WITH_THIRD_PARTY",
                                      "ACCIDENT_WITH_THIRD_PARTY_INJURY",
                                      "THIRD_PARTY_CLAIM_ONLY"})

    # Hospitalisation confirmed — requires both injury AND hospital name
    if "hospitalisation confirmed" in c:
        return (
            is_value_present(extracted.get("hospital_name")) and
            is_value_present(
                extracted.get("injuries_reported") or
                extracted.get("third_party_injury_reported", "")
            )
        )

    return True  # unknown condition — treat as required to be safe


def run_completeness_checks(
    documents_identified: list[dict],
    mandatory_fields_map: dict,
) -> tuple[list[dict], bool]:
    """
    For each identified document, checks whether all MANDATORY and
    applicable CONDITIONAL fields are present in the extracted output.
    Returns (completeness_checks, overall_passed).
    """
    completeness_checks = []
    overall_passed = True

    for doc in documents_identified:
        doc_type  = doc.get("doc_type")
        extracted = doc.get("fields", {})

        if doc_type in ("UNKNOWN", None):
            continue

        field_defs = mandatory_fields_map.get(doc_type, [])
        if not field_defs:
            completeness_checks.append({
                "doc_type":            doc_type,
                "page":                doc.get("page"),
                "status":              "SKIPPED",
                "fields_checked":      0,
                "fields_present":      0,
                "missing_mandatory":   [],
                "missing_conditional": [],
                "all_fields":          [],
            })
            continue

        missing_mandatory   = []
        missing_conditional = []
        present_fields      = []
        all_fields_detail   = []

        for fdef in field_defs:
            json_key  = fdef["json_key"]
            tier      = fdef["tier"]
            display   = fdef["display_name"]
            condition = fdef["condition"]

            val     = extracted.get(json_key)
            present = is_value_present(val, json_key)

            if tier == "MANDATORY":
                required = True
            elif tier == "CONDITIONAL":
                required = evaluate_conditional(condition, extracted)
            else:
                required = False  # OPTIONAL

            field_status = ("PRESENT" if present
                            else "MISSING" if required
                            else "ABSENT_NOT_REQUIRED")

            all_fields_detail.append({
                "display_name": display,
                "json_key":     json_key,
                "tier":         tier,
                "required":     required,
                "value":        val,
                "status":       field_status,
            })

            if present:
                present_fields.append(display)
            elif required:
                entry = {"display_name": display, "json_key": json_key}
                if tier == "MANDATORY":
                    missing_mandatory.append(entry)
                else:
                    missing_conditional.append({**entry, "condition": condition})

        if missing_mandatory:
            overall_passed = False

        if missing_mandatory:
            status = "INCOMPLETE_MANDATORY"
        elif missing_conditional:
            status = "INCOMPLETE_CONDITIONAL"
        else:
            status = "COMPLETE"

        completeness_checks.append({
            "doc_type":            doc_type,
            "page":                doc.get("page"),
            "status":              status,
            "fields_checked":      len(field_defs),
            "fields_present":      len(present_fields),
            "missing_mandatory":   missing_mandatory,
            "missing_conditional": missing_conditional,
            "all_fields":          all_fields_detail,
        })

    return completeness_checks, overall_passed


# ══════════════════════════════════════════════════════════════════════════
# CONSISTENCY CHECK
# ══════════════════════════════════════════════════════════════════════════

def normalise(value) -> str:
    """Uppercase, strip spaces for comparison."""
    if value is None:
        return None
    return str(value).upper().strip().replace(" ", "")


def normalise_loose(value) -> str:
    """Like normalise(), but also strips parenthetical annotations (e.g.
    'MH03RS4421 (Honda City)' -> 'MH03RS4421') before removing spaces.
    Used for fields where one source may add a bracketed description that
    another source doesn't carry."""
    if value is None:
        return None
    text = re.sub(r"\([^)]*\)", "", str(value))
    return text.upper().strip().replace(" ", "")


def values_contain_match(values: list) -> bool:
    """True if every normalised value is a substring of, or equal to, at
    least one other value in the list. Handles fields legitimately reported
    at different levels of detail across documents — e.g. vehicle make/model,
    where the FNOL form typically uses a shorthand name ('Maruti') and the
    RC uses the full manufacturer legal name and trim ('Maruti Suzuki India
    Ltd.'). Exact equality would fail on every such pair even though the
    values agree; this checks containment instead. Callers should pass
    already-normalised strings."""
    if len(values) < 2:
        return True
    for i, v1 in enumerate(values):
        if not v1:
            continue
        if not any(v1 in v2 or v2 in v1 for j, v2 in enumerate(values) if j != i and v2):
            return False
    return True


def run_consistency_checks(
    documents_identified: list[dict],
) -> tuple[list[dict], bool, list[str]]:
    """
    Cross-document field consistency checks.
    Returns (checks, all_passed, inconsistencies).
    """
    by_type = {
        doc["doc_type"]: doc.get("fields", {})
        for doc in documents_identified
        if doc.get("doc_type") not in ("UNKNOWN", None)
    }

    fnol  = by_type.get("FNOL_REGISTRATION_FORM", {})
    rc    = by_type.get("REGISTRATION_CERTIFICATE", {})
    dl    = by_type.get("DRIVING_LICENCE", {})
    fir   = by_type.get("FIR", {})
    photo = by_type.get("VEHICLE_DAMAGE_PHOTO_LOG", {})

    checks          = []
    inconsistencies = []

    def add_check(field_name: str, sources: list[tuple], description: str,
                  match_mode: str = "exact"):
        present = [
            (label, val) for label, val in sources
            if val not in (None, "", "null", "NULL")
        ]
        if len(present) < 2:
            checks.append({
                "field":           field_name,
                "description":     description,
                "sources_checked": [s[0] for s in sources],
                "values_found":    {s[0]: s[1] for s in present},
                "status":          "SKIPPED",
                "detail":          "Field present in fewer than 2 documents",
            })
            return

        if match_mode == "contains":
            # Loosely-normalised, substring-based match — for fields
            # legitimately reported at different levels of detail or with
            # extra bracketed annotation across documents (vehicle make/
            # model, TP vehicle registration). See values_contain_match().
            normalised = [normalise_loose(v) for _, v in present]
            all_match  = values_contain_match(normalised)
        else:
            normalised = [normalise(v) for _, v in present]
            all_match  = len(set(normalised)) == 1
        status     = "PASS" if all_match else "FAIL"
        detail     = ("All values match" if all_match
                      else f"Mismatch: {', '.join(f'{l}={v}' for l, v in present)}")

        if not all_match:
            inconsistencies.append(f"{field_name}: {detail}")

        checks.append({
            "field":           field_name,
            "description":     description,
            "sources_checked": [s[0] for s in present],
            "values_found":    {s[0]: s[1] for s in present},
            "status":          status,
            "detail":          detail,
        })

    # ── 10 checks ─────────────────────────────────────────────────────
    add_check("vehicle_registration_number",
              [("FNOL Form", fnol.get("vehicle_registration_number")),
               ("RC",        rc.get("registration_number")),
               ("FIR",       fir.get("complainant_vehicle_reg")),
               ("Photo Log", photo.get("vehicle_reg"))],
              "Vehicle registration number must be identical across all documents")

    add_check("vehicle_make",
              [("FNOL Form", fnol.get("vehicle_make")),
               ("RC",        rc.get("make"))],
              "Vehicle make must match between FNOL form and RC",
              match_mode="contains")

    add_check("vehicle_model",
              [("FNOL Form", fnol.get("vehicle_model")),
               ("RC",        rc.get("model"))],
              "Vehicle model must match between FNOL form and RC",
              match_mode="contains")

    add_check("fir_number",
              [("FNOL Form", fnol.get("fir_number")),
               ("FIR",       fir.get("fir_number"))],
              "FIR number on FNOL form must match the FIR document")

    def date_only(val):
        return str(val).split(" ")[0].split("T")[0] if val else None

    add_check("incident_date",
              [("FNOL Form", date_only(fnol.get("incident_date_time"))),
               ("FIR",       date_only(fir.get("date_of_incident")))],
              "Incident date on FNOL form must match the FIR")

    add_check("police_station",
              [("FNOL Form", fnol.get("police_station")),
               ("FIR",       fir.get("police_station"))],
              "Police station must match between FNOL form and FIR")

    add_check("policy_number",
              [("FNOL Form", fnol.get("policy_number")),
               ("RC",        rc.get("insurance_policy_number"))],
              "Policy number on FNOL form must match the RC")

    if fnol.get("third_party_vehicle_reg") and fir.get("third_party_vehicle_reg"):
        add_check("third_party_vehicle_reg",
                  [("FNOL Form", fnol.get("third_party_vehicle_reg")),
                   ("FIR",       fir.get("third_party_vehicle_reg"))],
                  "TP vehicle registration must match between FNOL form and FIR",
                  match_mode="contains")

    add_check("adjuster_id",
              [("FNOL Form", fnol.get("adjuster_id")),
               ("Photo Log", photo.get("adjuster_id"))],
              "Adjuster ID must match between FNOL form and Photo Log")

    if dl:
        dl_valid  = dl.get("dl_valid_on_incident_date", "")
        dl_status = dl.get("licence_status", "")
        passed    = ("YES" in str(dl_valid).upper() or
                     "ACTIVE" in str(dl_status).upper())
        status    = "PASS" if passed else "FAIL"
        detail    = (f"DL valid: {dl_valid} | Status: {dl_status}" if passed
                     else f"DL validity unclear — valid: {dl_valid}, status: {dl_status}")
        if not passed:
            inconsistencies.append(f"driving_licence_validity: {detail}")
        checks.append({
            "field":           "driving_licence_validity",
            "description":     "DL must be valid and effective on the incident date",
            "sources_checked": ["Driving Licence"],
            "values_found":    {"dl_valid_on_incident_date": dl_valid,
                                "licence_status": dl_status},
            "status":          status,
            "detail":          detail,
        })

    all_passed = all(c["status"] in ("PASS", "SKIPPED") for c in checks)
    return checks, all_passed, inconsistencies


# ══════════════════════════════════════════════════════════════════════════
# EXTRACTED SUMMARY BUILDER
# ══════════════════════════════════════════════════════════════════════════

def build_extracted_summary(
    fnol_fields: dict,
    documents_identified: list[dict],
) -> dict:
    """Flat summary of key fields collected across all documents."""
    get = lambda dt, k: next(
        (d["fields"].get(k) for d in documents_identified if d["doc_type"] == dt), None
    )
    return {
        "fnol_id":              fnol_fields.get("fnol_registration_number"),
        "member_id":            fnol_fields.get("member_id"),
        "policy_number":        fnol_fields.get("policy_number"),
        "vehicle_reg_fnol":     fnol_fields.get("vehicle_registration_number"),
        "vehicle_reg_rc":       get("REGISTRATION_CERTIFICATE", "registration_number"),
        "vehicle_reg_fir":      get("FIR", "complainant_vehicle_reg"),
        "vehicle_reg_photo":    get("VEHICLE_DAMAGE_PHOTO_LOG", "vehicle_reg"),
        "vehicle_make_fnol":    fnol_fields.get("vehicle_make"),
        "vehicle_make_rc":      get("REGISTRATION_CERTIFICATE", "make"),
        "vehicle_model_fnol":   fnol_fields.get("vehicle_model"),
        "vehicle_model_rc":     get("REGISTRATION_CERTIFICATE", "model"),
        "incident_type":        fnol_fields.get("incident_type"),
        "incident_date_fnol":   fnol_fields.get("incident_date_time"),
        "incident_date_fir":    get("FIR", "date_of_incident"),
        "fir_number_fnol":      fnol_fields.get("fir_number"),
        "fir_number_fir":       get("FIR", "fir_number"),
        "police_station_fnol":  fnol_fields.get("police_station"),
        "police_station_fir":   get("FIR", "police_station"),
        "policy_number_fnol":   fnol_fields.get("policy_number"),
        "policy_number_rc":     get("REGISTRATION_CERTIFICATE", "insurance_policy_number"),
        "adjuster_id_fnol":     fnol_fields.get("adjuster_id"),
        "adjuster_id_photo":    get("VEHICLE_DAMAGE_PHOTO_LOG", "adjuster_id"),
        "dl_licence_number":    get("DRIVING_LICENCE", "licence_number"),
        "dl_valid_on_incident": get("DRIVING_LICENCE", "dl_valid_on_incident_date"),
        "dl_status":            get("DRIVING_LICENCE", "licence_status"),
        "authorised_classes":   get("DRIVING_LICENCE", "authorised_classes"),
        "hypothecation_rc":     get("REGISTRATION_CERTIFICATE", "hypothecation"),
        "vehicle_class_rc":     get("REGISTRATION_CERTIFICATE", "vehicle_class"),
        "tp_vehicle_reg_fnol":  fnol_fields.get("third_party_vehicle_reg"),
        "tp_vehicle_reg_fir":   get("FIR", "third_party_vehicle_reg"),
        "injuries_fir":         get("FIR", "injuries_detail"),
        "adjuster_observation": get("VEHICLE_DAMAGE_PHOTO_LOG", "adjuster_observation"),
        "documents_found":      [d["doc_type"] for d in documents_identified
                                 if d["doc_type"] != "UNKNOWN"],
    }


# ══════════════════════════════════════════════════════════════════════════
# LANGGRAPH NODE
# ══════════════════════════════════════════════════════════════════════════

# Progress-reporting hook, set by run_intake_agent() right before invoking
# the graph, read inside document_intake_node(). A ContextVar rather than
# a plain module-level global or a key inside FNOLIntakeState, for two
# reasons:
#   1. StateGraph(FNOLIntakeState) builds its channels strictly from that
#      TypedDict's declared fields — an extra key like "_progress_callback"
#      stuffed into the input state gets silently dropped by LangGraph
#      before document_intake_node ever sees it, so the callback never
#      actually fires. (This is exactly what caused the UI to sit on one
#      static label the whole run — the callback was never invoked.)
#      A ContextVar bypasses the graph's state schema entirely.
#   2. A plain module-level global would leak between concurrent Streamlit
#      sessions if multiple users submit at the same time (each session
#      runs in its own thread) — a ContextVar is isolated per execution
#      context, so each session's callback stays private to that session's
#      thread, as long as set/get happen in the same thread (they do here:
#      run_intake_agent() sets it, then synchronously calls graph.invoke()
#      in that same thread; the ThreadPoolExecutor workers used for OCR/
#      classification/extraction never call the callback themselves).
_progress_callback_var: "contextvars.ContextVar" = contextvars.ContextVar(
    "progress_callback", default=None
)


def document_intake_node(state: FNOLIntakeState) -> FNOLIntakeState:
    """
    LangGraph node: FNOL Document Intake Agent.
    Steps:
      1.  Load mandatory fields from Excel.
      2.  Load document matrix from JSON.
      3.  PII Redaction — Google Vision OCR + ensemble anonymiser (non-LLM).
          Anonymised text replaces raw pages_text. pii_store kept locally.
      4.  Identify document type per page (GPT-4o mini — sees anonymised text).
      5.  Extract structured fields per document (GPT-4o mini — sees anonymised text).
      5a. Field split — route PII fields to pii_store, LLM-safe fields to triage.
      6.  Run mandatory field completeness check.
      7.  Run cross-document consistency checks.
      8.  Run document gap check (expected vs received by incident type).
      9.  Build extracted summary.
      10. Plain-language claimant summary (GPT-4o mini — sees validation results only).
    """
    # Progress reporting — see _progress_callback_var docstring above for
    # why this reads from a ContextVar rather than from `state`. Wrapped in
    # try/except so a broken UI callback can never take down the pipeline.
    _progress_callback = _progress_callback_var.get()

    # ── Per-run timing capture ──────────────────────────────────────────
    # Justification: the latency report we produced earlier was built from
    # estimates (no live credentials in that environment) — this captures
    # REAL durations for every actual run, replacing estimates with
    # measured numbers per submission rather than a one-off benchmark.
    # Reuses the same _report() call sites already in place for UI
    # progress, so no new instrumentation points are needed — one
    # mechanism serves both the live "which step is running" UI and the
    # saved timing report.
    _start_time = time.perf_counter()
    _step_marks: list[tuple[str, float]] = []

    def _report(step_label: str) -> None:
        _step_marks.append((step_label, time.perf_counter()))
        if _progress_callback:
            try:
                _progress_callback(step_label)
            except Exception:
                pass

    try:
        pdf_path = state["pdf_path"]

        _report("Loading claim configuration…")
        # Step 1: load mandatory fields
        mandatory_fields_map = load_mandatory_fields()
        state["mandatory_fields_map"] = mandatory_fields_map

        # Step 2: load document matrix
        document_matrix = load_document_matrix()

        # ── Step 3: PII Redaction ──────────────────────────────────────────
        # Justification: All PII is detected and replaced with <ENTITY_TYPE>
        # tokens before any page text reaches the LLM. The real values are
        # stored in pii_store — never sent to OpenAI or any external service.
        #
        # NOTE: redaction_report built here is PRELIMINARY — confidence_score
        # uses a placeholder spii_score since document classification/field
        # extraction hasn't run yet. It is finalized and overwritten below
        # (see finalize_redaction_confidence call) once extraction output is
        # available to run the real SPII leakage check against.
        _report("Scanning documents and protecting personal information…")
        redaction_result = redact_document(pdf_path)
        redaction_report = build_redaction_report(redaction_result)

        state["pii_store"] = {"by_page": redaction_result.pii_store}
        state["redaction_report"] = redaction_report

        # Use anonymised pages as pages_text for all downstream LLM steps
        # Justification: LLM sees <PERSON>, <IN_AADHAAR> etc. — enough to
        # identify document type and extract structural fields without PII.
        pages_text = redaction_result.anonymised_pages
        state["pages_text"] = pages_text

        _report(f"Identifying document types ({len(pages_text)} page"
                f"{'s' if len(pages_text) != 1 else ''})…")
        # Step 4: identify documents (LLM sees anonymised text only) — run
        # CONCURRENTLY across pages rather than one after another.
        # Justification: classifying page 2 doesn't depend on page 1's
        # result finishing first — nothing requires these calls to be
        # sequential. This changes WHEN the calls return, not how many are
        # made: same total LLM calls/tokens billed, just less wall-clock
        # time spent waiting for each one before starting the next.
        def _identify_one(i: int, page_text: str) -> tuple[int, dict]:
            id_result = identify_document(page_text, i)
            return i, {
                "page":        i,
                "doc_type":    id_result["doc_type"],
                "confidence":  id_result["confidence"],
                "reasoning":   id_result.get("reasoning", ""),
                "key_signals": id_result.get("key_signals", []),
                "fields":      {},
            }

        identify_results: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=min(len(pages_text), 8) or 1) as pool:
            futures = [
                pool.submit(_identify_one, i, page_text)
                for i, page_text in enumerate(pages_text, start=1)
            ]
            for future in as_completed(futures):
                i, doc_entry = future.result()
                identify_results[i] = doc_entry

        # as_completed() returns results out of order — restore page order,
        # since downstream logic (and the report) expects it.
        documents_identified = [identify_results[i] for i in sorted(identify_results)]

        _report("Extracting fields from your documents…")
        # Step 5: extract fields (LLM sees anonymised text only) — run
        # CONCURRENTLY across documents, same justification as step 4:
        # extracting one document's fields doesn't depend on another
        # document's extraction finishing first. Merging results back into
        # pii_store_fields / llm_safe_by_doctype happens afterward,
        # sequentially in the main thread, to avoid any shared-dict write
        # races between threads.
        # Step 5a: field split — PII fields → pii_store, safe fields → triage
        # Justification: Even after anonymising raw text, the LLM may infer
        # PII from structural cues. The field splitter deterministically
        # ensures those values never reach the triage LLM.
        fnol_fields      = {}
        pii_store_fields = {}   # accumulates PII fields across all doc types
        # llm_safe_by_doctype: accumulates the fields that actually reach the
        # LLM, per doc type. Used below to (a) run the real SPII leakage
        # check on exactly what the LLM saw, and (b) never used for
        # fields_protected — that comes from pii_store_fields (the PII
        # fields), not the LLM-safe ones.
        llm_safe_by_doctype: dict[str, dict] = {}

        known_docs = [
            (i, doc) for i, doc in enumerate(documents_identified)
            if doc["doc_type"] != "UNKNOWN"
        ]

        def _extract_one(i: int, doc: dict) -> tuple[int, str, dict, dict]:
            dt = doc["doc_type"]
            page_text = pages_text[doc["page"] - 1]
            all_fields = extract_document_fields(page_text, dt)

            # TEMP DIAGNOSTIC — remove once authorised_classes extraction is
            # confirmed correct. Prints exactly what page text the DL
            # extraction call received and what it returned, so we can tell
            # whether this is (a) a page-splitting/classification bug — the
            # DL page text is wrong/contaminated with RC content — or
            # (b) an extraction bug — the page text is correct but the LLM
            # still returned the wrong value.
            if dt == "DRIVING_LICENCE":
                print("=" * 70)
                print(f"[intake] DRIVING_LICENCE — page {doc['page']} text sent to extraction:")
                print(repr(page_text))
                print(f"[intake] DRIVING_LICENCE — fields returned by extraction:")
                print(all_fields)
                print("=" * 70)

            llm_safe, pii_fields = split_fields(all_fields, dt)
            return i, dt, llm_safe, pii_fields

        if known_docs:
            with ThreadPoolExecutor(max_workers=min(len(known_docs), 8) or 1) as pool:
                futures = [pool.submit(_extract_one, i, doc) for i, doc in known_docs]
                for future in as_completed(futures):
                    i, dt, llm_safe, pii_fields = future.result()

                    # Store only LLM-safe fields on the document for downstream use
                    documents_identified[i]["fields"] = llm_safe
                    llm_safe_by_doctype.setdefault(dt, {}).update(llm_safe)

                    # Accumulate PII fields into pii_store under doc_type key
                    if pii_fields:
                        pii_store_fields[dt] = pii_fields

                    if dt == "FNOL_REGISTRATION_FORM":
                        fnol_fields = llm_safe   # FNOL fields are already safe

        # ── Finalize redaction confidence — real SPII coverage check ───────
        # Justification: the preliminary confidence_score from redact_document()
        # used a placeholder spii_score (it couldn't know yet whether any real
        # SPII survived anonymisation). Now that field extraction has run, we
        # can scan exactly what the LLM saw (llm_safe_by_doctype) for
        # raw SPII-shaped values that should have been redacted tokens. This
        # recombines the already-computed residual/density/engine components
        # with a real spii_score — no OCR or detection work is redone.
        redaction_result = finalize_redaction_confidence(
            redaction_result, llm_safe_by_doctype
        )

        # ── fields_protected — the ACTUAL fields protected for THIS
        # submission, not a static list of every field pii_redactor is
        # capable of protecting across all possible doc types. Only fields
        # that were genuinely found (non-empty) and routed to pii_store for
        # the document types actually present in this upload are included.
        actual_protected_fields = sorted({
            k for fields in pii_store_fields.values()
            for k, v in fields.items()
            if v not in (None, "", "null", "NULL")
        })

        redaction_report = build_redaction_report(
            redaction_result, protected_fields=actual_protected_fields
        )
        state["redaction_report"] = redaction_report

        # ── Side-by-side PDF report ─────────────────────────────────────
        # Justification: a formatted PDF table is far more readable than a
        # plain-text dump for a page-by-page BEFORE/AFTER comparison. Saved
        # alongside the uploaded PDF (same temp dir) so it's cleaned up
        # together with it. Wrapped in try/except: if reportlab isn't
        # installed or PDF generation fails for any reason, the rest of
        # the intake pipeline must still succeed — the Privacy Protection
        # Report card itself doesn't depend on this file existing.
        try:
            side_by_side_pdf_path = os.path.join(
                os.path.dirname(pdf_path), "pii_redaction_report.pdf"
            )
            export_side_by_side_pdf(
                redaction_result, redaction_report, side_by_side_pdf_path
            )
            state["redaction_side_by_side_pdf_path"] = side_by_side_pdf_path
        except Exception as e:
            logger.warning(f"Side-by-side PDF generation failed: {e}")
            state["redaction_side_by_side_pdf_path"] = ""

        # Merge pii_store_fields into state pii_store
        # Justification: pii_store already has a "by_page" namespace for
        # per-page detection results (keyed by int page number). Field-level
        # PII from structured extraction goes in a separate "extracted_pii_fields"
        # namespace (keyed by doc_type string) so the two never collide under
        # the same dict — this keeps pii_store JSON-safe and unambiguous.
        state["pii_store"]["extracted_pii_fields"] = pii_store_fields

        state["documents_identified"] = documents_identified
        state["fnol_fields"]          = fnol_fields

        _report("Checking required documents are complete…")
        # Step 6: completeness
        comp_checks, comp_passed = run_completeness_checks(
            documents_identified, mandatory_fields_map
        )
        state["completeness_checks"] = comp_checks
        state["completeness_passed"] = comp_passed

        _report("Cross-checking details across your documents…")
        # Step 7: consistency
        cons_checks, cons_passed, inconsistencies = run_consistency_checks(
            documents_identified
        )
        state["consistency_checks"] = cons_checks
        state["consistency_passed"] = cons_passed
        state["inconsistencies"]    = inconsistencies

        # Step 8: document gap check
        gap_checks, gap_passed = run_document_gap_checks(
            documents_identified, fnol_fields, document_matrix
        )
        state["document_gap_checks"] = gap_checks
        state["document_gap_passed"] = gap_passed

        # Step 9: extracted summary
        state["extracted_summary"] = build_extracted_summary(
            fnol_fields, documents_identified
        )

        _report("Writing your submission summary…")
        # Step 10: plain-language claimant summary
        # Justification: _build_summary_context() sends only validation
        # results (completeness, consistency, gap check) — no raw fields,
        # no PII. This step was already safe before redaction integration.
        state["claimant_summary"] = generate_claimant_summary(state)

        # ── Build + save the per-run timing report ──────────────────────
        # Justification: turns the _step_marks captured above into a
        # per-step duration breakdown for THIS specific run — real
        # measured numbers, not the estimates from the standalone latency
        # report. duration_s for each step = time until the NEXT step's
        # _report() call fired; the last step's duration runs until now
        # (right after generate_claimant_summary returns).
        _end_time = time.perf_counter()
        timing_steps = []
        for idx, (label, ts) in enumerate(_step_marks):
            next_ts = _step_marks[idx + 1][1] if idx + 1 < len(_step_marks) else _end_time
            timing_steps.append({
                "step":           label,
                "duration_s":     round(next_ts - ts, 3),
                "start_offset_s": round(ts - _start_time, 3),
            })
        timing_report = {
            "total_duration_s": round(_end_time - _start_time, 3),
            "steps": timing_steps,
        }
        state["timing_report"] = timing_report

        # Saved alongside the uploaded PDF (same pattern as the side-by-side
        # PDF report) so each run's real numbers are preserved, not just
        # shown once and discarded. Wrapped in try/except: a save failure
        # here must never fail the whole intake pipeline.
        try:
            timing_report_path = os.path.join(
                os.path.dirname(pdf_path), "timing_report.json"
            )
            with open(timing_report_path, "w") as f:
                json.dump(timing_report, f, indent=2)
            state["timing_report_path"] = timing_report_path
        except Exception as e:
            logger.warning(f"Timing report save failed: {e}")
            state["timing_report_path"] = ""

        state["agent_status"]  = "COMPLETE"
        state["error_message"] = None

    except Exception as e:
        state["agent_status"]  = "ERROR"
        state["error_message"] = str(e)
        raise

    return state


# ══════════════════════════════════════════════════════════════════════════
# CLAIMANT SUMMARY GENERATOR
# ══════════════════════════════════════════════════════════════════════════

_DOC_LABELS = {
    "FNOL_REGISTRATION_FORM":   "FNOL Registration Form",
    "REGISTRATION_CERTIFICATE": "Vehicle Registration Certificate (RC)",
    "DRIVING_LICENCE":          "Driving Licence",
    "FIR":                      "First Information Report (FIR)",
    "VEHICLE_DAMAGE_PHOTO_LOG": "Vehicle Damage Photo Log",
    "KEY_SURRENDER":            "Key Surrender Confirmation",
    "UNKNOWN":                  "Unrecognised document",
}

_FIELD_LABELS = {
    "vehicle_registration_number": "vehicle registration number",
    "vehicle_make":                "vehicle make",
    "vehicle_model":               "vehicle model",
    "fir_number":                  "FIR number",
    "incident_date":               "incident date",
    "police_station":              "police station",
    "policy_number":               "policy number",
    "third_party_vehicle_reg":     "third-party vehicle registration",
    "adjuster_id":                 "adjuster ID",
    "driving_licence_validity":    "driving licence validity",
}


def _build_summary_context(state) -> str:
    """
    Assembles exactly what the three Streamlit output tables showed —
    nothing more, nothing less.

    Output 1 (Document Completeness) showed:
        - Document label   (e.g. "Driving Licence")
        - Status badge     (COMPLETE | INCOMPLETE_MANDATORY | INCOMPLETE_CONDITIONAL)
        - Fields present / total count
        - Names of missing mandatory fields (expander, when status != COMPLETE)
        - Names of missing conditional fields (expander, when applicable)

    Output 2 (Cross-Document Consistency) showed:
        - Only FAIL rows
        - Field label (e.g. "Vehicle Make")
        - Per-source value (e.g. "FNOL Form: Honda | RC: Honda Cars India Ltd.")

    Output 3 (Document Gap Check) showed:
        - Incident type
        - Document name
        - Status badge  (Submitted | Not submitted | Not applicable)
        - Criticality   (Expected at FNOL | Optional)

    Nothing from raw extracted fields, fnol_fields, pages_text,
    or documents_identified is included.
    """
    DOC_LABELS = {
        "FNOL_REGISTRATION_FORM":   "FNOL Registration Form",
        "REGISTRATION_CERTIFICATE": "Vehicle Registration Certificate (RC)",
        "DRIVING_LICENCE":          "Driving Licence",
        "FIR":                      "First Information Report (FIR)",
        "VEHICLE_DAMAGE_PHOTO_LOG": "Vehicle Damage Photo Log",
        "KEY_SURRENDER":            "Key Surrender Confirmation",
        "UNKNOWN":                  "Unrecognised document",
    }
    FIELD_LABELS = {
        "vehicle_registration_number": "Vehicle Registration Number",
        "vehicle_make":                "Vehicle Make",
        "vehicle_model":               "Vehicle Model",
        "fir_number":                  "FIR Number",
        "incident_date":               "Incident Date",
        "police_station":              "Police Station",
        "policy_number":               "Policy Number",
        "third_party_vehicle_reg":     "Third-Party Vehicle Registration",
        "adjuster_id":                 "Adjuster ID",
        "driving_licence_validity":    "Driving Licence Validity",
    }
    STATUS_DISPLAY = {
        "RECEIVED":            "Submitted",
        "MISSING_BLOCKING":    "Not submitted",
        "MISSING_MANDATORY":   "Not submitted",
        "MISSING_RECOMMENDED": "Not submitted",
        "NOT_REQUIRED":        "Not applicable",
    }
    CRIT_DISPLAY = {
        "BLOCKING":    "Expected at FNOL",
        "MANDATORY":   "Expected at FNOL",
        "RECOMMENDED": "Optional",
    }

    lines = []

    # ── OUTPUT 1: Document completeness table rows ─────────────────────────
    lines.append("OUTPUT 1 — DOCUMENTS FOUND AND FIELD COMPLETENESS")
    for c in state.get("completeness_checks", []):
        if c.get("doc_type") == "FNOL_REGISTRATION_FORM":
            continue   # never shown as a received supporting document
        label   = DOC_LABELS.get(c["doc_type"], c["doc_type"])
        status  = c["status"]
        present = c["fields_present"]
        total   = c["fields_checked"]
        mm      = c.get("missing_mandatory", [])
        mc      = c.get("missing_conditional", [])

        # Table row shown for every document
        lines.append(f"  Document: {label}")
        lines.append(f"  Status: {status.replace('_', ' ').title()}")
        lines.append(f"  Fields: {present}/{total} found")

        # Expander content — shown only when fields are missing
        if mm:
            lines.append(f"  Missing mandatory fields:")
            for f in mm:
                lines.append(f"    - {f['display_name']}")
        if mc:
            lines.append(f"  Missing conditional fields:")
            for f in mc:
                lines.append(f"    - {f['display_name']}")
        lines.append("")

    # ── OUTPUT 2: Consistency check — FAIL rows only ───────────────────────
    lines.append("OUTPUT 2 — CROSS-DOCUMENT CONSISTENCY")
    failures = [c for c in state.get("consistency_checks", [])
                if c["status"] == "FAIL"]
    if not failures:
        lines.append("  No mismatches found.")
    else:
        for c in failures:
            field_label = FIELD_LABELS.get(
                c["field"], c["field"].replace("_", " ").title()
            )
            lines.append(f"  Mismatch — {field_label}:")
            for src, val in c.get("values_found", {}).items():
                lines.append(f"    {src}: {val}")
    lines.append("")

    # ── OUTPUT 3: Document gap table rows ──────────────────────────────────
    # ONLY send documents that are NOT submitted — never send Submitted entries
    # to the LLM. This eliminates any risk of the LLM writing "not found"
    # bullets for documents that were actually received.
    lines.append("OUTPUT 3 — DOCUMENT GAP CHECK")
    has_missing = False
    for gap in state.get("document_gap_checks", []):
        if gap.get("status") == "SKIPPED":
            lines.append(f"  Skipped: {gap.get('detail', '')}")
            continue
        incident = (gap.get("incident_type") or "").replace("_", " ").title()
        missing_docs = [
            doc for doc in gap.get("documents", [])
            if doc["status"] in ("MISSING_BLOCKING", "MISSING_MANDATORY")
        ]
        if missing_docs:
            has_missing = True
            lines.append(f"  Incident type: {incident}")
            for doc in missing_docs:
                crit_label = CRIT_DISPLAY.get(doc["criticality"], doc["criticality"])
                lines.append(f"  - {doc['document']}: Not submitted ({crit_label})")
    if not has_missing:
        lines.append("  All expected documents were submitted.")
    lines.append("")

    return "\n".join(lines)


CLAIMANT_SUMMARY_SYSTEM = """
You are a helpful assistant for an Indian motor insurance company.
A claimant has uploaded claim documents. You have received the exact output
of three validation checks — Output 1, Output 2, and Output 3.

Your job: write a plain-language summary for the claimant based SOLELY on
what is stated in those three outputs. Do not add, infer, or invent anything.

STRICT RULES:
1. Use only the document names, field names, and values that appear in the
   input. Do not rename or paraphrase them.
2. Use simple conversational English. The claimant is not an insurance expert.
3. No technical codes. Replace "INCOMPLETE_MANDATORY" with plain language.
   Use "Not submitted" as-is where it appears in Output 3.
4. No decisive statements about claim acceptance or rejection.
5. No "you must" or "you cannot" — suggestive language only.
6. Begin with ONE warm opening sentence acknowledging the submission.
7. Write bullet points — one bullet per finding — following these rules:

   FROM OUTPUT 1:
   One bullet listing all documents that were found (status = Complete or
   Incomplete). Do not include documents marked as Skipped.
   IMPORTANT: Do NOT list the "FNOL Registration Form" — it is the claim
   submission form itself, not a supporting document. Only list supporting
   documents like RC, DL, FIR, Vehicle Damage Photo Log.
   If any document has missing mandatory fields, write one bullet per such
   document naming the document and listing each missing field exactly as
   it appears in the input.

   FROM OUTPUT 2:
   If there are no mismatches, do not write a bullet for Output 2.
   If there are mismatches, write ONE opening bullet then list each mismatch
   as a numbered sub-bullet showing the exact field name and exact values
   from each source. Use this pattern:
     "• We noticed a few differences across your submitted documents:
       1. [Field name] — [Source A] shows '[value]' while [Source B]
          shows '[value]'."

   FROM OUTPUT 3:
   Output 3 only lists documents that were NOT submitted.
   If Output 3 says "All expected documents were submitted", skip this bullet entirely.
   Otherwise write one bullet per missing document using this exact pattern:
     "[Document name] was not found in your submission. This is one of
      the documents typically expected for a [incident type] claim."

8. Write a bullet only if there is a finding to report. If Output 2 shows
   "No mismatches found", skip the Output 2 bullet entirely.
9. No closing line. No sign-off.
10. Total length: under 220 words.
"""



@traceable(name="intake.generate_claimant_summary", run_type="llm")
def generate_claimant_summary(state) -> str:
    """
    Calls GPT-4o mini to produce a plain-language claimant summary
    from all validation results in state.
    Called as Step 10 in document_intake_node.
    """
    context = _build_summary_context(state)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.0,  # zero — no creative freedom, exact terms only
            messages=[
                {"role": "system", "content": CLAIMANT_SUMMARY_SYSTEM},
                {"role": "user", "content": (
                    "Validation report for submitted documents:\n\n"
                    + context
                    + "\n\nWrite the claimant summary now."
                )},
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return (
            "Thank you for submitting your documents. "
            "Our team will review your submission and be in touch shortly."
        )


# ══════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════

def build_intake_graph():
    """
    Builds and compiles the single-node LangGraph graph.
    Additional nodes (policy_agent, history_agent, etc.) will be added here later.
    """
    graph = StateGraph(FNOLIntakeState)
    graph.add_node("document_intake", document_intake_node)
    graph.set_entry_point("document_intake")
    graph.add_edge("document_intake", END)
    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC API — called by main.py
# ══════════════════════════════════════════════════════════════════════════

_graph = None  # lazy-initialised singleton


@traceable(name="run_intake_agent", run_type="chain")
def run_intake_agent(pdf_path: str, progress_callback=None) -> FNOLIntakeState:
    """
    Run the FNOL Document Intake Agent on a PDF file.
    Called by main.py after the user uploads a file.

    Args:
        pdf_path: Absolute or relative path to the consolidated FNOL input PDF.
        progress_callback: Optional callable(step_label: str) -> None. Called
            at each major step boundary (redaction, classification,
            extraction, consistency checks, summary generation, ...) so the
            caller can show which step is currently running instead of one
            static "please wait" message. Never required — the pipeline
            runs identically with or without it, and a callback that raises
            is swallowed internally (see _report() in document_intake_node)
            so a UI bug can never break document processing.

    Returns:
        FNOLIntakeState with all results populated.
        Includes pii_store (local only) and redaction_report (for UI display).
    """
    global _graph
    if _graph is None:
        # Initialise PII redactor models once — spaCy + GLiNER take 5-15s
        # Justification: Loading at graph build time means the first user
        # request doesn't pay the cold-start penalty.
        init_redactor()
        _graph = build_intake_graph()

    initial_state: FNOLIntakeState = {
        "pdf_path":             pdf_path,
        "pages_text":           [],
        "documents_identified": [],
        "fnol_fields":          {},
        "extracted_summary":    {},
        "mandatory_fields_map": {},
        "completeness_checks":  [],
        "completeness_passed":  False,
        "consistency_checks":   [],
        "consistency_passed":   False,
        "inconsistencies":      [],
        "document_gap_checks":  [],
        "document_gap_passed":  False,
        "claimant_summary":     None,
        "pii_store":            {},    # populated by pii_redactor
        "redaction_report":     {},    # populated by pii_redactor
        "redaction_side_by_side_pdf_path": "",  # populated by pii_redactor
        "timing_report":       {},     # populated at end of document_intake_node
        "timing_report_path":  "",     # populated at end of document_intake_node
        "agent_status":         "PENDING",
        "error_message":        None,
    }

    # See _progress_callback_var's docstring (above document_intake_node)
    # for why this is a ContextVar rather than a key in initial_state.
    # reset() in the finally block so this session's callback never lingers
    # and gets accidentally reused by an unrelated later call in the same
    # thread (e.g. a Streamlit session thread being reused across reruns).
    token = _progress_callback_var.set(progress_callback)
    try:
        return _graph.invoke(initial_state)
    finally:
        _progress_callback_var.reset(token)

    return _graph.invoke(initial_state)
