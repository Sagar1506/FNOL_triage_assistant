"""
fraud_signal_agent.py
======================
Fraud Signal Detection Agent for the FNOL Triage pipeline.

Called as Step 5.5 inside fnol_triage_agent.run_triage_agent(), AFTER
check_waiting_period() and check_claims_history() so it can reuse their
already-computed dicts rather than re-deriving anything from Excel.

Public API:
    evaluate_fraud_signals(...) -> dict
        {
          "fraud_score":      int,
          "band":             "LOW" | "MEDIUM" | "HIGH",
          "confidence":       "HIGH" | "MODERATE" | "LOW",  # data completeness,
                               # per FNOL-GUIDE-001 Section 2 rule — not
                               # adjuster preference
          "signals":          list[dict],   # one entry per catalog signal
          "adjuster_summary": str,          # plain language, mandated phrasing
                               # from FNOL-GUIDE-001 Section 13.3 / 13.5,
                               # no "fraud" wording
        }

Design constraints carried over from the fnol_triage_agent.py conventions:
  - Never use the word "fraud" (or imply a fraud determination) in any
    LLM prompt or any user-facing string. Use "elevated review indicator(s)"
    language instead, matching FNOL-GUIDE-001 Section 13.
  - Deterministic checks are plain Python wherever the underlying data is
    categorical/structured. LLM judgment is used only for free-text
    consistency checks (narrative_coherence, key_status_concern,
    damage_severity_plausibility).
  - Signals reused from upstream (back_to_back_flag, repeat_claimant_flag,
    ncb_discrepancy, days_since_inception) are READ, never recomputed.
  - owner_name and any other PII field is read only from pii_store, and
    is NEVER included in any LLM prompt or in the returned "reason" text
    (only a boolean-safe description is returned).

Signal catalog (matches FNOL-GUIDE-001 Section 13.2 exactly — 9
deterministic + 3 LLM-judged = 12 total):
  1. BACK_TO_BACK_CLAIMS            major (3)  deterministic
  2. REPEAT_CLAIMANT                minor (1)  deterministic
  3. BIND_AND_GRIND                 major (3)  deterministic
  4. NCB_DISCREPANCY                minor (1)  deterministic
  5. HYPOTHECATION_INCONSISTENCY    minor (1)  deterministic
  6. DL_CLASS_MISMATCH              major (3)  deterministic
  7. OWNER_NAME_MISMATCH            major (3)  deterministic
  8. LATE_REPORTING                 minor (1)  deterministic
  9. CROSS_DOC_FIELD_MISMATCH       minor (1)  deterministic
 10. NARRATIVE_COHERENCE            major (3)  LLM
 11. KEY_STATUS_CONCERN             minor (1)  LLM  (THEFT_COMPLETE only)
 12. DAMAGE_SEVERITY_PLAUSIBILITY   minor (1)  LLM

Score bands (FNOL-GUIDE-001 Section 13.3):
  LOW:    0 - 2
  MEDIUM: 3 - 5
  HIGH:   6 or more   (also adds ELEVATED_REVIEW_INDICATORS escalation
                        condition — see fnol_triage_agent.evaluate_escalation)

LLM model is configurable via LLM_MODEL in .env (default: gpt-4o-mini),
same as fnol_triage_agent.py.
"""

import os
import json
import logging
import re
from datetime import date
from typing import Optional
from dateutil.parser import parse as parse_date

from openai import OpenAI
from dotenv import load_dotenv

from fraud_llm_schema import validate_fraud_llm_response

load_dotenv()

logger = logging.getLogger(__name__)

# ── LangSmith tracing (optional — only active when LANGCHAIN_TRACING_V2=true)
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn): return fn
        return decorator if args and callable(args[0]) else decorator

# ── LLM client — lazy initialisation, same pattern as fnol_triage_agent.py
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
_client   = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

MAJOR_WEIGHT = 3
MINOR_WEIGHT = 1

BAND_LOW_MAX    = 2   # 0-2   -> LOW
BAND_MEDIUM_MAX = 5   # 3-5   -> MEDIUM
                        # 6+    -> HIGH

# Bind-and-grind: deliberately wider than the mandatory waiting_period_days
# (usually 30) — this catches claims filed shortly after inception even
# once they're past the formal waiting period, which check_waiting_period()
# does not flag on its own.
BIND_AND_GRIND_THRESHOLD_DAYS = 45

# Late reporting — same 7-day threshold as the existing
# LATE_FNOL_OVER_7_DAYS escalation condition. Intentional overlap: that
# condition drives general escalation, this drives fraud scoring — same
# pattern already accepted for back_to_back_flag / repeat_claimant_flag.
LATE_REPORTING_THRESHOLD_DAYS = 7

TOTAL_SIGNAL_COUNT = 12


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _safe_date(val) -> Optional[date]:
    """Parse a date value from various formats. Returns None on failure."""
    if val is None:
        return None
    try:
        return parse_date(str(val), dayfirst=True).date()
    except Exception:
        return None


def _signal(code, label, weight, triggered, reason, data_complete=True):
    """Build one signal entry in the standard shape.

    data_complete=False marks a signal that could NOT be meaningfully
    evaluated because required data was missing/unparseable — used to
    derive the overall confidence level (GUIDE Section 2: confidence is
    determined by data completeness, not adjuster preference). This is
    distinct from "not applicable" (e.g. key_status_concern for non-theft
    incidents), which stays data_complete=True since nothing was missing.
    """
    return {
        "code":      code,
        "label":     label,
        "weight":    weight if triggered else 0,
        "max_weight": weight,
        "triggered": triggered,
        "reason":    reason,
        "data_complete": data_complete,
    }


@traceable(name="fraud.call_llm_json", run_type="llm")
def _call_llm_json(system: str, user: str) -> dict:
    resp = _get_client().chat.completions.create(
        model=LLM_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return json.loads(resp.choices[0].message.content.strip())


def _split_tokens(val) -> set:
    """Split a delimited/annotated categorical value ('LMV|MCWG', 'LMV, MCWG',
    'Motor Car (LMV)') into a normalised uppercase token set. Extracts
    contiguous alphanumeric runs via regex rather than replacing a fixed
    separator list — this also strips parentheses, so 'Motor Car (LMV)'
    yields {'MOTOR','CAR','LMV'} and correctly overlaps with a DL's 'LMV'
    token instead of leaving it as the unmatchable '(LMV)'.
    Returns empty set for None/blank."""
    if not val:
        return set()
    return {t for t in re.findall(r"[A-Z0-9]+", str(val).upper()) if t}


# ══════════════════════════════════════════════════════════════════════════
# DETERMINISTIC SIGNALS 1-2 — reused directly from claims_history
# ══════════════════════════════════════════════════════════════════════════

def _check_back_to_back(claims_history: dict) -> dict:
    triggered = bool(claims_history.get("back_to_back_flag"))
    days      = claims_history.get("days_since_last_claim")
    reason = (
        f"Prior claim on this vehicle was filed {days} days before this "
        f"incident (threshold: 30 or fewer days). Reused from claims history "
        f"— not re-evaluated here."
        if triggered else
        "No prior claim within 30 days of this incident on this vehicle."
    )
    return _signal("BACK_TO_BACK_CLAIMS", "Back-to-Back Claims",
                    MAJOR_WEIGHT, triggered, reason)


def _check_repeat_claimant(claims_history: dict) -> dict:
    triggered = bool(claims_history.get("repeat_claimant_flag"))
    count     = claims_history.get("prior_claims_12m_count", 0)
    reason = (
        f"{count} claims filed across all policies for this member in the "
        f"preceding 12 months (threshold: 3 or more). Reused from claims "
        f"history — not re-evaluated here."
        if triggered else
        "Fewer than 3 claims for this member in the preceding 12 months."
    )
    return _signal("REPEAT_CLAIMANT", "Repeat Claimant",
                    MINOR_WEIGHT, triggered, reason)


# ══════════════════════════════════════════════════════════════════════════
# DETERMINISTIC SIGNAL 3 — bind-and-grind, reuses waiting_period
# ══════════════════════════════════════════════════════════════════════════

def _check_bind_and_grind(waiting_period: dict) -> dict:
    days = waiting_period.get("days_since_inception")
    pre_inception = waiting_period.get("incident_pre_inception")

    if days is None:
        return _signal("BIND_AND_GRIND", "Claim Shortly After Policy Inception",
                        MAJOR_WEIGHT, False,
                        "Could not determine days since inception (date parse failure).",
                        data_complete=False)

    if pre_inception:
        # Already the most severe case (INCIDENT_BEFORE_POLICY_START in
        # escalation) — still worth surfacing here for score completeness.
        triggered = True
        reason = "Incident date is before policy inception."
    else:
        triggered = 0 <= days <= BIND_AND_GRIND_THRESHOLD_DAYS
        reason = (
            f"Incident occurred {days} days after policy inception "
            f"(threshold: {BIND_AND_GRIND_THRESHOLD_DAYS} days or fewer)."
            if triggered else
            f"Incident occurred {days} days after policy inception — outside "
            f"the {BIND_AND_GRIND_THRESHOLD_DAYS}-day window."
        )
    return _signal("BIND_AND_GRIND", "Claim Shortly After Policy Inception",
                    MAJOR_WEIGHT, triggered, reason)


# ══════════════════════════════════════════════════════════════════════════
# DETERMINISTIC SIGNAL 4 — NCB discrepancy, reused from claims_history
# ══════════════════════════════════════════════════════════════════════════

def _check_ncb_discrepancy(claims_history: dict) -> dict:
    note = claims_history.get("ncb_discrepancy")
    triggered = note is not None
    reason = note or "No NCB discrepancy against claims history."
    return _signal("NCB_DISCREPANCY", "No-Claim Bonus Discrepancy",
                    MINOR_WEIGHT, triggered, reason)


# ══════════════════════════════════════════════════════════════════════════
# DETERMINISTIC SIGNAL 5 — hypothecation inconsistency
# RC-extracted value (extracted_summary) vs. policy register value
# ══════════════════════════════════════════════════════════════════════════

def _check_hypothecation(extracted_summary: dict, policy_row: dict) -> dict:
    rc_val = extracted_summary.get("hypothecation_rc")
    register_hypothecated = bool(policy_row.get("hypothecated"))

    if rc_val is None or str(rc_val).strip() == "":
        return _signal("HYPOTHECATION_INCONSISTENCY", "Hypothecation Data Mismatch",
                        MINOR_WEIGHT, False,
                        "RC hypothecation field not available for comparison.",
                        data_complete=False)

    rc_says_financed = str(rc_val).strip().upper() not in ("NONE", "NIL", "NA", "N/A", "")
    triggered = rc_says_financed != register_hypothecated
    reason = (
        f"RC hypothecation field ('{rc_val}') does not match policy register "
        f"hypothecated flag ({register_hypothecated})."
        if triggered else
        "RC hypothecation field is consistent with the policy register."
    )
    return _signal("HYPOTHECATION_INCONSISTENCY", "Hypothecation Data Mismatch",
                    MINOR_WEIGHT, triggered, reason)


# ══════════════════════════════════════════════════════════════════════════
# DETERMINISTIC SIGNAL 6 — DL authorised class vs. RC vehicle class
# ══════════════════════════════════════════════════════════════════════════

def _check_dl_class(extracted_summary: dict) -> dict:
    authorised = extracted_summary.get("authorised_classes")
    vehicle_class = extracted_summary.get("vehicle_class_rc")

    if not authorised or not vehicle_class:
        return _signal("DL_CLASS_MISMATCH", "Driving Licence Class Mismatch",
                        MAJOR_WEIGHT, False,
                        "DL authorised classes or RC vehicle class not available "
                        "for comparison.",
                        data_complete=False)

    auth_tokens  = _split_tokens(authorised)
    class_tokens = _split_tokens(vehicle_class)

    # Any overlap between the two token sets counts as a match — DL and RC
    # vehicle-class vocabularies aren't perfectly standardised across
    # documents, so exact equality is too strict.
    triggered = auth_tokens.isdisjoint(class_tokens)
    reason = (
        f"DL authorised classes ('{authorised}') do not appear to cover the "
        f"insured vehicle's class ('{vehicle_class}')."
        if triggered else
        f"DL authorised classes ('{authorised}') cover the insured vehicle's "
        f"class ('{vehicle_class}')."
    )
    return _signal("DL_CLASS_MISMATCH", "Driving Licence Class Mismatch",
                    MAJOR_WEIGHT, triggered, reason)


# ══════════════════════════════════════════════════════════════════════════
# DETERMINISTIC SIGNAL 7 — RC owner name mismatch
# Reads ONLY from pii_store (both owner_name and claimant_name). Never
# sent to any LLM. Reason text never includes the actual name values —
# boolean-safe description only.
#
# POC SCOPE NOTE: the simulated test documents (RC, DL, FIR) use literal
# masked placeholders (e.g. "[Masked — MEM-1001]") instead of real names,
# by design — these were never meant to carry real PII. This means the
# check below is code-complete and will activate correctly once real
# names flow through pii_store, but cannot be meaningfully exercised
# against this POC's current test document set. This was a deliberate
# decision (not a defect) — regenerating all 15 test documents with real
# fictional names would touch too many documents and datasets for the
# value gained at this stage. Revisit if/when this moves past POC.
# ══════════════════════════════════════════════════════════════════════════

def _check_owner_name(pii_store: dict) -> dict:
    extracted_pii = (pii_store or {}).get("extracted_pii_fields", {})
    rc_pii   = extracted_pii.get("REGISTRATION_CERTIFICATE", {})
    fnol_pii = extracted_pii.get("FNOL_REGISTRATION_FORM", {})

    owner_name    = rc_pii.get("owner_name")
    claimant_name = fnol_pii.get("claimant_name")

    if not owner_name or not claimant_name:
        return _signal("OWNER_NAME_MISMATCH", "Registered Owner Mismatch",
                        MAJOR_WEIGHT, False,
                        "Not evaluated — RC owner name and/or FNOL claimant "
                        "name is not available for comparison.",
                        data_complete=False)

    # Loose normalised comparison — real deployments should use a proper
    # name-matching library (handles initials, transliteration, etc.)
    # rather than exact string equality.
    def _norm(n):
        return "".join(str(n).upper().split())

    triggered = _norm(owner_name) != _norm(claimant_name)
    reason = (
        "RC registered owner name does not match the claimant name on file. "
        "Names themselves are withheld from this report — verify directly "
        "in the source documents."
        if triggered else
        "RC registered owner name matches the claimant name on file."
    )
    return _signal("OWNER_NAME_MISMATCH", "Registered Owner Mismatch",
                    MAJOR_WEIGHT, triggered, reason)


# ══════════════════════════════════════════════════════════════════════════
# DETERMINISTIC SIGNAL 8 — late reporting
# ══════════════════════════════════════════════════════════════════════════

def _check_late_reporting(fnol_fields: dict, fnol_received_at: str) -> dict:
    incident_date = _safe_date(fnol_fields.get("incident_date_time"))
    received_date = _safe_date(fnol_received_at)

    if not incident_date or not received_date:
        return _signal("LATE_REPORTING", "Late Reporting",
                        MINOR_WEIGHT, False,
                        "Could not parse incident date or FNOL received date.",
                        data_complete=False)

    days_late = (received_date - incident_date).days
    triggered = days_late > LATE_REPORTING_THRESHOLD_DAYS
    reason = (
        f"FNOL received {days_late} days after the incident date "
        f"(threshold: {LATE_REPORTING_THRESHOLD_DAYS} days)."
        if triggered else
        f"FNOL received {days_late} day(s) after the incident date — within "
        f"the normal reporting window."
    )
    return _signal("LATE_REPORTING", "Late Reporting",
                    MINOR_WEIGHT, triggered, reason)


# ══════════════════════════════════════════════════════════════════════════
# DETERMINISTIC SIGNAL 9 — cross-document field mismatches
# Reuses consistency_checks already computed by fnol_intake_agent.py —
# does not re-derive anything.
# ══════════════════════════════════════════════════════════════════════════

def _check_cross_doc_mismatch(consistency_checks: list) -> dict:
    consistency_checks = consistency_checks or []
    failed = [
        c for c in consistency_checks
        if str(c.get("status", "")).upper() not in ("PASS", "SKIPPED")
    ]
    triggered = len(failed) > 0
    if triggered:
        fields = ", ".join(c.get("field", "unknown") for c in failed)
        reason = f"{len(failed)} cross-document field mismatch(es) found: {fields}."
    else:
        reason = "No cross-document field mismatches found."
    return _signal("CROSS_DOC_FIELD_MISMATCH", "Cross-Document Field Mismatch",
                    MINOR_WEIGHT, triggered, reason)


# ══════════════════════════════════════════════════════════════════════════
# LLM SIGNALS 10-12 — single combined call
# Tightly scoped per the design discussion:
#   - narrative_coherence: ONLY flag genuine contradictions between the
#     claimant's own account and physical/documentary evidence already
#     surfaced elsewhere in this triage run. NEVER flag tone, brevity,
#     translated-English phrasing, or nervousness. If in doubt, do not flag.
#   - key_status_concern: evaluated ONLY for THEFT_COMPLETE incidents.
#   - damage_severity_plausibility: compares damage description against
#     adjuster observation and incident type, not a photo-forensics check.
# ══════════════════════════════════════════════════════════════════════════

FRAUD_LLM_SYSTEM = f"""
You are assisting a motor insurance triage system in India. You will receive
FNOL (First Notice of Loss) narrative details for ONE claim. Evaluate exactly
three things and return a JSON object with three keys:
  "narrative_coherence"
  "key_status_concern"
  "damage_severity_plausibility"

GENERAL RULES — apply to all three:
- Never use the word "fraud" or any synonym implying a fraud determination.
- Never make a determination — only report whether an indicator is present,
  with a short factual reason grounded in the specific text provided.
- If the available text is too sparse to judge either way, set
  triggered=false and confidence="LOW" — absence of evidence is not itself
  a finding.
- Model: {LLM_MODEL}

1. narrative_coherence (object: triggered, confidence, reason)
   ONLY flag a genuine contradiction — where the claimant's own account
   directly conflicts with itself, or with other data already provided
   in this same input (e.g. incident type, damage description, adjuster
   observation). Confidence: HIGH | MODERATE | LOW.
   DO NOT flag, even if present:
     - short, terse, or minimal answers
     - grammar, spelling, or translated-English phrasing
     - a claimant who sounds nervous, upset, or vague about minor details
     - simply lacking detail (that is a document-gap issue, not this)
   Example of what SHOULD be flagged: claimant states the vehicle was
   parked and stationary when hit, but the damage description or adjuster
   observation describes damage consistent with a moving, high-speed
   collision.
   Example of what should NOT be flagged: a short, plainly-worded incident
   description with no internal contradiction, even if it reads tersely.

2. key_status_concern (object: applicable, triggered, confidence, reason)
   Set applicable=false immediately (with triggered=false) unless
   incident_type is exactly THEFT_COMPLETE — this check has no meaning
   for any other incident type.
   For THEFT_COMPLETE only: flag only if the narrative describes a
   secure/attended location for the vehicle AND simultaneously indicates
   an unresolved discrepancy in key accounting (e.g. not all issued keys
   can be produced, with no explanation given).

3. damage_severity_plausibility (object: triggered, confidence, reason)
   Flag only if the damage description and the adjuster observation (if
   provided) describe meaningfully different severity or type of damage
   for the same incident, in a way that is not explained by the incident
   type. Do not flag minor wording differences between the two sources.
"""


def _run_llm_signals(fnol_fields: dict, extracted_summary: dict) -> dict:
    incident_type = str(fnol_fields.get("incident_type", "")).strip().upper().replace(" ", "_")

    user_prompt = f"""FNOL DETAILS:
  Incident type       : {incident_type}
  Incident date       : {fnol_fields.get('incident_date_time')}
  Incident location    : {fnol_fields.get('incident_location')}
  Damage description   : {fnol_fields.get('damage_description')}
  Third party involved : {fnol_fields.get('third_party_involved')}
  FIR filed            : {fnol_fields.get('fir_filed')}

ADJUSTER / PHOTO LOG OBSERVATION: {extracted_summary.get('adjuster_observation')}

Evaluate narrative_coherence, key_status_concern, and
damage_severity_plausibility as instructed."""

    llm_call_failed = False
    try:
        raw_result = _call_llm_json(FRAUD_LLM_SYSTEM, user_prompt)
        validated  = validate_fraud_llm_response(raw_result)
        if validated is None:
            # Malformed shape (missing key, wrong type, invalid confidence
            # value, etc.) — treated the same as a failed call so it flows
            # into data_complete=False / MODERATE-or-lower confidence below,
            # rather than silently defaulting to "not triggered" on
            # whatever partial dict came back.
            result = {}
            llm_call_failed = True
        else:
            result = validated.model_dump()
    except Exception as e:
        logger.warning(f"Fraud LLM signal call failed: {e}")
        result = {}
        llm_call_failed = True

    nc = result.get("narrative_coherence", {}) or {}
    ks = result.get("key_status_concern", {}) or {}
    ds = result.get("damage_severity_plausibility", {}) or {}

    # When the LLM call or schema validation failed, say so explicitly in
    # the persisted reason text — otherwise this is indistinguishable in
    # triage.json / the PDF from "the LLM checked and found nothing wrong,"
    # which is a materially different situation an adjuster should be able
    # to tell apart without having to check server logs.
    _llm_failure_note = (
        " [LLM check could not be completed for this claim — treat as "
        "NOT evaluated, not as confirmed clean. See server logs.]"
        if llm_call_failed else ""
    )

    narrative_signal = _signal(
        "NARRATIVE_COHERENCE", "Narrative Coherence",
        MAJOR_WEIGHT, bool(nc.get("triggered")),
        (nc.get("reason") or "No narrative contradiction identified.") + _llm_failure_note,
        data_complete=not llm_call_failed,
    )

    # Gate strictly in Python too — never trust the model's own
    # "applicable" flag as the sole gate for a theft-only check.
    key_status_applicable = incident_type == "THEFT_COMPLETE"
    key_status_signal = _signal(
        "KEY_STATUS_CONCERN", "Key Status Concern (Theft Claims Only)",
        MINOR_WEIGHT,
        key_status_applicable and bool(ks.get("triggered")),
        ((ks.get("reason") or "No key status concern identified.") + _llm_failure_note)
        if key_status_applicable else
        "Not applicable — incident type is not THEFT_COMPLETE.",
        # Not applicable ≠ incomplete data — only mark incomplete when the
        # check was actually applicable and the LLM call failed.
        data_complete=(not llm_call_failed) if key_status_applicable else True,
    )

    damage_signal = _signal(
        "DAMAGE_SEVERITY_PLAUSIBILITY", "Damage Severity Plausibility",
        MINOR_WEIGHT, bool(ds.get("triggered")),
        (ds.get("reason") or "No damage severity inconsistency identified.") + _llm_failure_note,
        data_complete=not llm_call_failed,
    )

    return {
        "narrative_coherence": narrative_signal,
        "key_status_concern":  key_status_signal,
        "damage_severity":     damage_signal,
    }


# ══════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════

def _compute_band(score: int) -> str:
    if score <= BAND_LOW_MAX:
        return "LOW"
    if score <= BAND_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def _compute_confidence(signals: list) -> str:
    """
    Confidence reflects data completeness (FNOL-GUIDE-001 Section 2 rule:
    'Confidence is determined by data completeness — not adjuster
    preference'), applied here to the fraud signal set specifically.

    HIGH:     all 12 signals had sufficient data to evaluate.
    MODERATE: 1-3 signals could not be evaluated due to missing/unparseable
              data (each counted once, regardless of MAJOR/MINOR weight).
    LOW:      more than 3 signals could not be evaluated.
    """
    incomplete = sum(1 for s in signals if not s.get("data_complete", True))
    if incomplete == 0:
        return "HIGH"
    if incomplete <= 3:
        return "MODERATE"
    return "LOW"


def _build_adjuster_summary(triggered_signals: list, band: str, confidence: str) -> str:
    """
    Plain-language summary built in Python (not LLM-generated) so the
    exact mandated phrasing from FNOL-GUIDE-001 Section 13.3 and the
    Section 13.5 language substitution table is always used verbatim,
    with zero risk of model drift.
    """
    n = len(triggered_signals)
    base = (
        f"{n} of {TOTAL_SIGNAL_COUNT} elevated-review indicators are "
        f"present — Confidence: {confidence}."
    )
    if band == "HIGH":
        # GUIDE Section 13.5 mandated phrase for "Refer this to the
        # police / deny this claim." substitution.
        action = " Recommend referral to SIU for further review. Adjuster to determine next steps."
    elif band == "MEDIUM":
        # GUIDE Section 13.3 MEDIUM band action, verbatim.
        action = " Review the indicator list before proceeding. Does not independently escalate."
    else:
        action = " No action required beyond standard triage."
    return base + action


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════

@traceable(name="fraud.evaluate_fraud_signals", run_type="chain")
def evaluate_fraud_signals(
    fnol_fields: dict,
    policy_row: dict,
    extracted_summary: dict,
    waiting_period: dict,
    claims_history: dict,
    consistency_checks: list,
    pii_store: dict,
    fnol_received_at: str,
) -> dict:
    """
    Evaluate all 12 fraud signal indicators for one FNOL and return the
    combined score, band, per-signal breakdown, and adjuster summary.

    Args:
        fnol_fields        : intake_state["fnol_fields"]
        policy_row          : policy_row returned by build_claim_snapshot()
        extracted_summary   : intake_state["extracted_summary"]
        waiting_period      : state["waiting_period"] (already computed)
        claims_history      : state["claims_history"] (already computed)
        consistency_checks  : intake_state["consistency_checks"]
        pii_store           : intake_state["pii_store"] — read-only, never
                               forwarded to any LLM call or included in
                               returned reason text. Supplies both the RC
                               owner_name and the FNOL claimant_name for
                               the owner-mismatch check.
        fnol_received_at    : same timestamp string passed to run_triage_agent()

    Returns:
        {
          "fraud_score":      int,
          "band":             "LOW" | "MEDIUM" | "HIGH",
          "signals":          list[dict],
          "adjuster_summary": str,
        }
    """
    signals = []

    # ── Deterministic signals ────────────────────────────────────────────
    signals.append(_check_back_to_back(claims_history))
    signals.append(_check_repeat_claimant(claims_history))
    signals.append(_check_bind_and_grind(waiting_period))
    signals.append(_check_ncb_discrepancy(claims_history))
    signals.append(_check_hypothecation(extracted_summary, policy_row))
    signals.append(_check_dl_class(extracted_summary))
    signals.append(_check_owner_name(pii_store))
    signals.append(_check_late_reporting(fnol_fields, fnol_received_at))
    signals.append(_check_cross_doc_mismatch(consistency_checks))

    # ── LLM-judged signals (single combined call) ───────────────────────
    llm_results = _run_llm_signals(fnol_fields, extracted_summary)
    signals.append(llm_results["narrative_coherence"])
    signals.append(llm_results["key_status_concern"])
    signals.append(llm_results["damage_severity"])

    # ── Score + band ──────────────────────────────────────────────────────
    fraud_score = sum(s["weight"] for s in signals)
    band = _compute_band(fraud_score)
    triggered_signals = [s for s in signals if s["triggered"]]
    confidence = _compute_confidence(signals)
    adjuster_summary = _build_adjuster_summary(triggered_signals, band, confidence)

    return {
        "fraud_score":      fraud_score,
        "band":             band,
        "confidence":       confidence,
        "signals":          signals,
        "adjuster_summary": adjuster_summary,
    }
