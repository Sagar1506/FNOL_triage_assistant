# -*- coding: utf-8 -*-
"""
fraud_llm_schema.py
====================
Pydantic schemas validating the JSON returned by the FRAUD_LLM_SYSTEM call
in fraud_signal_agent.py's _run_llm_signals().

Why this matters here specifically: today's DL_CLASS_MISMATCH /
NARRATIVE_COHERENCE bugs were both cases where a field silently contained
the wrong value with no error anywhere in the pipeline — a mismatched
string just flowed straight through. Validation doesn't prevent a WRONG
but well-formed answer (that's still a prompt/extraction problem), but it
DOES catch a MALFORMED one — missing keys, wrong types, an out-of-range
confidence value, an unexpected key the LLM invented — immediately, at the
call site, instead of three steps downstream where it's much harder to
trace back.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError


Confidence = Literal["HIGH", "MODERATE", "LOW"]


class NarrativeCoherenceResult(BaseModel):
    triggered: bool
    confidence: Confidence
    reason: str = Field(min_length=1)


class KeyStatusConcernResult(BaseModel):
    applicable: bool
    triggered: bool
    confidence: Confidence
    reason: str = Field(min_length=1)


class DamageSeverityResult(BaseModel):
    triggered: bool
    confidence: Confidence
    reason: str = Field(min_length=1)


class FraudLLMResponse(BaseModel):
    """Top-level shape returned by _call_llm_json(FRAUD_LLM_SYSTEM, ...)."""
    narrative_coherence:            NarrativeCoherenceResult
    key_status_concern:             KeyStatusConcernResult
    damage_severity_plausibility:   DamageSeverityResult


def validate_fraud_llm_response(raw: dict) -> Optional[FraudLLMResponse]:
    """
    Validate the raw dict returned by _call_llm_json() against the schema
    above. Returns the validated model on success, or None on failure
    (logging the exact validation error) so callers can fall back to the
    existing "treat as failed call" behaviour already in
    _run_llm_signals() (data_complete=False) rather than crashing.
    """
    try:
        return FraudLLMResponse.model_validate(raw)
    except ValidationError as e:
        import logging
        logging.getLogger(__name__).warning(
            f"Fraud LLM response failed schema validation: {e}\nRaw response: {raw}"
        )
        return None
