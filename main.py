"""
main.py
=======
FNOL Assistance — Streamlit UI

Flow:
  1. User enters Member ID
  2. Matched against policy_register_sample.xlsx
  3. If match: show 3 options
       a. Register new FNOL  -> intake agent + registration
       b. Generate FNOL      -> "service coming soon"
       c. Query existing FNOL -> "service coming soon"
  4. Register new FNOL sub-flow:
       - Upload consolidated PDF
       - Run intake agent
       - Show LLM summary
       - Two choices: Submit revised | Proceed with registration
       - On Proceed: generate FNOL, save PDF, write register, show success

Run:
    streamlit run main.py
"""

import os
import json
import tempfile
import shutil
import logging
from datetime import datetime

# Persist WARNING+ log messages (including Pydantic validation failures in
# fraud_signal_agent.py) to a file, not just stderr. Terminal scrollback
# disappears on restart; this survives it, and is the same place to check
# on Render (or anywhere else) without needing live access to the console.
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("fnol_app.log"),
        logging.StreamHandler(),   # keep console output too
    ],
)

import streamlit as st
from dotenv import load_dotenv

from rag_agent import ask_rag
from fnol_intake_agent import run_intake_agent
from fnol_triage_agent import run_triage_agent
from fnol_register import look_up_member
from fnol_register import generate_fnol_number
from fnol_register import save_intake_document
from fnol_register import register_fnol
from fnol_register import look_up_fnol

load_dotenv()

st.set_page_config(
    page_title="FNOL Assistance",
    page_icon="📋",
    layout="wide",
)

def _header(text):
    st.markdown(
        f"<h1 style='color:#1F4E79;margin-bottom:0;'>{text}</h1>",
        unsafe_allow_html=True,
    )

def _subtext(text):
    st.markdown(
        f"<p style='color:#555;margin-top:4px;'>{text}</p>",
        unsafe_allow_html=True,
    )

# ── Session state ──────────────────────────────────────────────────────────
_DEFAULTS = {
    "member_id":       None,
    "member_data":     None,
    "page":            "home",
    "intake_result":   None,
    "tmp_pdf_path":    None,
    "last_filename":   None,
    "fnol_id":         None,
    "action":          None,
    "upload_counter":  0,     # incremented on every revise to force fresh uploader
    "triage_result": None,
    "triage_error":  None,
    "rag_history":   [],    # RAG conversation [{question, answer}]
    "rag_record":    None,  # currently queried FNOL record
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def _reset():
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════
# FRAUD SIGNALS BANNER — shared by page_done() and _render_fnol_summary()
# ══════════════════════════════════════════════════════════════════════════
def _render_fraud_banner(fraud: dict):
    """
    Renders the elevated-review-indicators banner for a completed triage.
    Shown for MEDIUM/HIGH bands only — a LOW band with no indicators
    triggered doesn't need to occupy screen space, same pattern as the
    escalation banner only rendering when esc_flag is true.

    This UI is adjuster-operated end to end (FNOL-SOP-001 Section 3), so
    fraud signal detail is shown consistently here, on the Query FNOL
    screen, and in the full triage PDF — see FNOL-GUIDE-001 Section 13.6.
    No "fraud" wording is used anywhere in this banner.
    """
    if not fraud:
        return
    band = str(fraud.get("band", "LOW")).upper()
    if band not in ("MEDIUM", "HIGH"):
        return

    band_color = "#8B0000" if band == "HIGH" else "#7B4F00"
    band_bg    = "#FCE4D6" if band == "HIGH" else "#FFF2CC"

    signal_lines = "".join(
        f"<li style='margin-bottom:4px;'>{s.get('reason', '')}</li>"
        for s in fraud.get("signals", []) if s.get("triggered")
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='background:{band_bg};border-left:4px solid {band_color};"
        f"border-radius:4px;padding:12px 16px;'>"
        f"<p style='color:{band_color};font-weight:600;margin:0 0 6px 0;'>"
        f"🔎 Elevated review indicators — Band: {band} "
        f"({fraud.get('fraud_score', 0)} pts)</p>"
        f"<ul style='margin:0;padding-left:18px;color:#3A3A3A;font-size:0.9rem;'>"
        f"{signal_lines}</ul>"
        f"<p style='color:{band_color};margin:8px 0 0 0;font-size:0.85rem;'>"
        f"{fraud.get('adjuster_summary', '')}</p></div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════
def page_home():
    _header("📋 FNOL Assistance")
    _subtext("Welcome to the FNOL Assistance portal. Please enter your Member ID to get started.")
    st.markdown("---")

    member_input = st.text_input(
        "Member ID",
        placeholder="e.g. MEM-1001",
        max_chars=20,
    ).strip().upper()

    if st.button("Continue", type="primary"):
        if not member_input:
            st.warning("Please enter your Member ID.")
            return
        with st.spinner("Looking up your details…"):
            try:
                data = look_up_member(member_input)
            except FileNotFoundError as e:
                st.warning(str(e))
                return
        if data is None:
            st.info(
                f"No records found for Member ID **{member_input}**. "
                "Please check the ID and try again."
            )
        else:
            st.session_state.member_id   = member_input
            st.session_state.member_data = data
            st.session_state.page        = "options"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PAGE: OPTIONS
# ══════════════════════════════════════════════════════════════════════════
def page_options():
    _header("📋 FNOL Assistance")
    n = len(st.session_state.member_data["policies"])
    st.markdown(
        f"<p style='color:#555;margin-top:4px;'>Welcome, "
        f"<b>{st.session_state.member_id}</b>. You have "
        f"<b>{n}</b> polic{'y' if n==1 else 'ies'} on record. "
        f"How can we help you today?</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            "<div style='background:#D5E8F0;border-radius:8px;padding:20px;"
            "text-align:center;min-height:110px;'>"
            "<span style='color:#1F4E79;font-size:1rem;font-weight:600;'>"
            "📝<br><br>Register new FNOL</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Select", key="btn_register", use_container_width=True):
            st.session_state.page = "register_fnol"
            st.rerun()

    with col2:
        st.markdown(
            "<div style='background:#FFF2CC;border-radius:8px;padding:20px;"
            "text-align:center;min-height:110px;'>"
            "<span style='color:#7B4F00;font-size:1rem;font-weight:600;'>"
            "🔍<br><br>Query existing FNOL</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Select", key="btn_query", use_container_width=True):
            st.session_state.page = "query_fnol"
            st.rerun()

    st.markdown("---")
    if st.button("← Change Member ID", type="secondary"):
        _reset()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PAGE: COMING SOON
# ══════════════════════════════════════════════════════════════════════════
def page_coming_soon():
    _header("📋 FNOL Assistance")
    st.markdown("---")
    st.info("🕐 This service will be available shortly. Please check back later.")
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("← Back", type="secondary"):
        st.session_state.page = "options"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PRIVACY PROTECTION REPORT
# ══════════════════════════════════════════════════════════════════════════

def _render_redaction_report(report: dict, side_by_side_pdf_path: str = "", fnol_id: str = ""):
    """
    Render the Privacy Protection Report card in the Streamlit UI.

    Justification: The claimant and adjuster need to see exactly what PII
    was detected and protected before the document was analysed by AI.
    This builds trust and provides a visible audit trail.

    Shows:
      - Anonymisation confidence score and proceed/warn/hold status
      - A downloadable page-by-page before/after PDF report (replaces the
        in-page entity table — the full detail belongs in a document the
        person can keep, not a UI element that disappears on refresh)
      - Which specific fields were never sent to any AI service
    """
    if not report:
        return

    proceed_status = report.get("proceed_status", "HOLD")
    score          = report.get("confidence_score", 0.0)
    status_icon    = report.get("status_icon", "🔴")
    status_message = report.get("status_message", "")
    entity_lines   = report.get("entity_lines", [])
    engines_used   = report.get("engines_used", [])
    error          = report.get("error")

    # ── Card colour by status ──────────────────────────────────────────
    STATUS_STYLE = {
        "PROCEED": ("background:#E2EFDA;border:1px solid #1E4D0F;", "#1E4D0F"),
        "WARN":    ("background:#FFF2CC;border:1px solid #7B4F00;", "#7B4F00"),
        "HOLD":    ("background:#FCE4D6;border:1px solid #8B0000;", "#8B0000"),
    }
    box_style, text_color = STATUS_STYLE.get(
        proceed_status,
        ("background:#D5E8F0;border:1px solid #1F4E79;", "#1F4E79")
    )

    # ── Header card ────────────────────────────────────────────────────
    st.markdown(
        f"<div style='{box_style}border-radius:8px;padding:14px 18px;"
        f"margin-bottom:12px;'>"
        f"<p style='color:{text_color};font-weight:700;font-size:0.95rem;"
        f"margin:0 0 4px 0;'>{status_icon} Privacy Protection Report</p>"
        f"<p style='color:#3A3A3A;font-size:0.88rem;margin:0 0 6px 0;'>"
        f"{status_message}</p>"
        f"<p style='color:#1F4E79;font-size:0.82rem;margin:0 0 6px 0;"
        f"font-weight:600;'>"
        f"🔐 Detected PII is redacted locally and protected before AI analysis "
        f"(OpenAI GPT-4o mini) — sensitive values never leave this system.</p>"
        f"<p style='color:#555;font-size:0.82rem;margin:0;'>"
        f"Anonymisation confidence: <b>{int(score * 100)}%</b> &nbsp;|&nbsp; "
        f"Engines used: <b>{', '.join(engines_used) if engines_used else 'N/A'}</b>"
        f"</p></div>",
        unsafe_allow_html=True,
    )

    # ── Error detail ───────────────────────────────────────────────────
    if error:
        st.markdown(
            f"<div style='background:#FCE4D6;border-left:4px solid #8B0000;"
            f"border-radius:4px;padding:10px 14px;margin-bottom:10px;'>"
            f"<p style='color:#8B0000;font-size:0.85rem;margin:0;'>"
            f"⚠️ Redaction error: {error}</p></div>",
            unsafe_allow_html=True,
        )
        return

    # ── Downloadable side-by-side PDF report (replaces in-page entity table) ──
    # Justification: contains the real before/after text, so it's offered
    # as a download for the document owner to keep, not rendered inline.
    # Read from disk (not an in-memory string) since export_side_by_side_pdf
    # writes a real PDF file — same pattern as the triage PDF download.
    if side_by_side_pdf_path and os.path.exists(side_by_side_pdf_path):
        with open(side_by_side_pdf_path, "rb") as f:
            st.download_button(
                label="📄  Download detailed PII redaction report (PDF)",
                data=f.read(),
                file_name=f"{fnol_id or 'fnol'}_pii_redaction_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    elif not entity_lines:
        st.markdown(
            "<p style='color:#888;font-size:0.85rem;margin:0 0 8px 0;'>"
            "No PII entities detected in this document.</p>",
            unsafe_allow_html=True,
        )

    # ── Fields never sent to AI ────────────────────────────────────────
    protected_fields = report.get("fields_protected", [])
    if protected_fields:
        fields_display = ", ".join(
            f.replace("_", " ").title() for f in sorted(protected_fields)
        )
        st.markdown(
            f"<p style='color:#555;font-size:0.82rem;margin:6px 0 0 0;'>"
            f"🛡️ <b>Fields never sent to AI:</b> {fields_display}</p>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════
# PAGE: REGISTER NEW FNOL
# ══════════════════════════════════════════════════════════════════════════
def page_register_fnol():
    _header("📋 FNOL Assistance")
    _subtext("Register New FNOL — please upload your claim documents to proceed.")
    st.markdown("---")

    action = st.session_state.action

    if action == "proceed":
        _do_register()
        return

    upload_label = (
        "Upload revised documents (PDF)"
        if action == "revise"
        else "Upload your claim documents (PDF — all documents in one file)"
    )

    # Key includes upload_counter so clicking "Upload revised documents"
    # forces Streamlit to mount a completely fresh uploader widget.
    uploader_key  = f"fnol_pdf_uploader_{st.session_state.upload_counter}"

    uploaded_file = st.file_uploader(
        upload_label,
        type=["pdf"],
        key=uploader_key,
        help="Please combine your FNOL form and all supporting documents into one PDF.",
    )

    if uploaded_file is not None:
        is_new = uploaded_file.name != st.session_state.last_filename

        if is_new or st.session_state.intake_result is None:
            st.session_state.last_filename = uploaded_file.name
            st.session_state.action        = None

            # Save to a named temp file that persists until registration
            tmp_dir  = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())

            # Clean up previous temp
            if st.session_state.tmp_pdf_path:
                try:
                    shutil.rmtree(
                        os.path.dirname(st.session_state.tmp_pdf_path),
                        ignore_errors=True,
                    )
                except Exception:
                    pass

            st.session_state.tmp_pdf_path = tmp_path

            # ── Live step-by-step progress ──────────────────────────────
            # Justification: run_intake_agent() runs 7+ distinct steps
            # (redaction, classification, extraction, consistency checks,
            # summary generation, ...) that can take 20-60+ seconds total.
            # A single static "please wait" spinner gives no sense of
            # progress or where time is going; st.status() lets
            # document_intake_node report which step is currently running
            # via progress_callback, so the person sees real movement
            # instead of one message the whole time.
            with st.status("Reviewing your documents…", expanded=True) as status:
                def _update_status(step_label: str) -> None:
                    status.update(label=step_label)
                    status.write(f"▸ {step_label}")

                try:
                    result = run_intake_agent(tmp_path, progress_callback=_update_status)
                    st.session_state.intake_result = result
                    status.update(
                        label="Review complete", state="complete", expanded=False
                    )
                except Exception as e:
                    status.update(
                        label="We encountered an issue reviewing your documents",
                        state="error", expanded=False,
                    )
                    st.warning(
                        f"We encountered an issue reviewing your documents: {e}. "
                        "Please try again."
                    )
                    st.session_state.intake_result = None

    result = st.session_state.intake_result

    if result is not None and result.get("agent_status") != "ERROR":

        # ── Privacy Protection Report ──────────────────────────────────
        # Justification: Show the user what PII was detected and protected
        # before any AI analysis. Displayed first — above everything else —
        # so the user sees it regardless of scroll position.
        redaction_report = result.get("redaction_report", {})
        side_by_side_pdf_path = result.get("redaction_side_by_side_pdf_path", "")
        # fnol_id doesn't exist yet at this stage (assigned at registration),
        # so use the uploaded filename (minus extension) for the download
        # filename instead.
        file_stub = os.path.splitext(st.session_state.last_filename or "fnol")[0]
        _render_redaction_report(redaction_report, side_by_side_pdf_path, file_stub)

        # ── Hold gate — block submission if confidence is too low ──────
        # Justification: If the anonymiser confidence is below 0.65, there
        # is meaningful risk that PII was not fully redacted. We do not
        # proceed to the LLM or registration in this case.
        if redaction_report.get("proceed_status") == "HOLD":
            st.markdown(
                "<div style='background:#FCE4D6;border:1px solid #8B0000;"
                "border-radius:8px;padding:16px 20px;margin:12px 0;'>"
                "<p style='color:#8B0000;font-weight:600;margin:0 0 8px 0;"
                "font-size:1rem;'>⛔ Submission on hold — manual review required</p>"
                "<p style='color:#3A3A3A;margin:0;font-size:0.88rem;line-height:1.5;'>"
                "Our privacy protection check was unable to confirm that all "
                "sensitive information in your document has been protected. "
                "For your security, this submission has been paused. "
                "Please contact our support team or try uploading a clearer "
                "copy of your documents.</p></div>",
                unsafe_allow_html=True,
            )
            if st.button(
                "📤  Upload revised documents",
                use_container_width=True,
                type="secondary",
                key="btn_revise_hold",
            ):
                st.session_state.intake_result   = None
                st.session_state.last_filename   = None
                st.session_state.action          = "revise"
                st.session_state.upload_counter += 1
                st.rerun()
            st.markdown("---")
            if st.button("← Back to options", type="secondary", key="btn_back_hold"):
                st.session_state.page          = "options"
                st.session_state.intake_result = None
                st.session_state.action        = None
                st.session_state.last_filename = None
                st.rerun()
            return   # stop here — nothing else rendered

        st.markdown("---")

        # ── Member ID check — FIRST, before anything else is shown ────
        fnol_member = (
            (result.get("fnol_fields") or {})
            .get("member_id", "")
            or ""
        ).strip().upper()
        ui_member = (st.session_state.member_id or "").strip().upper()
        member_match = (fnol_member == ui_member) or (fnol_member == "")

        if not member_match:
            st.markdown(
                f"<div style='background:#FCE4D6;border:1px solid #8B0000;"
                f"border-radius:8px;padding:16px 20px;margin-bottom:16px;'>"
                f"<p style='color:#8B0000;font-weight:600;margin:0 0 8px 0;"
                f"font-size:1rem;'>We are unable to proceed with this submission</p>"
                f"<p style='color:#3A3A3A;margin:0;font-size:0.93rem;line-height:1.5;'>"
                f"The Member ID found in your submitted FNOL form "
                f"(<b>{fnol_member}</b>) does not match the Member ID you "
                f"used to log in (<b>{ui_member}</b>). "
                f"For security reasons, we are unable to register an FNOL on "
                f"behalf of a different member. Kindly review your documents "
                f"and upload the correct set."
                f"</p></div>",
                unsafe_allow_html=True,
            )
            if st.button(
                "📤  Upload revised documents",
                use_container_width=True,
                type="secondary",
                key="btn_revise",
            ):
                st.session_state.intake_result   = None
                st.session_state.last_filename   = None
                st.session_state.action          = "revise"
                st.session_state.upload_counter += 1
                st.rerun()
            return   # stop here — nothing else rendered

        # ── Member ID matches — show file info, summary, action buttons ──

        # ── WARN banner — visible but not blocking ─────────────────────
        # Justification: WARN means confidence is moderate (0.65–0.84).
        # We allow the user to proceed but flag the submission for review
        # by the claims team. Logged in redaction_report in the triage PDF.
        if redaction_report.get("proceed_status") == "WARN":
            st.markdown(
                "<div style='background:#FFF2CC;border:1px solid #7B4F00;"
                "border-radius:8px;padding:12px 16px;margin-bottom:12px;'>"
                "<p style='color:#7B4F00;font-weight:600;margin:0 0 4px 0;"
                "font-size:0.88rem;'>⚠️ Privacy check — moderate confidence</p>"
                "<p style='color:#3A3A3A;margin:0;font-size:0.85rem;'>"
                "Anonymisation confidence is moderate. You may proceed, but "
                "this submission will be flagged for privacy review by the "
                "claims team before processing.</p></div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<p style='color:#888;font-size:0.85rem;'>"
            f"File reviewed: <b>{st.session_state.last_filename}</b></p>",
            unsafe_allow_html=True,
        )

        # ── Processing time breakdown ────────────────────────────────────
        # Justification: real, measured per-step durations for this run
        # (see fnol_intake_agent.py's timing_report) — replaces the
        # standalone latency report's estimates with actual numbers per
        # submission. Collapsed by default: useful for diagnosing where
        # time goes, not something the claimant needs to see by default.
        timing_report = result.get("timing_report") or {}
        if timing_report.get("steps"):
            with st.expander(
                f"⏱️ Processing time breakdown "
                f"({timing_report.get('total_duration_s', 0):.1f}s total)",
                expanded=False,
            ):
                rows_html = "".join(
                    f"<tr>"
                    f"<td style='padding:4px 10px;font-size:0.84rem;'>{s['step']}</td>"
                    f"<td style='padding:4px 10px;font-size:0.84rem;text-align:right;'>"
                    f"{s['duration_s']:.2f}s</td>"
                    f"</tr>"
                    for s in timing_report["steps"]
                )
                st.markdown(
                    f"<table style='width:100%;border-collapse:collapse;'>"
                    f"<thead><tr>"
                    f"<th style='text-align:left;padding:4px 10px;font-size:0.82rem;"
                    f"color:#888;border-bottom:1px solid #DDD;'>Step</th>"
                    f"<th style='text-align:right;padding:4px 10px;font-size:0.82rem;"
                    f"color:#888;border-bottom:1px solid #DDD;'>Duration</th>"
                    f"</tr></thead><tbody>{rows_html}</tbody></table>",
                    unsafe_allow_html=True,
                )
                timing_report_path = result.get("timing_report_path", "")
                if timing_report_path and os.path.exists(timing_report_path):
                    with open(timing_report_path, "rb") as f:
                        st.download_button(
                            label="Download timing report (JSON)",
                            data=f.read(),
                            file_name="fnol_timing_report.json",
                            mime="application/json",
                            key="timing_report_download",
                        )

        st.markdown("---")

        summary = result.get("claimant_summary") or (
            "Thank you for your submission. "
            "Our team will review your documents and be in touch shortly."
        )
        st.markdown(
            "<h3 style='color:#1F4E79;'>Summary of your submission</h3>",
            unsafe_allow_html=True,
        )
        _render_summary(summary)
        st.markdown("---")

        _subtext("How would you like to proceed?")
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            if st.button(
                "📤  Upload revised documents",
                use_container_width=True,
                type="secondary",
                key="btn_revise",
            ):
                st.session_state.intake_result   = None
                st.session_state.last_filename   = None
                st.session_state.action          = "revise"
                st.session_state.upload_counter += 1
                st.rerun()

        with col2:
            if st.button(
                "✅  Proceed with FNOL registration",
                use_container_width=True,
                type="primary",
                key="btn_proceed",
            ):
                st.session_state.action = "proceed"
                st.rerun()

    elif uploaded_file is None and st.session_state.intake_result is None:
        st.info("👆 Please upload your documents to get started.")

    st.markdown("---")
    if st.button("← Back to options", type="secondary", key="btn_back"):
        st.session_state.page          = "options"
        st.session_state.intake_result = None
        st.session_state.action        = None
        st.session_state.last_filename = None
        st.rerun()


def _do_register():
    """Generate FNOL number, save PDF, write register row, go to done page."""
    result = st.session_state.intake_result
    fnol   = result.get("fnol_fields", {}) if result else {}

    with st.spinner("Registering your FNOL…"):
        try:
            fnol_id = generate_fnol_number()

            doc_link = save_intake_document(
                st.session_state.tmp_pdf_path, fnol_id
            )

            register_fnol(
                fnol_id       = fnol_id,
                member_id     = st.session_state.member_id,
                policy_number = fnol.get("policy_number") or "",
                vehicle_reg   = fnol.get("vehicle_registration_number") or "",
                incident_date = fnol.get("incident_date_time") or "",
                incident_type = fnol.get("incident_type") or "",
                adjuster_id   = fnol.get("adjuster_id") or "",
                doc_links     = doc_link,
            )

            st.session_state.fnol_id = fnol_id

        except Exception as e:
            st.warning(
                f"We encountered an issue while registering your FNOL: {e}. "
                "Please try again or contact our support team."
            )
            st.session_state.action = None
            return

    # ── Run triage agent immediately after registration ────────────────
    # Justification: same live-progress pattern as the intake agent — a
    # single static spinner gives no sense of where time is going across
    # the 11 triage steps (2 of which are LLM calls).
    with st.status("Generating FNOL Triage Summary…", expanded=True) as status:
        def _update_triage_status(step_label: str) -> None:
            status.update(label=step_label)
            status.write(f"▸ {step_label}")

        try:
            triage_result = run_triage_agent(
                intake_state     = st.session_state.intake_result,
                fnol_id          = st.session_state.fnol_id,
                fnol_received_at = datetime.now().strftime("%d-%b-%Y %H:%M"),
                progress_callback= _update_triage_status,
            )
            st.session_state.triage_result = triage_result
            status.update(label="Triage summary ready", state="complete", expanded=False)
        except Exception as e:
            import traceback
            err_detail = traceback.format_exc()
            st.session_state.triage_result    = None
            st.session_state.triage_error     = str(e)
            st.session_state.triage_traceback = err_detail
            status.update(
                label="We encountered an issue generating the triage summary",
                state="error", expanded=False,
            )

    st.session_state.page = "done"
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PAGE: DONE
# ══════════════════════════════════════════════════════════════════════════
def page_query_fnol():
    _header("📋 FNOL Assistance")
    _subtext("Query existing FNOL — enter your FNOL reference number to view status.")
    st.markdown("---")

    fnol_input = st.text_input(
        "FNOL Reference Number",
        placeholder="e.g. FNOL-2026-0002",
        max_chars=20,
    ).strip().upper()

    if st.button("Search", type="primary"):
        if not fnol_input:
            st.warning("Please enter a FNOL reference number.")
        else:
            with st.spinner("Looking up your FNOL..."):
                record = look_up_fnol(fnol_input)

            if record is None:
                st.info(
                    f"No FNOL found with reference **{fnol_input}**. "
                    "Please check the number and try again."
                )
            elif str(record.get("member_id", "")).strip().upper() !=                  str(st.session_state.member_id or "").strip().upper():
                st.markdown(
                    f"<div style='background:#FCE4D6;border:1px solid #8B0000;"
                    f"border-radius:8px;padding:14px 18px;'>"
                    f"<p style='color:#8B0000;font-weight:600;margin:0 0 6px 0;'>"
                    f"Access denied</p>"
                    f"<p style='color:#3A3A3A;margin:0;font-size:0.93rem;'>"
                    f"This FNOL is registered under a different Member ID. "
                    f"You can only query FNOLs registered under your own Member ID.</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                # Store record and clear RAG history when a new FNOL is loaded
                if (st.session_state.rag_record is None or
                        st.session_state.rag_record.get("fnol_id") != record.get("fnol_id")):
                    st.session_state.rag_history = []
                st.session_state.rag_record = record

    # Show two-column layout if a record is loaded
    record = st.session_state.rag_record
    if record:
        st.markdown("---")
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            _render_fnol_summary(record)

        with col_right:
            _render_rag_panel()

    st.markdown("---")
    if st.button("<- Back to options", type="secondary"):
        st.session_state.page       = "options"
        st.session_state.rag_record = None
        st.session_state.rag_history = []
        st.rerun()


def _render_fnol_summary(record: dict):
    """Left column — FNOL triage summary card."""
    STATUS_STYLE = {
        "COMPLETE":          ("background:#E2EFDA;border:1px solid #1E4D0F;",
                              "#1E4D0F", "Complete"),
        "ESCALATED":         ("background:#FCE4D6;border:1px solid #8B0000;",
                              "#8B0000", "Escalated - Senior Review Required"),
        "PENDED_FOR_INPUTS": ("background:#FFF2CC;border:1px solid #7B4F00;",
                              "#7B4F00", "Pending - Documents Required"),
        "EXCEPTION_RAISED":  ("background:#FCE4D6;border:1px solid #8B0000;",
                              "#8B0000", "Exception Raised - Policy Issue"),
        "PENDING":           ("background:#D5E8F0;border:1px solid #1F4E79;",
                              "#1F4E79", "Triage Pending"),
    }

    triage_status = str(record.get("triage_status") or "PENDING").strip()
    style_tuple   = STATUS_STYLE.get(
        triage_status,
        ("background:#D5E8F0;border:1px solid #1F4E79;", "#1F4E79", triage_status)
    )
    box_style, text_color, status_label = style_tuple
    incident_nice = str(record.get("incident_type") or "").replace("_", " ").title()

    # Status card
    st.markdown(
        f"<div style='{box_style}border-radius:8px;padding:14px 18px;margin-bottom:12px;'>"
        f"<b style='color:{text_color};font-size:0.95rem;'>"
        f"{record.get('fnol_id')} - {status_label}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Details
    fields = [
        ("Member ID",       record.get("member_id")),
        ("Policy Number",   record.get("policy_number")),
        ("Vehicle Reg",     record.get("vehicle_reg")),
        ("Incident Type",   incident_nice),
        ("Incident Date",   record.get("incident_date")),
        ("FNOL Received",   record.get("fnol_received_at")),
        ("Adjuster ID",     record.get("adjuster_id")),
        ("Aging Risk",      record.get("aging_risk")),
        ("Escalation",      "Yes" if record.get("escalation_flag") else "No"),
    ]
    for label, value in fields:
        st.markdown(
            f"<p style='margin:3px 0;font-size:0.88rem;'>"
            f"<span style='color:#888;'>{label}:</span> "
            f"<b>{value or '—'}</b></p>",
            unsafe_allow_html=True,
        )

    # Escalation reasons
    esc_reason = record.get("escalation_reason")
    if esc_reason:
        reasons      = esc_reason.split("|")
        reason_items = "".join(
            f"<li style='margin-bottom:3px;font-size:0.85rem;'>"
            f"{r.replace('_', ' ').title()}</li>"
            for r in reasons
        )
        st.markdown(
            f"<div style='background:#FCE4D6;border-left:3px solid #8B0000;"
            f"border-radius:4px;padding:8px 12px;margin-top:8px;'>"
            f"<p style='color:#8B0000;font-weight:600;margin:0 0 4px 0;"
            f"font-size:0.85rem;'>Escalation conditions:</p>"
            f"<ul style='margin:0;padding-left:16px;color:#3A3A3A;'>"
            f"{reason_items}</ul></div>",
            unsafe_allow_html=True,
        )

    # Elevated review indicators (fraud signals) — not stored in the flat
    # fnol_register row, so load it from the saved triage JSON if present.
    # Wrapped in try/except: a missing/unreadable JSON must never break
    # the rest of this summary card.
    triage_json_path = record.get("triage_json_path")
    if triage_json_path and os.path.exists(str(triage_json_path)):
        try:
            with open(triage_json_path, "r") as f:
                triage_full = json.load(f)
            _render_fraud_banner(triage_full.get("fraud_signals", {}))
        except Exception:
            pass

    # Triage PDF
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("**Triage Summary**")
    pdf_path     = record.get("triage_pdf_path")
    triage_status = str(record.get("triage_status") or "PENDING").strip()

    if pdf_path and os.path.exists(str(pdf_path)):
        # Download button
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label               = "📄  Download Triage Summary (PDF)",
            data                = pdf_bytes,
            file_name           = f"{record.get('fnol_id')}_triage.pdf",
            mime                = "application/pdf",
            type                = "primary",
            use_container_width = True,
            key                 = f"dl_triage_{record.get('fnol_id')}",
        )
        # Inline PDF viewer
        import base64
        b64 = base64.b64encode(pdf_bytes).decode()
        st.markdown(
            f"<iframe src='data:application/pdf;base64,{b64}' "
            f"width='100%' height='500px' "
            f"style='border:1px solid #CCCCCC;border-radius:4px;"
            f"margin-top:8px;'></iframe>",
            unsafe_allow_html=True,
        )
    elif triage_status == "PENDING":
        st.info("ℹ️ Triage summary has not been generated yet for this FNOL.")
    else:
        st.info(
            "ℹ️ Triage PDF is not available at the stored path. "
            "It may have been generated on a different machine."
        )


def _render_rag_panel():
    """Right column - RAG Q&A panel for FNOL Guidelines and SOP."""
    st.markdown(
        "<h4 style='color:#1F4E79;margin:0 0 8px 0;'>Ask FNOL Guidelines</h4>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#888;font-size:0.82rem;margin-bottom:10px;'>"
        "Ask questions about FNOL guidelines and procedures. "
        "Answers are grounded in the FNOL Guide and SOP documents.</p>",
        unsafe_allow_html=True,
    )

    # Show conversation history
    history = st.session_state.rag_history
    if history:
        for turn in history:
            st.markdown(
                f"<div style='background:#F2F2F2;border-radius:6px;"
                f"padding:8px 12px;margin-bottom:6px;'>"
                f"<p style='color:#1F4E79;font-weight:600;font-size:0.85rem;"
                f"margin:0 0 4px 0;'>You</p>"
                f"<p style='color:#3A3A3A;font-size:0.88rem;margin:0;'>"
                f"{turn['question']}</p></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='background:#E2EFDA;border-radius:6px;"
                f"padding:8px 12px;margin-bottom:10px;'>"
                f"<p style='color:#1E4D0F;font-weight:600;font-size:0.85rem;"
                f"margin:0 0 4px 0;'>Guidelines Assistant</p>"
                f"<p style='color:#3A3A3A;font-size:0.88rem;margin:0;'>"
                f"{turn['answer']}</p></div>",
                unsafe_allow_html=True,
            )

    # Question input
    question = st.text_input(
        "Your question",
        placeholder="e.g. What documents are required for a theft claim?",
        key="rag_question_input",
        label_visibility="collapsed",
    )

    col_ask, col_clear = st.columns([3, 1], gap="small")
    with col_ask:
        ask_clicked = st.button("Ask", type="primary",
                                use_container_width=True, key="btn_ask_rag")
    with col_clear:
        if st.button("Clear", type="secondary",
                     use_container_width=True, key="btn_clear_rag"):
            st.session_state.rag_history = []
            st.rerun()

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching guidelines..."):
                result = ask_rag(question.strip(), history)

            if result.get("error"):
                st.warning(
                    f"Could not retrieve an answer: {result['error']}"
                )
            else:
                answer = result["answer"]
                # Store sources and add to history
                st.session_state["_last_rag_sources"] = result.get("sources", [])
                st.session_state.rag_history.append({
                    "question": question.strip(),
                    "answer":   answer,
                })
                st.rerun()

    # Sources expander - show only if history exists
    if history and st.session_state.get("_last_rag_sources"):
        sources = st.session_state._last_rag_sources
        if sources:
            with st.expander("View sources", expanded=False):
                for s in sources:
                    st.markdown(
                        f"**{s['source_label']}** - "
                        f"Page {s['page_start']} "
                        f"(relevance: {s['score']})",
                        unsafe_allow_html=False,
                    )


def page_done():
    _header("📋 FNOL Assistance")
    st.markdown("---")

    fnol_id      = st.session_state.fnol_id
    triage       = st.session_state.triage_result or {}
    triage_error  = st.session_state.get("triage_error")
    triage_status = (triage.get("triage_status")
                     or ("ERROR" if triage_error else None))
    if not triage_status or triage_status == "IN_PROGRESS":
        triage_status = "ERROR" if triage_error else "ERROR"
    esc_flag     = triage.get("escalation_flag", False)
    esc_reason   = triage.get("escalation_reason") or ""
    pdf_path     = triage.get("triage_pdf_path")

    # ── Status badge colour ────────────────────────────────────────────
    # Internal status → (CSS, text colour, icon, user-facing label)
    STATUS_STYLE = {
        "COMPLETE":          ("background:#E2EFDA;border:1px solid #1E4D0F;",
                              "#1E4D0F", "✅", "Complete"),
        "ESCALATED":         ("background:#FCE4D6;border:1px solid #8B0000;",
                              "#8B0000", "⚠️", "Escalated — Senior Review Required"),
        "PENDED_FOR_INPUTS": ("background:#FFF2CC;border:1px solid #7B4F00;",
                              "#7B4F00", "⏸", "Pending — Documents Required"),
        "EXCEPTION_RAISED":  ("background:#FCE4D6;border:1px solid #8B0000;",
                              "#8B0000", "🚫", "Exception Raised — Policy Issue"),
        "ERROR":             ("background:#FCE4D6;border:1px solid #8B0000;",
                              "#8B0000", "❌", "Triage Generation Failed"),
    }
    style_tuple  = STATUS_STYLE.get(
        triage_status,
        ("background:#D5E8F0;border:1px solid #1F4E79;", "#1F4E79", "ℹ️",
         triage_status or "Pending")
    )
    box_style, text_color, icon, status_label = style_tuple

    # ── Registration success card ──────────────────────────────────────
    st.markdown(
        f"<div style='{box_style}border-radius:8px;padding:20px 24px;'>"
        f"<h3 style='color:{text_color};margin:0 0 8px 0;'>"
        f"{icon} FNOL Registered — Triage Status: {status_label}</h3>"
        f"<p style='color:#3A3A3A;margin:0;font-size:1rem;'>"
        f"Your claim notification has been registered and the triage summary "
        f"has been generated. Our claims team will be in touch shortly.</p>"
        f"<p style='margin:12px 0 4px 0;font-size:1rem;color:#3A3A3A;'>"
        f"FNOL Reference: "
        f"<span style='font-size:1.2rem;font-weight:700;color:#1F4E79;'>"
        f"{fnol_id}</span></p>"
        f"<p style='margin:4px 0 0 0;font-size:0.85rem;color:#555;'>"
        f"Please keep this reference number for your records.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Escalation detail ──────────────────────────────────────────────
    if esc_flag and esc_reason:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        conditions = triage.get("escalation", {}).get("triggered_conditions", [])
        cond_lines = "".join(
            f"<li style='margin-bottom:4px;'><b>{c['code']}</b> — {c['label']}</li>"
            for c in conditions
        )
        st.markdown(
            f"<div style='background:#FCE4D6;border-left:4px solid #8B0000;"
            f"border-radius:4px;padding:12px 16px;'>"
            f"<p style='color:#8B0000;font-weight:600;margin:0 0 6px 0;'>"
            f"⚠️ Escalation conditions triggered — Senior Claims Manager review required</p>"
            f"<ul style='margin:0;padding-left:18px;color:#3A3A3A;font-size:0.9rem;'>"
            f"{cond_lines}</ul></div>",
            unsafe_allow_html=True,
        )

    # ── Elevated review indicators (fraud signals) ──────────────────────
    _render_fraud_banner(triage.get("fraud_signals", {}))

    # ── Pended detail ──────────────────────────────────────────────────
    if triage_status == "PENDED_FOR_INPUTS":
        missing = [
            d["name"] for d in triage.get("doc_checklist", [])
            if not d.get("received")
        ]
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if missing:
            items_html = "".join(f"<li style='margin-bottom:4px;'>{m}</li>"
                                 for m in missing)
            st.markdown(
                f"<div style='background:#FFF2CC;border-left:4px solid #7B4F00;"
                f"border-radius:4px;padding:12px 16px;'>"
                f"<p style='color:#7B4F00;font-weight:600;margin:0 0 6px 0;'>"
                f"⏸ The following documents are required to progress this claim:</p>"
                f"<ul style='margin:0;padding-left:18px;color:#3A3A3A;font-size:0.9rem;'>"
                f"{items_html}</ul>"
                f"<p style='color:#7B4F00;margin:8px 0 0 0;font-size:0.85rem;'>"
                f"Please upload the revised document set including these documents "
                f"to allow the claim to proceed.</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("ℹ️ Some documents could not be verified. "
                    "Please review your submission.")

    # ── Triage error detail ───────────────────────────────────────────
    if triage_error:
        tb = st.session_state.get("triage_traceback", "")
        with st.expander("⚠️ Triage error details (for support)", expanded=True):
            st.code(triage_error)
            if tb:
                st.code(tb)

    # ── Download triage PDF ────────────────────────────────────────────
    if pdf_path and os.path.exists(pdf_path):
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label     = "📄  Download Triage Summary (PDF)",
                data      = f.read(),
                file_name = f"{fnol_id}_triage.pdf",
                mime      = "application/pdf",
                type      = "primary",
                use_container_width=True,
            )

    # ── Triage processing time breakdown ────────────────────────────────
    # Justification: same pattern as the intake side — real, measured
    # per-step durations for this triage run, collapsed by default since
    # it's a diagnostic, not something the claimant needs to see up front.
    triage_timing = triage.get("timing_report") or {}
    if triage_timing.get("steps"):
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        with st.expander(
            f"⏱️ Triage processing time breakdown "
            f"({triage_timing.get('total_duration_s', 0):.1f}s total)",
            expanded=False,
        ):
            rows_html = "".join(
                f"<tr>"
                f"<td style='padding:4px 10px;font-size:0.84rem;'>{s['step']}</td>"
                f"<td style='padding:4px 10px;font-size:0.84rem;text-align:right;'>"
                f"{s['duration_s']:.2f}s</td>"
                f"</tr>"
                for s in triage_timing["steps"]
            )
            st.markdown(
                f"<table style='width:100%;border-collapse:collapse;'>"
                f"<thead><tr>"
                f"<th style='text-align:left;padding:4px 10px;font-size:0.82rem;"
                f"color:#888;border-bottom:1px solid #DDD;'>Step</th>"
                f"<th style='text-align:right;padding:4px 10px;font-size:0.82rem;"
                f"color:#888;border-bottom:1px solid #DDD;'>Duration</th>"
                f"</tr></thead><tbody>{rows_html}</tbody></table>",
                unsafe_allow_html=True,
            )
            triage_timing_path = triage.get("timing_report_path")
            if triage_timing_path and os.path.exists(triage_timing_path):
                with open(triage_timing_path, "rb") as f:
                    st.download_button(
                        label="Download timing report (JSON)",
                        data=f.read(),
                        file_name=f"{fnol_id}_triage_timing_report.json",
                        mime="application/json",
                        key="triage_timing_report_download",
                    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    if st.button("← Back to main menu", type="secondary"):
        _reset()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY RENDERER — opening sentence + bullets + numbered sub-bullets
# ══════════════════════════════════════════════════════════════════════════
def _render_summary(summary: str):
    lines   = [l.strip() for l in summary.split("\n") if l.strip()]
    opening = ""
    items   = []   # (kind, text) — kind = "bullet" | "sub"

    for line in lines:
        if line.startswith(("•", "-", "*", "–")):
            items.append(("bullet", line.lstrip("•-*– ").strip()))
        elif len(line) >= 2 and line[0].isdigit() and line[1] in ".):":
            # Strip leading number prefix (e.g. "1. " or "1) ") — <ol> adds its own
            import re as _re  # noqa
            cleaned = _re.sub(r"^\d+[.):] *", "", line).strip()
            items.append(("sub", cleaned))
        else:
            if not opening:
                opening = line
            else:
                items.append(("bullet", line))

    if opening:
        st.markdown(
            f"<p style='color:#3A3A3A;font-size:1rem;margin-bottom:12px;'>"
            f"{opening}</p>",
            unsafe_allow_html=True,
        )

    html = "<ul style='padding-left:20px;'>"
    i = 0
    while i < len(items):
        kind, text = items[i]
        if kind == "bullet":
            subs = []
            j    = i + 1
            while j < len(items) and items[j][0] == "sub":
                subs.append(items[j][1])
                j += 1
            sub_html = ""
            if subs:
                sub_html = (
                    "<ol style='margin-top:4px;margin-bottom:4px;'>"
                    + "".join(
                        f"<li style='margin-bottom:4px;color:#3A3A3A;"
                        f"font-size:0.9rem;'>{s}</li>"
                        for s in subs
                    )
                    + "</ol>"
                )
            html += (
                f"<li style='margin-bottom:8px;color:#3A3A3A;"
                f"font-size:0.95rem;'>{text}{sub_html}</li>"
            )
            i = j if subs else i + 1
        else:
            html += (
                f"<li style='margin-bottom:4px;color:#3A3A3A;"
                f"font-size:0.9rem;list-style-type:decimal;"
                f"margin-left:16px;'>{text}</li>"
            )
            i += 1
    html += "</ul>"
    st.markdown(html, unsafe_allow_html=True)


def _render_logout_sidebar():
    """
    Shown on every page once a member is logged in (i.e. every page except
    'home', which IS the login screen). Reuses the existing _reset() helper
    so logout clears the exact same state a fresh session starts with —
    no separate list to keep in sync as new session keys get added later.
    """
    if st.session_state.page == "home":
        return
    with st.sidebar:
        st.markdown(f"**Logged in as:** {st.session_state.member_id}")
        if st.button("🚪 Log out", use_container_width=True):
            _reset()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════
def main():
    _render_logout_sidebar()
    page = st.session_state.page
    if   page == "home":          page_home()
    elif page == "options":       page_options()
    elif page == "coming_soon":   page_coming_soon()
    elif page == "query_fnol":    page_query_fnol()
    elif page == "register_fnol": page_register_fnol()
    elif page == "done":          page_done()
    else:
        _reset()
        st.rerun()


if __name__ == "__main__":
    main()
