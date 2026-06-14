"""
fnol_triage_pdf.py
==================
Renders the FNOL Triage Summary PDF.
Matches the 8-section format of the sample FNOL-2026-0002_triage.pdf exactly.

Public API:
    render_triage_pdf(state: FNOLTriageState, output_path: str)
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate

PW, PH = A4
LM = RM = 1.8 * cm
TM = BM = 1.8 * cm
W  = PW - LM - RM

# ── Palette ────────────────────────────────────────────────────────────────
NAVY    = HexColor("#1F4E79")
BLUE    = HexColor("#2E75B6")
LT_BLUE = HexColor("#D5E8F0")
TEAL    = HexColor("#0F6E56")
TEAL_BG = HexColor("#E1F5EE")
RED     = HexColor("#8B0000")
RED_BG  = HexColor("#FCE4D6")
AMBER   = HexColor("#7B4F00")
AMB_BG  = HexColor("#FFF2CC")
GREEN   = HexColor("#1E4D0F")
GRN_BG  = HexColor("#E2EFDA")
GRAY    = HexColor("#3A3A3A")
LT_GRAY = HexColor("#F2F2F2")
MG      = HexColor("#CCCCCC")
WHITE   = white

# ── Styles ─────────────────────────────────────────────────────────────────
def S(n, **k): return ParagraphStyle(n, **k)

ST = {
    "doc_title": S("dt", fontName="Helvetica-Bold", fontSize=14, textColor=NAVY,
                   alignment=TA_CENTER, spaceBefore=0, spaceAfter=4, leading=18),
    "doc_sub":   S("ds", fontName="Helvetica",      fontSize=9,  textColor=BLUE,
                   alignment=TA_CENTER, spaceBefore=0, spaceAfter=2),
    "doc_notice":S("dn", fontName="Helvetica-Bold", fontSize=8,  textColor=RED,
                   alignment=TA_CENTER, spaceBefore=2, spaceAfter=6),
    "sec_hdr":   S("sh", fontName="Helvetica-Bold", fontSize=11, textColor=WHITE,
                   spaceBefore=0, spaceAfter=0, leading=14),
    "body":      S("b",  fontName="Helvetica",       fontSize=9,  textColor=GRAY,
                   spaceBefore=1, spaceAfter=1, leading=13),
    "bold":      S("bl", fontName="Helvetica-Bold",  fontSize=9,  textColor=GRAY,
                   spaceBefore=1, spaceAfter=1, leading=13),
    "small":     S("sm", fontName="Helvetica",       fontSize=8,  textColor=GRAY,
                   spaceBefore=1, spaceAfter=1, leading=11),
    "esc_hdr":   S("eh", fontName="Helvetica-Bold",  fontSize=10, textColor=RED,
                   spaceBefore=2, spaceAfter=2),
    "esc_body":  S("eb", fontName="Helvetica",       fontSize=9,  textColor=GRAY,
                   spaceBefore=1, spaceAfter=1, leading=13),
    "footer":    S("ft", fontName="Helvetica",       fontSize=7.5,textColor=BLUE,
                   alignment=TA_CENTER),
    "status_green": S("sg", fontName="Helvetica-Bold", fontSize=9, textColor=GREEN),
    "status_red":   S("sr", fontName="Helvetica-Bold", fontSize=9, textColor=RED),
    "status_amber": S("sa", fontName="Helvetica-Bold", fontSize=9, textColor=AMBER),
    "status_blue":  S("sb", fontName="Helvetica-Bold", fontSize=9, textColor=NAVY),
}

from reportlab.lib import colors
from reportlab.platypus.flowables import HRFlowable

def sp(h=4):    return Spacer(1, h)
def hr(c=MG, t=0.4): return HRFlowable(width="100%", thickness=t, color=c,
                                         spaceAfter=3, spaceBefore=3)
def p(t, st="body"): return Paragraph(str(t) if t is not None else "—", ST[st])

def _bdr(top=None, bottom=None, left=None, right=None):
    return None  # placeholder — not used


# ── Table builder ──────────────────────────────────────────────────────────
def _tbl(data, col_widths, style_cmds=None, row_heights=None):
    from reportlab.platypus import Table, TableStyle
    tbl = Table(data, colWidths=col_widths)
    base = [
        ("FONTNAME",      (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("GRID",          (0,0), (-1,-1), 0.3, MG),
    ]
    if style_cmds:
        base.extend(style_cmds)
    tbl.setStyle(TableStyle(base))
    return tbl


def _sec_header(num: str, title: str) -> Table:
    """Navy section header bar."""
    cell = Paragraph(f"<b><font color='white'>{num}. {title}</font></b>",
                     ParagraphStyle("sh2", fontName="Helvetica-Bold", fontSize=10,
                                    textColor=WHITE, leading=13))
    tbl = Table([[cell]], colWidths=[W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), NAVY),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
    ]))
    return tbl


def _status_para(status: str) -> Paragraph:
    status = str(status or "")
    if status in ("COVERED", "RECEIVED", "COMPLETE"):
        st = "status_green"
    elif status in ("NOT_COVERED", "MISSING", "ESCALATED", "EXCEPTION_RAISED"):
        st = "status_red"
    elif status in ("REQUIRES_VERIFICATION", "PENDED_FOR_INPUTS"):
        st = "status_amber"
    else:
        st = "status_blue"
    return p(status, st)


def _conf_para(conf: str) -> Paragraph:
    conf = str(conf or "")
    if conf == "HIGH":
        return p(conf, "status_green")
    elif conf == "LOW":
        return p(conf, "status_red")
    return p(conf, "status_amber")


# ══════════════════════════════════════════════════════════════════════════
# PAGE HEADER / FOOTER
# ══════════════════════════════════════════════════════════════════════════

def _make_doc(output_path: str, state: dict) -> BaseDocTemplate:
    fnol_id      = state.get("fnol_id", "")
    triage_at    = state.get("triage_generated_at", "")
    adj_id       = state.get("adjuster_id", "")
    triage_status= state.get("triage_status", "")

    def _on_page(canvas, doc):
        canvas.saveState()
        # Footer
        footer_text = (f"{fnol_id} Triage Summary  ·  AI-OPS-FNOL  ·  "
                       f"{triage_at[:10] if triage_at else ''}  ·  {adj_id}")
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(BLUE)
        canvas.drawCentredString(PW / 2, BM * 0.6, footer_text)
        # Page number
        canvas.drawRightString(PW - RM, BM * 0.6,
                               f"Page {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(
        output_path, pagesize=A4,
        leftMargin=LM, rightMargin=RM,
        topMargin=TM, bottomMargin=BM * 1.4,
        title=f"{fnol_id} FNOL Triage Summary",
        author="AI-OPS-FNOL",
    )
    frame = Frame(LM, BM * 1.4, W, PH - TM - BM * 1.4, id="main")
    template = PageTemplate(id="main", frames=[frame], onPage=_on_page)
    doc.addPageTemplates([template])
    return doc


# ══════════════════════════════════════════════════════════════════════════
# RENDER FUNCTIONS — one per section
# ══════════════════════════════════════════════════════════════════════════

def _render_header(state: dict) -> list:
    """Document header + escalation banner if triggered."""
    items = []
    fnol_id      = state.get("fnol_id", "")
    snap         = state.get("claim_snapshot", {})
    triage_at    = state.get("triage_generated_at", "")[:16].replace("T", " ")
    adj_id       = state.get("adjuster_id", "")
    triage_status= state.get("triage_status", "")
    incident_type= str(snap.get("incident_type") or "").replace("_", " ").title()

    items.append(p("ABC GENERAL INSURANCE LTD.", "doc_sub"))
    items.append(p("AI Copilot — FNOL Triage Summary", "doc_title"))
    items.append(p(f"{fnol_id}  ·  {incident_type}  ·  {snap.get('incident_date', '')[:10]}",
                   "doc_sub"))
    items.append(p("FOR ADJUSTER REVIEW ONLY — NOT A TRIAGE DETERMINATION", "doc_notice"))
    items.append(p("This summary is generated by the AI Operations Copilot as decision support. "
                   "All coverage, escalation, and admissibility determinations are the "
                   "adjuster's sole responsibility.", "small"))
    items.append(p(f"Generated: {triage_at}  ·  Adjuster: {adj_id}  ·  "
                   f"Triage status: {triage_status}", "small"))
    items.append(sp(6))

    # Escalation banner
    escalation = state.get("escalation", {})
    if escalation.get("escalation_flag"):
        conditions = escalation.get("triggered_conditions", [])
        esc_rows = [[
            Paragraph("<b><font color='white'>■ ESCALATION FLAG — "
                      "Senior Claims Manager Review Required</font></b>",
                      ParagraphStyle("eb2", fontName="Helvetica-Bold", fontSize=10,
                                     textColor=WHITE, leading=13)),
        ]]
        esc_tbl = Table(esc_rows, colWidths=[W])
        esc_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), RED),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ]))
        items.append(esc_tbl)

        cond_text = "This claim has triggered the following sensitive case condition(s):"
        items.append(p(cond_text, "small"))
        for i, c in enumerate(conditions, 1):
            items.append(p(f"{i}. <b>{c['code']}</b> — {c['label']}.", "small"))

        items.append(p("Triage determination must NOT be communicated to the claimant "
                       "until the senior claims manager has reviewed and instructed.", "small"))
        items.append(p(f"Escalation reason logged in fnol_register: "
                       f"{escalation.get('escalation_reason', '')}", "small"))
        items.append(sp(6))

    return items


def _render_claim_snapshot(state: dict) -> list:
    items = [_sec_header("1", "Claim Snapshot"), sp(4)]
    snap  = state.get("claim_snapshot", {})
    aging = state.get("aging_risk", {})

    aging_str = (f"{'■' if aging.get('sla_breached') else '■'} "
                 f"{aging.get('aging_risk_level', '')} "
                 f"({aging.get('fnol_age_hours', 0):.0f} hours)")

    left_rows = [
        ("FNOL ID",        state.get("fnol_id", "")),
        ("Member ID",      snap.get("member_id", "")),
        ("Vehicle",        snap.get("vehicle", "")),
        ("Incident Type",  str(snap.get("incident_type","")).replace("_"," ").title()),
        ("Incident Location", snap.get("incident_location", "")),
        ("Coverage Type",  snap.get("coverage_type", "")),
        ("NCB at Inception", f"{snap.get('ncb_percent', 0)}%"),
        ("Hypothecated",   "Yes" if snap.get("hypothecated") else "No"),
        ("Policy Status",  snap.get("policy_status", "")),
    ]
    right_rows = [
        ("Aging Risk",     aging_str),
        ("Policy Number",  snap.get("policy_number", "")),
        ("Insurer",        snap.get("insurer", "")),
        ("Incident Date",  str(snap.get("incident_date", ""))[:16]),
        ("FNOL Received",  str(snap.get("fnol_received_at", ""))[:16]),
        ("IDV",            f"Rs.{snap.get('idv_inr', 0):,}"),
        ("Add-ons",        " | ".join(snap.get("addons") or ["None"])),
        ("Financer",       snap.get("financer_name") or "None"),
        ("Policy Expiry",  snap.get("policy_expiry", "")),
    ]

    # Two-column snapshot table
    col_w = W / 2 - 0.1 * cm
    tbl_data = []
    for (lk, lv), (rk, rv) in zip(left_rows, right_rows):
        tbl_data.append([
            Paragraph(f"<b>{lk}</b>", ST["bold"]),
            Paragraph(str(lv), ST["body"]),
            Paragraph(f"<b>{rk}</b>", ST["bold"]),
            Paragraph(str(rv), ST["body"]),
        ])
    tbl = _tbl(tbl_data, [col_w*0.38, col_w*0.62, col_w*0.38, col_w*0.62])
    items.append(tbl)

    # Policy expiry warning
    warn = snap.get("policy_expiry_warning")
    if warn:
        items.append(sp(4))
        warn_tbl = Table(
            [[Paragraph(f"<b>Policy expiry note</b><br/>{warn}", ST["small"])]],
            colWidths=[W]
        )
        warn_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), AMB_BG),
            ("BOX",           (0,0), (-1,-1), 0.5, AMBER),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ]))
        items.append(warn_tbl)

    items.append(sp(6))
    return items


def _render_coverage(state: dict) -> list:
    items = [_sec_header("2", "Coverage Applicability"), sp(4)]
    snap   = state.get("claim_snapshot", {})
    it     = str(snap.get("incident_type","")).replace("_"," ").title()
    conf   = state.get("coverage_confidence_overall", "LOW")
    items.append(p(f"Incident type: <b>{it}</b>. "
                   f"Overall coverage confidence: <b>{conf}</b>.", "small"))
    warn = snap.get("policy_expiry_warning")
    if warn:
        items.append(p(f"Note: {warn}", "small"))
    items.append(sp(4))

    hdr = [
        Paragraph("<b><font color='white'>Section</font></b>",
                  ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9,
                                 textColor=WHITE)),
        Paragraph("<b><font color='white'>Clause ref</font></b>",
                  ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9,
                                 textColor=WHITE)),
        Paragraph("<b><font color='white'>Status</font></b>",
                  ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9,
                                 textColor=WHITE)),
        Paragraph("<b><font color='white'>Confidence</font></b>",
                  ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9,
                                 textColor=WHITE)),
        Paragraph("<b><font color='white'>Notes</font></b>",
                  ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9,
                                 textColor=WHITE)),
    ]
    rows = [hdr]
    col_w = [W*0.22, W*0.16, W*0.14, W*0.10, W*0.38]

    for i, c in enumerate(state.get("coverage_result", [])):
        bg = LT_GRAY if i % 2 == 0 else WHITE
        rows.append([
            p(c.get("section", ""), "body"),
            p(c.get("clause_ref", ""), "small"),
            _status_para(c.get("status", "")),
            _conf_para(c.get("confidence", "")),
            p(c.get("note", ""), "small"),
        ])

    tbl = _tbl(rows, col_w, [
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        *[("BACKGROUND", (0,i+1), (-1,i+1), LT_GRAY if i%2==0 else WHITE)
          for i in range(len(rows)-1)]
    ])
    items.append(tbl)
    items.append(p(f"Overall coverage confidence: <b>{conf}</b>", "small"))
    items.append(sp(6))
    return items


def _render_exclusions(state: dict) -> list:
    items = [_sec_header("3", "Active Exclusions — Requires Verification"), sp(4)]
    items.append(p("Exclusions are flagged for adjuster review. "
                   "Applicability is always REQUIRES_VERIFICATION.", "small"))
    items.append(sp(4))

    hdr = [
        Paragraph("<b><font color='white'>Exclusion</font></b>",
                  ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
        Paragraph("<b><font color='white'>Clause ref</font></b>",
                  ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
        Paragraph("<b><font color='white'>Confidence</font></b>",
                  ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
        Paragraph("<b><font color='white'>Reason flagged</font></b>",
                  ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
    ]
    rows = [hdr]
    col_w = [W*0.25, W*0.18, W*0.10, W*0.47]

    for i, ex in enumerate(state.get("exclusion_result", [])):
        rows.append([
            p(ex.get("name", ""), "body"),
            p(ex.get("clause_ref", ""), "small"),
            _conf_para(ex.get("confidence", "")),
            p(ex.get("reason_relevant", ""), "small"),
        ])

    if len(rows) == 1:
        rows.append([p("No exclusions flagged for this policy / incident type.", "small"),
                     p(""), p(""), p("")])

    tbl = _tbl(rows, col_w, [
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        *[("BACKGROUND", (0,i+1), (-1,i+1), LT_GRAY if i%2==0 else WHITE)
          for i in range(len(rows)-1)]
    ])
    items.append(tbl)
    items.append(sp(6))
    return items


def _render_waiting_period(state: dict) -> list:
    items = [_sec_header("4", "Waiting Period Status"), sp(4)]
    wp = state.get("waiting_period", {})

    rows = [
        ["Policy inception date",   wp.get("policy_inception", "")],
        ["Waiting period",          f"{wp.get('waiting_period_days', 30)} days"],
        ["Incident date",           str(wp.get("incident_date", ""))[:16]],
        ["Days since inception",    f"{wp.get('days_since_inception', '?')} days"],
        ["Conclusion",              wp.get("note", "")],
        ["Confidence",              wp.get("confidence", "")],
    ]
    tbl_data = [
        [Paragraph(f"<b>{r[0]}</b>", ST["bold"]),
         Paragraph(str(r[1]), ST["body"])]
        for r in rows
    ]
    items.append(_tbl(tbl_data, [W*0.30, W*0.70]))
    items.append(sp(4))

    if wp.get("incident_pre_inception"):
        label = "INCIDENT BEFORE POLICY START DATE — CRITICAL EXCEPTION"
        color = RED
    elif wp.get("waiting_period_active"):
        label = f"WAITING PERIOD ACTIVE — {wp.get('days_since_inception')} days since inception"
        color = AMBER
    else:
        label = "Waiting period: NOT ACTIVE"
        color = TEAL

    status_tbl = Table(
        [[Paragraph(f"<b>{label}</b>",
                    ParagraphStyle("wps", fontName="Helvetica-Bold", fontSize=9,
                                   textColor=WHITE))]],
        colWidths=[W]
    )
    status_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), color),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
    ]))
    items.append(status_tbl)
    items.append(sp(6))
    return items


def _render_claims_history(state: dict) -> list:
    items = [_sec_header("5", "Claims History"), sp(4)]
    ch = state.get("claims_history", {})
    snap = state.get("claim_snapshot", {})
    items.append(p(f"36-month claim count — vehicle {snap.get('vehicle_reg', '')}", "small"))
    items.append(sp(4))

    prior = ch.get("prior_claims_36m", [])
    if prior:
        hdr = [
            Paragraph("<b><font color='white'>Claim ID</font></b>",
                      ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
            Paragraph("<b><font color='white'>Claim date</font></b>",
                      ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
            Paragraph("<b><font color='white'>Incident type</font></b>",
                      ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
            Paragraph("<b><font color='white'>Status</font></b>",
                      ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
            Paragraph("<b><font color='white'>Settled amount</font></b>",
                      ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
            Paragraph("<b><font color='white'>NCB impact</font></b>",
                      ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
        ]
        rows = [hdr]
        for i, cl in enumerate(prior):
            amt = cl.get("settled_amount_inr") or cl.get("settled_amount") or 0
            rows.append([
                p(cl.get("claim_id", ""), "small"),
                p(str(cl.get("claim_date", ""))[:10], "small"),
                p(str(cl.get("incident_type","")).replace("_"," ").title(), "small"),
                p(cl.get("status", ""), "small"),
                p(f"Rs.{int(amt):,}", "small"),
                p("Yes" if cl.get("ncb_impact") else "No", "small"),
            ])
        items.append(_tbl(rows, [W*0.15, W*0.12, W*0.22, W*0.12, W*0.22, W*0.12], [
            ("BACKGROUND", (0,0), (-1,0), NAVY),
        ]))
        items.append(sp(4))
    else:
        items.append(p("No prior claims found for this vehicle in the last 36 months.", "small"))
        items.append(sp(4))

    metrics = [
        ["Prior claims on this vehicle (36 months)", str(ch.get("prior_claims_36m_count", 0))],
        ["Last claim date", str(ch.get("last_claim_date", "None") or "None")[:10]],
        ["Days since last claim", str(ch.get("days_since_last_claim", "N/A"))],
        ["Back-to-back flag",
         "ACTIVE — prior claim within 30 days" if ch.get("back_to_back_flag") else "Not triggered"],
        ["12-month cross-policy claims", str(ch.get("prior_claims_12m_count", 0))],
        ["Repeat claimant flag",
         f"ACTIVE (threshold: 3+)" if ch.get("repeat_claimant_flag") else "Not triggered"],
        ["NCB cross-check", ch.get("ncb_discrepancy") or "No discrepancy found"],
    ]
    m_data = [
        [Paragraph(f"<b>{r[0]}</b>", ST["bold"]), Paragraph(r[1], ST["body"])]
        for r in metrics
    ]
    items.append(_tbl(m_data, [W*0.45, W*0.55]))

    if ch.get("back_to_back_flag"):
        items.append(sp(4))
        btb_tbl = Table(
            [[Paragraph(
                "<b>Back-to-back claims flag: ACTIVE</b><br/>"
                f"Prior claim {(ch.get('prior_claims_36m') or [{}])[0].get('claim_id','')} "
                f"settled {ch.get('days_since_last_claim','?')} days before this incident. "
                "This is a data observation — not a fraud determination.",
                ParagraphStyle("btb", fontName="Helvetica", fontSize=8.5,
                               textColor=AMBER, leading=12))]],
            colWidths=[W]
        )
        btb_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), AMB_BG),
            ("BOX",           (0,0), (-1,-1), 0.5, AMBER),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ]))
        items.append(btb_tbl)

    items.append(sp(6))
    return items


def _render_doc_checklist(state: dict) -> list:
    items = [_sec_header("6", "Document Gap Checklist"), sp(4)]
    snap = state.get("claim_snapshot", {})
    it   = str(snap.get("incident_type","")).replace("_"," ").title()
    items.append(p(f"Required documents for incident type: <b>{it}</b>.", "small"))
    items.append(sp(4))

    hdr = [
        Paragraph("<b><font color='white'>Document</font></b>",
                  ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
        Paragraph("<b><font color='white'>Criticality</font></b>",
                  ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
        Paragraph("<b><font color='white'>Status</font></b>",
                  ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
        Paragraph("<b><font color='white'>Action required</font></b>",
                  ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
    ]
    rows = [hdr]

    for i, doc in enumerate(state.get("doc_checklist", [])):
        crit    = "BLOCKING" if doc.get("blocking") else "MANDATORY"
        status  = doc.get("status", "")
        action  = "No action needed." if doc.get("received") else (
            doc.get("collection_target") or "To be collected."
        )
        rows.append([
            p(doc.get("name", ""), "body"),
            _status_para(crit),
            _status_para("RECEIVED" if doc.get("received") else "MISSING"),
            p(action, "small"),
        ])

    if len(rows) == 1:
        rows.append([p("No checklist available.", "small"), p(""), p(""), p("")])

    items.append(_tbl(rows, [W*0.30, W*0.14, W*0.14, W*0.42], [
        ("BACKGROUND", (0,0), (-1,0), NAVY),
    ]))
    items.append(sp(6))
    return items


def _render_escalation_detail(state: dict) -> list:
    items = [_sec_header("7", "Escalation Flag — Detail"), sp(4)]
    esc = state.get("escalation", {})

    if not esc.get("escalation_flag"):
        no_esc = Table(
            [[Paragraph("No escalation conditions were triggered for this FNOL.", ST["body"])]],
            colWidths=[W]
        )
        no_esc.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), GRN_BG),
            ("BOX",           (0,0), (-1,-1), 0.5, TEAL),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ]))
        items.append(no_esc)
        items.append(sp(6))
        return items

    hdr_tbl = Table(
        [[Paragraph("<b><font color='white'>Escalation is ACTIVE — "
                    "Senior Claims Manager Review Required</font></b>",
                    ParagraphStyle("et", fontName="Helvetica-Bold", fontSize=9,
                                   textColor=WHITE))]],
        colWidths=[W]
    )
    hdr_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), RED),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
    ]))
    items.append(hdr_tbl)
    items.append(p("Escalation flag set by Python conditional logic. "
                   "Cannot be overridden by model or adjuster.", "small"))
    items.append(sp(4))

    hdr = [
        Paragraph("<b><font color='white'>Condition</font></b>",
                  ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
        Paragraph("<b><font color='white'>Trigger logic</font></b>",
                  ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
        Paragraph("<b><font color='white'>Status</font></b>",
                  ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
        Paragraph("<b><font color='white'>Adjuster instruction</font></b>",
                  ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
    ]
    rows = [hdr]
    for c in esc.get("triggered_conditions", []):
        rows.append([
            p(c.get("code", ""), "bold"),
            p(c.get("trigger", ""), "small"),
            Paragraph("<b>TRIGGERED</b>",
                      ParagraphStyle("tr", fontName="Helvetica-Bold", fontSize=9,
                                     textColor=RED)),
            p(c.get("adjuster_instruction", ""), "small"),
        ])

    items.append(_tbl(rows, [W*0.20, W*0.28, W*0.10, W*0.42], [
        ("BACKGROUND", (0,0), (-1,0), NAVY),
    ]))
    items.append(sp(6))
    return items


def _render_aging_risk(state: dict) -> list:
    items = [_sec_header("8", "Aging Risk Indicator"), sp(4)]
    ar   = state.get("aging_risk", {})
    snap = state.get("claim_snapshot", {})

    rows = [
        ["FNOL received at",        snap.get("fnol_received_at", "")[:16]],
        ["Triage generated at",     state.get("triage_generated_at", "")[:16].replace("T"," ")],
        ["FNOL age at triage",      f"{ar.get('fnol_age_hours', 0):.2f} hours"],
        ["Aging risk",              f"{ar.get('aging_risk_level', '')}"],
        ["IRDAI 24-hour SLA due",   ar.get("irdai_sla_due", "")],
        ["SLA status",              "BREACHED" if ar.get("sla_breached") else "No breach"],
    ]
    tbl_data = [
        [Paragraph(f"<b>{r[0]}</b>", ST["bold"]),
         Paragraph(str(r[1]), ST["body"])]
        for r in rows
    ]
    items.append(_tbl(tbl_data, [W*0.40, W*0.60]))
    items.append(sp(6))
    return items


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════

def render_triage_pdf(state: dict, output_path: str):
    """
    Render the complete 8-section FNOL Triage Summary PDF.

    Args:
        state       : FNOLTriageState dict
        output_path : Full path for the output PDF
    """
    doc   = _make_doc(output_path, state)
    story = []

    story += _render_header(state)
    story += _render_claim_snapshot(state)
    story.append(PageBreak())

    story += _render_coverage(state)
    story.append(PageBreak())

    story += _render_exclusions(state)
    story += _render_waiting_period(state)
    story.append(PageBreak())

    story += _render_claims_history(state)
    story.append(PageBreak())

    story += _render_doc_checklist(state)
    story += _render_escalation_detail(state)
    story.append(PageBreak())

    story += _render_aging_risk(state)

    doc.build(story)
