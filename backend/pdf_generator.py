"""
RecoverFlow — L3 Escalation PDF Generator
=========================================
Builds a clean, legal-style "FORMAL PAYMENT RECOVERY NOTICE" PDF using reportlab.

Generated once when a session escalates (all negotiation steps exhausted, a
legal threat, or no progress) and cached on the session so the download
endpoint never regenerates it. The document doubles as the compliant
paper-trail of recovery attempts made *before* escalation, in line with the
RBI Fair Practices Code.

Usage:
    from backend.pdf_generator import generate_escalation_pdf
    path = generate_escalation_pdf(session, invoice)   # -> "/tmp/escalation_INV-xxxx.pdf"
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Where escalation PDFs are written (matches the L3 spec).
PDF_DIR = Path("/tmp")

# Keep in sync with backend.agent.MERCHANT_NAME — the PDF is a merchant-facing
# legal notice and should carry the same merchant identity the agent speaks as.
MERCHANT_NAME = "RecoverFlow Demo Merchant"

_IST = timezone(timedelta(hours=5, minutes=30))


def _ist_now() -> datetime:
    """Current time in IST (UTC+5:30), using the IANA zone when available."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Kolkata"))
    except Exception:  # pragma: no cover - fallback when tzdata is unavailable
        return datetime.now(_IST)


def _inr(amount) -> str:
    """Format a rupee amount (int/float, whole rupees) with Indian grouping.

    Uses the ISO code "INR" rather than the ₹ glyph — the standard Helvetica
    fonts bundled with reportlab can't render ₹ (it shows as a stray "I"), so we
    keep the PDF ASCII-safe and portable across hosts.
    """
    try:
        rupees = int(round(float(amount)))
    except (TypeError, ValueError):
        rupees = 0
    sign = "-" if rupees < 0 else ""
    s = str(abs(rupees))
    if len(s) <= 3:
        body = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        groups = []
        while len(rest) > 2:
            groups.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.append(rest)
        groups.reverse()
        body = ",".join(groups) + "," + last3
    return f"{sign}INR {body}"


def _fmt_date(iso: str) -> str:
    """Format an ISO date ('YYYY-MM-DD') as '25 Aug 2025'."""
    if not iso:
        return "—"
    try:
        return datetime.strptime(str(iso)[:10], "%Y-%m-%d").strftime("%d %b %Y")
    except ValueError:
        return str(iso)


def _fmt_ts(iso) -> str:
    """Render an ISO timestamp as IST 'YYYY-MM-DD HH:MM IST'."""
    if not iso:
        return ""
    s = str(iso)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return s
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_IST).strftime("%Y-%m-%d %H:%M IST")


def _days_overdue(invoice: dict, session: dict) -> int:
    """Days overdue — prefer the invoice's dpd, else compute from due_date."""
    dpd = invoice.get("dpd")
    if isinstance(dpd, (int, float)):
        return int(dpd)
    due = invoice.get("due_date")
    if due:
        try:
            return max(0, (datetime.now(_IST).date() - datetime.strptime(
                str(due)[:10], "%Y-%m-%d").date()).days)
        except ValueError:
            pass
    return int(session.get("dpd", 0) or 0)


def _short(text, limit: int = 160) -> str:
    """Collapse a message to a single line, truncated for the audit list."""
    if text is None:
        return ""
    line = str(text).replace("\n", " ").replace("\r", " ").strip()
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _describe_event(entry: dict) -> str:
    """Turn one audit_log entry into a human-readable recovery-attempt line."""
    ev = entry.get("event") or "event"
    msg = entry.get("message")

    if ev == "session_created":
        return (f"Recovery session opened for {entry.get('invoice_id', '')} "
                f"(tier {entry.get('tier', '—')}, score {entry.get('score', '—')})")
    if ev == "agent_turn":
        if entry.get("turn") == 0:
            return "Agent first contacted the debtor"
        if entry.get("speaker") == "agent":
            return f"Agent: {_short(msg)}"
        return f"Debtor: {_short(msg)}"
    if ev == "debtor_turn":
        if (msg or "").strip() == "[silent]":
            return "Debtor did not respond"
        return f"Debtor: {_short(msg)}"
    if ev == "intent_extracted":
        return f"Detected intent: {entry.get('intent')}"
    if ev == "state_transition":
        reason = entry.get("reason")
        suffix = f" ({reason})" if reason else ""
        return f"Negotiation moved to “{entry.get('to_state', '')}”{suffix}"
    if ev == "negotiation_step":
        return f"Negotiation step {entry.get('step')}"
    if ev == "counter_offer":
        return (f"Debtor offered {_inr(entry.get('offered'))} — "
                f"agent countered {_inr(entry.get('counter'))}")
    if ev == "debtor_question":
        return "Debtor asked a clarifying question"
    if ev == "reason_mcq_answered":
        return f"Debtor's reason for non-payment: {entry.get('reason') or entry.get('button_id')}"
    if ev == "dispute_flagged":
        return "Debtor disputed the invoice"
    if ev == "document_upload_requested":
        return "Requested a supporting document from the debtor"
    if ev == "document_verified":
        return f"Document reviewed — verdict {entry.get('verdict')}"
    if ev == "hardship_verified":
        return "Inability-to-pay claim verified (hardship floor applied)"
    if ev == "installment_scheduled":
        return f"Scheduled {_inr(entry.get('amount'))} on {_fmt_date(entry.get('date'))}"
    if ev == "plan_ready":
        return "Payment plan presented to the debtor"
    if ev == "deferred_scheduled":
        return "Deferred payment scheduled"
    if ev == "razorpay_order_created":
        return "Payment link generated"
    if ev == "finalize_agreement":
        return "Payment agreement finalised"
    if ev == "stopping_rule":
        return f"Stopping rule fired: {entry.get('reason')}"
    if ev == "escalation_triggered":
        return f"Escalation triggered: {entry.get('reason')}"
    if ev == "L3_triggered":
        return "Escalation (L3) triggered"
    if ev == "trust_score":
        return f"Trust score {entry.get('score')} (tier {entry.get('tier')})"
    if ev == "upload_requested":
        return "Document upload requested"
    return str(ev).replace("_", " ").capitalize()


def _escape(text) -> str:
    """Escape text destined for a reportlab Paragraph mini-HTML body."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

_BRAND = colors.HexColor("#0F172A")
_ACCENT = colors.HexColor("#2563EB")
_MUTED = colors.HexColor("#64748B")
_LINE = colors.HexColor("#CBD5E1")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=20, leading=24, alignment=TA_CENTER,
            textColor=_BRAND, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=10, leading=13, alignment=TA_CENTER, textColor=_MUTED,
        ),
        "section": ParagraphStyle(
            "section", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=12, leading=15, alignment=TA_LEFT,
            textColor=_BRAND, spaceBefore=14, spaceAfter=6,
        ),
        "label": ParagraphStyle(
            "label", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=9, leading=12, textColor=_MUTED,
        ),
        "value": ParagraphStyle(
            "value", parent=base["Normal"], fontName="Helvetica",
            fontSize=10, leading=13, textColor=_BRAND,
        ),
        "audit": ParagraphStyle(
            "audit", parent=base["Normal"], fontName="Helvetica",
            fontSize=9, leading=13, textColor=colors.HexColor("#334155"),
            leftIndent=2, spaceAfter=3,
        ),
        "notice": ParagraphStyle(
            "notice", parent=base["Normal"], fontName="Helvetica",
            fontSize=10, leading=15, textColor=colors.HexColor("#334155"),
            spaceAfter=8,
        ),
    }


def _footer(canvas, doc, doc_id: str) -> None:
    """Draw the footer (merchant line + document id) on every page."""
    canvas.saveState()
    width, _height = A4
    canvas.setStrokeColor(_LINE)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 15 * mm, width - 20 * mm, 15 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(_MUTED)
    canvas.drawCentredString(width / 2, 10.5 * mm, "RecoverFlow | Powered by Razorpay")
    canvas.setFont("Helvetica", 7.5)
    canvas.drawCentredString(width / 2, 6.5 * mm, f"Document ID: {doc_id}")
    canvas.restoreState()


def _invoice_details_table(session: dict, invoice: dict, st) -> Table:
    """Two-column label/value table for Section 1."""
    overdue = _days_overdue(invoice, session)
    overdue_label = f"{overdue} day" if overdue == 1 else f"{overdue} days"
    rows = [
        ["Merchant", _escape(invoice.get("merchant_name") or session.get("merchant_name") or MERCHANT_NAME)],
        ["Debtor", _escape(session.get("company_name") or session.get("debtor_name") or "—")],
        ["Contact", _escape(session.get("debtor_name") or "—")],
        ["Invoice ID", _escape(session.get("invoice_id") or invoice.get("invoice_id") or "—")],
        ["Invoice amount", _inr(invoice.get("amount") or session.get("invoice_amount"))],
        ["Due date", _fmt_date(invoice.get("due_date"))],
        ["Days overdue", overdue_label],
    ]
    data = [[Paragraph(_escape(label), st["label"]), Paragraph(value, st["value"])]
            for label, value in rows]
    table = Table(data, colWidths=[40 * mm, 120 * mm])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, _LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _recovery_attempts(session: dict, st) -> list:
    """Build the Section 2 list of audit-log entries, plus the escalation reason."""
    log = session.get("audit_log") or []
    flow = []
    if not log:
        flow.append(Paragraph("No recovery attempts recorded.", st["audit"]))
        return flow

    for entry in log:
        if not isinstance(entry, dict):
            continue
        ts = _fmt_ts(entry.get("timestamp"))
        desc = _describe_event(entry)
        # Bullet (U+2022) is WinAnsi-safe and renders in reportlab's Helvetica,
        # unlike ✓ (U+2713) which drops out of the standard 14 fonts.
        flow.append(Paragraph(
            f"• [{ts}] — {_escape(desc)}", st["audit"],
        ))
    return flow


def _document_verification_section(session: dict, st) -> list:
    """Section 3 — only shown when a document was uploaded for verification."""
    uploaded = session.get("document_uploaded") or bool(session.get("document_verification"))
    if not uploaded:
        return []

    doc = session.get("document_verification") or {}
    verdict = doc.get("verdict") or "—"
    reason = (doc.get("merchant_flag")
              or doc.get("recommended_action")
              or doc.get("debtor_friendly_response")
              or "—")
    rows = [
        ["Document uploaded", "Yes"],
        ["Verification verdict", _escape(str(verdict))],
        ["Reason / note", _escape(str(reason))],
    ]
    if doc.get("extracted_utr"):
        rows.append(["Extracted UTR", _escape(str(doc["extracted_utr"]))])

    data = [[Paragraph(_escape(l), st["label"]), Paragraph(v, st["value"])]
            for l, v in rows]
    table = Table(data, colWidths=[40 * mm, 120 * mm])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, _LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return [table]


def _debtor_stated_amount(session: dict) -> int | None:
    """The rupee amount the debtor said they could pay.

    Primary source is ``last_debtor_offer`` (Python's most-recent-offer field).
    When that's absent — e.g. the Mongo flow doesn't persist it across restarts —
    fall back to scanning the debtor's own messages for the most recent amount
    they mentioned, so an offer made early in the chat (even if later messages
    omitted a figure) is never dropped.
    """
    offer = session.get("last_debtor_offer")
    if isinstance(offer, (int, float)) and offer > 0:
        return int(offer)

    try:
        from backend.agent import _extract_amount_rupees
    except Exception:  # pragma: no cover - never expected; keep the PDF standalone
        _extract_amount_rupees = None

    amount: int | None = None
    if _extract_amount_rupees is not None:
        for m in session.get("messages") or []:
            if isinstance(m, dict) and m.get("role") == "user":
                found = _extract_amount_rupees(m.get("content") or "")
                if found is not None:
                    amount = found  # keep the last stated amount
    return amount


def _negotiation_summary_section(session: dict, st) -> list:
    """Section 3 — the gap between the debtor's stated amount and the AI's ask.

    The AI's minimum ask is a function of the debtor's trust score (score →
    tier → floor percentage), computed once by the NegotiationEngine and frozen
    at session start. The debtor's stated amount is what they told us they
    could pay (most recent offer, or the last amount in their messages).
    """
    engine = session.get("negotiation_engine") or {}

    debtor_offer = _debtor_stated_amount(session)
    offer_text = _inr(debtor_offer) if debtor_offer is not None else "—"

    min_today = engine.get("min_today")
    ask_text = _inr(min_today) if min_today is not None else "—"

    trust_score = session.get(
        "display_trust_score", session.get("trust_score", session.get("score", 0))
    )
    trust_tier = (
        session.get("display_trust_tier")
        or (session.get("trust_score_result") or {}).get("tier")
        or engine.get("tier")
        or session.get("tier")
        or "—"
    )

    rows = [
        ["Debtor's stated payable amount", offer_text],
        ["AI's minimum ask (trust score-based)", ask_text],
        ["Trust score", _escape(f"{trust_score} (Tier {trust_tier})")],
    ]

    data = [[Paragraph(_escape(l), st["label"]), Paragraph(v, st["value"])]
            for l, v in rows]
    table = Table(data, colWidths=[40 * mm, 120 * mm])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, _LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return [table]


def generate_escalation_pdf(session: dict, invoice: dict) -> str:
    """Generate the L3 escalation PDF and return its file path.

    ``session`` is the negotiation session dict; ``invoice`` is the invoice
    record (the session's ``current_invoice``). The PDF is written to
    ``/tmp/escalation_{invoice_id}.pdf`` (overwriting any previous copy) and the
    returned path is cached by the caller so it is generated only once.
    """
    invoice = invoice or {}
    invoice_id = session.get("invoice_id") or invoice.get("invoice_id") or "UNKNOWN"
    now = _ist_now()
    doc_id = f"ESC-{invoice_id}-{now.strftime('%Y%m%d%H%M%S')}"
    out_path = PDF_DIR / f"escalation_{invoice_id}.pdf"

    st = _styles()
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=24 * mm,
        title="Formal Payment Recovery Notice",
        author="RecoverFlow",
    )

    story = [
        Paragraph("FORMAL PAYMENT RECOVERY NOTICE", st["title"]),
        Paragraph("Generated by RecoverFlow", st["subtitle"]),
        Paragraph(f"Date &amp; time (IST): {now.strftime('%d %b %Y, %I:%M %p')}", st["subtitle"]),
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1.2, color=_ACCENT),
    ]

    # SECTION 1 — Invoice details
    story.append(Paragraph("1. INVOICE DETAILS", st["section"]))
    story.append(_invoice_details_table(session, invoice, st))

    # SECTION 2 — Recovery attempts
    story.append(Paragraph("2. RECOVERY ATTEMPTS", st["section"]))
    story.extend(_recovery_attempts(session, st))

    # SECTION 3 — Negotiation summary (debtor's stated amount vs AI's ask)
    story.append(Paragraph("3. NEGOTIATION SUMMARY", st["section"]))
    story.extend(_negotiation_summary_section(session, st))

    # SECTION 4 — Document verification (conditional)
    doc_section = _document_verification_section(session, st)
    if doc_section:
        story.append(Paragraph("4. DOCUMENT VERIFICATION", st["section"]))
        story.extend(doc_section)
        notice_idx = "5"
    else:
        notice_idx = "4"

    # SECTION 5 — Escalation notice
    story.append(Paragraph(f"{notice_idx}. ESCALATION NOTICE", st["section"]))
    story.append(Paragraph(
        "Despite multiple attempts to resolve this payment amicably, no agreement "
        "was reached. This invoice is now referred for formal recovery action.",
        st["notice"],
    ))
    story.append(Paragraph(
        "This document serves as evidence of compliant recovery attempts made "
        "before escalation, in accordance with the RBI Fair Practices Code.",
        st["notice"],
    ))

    doc.build(
        story,
        onFirstPage=lambda c, d: _footer(c, d, doc_id),
        onLaterPages=lambda c, d: _footer(c, d, doc_id),
    )
    return str(out_path)
