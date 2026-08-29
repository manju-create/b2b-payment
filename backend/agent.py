"""
RecoverFlow — Negotiation Agent
=================================
Conducts real-time B2B payment recovery conversations as "Aria" — a warm,
human-sounding financial advisor. The conversation itself is driven by DeepSeek
(a single intelligent system prompt plus function-calling tools). Python keeps
only the deterministic parts: tool execution, session state updates, stopping
rules, and trust-score calculation.

Sessions are in-memory dicts (no DB yet).

Environment
-----------
Requires DEEPSEEK_API_KEY for LLM-driven turns.

Public API
----------
create_session(invoice_id)          -> session_dict
open_turn(session)                  -> (agent_reply, session)
process_turn(session, message)      -> (agent_reply, session)
handle_document_verdict(session, situation, result) -> (agent_reply, session)
simulate_debtor_turn(session, ...)  -> str
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Auto-load .env if python-dotenv is installed (graceful fallback if not)
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from backend.scoring import (  # noqa: E402
    score_debtor,
    get_negotiation_stance,
    project_score_change,
    calculate_trust_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MERCHANT_NAME = "RecoverFlow Demo Merchant"
AGENT_NAME = "Aria"
MODEL = "deepseek-chat"
MAX_TOKENS = 1024
DATA_DIR = REPO_ROOT / "data"
logger = logging.getLogger(__name__)

# Situation labels (used by the document-verification flow + dashboard).
SITUATION_CASHFLOW = "CASHFLOW"
SITUATION_DISPUTE = "DISPUTE"
SITUATION_ALREADY_PAID = "ALREADY_PAID"
SITUATION_INSTALLMENTS = "INSTALLMENTS"

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

_DEBTORS_CACHE: dict[str, dict] | None = None
_INVOICES_CACHE: dict[str, dict] | None = None


def _load_debtors() -> dict[str, dict]:
    global _DEBTORS_CACHE
    if _DEBTORS_CACHE is None:
        raw = json.loads((DATA_DIR / "debtors.json").read_text())
        _DEBTORS_CACHE = {d["debtor_id"]: d for d in raw}
    return _DEBTORS_CACHE


def _load_invoices() -> dict[str, dict]:
    global _INVOICES_CACHE
    if _INVOICES_CACHE is None:
        raw = json.loads((DATA_DIR / "invoices.json").read_text())
        _INVOICES_CACHE = {inv["invoice_id"]: inv for inv in raw}
    return _INVOICES_CACHE


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _rupees(paise: int) -> str:
    """Format paise integer as ₹ with Indian comma grouping (lakhs/crores)."""
    rupees = paise // 100
    if rupees < 0:
        return f"-{_rupees(-paise)}"
    s = str(rupees)
    if len(s) <= 3:
        return f"₹{s}"
    # Indian grouping: last 3 digits, then groups of 2
    last3 = s[-3:]
    rest = s[:-3]
    groups = []
    while len(rest) > 2:
        groups.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.append(rest)
    groups.reverse()
    return f"₹{','.join(groups)},{last3}"


def _rupees_digits(paise: int) -> str:
    """Grouped rupee digits without the ₹ symbol (for prompt templates)."""
    return _rupees(paise).replace("₹", "").lstrip("-")


def format_date(iso: str) -> str:
    """Format an ISO 'YYYY-MM-DD' date as a human-readable string (e.g. '26 Aug 2026')."""
    try:
        return datetime.strptime(iso, "%Y-%m-%d").strftime("%d %b %Y")
    except ValueError:
        return iso


def _first_name(name: str) -> str:
    return (name or "there").strip().split()[0]


def _normalize_no_discount_plan(session: dict, plan: dict) -> dict:
    """Return payment terms with no discount applied."""
    invoice_paise = session["invoice_amount_paise"]
    normalized = dict(plan)
    upfront = normalized.get("upfront_amount", 0)
    deferred = max(0, invoice_paise - upfront)

    normalized["deferred_amount_raw"] = deferred
    normalized["deferred_amount"] = deferred
    normalized["discount_amount"] = 0
    normalized["total_payable"] = upfront + deferred
    normalized["deferred_display"] = _rupees(deferred)
    normalized["discount_display"] = "₹0"
    normalized["total_display"] = _rupees(upfront + deferred)
    return normalized


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit(session: dict, event: str, **kwargs) -> None:
    session["audit_log"].append({
        "event": event,
        "timestamp": _ts(),
        "invoice_id": session["invoice_id"],
        "session_id": session["session_id"],
        **kwargs,
    })


# ---------------------------------------------------------------------------
# Live trust score — recalculated at session start and after every debtor turn
# ---------------------------------------------------------------------------

def _signal_reason(old_signals: dict, new_signals: dict) -> str:
    """Technical list of which signals changed between two turns (agent-facing)."""
    changes: list[str] = []
    for key in sorted(set(old_signals) | set(new_signals)):
        before = int(old_signals.get(key, 0))
        after = int(new_signals.get(key, 0))
        if after != before:
            changes.append(f"{key} {after - before:+d}")
    return ", ".join(changes) if changes else "no signal change"


def _friendly_signal_reason(session: dict, old_signals: dict, new_signals: dict) -> str:
    """Debtor-facing, plain-language reason for the score change this turn."""
    phrases: list[str] = []
    for key in sorted(set(old_signals) | set(new_signals)):
        before = int(old_signals.get(key, 0))
        after = int(new_signals.get(key, 0))
        if after == before:
            continue
        if key == "voluntary_partial_offer":
            phrases.append("You offered a partial payment")
        elif key == "response_engagement":
            phrases.append("You responded quickly" if after > before else "Slow response")
        elif key == "negotiation_behaviour":
            if session.get("accepted_first_offer"):
                phrases.append("You accepted the first offer")
            elif session.get("offers_rejected", 0) >= 2:
                phrases.append("Multiple offers rejected")
            elif session.get("negotiated_down"):
                phrases.append("You negotiated the terms down")
            else:
                phrases.append("Your negotiation behaviour")
        elif key == "current_dpd":
            phrases.append("The invoice is overdue")
        elif key == "on_time_rate":
            phrases.append("Your payment history")
        elif key == "avg_days_late":
            phrases.append("Your average payment delay")
        elif key == "dispute_history":
            phrases.append("Your dispute history")
        elif key == "repeat_customer":
            phrases.append("Your purchase history")
        elif key == "invoice_size_vs_typical":
            phrases.append("This invoice amount")
        else:
            phrases.append(key)
    return ", ".join(phrases) if phrases else "Your trust score is unchanged"


def _refresh_trust_score(session: dict) -> None:
    """Recompute the live trust score and rebuild the system prompt."""
    prev = session.get("trust_score_result")
    result = calculate_trust_score(
        session["debtor_history"], session["current_invoice"], session
    )

    if prev is None:
        delta = 0
        signal_reason = "initial assessment"
        friendly_reason = "initial assessment"
    else:
        delta = int(result["score"]) - int(prev["score"])
        signal_reason = _signal_reason(prev.get("signals", {}), result.get("signals", {}))
        friendly_reason = _friendly_signal_reason(
            session, prev.get("signals", {}), result.get("signals", {})
        )

    session["trust_score_result"] = result
    session["trust_score"] = int(result["score"])
    session["trust_score_delta"] = delta
    session["trust_score_reason"] = friendly_reason          # debtor-facing
    session["trust_score_signal_reason"] = signal_reason      # agent/internal

    _audit(
        session,
        "trust_score",
        score=result["score"],
        tier=result["tier"],
        delta=delta,
        signal_reason=signal_reason,
        reason=friendly_reason,
        signals=result["signals"],
        min_acceptance_pct=result["negotiation_flex"]["min_acceptance_pct"],
        tone=result["negotiation_flex"]["tone"],
    )

    # Rebuild the system prompt so the trust block reflects the latest turn.
    session["system_prompt"] = build_system_prompt(session)


def _finalize_turn(session: dict, reply: str) -> tuple[str, dict]:
    """Refresh the trust score after a debtor turn, then stamp the reply time."""
    _refresh_trust_score(session)
    session["last_agent_ts"] = _ts()
    return reply, session


# ---------------------------------------------------------------------------
# PART 1: SESSION INITIALISATION
# ---------------------------------------------------------------------------

def create_session(invoice_id: str) -> dict:
    """Load invoice + debtor, recompute score, return initialised session dict."""
    invoices = _load_invoices()
    debtors = _load_debtors()

    if invoice_id not in invoices:
        raise ValueError(f"Invoice {invoice_id!r} not found")
    invoice = invoices[invoice_id]
    debtor_id = invoice["debtor_id"]
    if debtor_id not in debtors:
        raise ValueError(f"Debtor {debtor_id!r} not found")
    debtor = debtors[debtor_id]

    # Always recompute — never trust stale tier field in JSON
    score_result = score_debtor(debtor, invoice)
    tier  = score_result["tier"]
    score = int(score_result["score"])

    # Negotiate based on trust score, not fixed tier buckets
    stance = get_negotiation_stance(score)

    projected_full     = project_score_change(score, "full_upfront")
    projected_partial  = project_score_change(score, "partial_deferred")
    projected_escalate = project_score_change(score, "escalated")

    # Store amounts in paise internally
    invoice_amount_paise = invoice["amount"] * 100
    # Floor is always 20% — universal hard floor
    HARD_FLOOR_PCT = 20
    min_now_paise = round(invoice_amount_paise * HARD_FLOOR_PCT / 100)

    session: dict[str, Any] = {
        "session_id": str(uuid4()),
        "invoice_id": invoice_id,
        "debtor_id": debtor_id,
        "debtor_name": debtor["contact_name"],
        "company_name": debtor["company_name"],
        "invoice_amount_paise": invoice_amount_paise,
        "invoice_amount": invoice["amount"],   # rupees, for display/math
        "dpd": invoice["dpd"],
        "simulated_outcome": invoice.get("simulated_outcome", "clean_settlement"),
        "score":             score,
        "tier":              tier,              # kept for display only
        "stance":            stance,            # drives negotiation logic
        "negotiation_floor": HARD_FLOOR_PCT,
        "min_now_paise":     min_now_paise,
        "score_projections": {
            "full_upfront":     projected_full,
            "partial_deferred": projected_partial,
            "escalated":        projected_escalate,
        },
        "turn_count": 0,
        "max_turns": 8,
        "status": "active",
        "messages": [],
        "audit_log": [],
        "razorpay_order_id": None,
        "payment_order": None,
        "payment_amount": None,
        "agreed_terms": None,
        "recovered_paise": 0,
        "system_prompt": "",
        # --- conversation-flow state ---
        "identified_situation": None,   # set by flag_dispute / document flow
        "situation_claim": None,        # the debtor's own words that triggered the situation
        "dispute_evidence": None,
        "payment_claim_evidence": None,
        "promise_to_pay": None,         # set by set_promise_to_pay
        "pending_upload": None,         # set by request_document_upload
        # --- document upload / verification state ---
        "upload_attempts": 0,
        "document_verification": None,   # latest verify_document() result (merged)
        "merchant_flag": None,           # one-line merchant dashboard flag
        # --- live trust-score state ---
        "debtor_history": debtor,
        "current_invoice": invoice,
        "last_agent_ts": None,
        "last_debtor_ts": None,
        "voluntary_partial_offered": False,
        "partial_after_suggested": False,
        "accepted_first_offer": False,
        "offers_rejected": 0,
        "negotiated_down": False,
        "trust_score_result": None,
        "trust_score": 0,
        "trust_score_delta": 0,
        "trust_score_reason": "initial assessment",
        "trust_score_signal_reason": "initial assessment",
    }
    _refresh_trust_score(session)  # computes trust score + builds system prompt

    # Freeze the payment-history trust score for display. The live score
    # (session["trust_score"]) keeps moving with negotiation signals on every
    # turn, but the debtor card and the merchant dashboard must show the SAME
    # number — so both read this stable snapshot taken at session start.
    session["display_trust_score"] = session["trust_score"]
    session["display_trust_tier"]  = (session["trust_score_result"] or {}).get("tier", tier)

    _audit(session, "session_created",
           tier=tier, score=score,
           invoice_amount_paise=invoice_amount_paise,
           cold_start=score_result["cold_start"])
    return session


# ---------------------------------------------------------------------------
# PART 2: SYSTEM PROMPT — a single intelligent prompt (no flow logic)
# ---------------------------------------------------------------------------

def _render_history(messages: list[dict]) -> str:
    """Render the conversation so far as a readable transcript for the prompt."""
    lines: list[str] = []
    for m in messages:
        role = "Debtor" if m.get("role") == "user" else "Aria"
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(no conversation yet)"


def build_system_prompt(session: dict) -> str:
    """Build Aria's system prompt — instructions to a smart human, not a flowchart."""
    amount = _rupees_digits(session["invoice_amount_paise"])
    min_amount = _rupees_digits(session.get("min_now_paise", round(session["invoice_amount_paise"] * 0.20)))
    trust_score = session.get("trust_score", 0)
    history = _render_history(session.get("messages", []))

    return f"""You are Aria, a warm and intelligent payment recovery specialist at {MERCHANT_NAME}. You are having a real human conversation with {session['debtor_name']} about their overdue invoice of ₹{amount}.

YOUR PERSONALITY:
- You speak like a real person — natural, warm, never robotic
- Short messages — 2 sentences maximum per reply
- You actually listen and remember everything said in this conversation
- You never ask for something the debtor already told you
- You never repeat yourself
- You adapt completely to what the debtor says

YOUR GOAL:
Recover as much of ₹{amount} as possible, as soon as possible — but in a way that feels helpful, not pushy. The debtor should feel like you're on their side.

WHAT YOU KNOW SO FAR:
Invoice ID: {session['invoice_id']}
Amount due: ₹{amount}
Days overdue: {session['dpd']}
Debtor trust score: {trust_score}/100 (DO NOT mention this)
Minimum you can accept: ₹{min_amount} (DO NOT mention this)

CONVERSATION HISTORY:
{history}

HOW TO HANDLE COMMON SITUATIONS:
Use your own judgment — but here are guidelines:

If they say they'll pay tomorrow or on a specific date:
→ Believe them, confirm the date, wish them well, end warmly
→ Do NOT push for payment today

If they offer a partial amount:
→ If it's reasonable — accept it immediately, sort the rest later
→ If it's very low — ask what's making it tight, understand first
→ Never flatly reject an offer

If they say they already paid:
→ Take it seriously, ask for UTR or transaction reference
→ Tell them you'll get it checked right away

If they dispute the invoice:
→ Stop asking for money completely
→ Ask what specifically looks wrong
→ Tell them you'll flag it for review

If they give a reason they can't pay:
→ Acknowledge it genuinely
→ Work around their reality, not your minimum

If they seem confused or upset:
→ Slow down, acknowledge how they feel
→ One simple question at a time

TOOLS AVAILABLE TO YOU:
- generate_payment_link: use when debtor agrees to pay NOW
- set_promise_to_pay: use when debtor commits to a future date
- flag_dispute: use when debtor disputes the invoice
- request_document_upload: use when you need proof from debtor
- escalate: use only when debtor is completely unresponsive

RULES:
- Never mention tier, trust score, minimum thresholds
- Never use: "kindly", "as per", "please be advised", "I understand that", "I appreciate"
- Never ask two questions in one message
- Never repeat what debtor just said back to them
- If debtor said it already — never ask again
- Always end on a warm, human note"""


# ---------------------------------------------------------------------------
# PART 3: TOOL DEFINITIONS + HANDLERS
# ---------------------------------------------------------------------------

# OpenAI function-calling format — each entry wraps the schema under
# {"type": "function", "function": {...}}.
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "generate_payment_link",
            "description": (
                "Create a payment link for the debtor to pay the agreed amount now. "
                "Call only when the debtor has explicitly agreed to pay now."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount in rupees the debtor agreed to pay now"},
                },
                "required": ["amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_promise_to_pay",
            "description": (
                "Record the debtor's commitment to pay on a future date. "
                "Use when the debtor commits to a specific date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "The date the debtor committed to pay, e.g. '2026-09-05' or 'next Friday'"},
                    "amount": {"type": "number", "description": "The amount they committed to pay (optional)"},
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_dispute",
            "description": "Flag the invoice as disputed and stop asking for payment. Use when the debtor disputes the invoice.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "What the debtor says is wrong with the invoice"},
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_document_upload",
            "description": (
                "Ask the debtor to upload a document as proof (payment receipt, "
                "invoice copy, bank statement, business closure letter, etc.)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_type": {"type": "string", "description": "What kind of document is needed, e.g. 'payment receipt' or 'bank statement'"},
                    "reason": {"type": "string", "description": "Why the document is needed"},
                },
                "required": ["document_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate",
            "description": "Escalate to a human. Use only when the debtor is completely unresponsive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "enum": ["debtor_unresponsive"], "description": "Reason for escalation"},
                },
                "required": ["reason"],
            },
        },
    },
]


# ---- Handlers ---------------------------------------------------------------

def _handle_get_invoice_details(inputs: dict, session: dict) -> dict:
    invoices = _load_invoices()
    debtors = _load_debtors()
    inv_id = inputs["invoice_id"]
    if inv_id not in invoices:
        return {"error": f"Invoice {inv_id} not found"}
    inv = invoices[inv_id]
    debtor = debtors.get(inv["debtor_id"], {})
    return {
        "invoice_id": inv_id,
        "amount_rupees": inv["amount"],
        "amount_formatted": _rupees(inv["amount"] * 100),
        "dpd": inv["dpd"],
        "due_date": inv["due_date"],
        "debtor_name": debtor.get("contact_name", "Unknown"),
        "company_name": debtor.get("company_name", "Unknown"),
        "merchant_name": MERCHANT_NAME,
    }


def _handle_validate_proposed_terms(inputs: dict, session: dict) -> dict:
    now_pct    = inputs["now_pct"]
    # Accept either now_amount_rupees (rupees) or upfront_offered_paise (paise)
    invoice_paise = session["invoice_amount_paise"]
    if "now_amount_rupees" in inputs:
        now_amount = round(inputs["now_amount_rupees"] * 100)
    elif "upfront_offered_paise" in inputs:
        now_amount = inputs["upfront_offered_paise"]
    else:
        now_amount = round(invoice_paise * now_pct / 100)

    HARD_FLOOR = 20
    floor_paise = round(invoice_paise * HARD_FLOOR / 100)

    if now_pct < HARD_FLOOR:
        return {
            "valid": False,
            "violations": [
                f"Cannot accept below 20% upfront. "
                f"Minimum is ₹{invoice_paise * 0.20:,.0f}"
            ],
        }

    if now_amount > invoice_paise:
        return {
            "valid": False,
            "violations": [
                f"Amount {_rupees(now_amount)} exceeds invoice total {_rupees(invoice_paise)}"
            ],
        }

    deferred = invoice_paise - now_amount
    upfront_pct = round(now_amount / invoice_paise * 100, 1)
    st = session.get("stance", {})

    _audit(session, "terms_validated",
           now_pct=now_pct, upfront_amount=now_amount, deferred_amount=deferred)

    return {
        "valid": True,
        "violations": [],
        "computed_plan": {
            "upfront_amount":  now_amount,
            "upfront_pct":     upfront_pct,
            "deferred_amount": deferred,
            "deferred_pct":    round(100 - upfront_pct, 1),
        },
        "note": (
            "above_target" if now_pct >= st.get("target", 0)
            else "below_target_but_valid"
        ),
    }


def _handle_generate_payment_link(inputs: dict, session: dict) -> dict:
    """Create a Razorpay Order for the Checkout JS flow.

    Enforces the 20% floor in Python (the LLM is told the minimum but this is
    the hard guardrail). Returns the order info dict the frontend needs to open
    the Checkout modal.
    """
    from backend.razorpay_client import create_order

    amount_inr  = inputs["amount"]            # rupees
    amount_paise = round(amount_inr * 100)
    min_now = session.get("min_now_paise", round(session["invoice_amount_paise"] * 0.20))

    if amount_paise < min_now:
        return {"error": "Amount is below the minimum the agent can accept."}
    if amount_paise > session["invoice_amount_paise"]:
        return {"error": "Amount exceeds the invoice total."}

    invoice_id  = inputs.get("invoice_id", session["invoice_id"])

    order = create_order(
        amount_inr=amount_inr,
        invoice_id=invoice_id,
        session_id=session["session_id"],
        debtor_name=session["debtor_name"],
    )

    session["razorpay_order_id"] = order["id"]
    session["payment_amount"]    = amount_inr
    session["status"]            = "awaiting_payment"

    # Build agreed_terms if not already set (LLM called this directly)
    if not session.get("agreed_terms"):
        invoice_paise = session["invoice_amount_paise"]
        upfront_paise = amount_paise
        deferred_paise = max(0, invoice_paise - upfront_paise)
        st = session.get("stance", {})
        max_days = st.get("max_days", 30)
        due_date_str = (date.today() + timedelta(days=max_days)).isoformat()
        session["agreed_terms"] = _normalize_no_discount_plan(session, {
            "upfront_amount":      upfront_paise,
            "upfront_pct":         round(upfront_paise / invoice_paise * 100, 1),
            "deferred_amount_raw": deferred_paise,
            "deferred_pct":        round(deferred_paise / invoice_paise * 100, 1),
            "deferred_days":       max_days,
            "deferred_due_date":   due_date_str,
            "upfront_display":     _rupees(upfront_paise),
            "due_date_display":    format_date(due_date_str),
        })
        if deferred_paise > 0:
            _audit(session, "deferred_scheduled",
                   deferred_amount=deferred_paise,
                   due_date=due_date_str)
    else:
        session["agreed_terms"] = _normalize_no_discount_plan(session, session["agreed_terms"])

    # RAZORPAY_KEY_ID is public — the frontend uses it to open Checkout JS.
    order_info = {
        "order_id":       order["id"],
        "amount":         amount_inr,
        "amount_display": _rupees(round(amount_inr * 100)),
        "key_id":         os.environ.get("RAZORPAY_KEY_ID", ""),
        "debtor_name":    session["debtor_name"],
        "invoice_id":     invoice_id,
        "session_id":     session["session_id"],
    }
    session["payment_order"] = order_info

    _audit(session, "razorpay_order_created",
           order_id=order["id"], amount=amount_inr)

    return order_info


def _handle_set_promise_to_pay(inputs: dict, session: dict) -> dict:
    """Record the debtor's commitment to pay on a future date."""
    promise_date = inputs.get("date", "")
    amount = inputs.get("amount")
    session["promise_to_pay"] = {
        "date": promise_date,
        "amount": amount,
        "recorded_at": _ts(),
    }
    session["status"] = "promise_to_pay"
    _audit(session, "promise_to_pay_set", date=promise_date, amount=amount)
    return {"status": "promise_to_pay", "date": promise_date, "amount": amount}


def _handle_flag_dispute(inputs: dict, session: dict) -> dict:
    """Flag the invoice as disputed and stop collecting."""
    reason = inputs.get("reason", "")
    session["status"] = "disputed"
    session["identified_situation"] = SITUATION_DISPUTE
    session["dispute_evidence"] = {"reason": reason}
    _audit(session, "dispute_flagged", reason=reason)
    return {"status": "disputed", "reason": reason}


def _situation_for_document_type(document_type: str) -> str:
    """Map a requested document type to a verifier situation label."""
    d = (document_type or "").lower()
    if any(w in d for w in ("payment", "receipt", "utr", "transfer", "paid", "transaction")):
        return "ALREADY_PAID"
    if any(w in d for w in ("invoice", "dispute", "contract", "agreement", "order", "quotation")):
        return "DISPUTE"
    if any(w in d for w in ("bank", "statement", "closure", "medical", "cashflow", "cash flow", "loss", "hardship")):
        return "CANNOT_PAY"
    return "GENERAL"


def _handle_request_document_upload(inputs: dict, session: dict) -> dict:
    """Ask the debtor to upload a document; flag the frontend to show the card."""
    document_type = inputs.get("document_type", "")
    reason = inputs.get("reason", "")
    situation = _situation_for_document_type(document_type)
    session["pending_upload"] = {
        "document_type": document_type,
        "reason": reason,
        "situation": situation,
    }
    _audit(session, "document_upload_requested",
           document_type=document_type, reason=reason, situation=situation)
    return {"requested": True, "document_type": document_type}


_ESCALATION_MESSAGES = {
    "max_turns_reached": (
        "I've taken this as far as I can here. Our team will reach out to you shortly."
    ),
    "debtor_dispute": (
        "I've passed everything across to the team. Sit tight — they'll be in touch within 2 business days."
    ),
    "debtor_requested_human": (
        "No problem — I'll connect you with a real person on our team who'll reach out shortly."
    ),
    "debtor_unresponsive": (
        "No rush — I'll leave this with our team and they'll follow up when you're ready."
    ),
}


def _handle_escalate(inputs: dict, session: dict) -> dict:
    reason = inputs.get("reason", "debtor_unresponsive")
    session["status"] = "escalated"
    _audit(session, "escalation_triggered", reason=reason)
    return {
        "closing_message": _ESCALATION_MESSAGES.get(reason, "This matter has been escalated."),
        "status": session["status"],
    }


def _execute_tool(name: str, tool_input: dict, session: dict) -> Any:
    dispatch = {
        "get_invoice_details":     _handle_get_invoice_details,
        "validate_proposed_terms": _handle_validate_proposed_terms,
        "generate_payment_link":   _handle_generate_payment_link,
        "set_promise_to_pay":      _handle_set_promise_to_pay,
        "flag_dispute":            _handle_flag_dispute,
        "request_document_upload": _handle_request_document_upload,
        "escalate":                _handle_escalate,
    }
    handler = dispatch.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}
    return handler(tool_input, session)


# ---------------------------------------------------------------------------
# PART 4: DOCUMENT VERIFICATION — act on the verifier's verdict
# ---------------------------------------------------------------------------
#
# `situation` here uses the verifier's labels ("DISPUTE" | "ALREADY_PAID" |
# "CANNOT_PAY" | "GENERAL"). CANNOT_PAY maps to the agent's CASHFLOW situation.

_MAX_UPLOAD_ATTEMPTS = 2


def _flag_for_accept(situation: str, result: dict) -> str:
    """Build the merchant dashboard flag line for an ACCEPT_CLAIM verdict."""
    if situation == "ALREADY_PAID":
        utr = result.get("extracted_utr")
        return f"✅ Payment proof verified — UTR: {utr}" if utr else "✅ Payment proof verified"
    if situation == "DISPUTE":
        disc = result.get("amount_discrepancy")
        return (
            f"⚠️ Dispute verified — amount discrepancy: {disc}" if disc
            else "⚠️ Dispute verified"
        )
    return "✅ Unable-to-pay proof verified — installment plan offered"


def _pivot_to_negotiation_reply(session: dict) -> str:
    """Anchor the conversation on a concrete upfront amount instead of closing.

    Used when a document can't be verified — keep the chat open, state the
    amount we're hoping to collect today, and invite the debtor to negotiate.
    """
    stance = session.get("stance", {})
    opening_pct = stance.get("opening", 50)
    expected_paise = round(session["invoice_amount_paise"] * opening_pct / 100)
    expected = _rupees(expected_paise)
    return (
        f"Thanks for sharing that — I couldn't fully verify the document, but "
        f"let's keep this moving. Based on your account, around {expected} today "
        f"would settle it. Does that work, or what can you manage?"
    )


def handle_document_verdict(
    session: dict, situation: str, result: dict
) -> tuple[str, dict]:
    """
    Act on a document verification result. Returns (agent_reply, session).

    Mutates the session: sets status / merchant_flag / document_verification,
    appends to messages + audit log, and (for CANNOT_PAY ACCEPT) moves the
    conversation toward an installment offer.
    """
    attempt = session.get("upload_attempts", 0)
    action = result.get("recommended_action", "ESCALATE_TO_MERCHANT")

    # Enforce the 2-attempt cap: a second unreadable/inconclusive document
    # is escalated rather than asking for proof a third time.
    if action == "REQUEST_BETTER_PROOF" and attempt >= _MAX_UPLOAD_ATTEMPTS:
        action = "ESCALATE_TO_MERCHANT"

    debtor_friendly = (result.get("debtor_friendly_response") or "").strip()
    merchant_flag: str

    if action == "ACCEPT_CLAIM":
        if situation == "ALREADY_PAID":
            session["status"] = "payment_claimed_verified"
            merchant_flag = _flag_for_accept(situation, result)
            reply = (
                f"{debtor_friendly} I've flagged this to {MERCHANT_NAME} — you "
                f"won't receive any further payment requests while they confirm."
            ).strip()
        elif situation == "DISPUTE":
            session["status"] = "disputed_verified"
            merchant_flag = _flag_for_accept(situation, result)
            reply = (
                f"{debtor_friendly} I've paused all payment requests and sent "
                f"this to {MERCHANT_NAME} for review."
            ).strip()
        elif situation == "GENERAL":
            # No specific claim — acknowledge receipt, surface to merchant, and
            # keep the conversation open.
            merchant_flag = "📄 Document received — awaiting merchant review"
            reply = debtor_friendly or "Thanks — I've noted that document down."
        else:  # CANNOT_PAY — offer an installment plan and proceed to that flow
            merchant_flag = _flag_for_accept(situation, result)
            reply = (
                f"{debtor_friendly} Based on what you've shared, let's work out "
                f"a plan that's manageable. How does splitting this into 3 "
                f"payments sound?"
            ).strip()
    elif action == "REQUEST_BETTER_PROOF":
        # Warm, never accusatory. The upload card is shown again for one more
        # attempt (the server reads the final recommended_action).
        merchant_flag = "⚠️ Requesting better proof — document inconclusive"
        reply = debtor_friendly or (
            "Thanks — could you share a clearer copy so I can verify this properly?"
        )
    else:  # ESCALATE_TO_MERCHANT → pivot to payment negotiation (keep chat open)
        merchant_flag = "🔴 Manual review needed — document inconclusive"
        reply = _pivot_to_negotiation_reply(session)
        _audit(session, "L3_triggered", reason="document_inconclusive",
               situation=situation, upload_attempt=attempt)

    # Persist the full (merged) result for the dashboard + audit trail.
    session["document_verification"] = {
        **result,
        "situation": situation,
        "upload_attempt": attempt,
        "recommended_action": action,   # final action after the attempt cap
        "merchant_flag": merchant_flag,
    }
    session["merchant_flag"] = merchant_flag

    session["messages"].append({"role": "assistant", "content": reply})
    _audit(
        session,
        "document_verified",
        situation=situation,
        verdict=result.get("verdict"),
        confidence=result.get("confidence"),
        checks=result.get("checks"),
        red_flags=result.get("red_flags"),
        recommended_action=action,
        upload_attempt=attempt,
    )
    _audit(session, "agent_turn", turn=session.get("turn_count", 0),
           speaker="agent", message=reply)
    return reply, session


# ---------------------------------------------------------------------------
# PART 5: TURN FUNCTION (LLM-driven)
# ---------------------------------------------------------------------------

def _get_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def _call_llm(session: dict, client: OpenAI, user_message: str) -> str:
    """Call DeepSeek, run the tool-call loop, and return Aria's final text.

    The conversation history lives in the system prompt, so the API messages
    carry only the current user message plus any tool round-trips for this turn.
    """
    messages: list[dict] = [
        {"role": "system", "content": session["system_prompt"]},
        {"role": "user", "content": user_message},
    ]

    for _ in range(6):   # safety cap on tool-call loops
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        choice = response.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            return (msg.content or "").strip()

        tool_call_dicts = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": tool_call_dicts,
        })

        for tc in tool_calls:
            try:
                tool_input = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_input = {}
            result = _execute_tool(tc.function.name, tool_input, session)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    return "Thanks — let me get that sorted for you."


def _no_key_reply(session: dict) -> str:
    """Warm fallback when no API key is configured (e.g. some tests)."""
    name = _first_name(session["debtor_name"])
    return (
        f"Thanks {name} — I've got that. Our payment team will reach out shortly "
        f"to sort this out for you."
    )


def _is_legal_threat(message: str) -> bool:
    m = message.lower()
    return any(w in m for w in ("lawyer", "legal", "consumer forum", "rbi", "advocate", "court"))


def _requests_human(message: str) -> bool:
    m = message.lower()
    return any(w in m for w in (
        "talk to a human", "speak to a human", "talk to someone", "speak to someone",
        "real person", "human representative", "talk to a real", "call me back",
    ))


def process_turn(session: dict, debtor_message: str) -> tuple[str, dict]:
    """
    Process one debtor message. The conversation is driven by DeepSeek; Python
    only enforces stopping rules and applies tool results to the session.
    """
    if session["status"] != "active":
        return f"[Session {session['status']} — no further turns]", session

    session["turn_count"] += 1
    turn = session["turn_count"]
    session["last_debtor_ts"] = _ts() if debtor_message.strip() else None

    # --- Stopping rule: unresponsive (silent / empty message) ---
    if not debtor_message.strip():
        _audit(session, "debtor_turn", turn=turn, speaker="debtor", message="[silent]")
        if turn >= session["max_turns"]:
            session["status"] = "escalated"
            _audit(session, "stopping_rule", status="escalated", reason="debtor_unresponsive")
            reply = "No rush — I'll leave this with our team and they'll follow up when you're ready."
        else:
            reply = "I'm still here when you're ready to talk this through."
        _audit(session, "agent_turn", turn=turn, speaker="agent", message=reply)
        return _finalize_turn(session, reply)

    _audit(session, "debtor_turn", turn=turn, speaker="debtor", message=debtor_message)

    # --- Stopping rule: legal threat → legal_hold ---
    if _is_legal_threat(debtor_message):
        session["status"] = "legal_hold"
        reply = "Understood — I'll pass this to our team and we won't contact you further on this."
        _audit(session, "stopping_rule", status="legal_hold", reason="debtor_legal_threat")
        session["messages"].append({"role": "user", "content": debtor_message})
        session["messages"].append({"role": "assistant", "content": reply})
        _audit(session, "agent_turn", turn=turn, speaker="agent", message=reply)
        return _finalize_turn(session, reply)

    # --- Stopping rule: debtor asks for a human ---
    if _requests_human(debtor_message):
        session["status"] = "escalated"
        reply = "No problem — I'll connect you with a real person on our team who'll reach out shortly."
        _audit(session, "stopping_rule", status="escalated", reason="debtor_requested_human")
        session["messages"].append({"role": "user", "content": debtor_message})
        session["messages"].append({"role": "assistant", "content": reply})
        _audit(session, "agent_turn", turn=turn, speaker="agent", message=reply)
        return _finalize_turn(session, reply)

    # --- LLM-driven turn ---
    # Build the system prompt from the conversation so far (excludes this turn),
    # then let DeepSeek reason + call tools.
    session["system_prompt"] = build_system_prompt(session)
    session["messages"].append({"role": "user", "content": debtor_message})

    try:
        client = _get_client()
        reply = _call_llm(session, client, debtor_message)
    except EnvironmentError:
        reply = _no_key_reply(session)
    except Exception:
        logger.exception("LLM turn failed")
        reply = "Sorry, I hit a little snag — could you say that again?"

    if not reply or not reply.strip():
        reply = "Thanks — let me get that sorted for you."

    session["messages"].append({"role": "assistant", "content": reply})
    _audit(session, "agent_turn", turn=turn, speaker="agent", message=reply)

    return _finalize_turn(session, reply)


def _build_opening_message(session: dict) -> str:
    """Return Aria's warm opening — greet and invite the debtor to start."""
    name = _first_name(session["debtor_name"])
    amount = _rupees(session["invoice_amount_paise"])
    return (
        f"Hi {name}! I'm {AGENT_NAME} from {MERCHANT_NAME} — your invoice of {amount} "
        f"is a little overdue. Would you like to pay it in full, pay a little now "
        f"and the rest later, or is something else going on?"
    )


def open_turn(session: dict) -> tuple[str, dict]:
    """
    Send Aria's warm opening message. Pure Python — no LLM call needed.
    """
    opening = _build_opening_message(session)
    session["messages"] = [{"role": "assistant", "content": opening}]
    session["last_agent_ts"] = _ts()
    _audit(session, "agent_turn", turn=0, speaker="agent", message=opening)
    return opening, session


# ---------------------------------------------------------------------------
# PART 6: SIMULATED DEBTOR
# ---------------------------------------------------------------------------

def simulate_debtor_turn(
    session: dict,
    simulated_outcome: str | None = None,
    turn_override: int | None = None,
) -> str:
    """
    Return a simulated debtor message for demo/batch runs.

    The LLM-driven agent has no rigid situation stages, so this mostly returns
    an opening message keyed on the simulated outcome.
    """
    outcome = simulated_outcome or session.get("simulated_outcome", "clean_settlement")

    if session.get("turn_count", 0) == 0 and not session.get("messages"):
        if outcome == "dispute":
            return "I don't think this amount is right — we agreed on less."
        if outcome == "repeat_extension":
            return "Can I get more time to pay this? Maybe in a few installments?"
        return "Business has been slow, so I can't pay the full amount right now."

    return ""
