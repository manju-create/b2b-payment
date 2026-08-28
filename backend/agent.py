"""
RecoverFlow — Negotiation Agent
=================================
Conducts real-time B2B payment negotiations via DeepSeek
(OpenAI-compatible API) with function-calling.
Sessions are in-memory dicts (no DB yet).

Environment
-----------
Set DEEPSEEK_API_KEY in your shell or create a .env file at the repo root.
python-dotenv is used if installed; otherwise the env var must be set manually.

Public API
----------
create_session(invoice_id)          -> session_dict
generate_payment_link(...)         -> order info dict
process_turn(session, message)      -> (agent_reply: str, session: dict)
simulate_debtor_turn(session, ...)  -> str
open_turn(session)                  -> (agent_reply: str, session: dict)
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

from backend.scoring import score_debtor  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MERCHANT_NAME = "RecoverFlow Demo Merchant"
MODEL = "deepseek-chat"
MAX_TOKENS = 1024
DATA_DIR = REPO_ROOT / "data"
logger = logging.getLogger(__name__)

TIER_BOUNDS: dict[str, dict[str, int]] = {
    "A": {"min_now_pct": 25, "max_defer_pct": 75, "max_days": 60, "max_discount_pct": 15},
    "B": {"min_now_pct": 40, "max_defer_pct": 60, "max_days": 45, "max_discount_pct": 10},
    "C": {"min_now_pct": 60, "max_defer_pct": 40, "max_days": 30, "max_discount_pct":  5},
    "D": {"min_now_pct": 85, "max_defer_pct": 15, "max_days": 15, "max_discount_pct":  0},
}

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


def format_date(iso: str) -> str:
    """Format an ISO 'YYYY-MM-DD' date as a human-readable string (e.g. '26 Aug 2026')."""
    try:
        return datetime.strptime(iso, "%Y-%m-%d").strftime("%d %b %Y")
    except ValueError:
        return iso


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
    tier = score_result["tier"]
    score = score_result["score"]
    bounds = TIER_BOUNDS[tier]

    # Store amounts in paise internally
    invoice_amount_paise = invoice["amount"] * 100
    min_now_paise = round(invoice_amount_paise * bounds["min_now_pct"] / 100)

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
        "score": score,
        "tier": tier,
        "tier_bounds": bounds,
        "min_now_paise": min_now_paise,
        "turn_count": 0,
        "max_turns": 8,
        "status": "active",
        "messages": [],
        "audit_log": [],
        "razorpay_order_id": None,
        "payment_order": None,          # order info dict returned to the frontend
        "payment_amount": None,         # rupees — the upfront amount for the order
        "agreed_terms": None,
        "recovered_paise": 0,
        "system_prompt": "",
        # --- negotiation-flow state flags ---
        "awaiting_intent": False,               # True after A/B opening sent
        "awaiting_amount": False,               # True after debtor chooses partial
        "awaiting_plan_confirmation": False,    # True after plan presented
        "pending_plan": None,                   # plan dict awaiting confirmation
    }
    session["system_prompt"] = build_system_prompt(session)

    _audit(session, "session_created",
           tier=tier, score=score,
           invoice_amount_paise=invoice_amount_paise,
           cold_start=score_result["cold_start"])
    return session


# ---------------------------------------------------------------------------
# PART 2: SYSTEM PROMPT
# ---------------------------------------------------------------------------

def build_system_prompt(session: dict) -> str:
    b = session["tier_bounds"]
    amount_paise = session["invoice_amount_paise"]
    min_now_paise = session["min_now_paise"]

    return f"""You are a professional payment recovery assistant for {MERCHANT_NAME}.
Your goal is to reach a payment settlement on the outstanding invoice below.

PERSONA:
- Empathetic but firm. Acknowledge cash flow difficulties without conceding terms.
- If asked directly whether you are an AI, confirm you are an automated payment assistant.
- Never reveal the debtor's internal tier rating (A/B/C/D) to the debtor.
- Never fabricate invoice details — use only what is provided here.
- Never make threats not backed by the escalation process described below.

TOOLS AVAILABLE:
- get_invoice_details: look up invoice details mid-conversation if needed
- validate_proposed_terms: MUST call before accepting any debtor counter-offer
- generate_payment_link: call when debtor explicitly agrees to specific terms
- escalate: call for disputes, human requests, unresponsive debtor, or max turns

INVOICE:
  ID       : {session['invoice_id']}
  Amount   : {_rupees(amount_paise)}
  Overdue  : {session['dpd']} days
  Debtor   : {session['debtor_name']} ({session['company_name']})

AUTHORISED TERMS (do not quote exact percentages to the debtor):
  Minimum collect now : {b['min_now_pct']}% = {_rupees(min_now_paise)}
  Maximum defer       : {b['max_defer_pct']}%
  Maximum defer period: {b['max_days']} days
  Maximum discount    : {b['max_discount_pct']}%

HARD RULES (inviolable):
1. Cannot offer better terms than above. If debtor demands better, say you are not
   authorised and offer to escalate to the merchant.
2. Settlement only occurs when you call generate_payment_link after debtor agrees.
3. Always call validate_proposed_terms before accepting any counter-offer.
   If violations exist, you cannot accept those terms.
4. After {session['max_turns']} turns without settlement, call escalate("max_turns_reached").
5. If debtor disputes the invoice amount, call escalate("debtor_dispute").
6. If debtor asks to speak to a human, call escalate("debtor_requested_human").

Begin by greeting the debtor by name, citing the specific invoice, and asking
how you can help them resolve it today."""


# ---------------------------------------------------------------------------
# PART 3: TOOL DEFINITIONS + HANDLERS
# ---------------------------------------------------------------------------

# OpenAI function-calling format — each entry wraps the schema under {"type": "function", "function": {...}}
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_invoice_details",
            "description": "Retrieve invoice details by invoice_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {
                        "type": "string",
                        "description": "Invoice ID e.g. INV-0001",
                    },
                },
                "required": ["invoice_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_proposed_terms",
            "description": (
                "Validate proposed settlement terms against tier bounds. "
                "MUST be called before accepting any debtor counter-offer. "
                "Returns {valid: bool, violations: [str]}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "now_pct":      {"type": "number",  "description": "% to pay now (0-100)"},
                    "defer_pct":    {"type": "number",  "description": "% to defer (0-100)"},
                    "defer_days":   {"type": "integer", "description": "Days for deferred portion"},
                    "discount_pct": {"type": "number",  "description": "% discount offered (0-100)"},
                },
                "required": ["now_pct", "defer_pct", "defer_days", "discount_pct"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_payment_link",
            "description": (
                "Create a Razorpay Order for the agreed upfront amount (Checkout JS flow). "
                "Call only after debtor explicitly agrees and validate_proposed_terms returned valid=true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount":      {"type": "number", "description": "Amount in rupees"},
                    "invoice_id":  {"type": "string", "description": "Invoice ID"},
                },
                "required": ["amount", "invoice_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate",
            "description": "Escalate the negotiation. Returns a closing message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "enum": [
                            "max_turns_reached",
                            "debtor_dispute",
                            "debtor_requested_human",
                            "debtor_unresponsive",
                        ],
                        "description": "Reason for escalation",
                    },
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
    b = session["tier_bounds"]
    now_pct      = inputs["now_pct"]
    defer_pct    = inputs["defer_pct"]
    defer_days   = inputs["defer_days"]
    discount_pct = inputs["discount_pct"]
    tier         = session["tier"]

    invoice_paise = session["invoice_amount_paise"]

    # The exact amount the debtor offered (structured flow passes this so the
    # deferred remainder is exact). Fall back to deriving it from now_pct for
    # the free-form LLM path, which only supplies percentages.
    upfront_offered = inputs.get("upfront_offered_paise")
    if upfront_offered is None:
        upfront_offered = round(invoice_paise * now_pct / 100)

    violations: list[str] = []

    # Minimum upfront check (exact amount — tier percentages validate only)
    min_now_paise = round(invoice_paise * b["min_now_pct"] / 100)
    if upfront_offered < min_now_paise:
        violations.append(
            f"Upfront {_rupees(upfront_offered)} is below minimum "
            f"{_rupees(min_now_paise)} ({b['min_now_pct']}% for your account)"
        )

    # Overpayment check (debtor cannot pay more than the invoice)
    if upfront_offered > invoice_paise:
        violations.append(
            f"Amount {_rupees(upfront_offered)} exceeds invoice "
            f"total {_rupees(invoice_paise)}"
        )

    # Remaining tier guardrails (used by the free-form LLM path)
    if defer_pct > b["max_defer_pct"]:
        violations.append(
            f"defer_pct {defer_pct}% exceeds maximum {b['max_defer_pct']}% for Tier {tier}"
        )
    if defer_days > b["max_days"]:
        violations.append(
            f"defer_days {defer_days} exceeds maximum {b['max_days']} days for Tier {tier}"
        )
    if discount_pct > b["max_discount_pct"]:
        violations.append(
            f"discount_pct {discount_pct}% exceeds maximum {b['max_discount_pct']}% for Tier {tier}"
        )
    if abs(now_pct + defer_pct - 100) > 0.5:
        violations.append(
            f"now_pct ({now_pct}%) + defer_pct ({defer_pct}%) must sum to 100%"
        )

    if violations:
        return {"valid": False, "violations": violations}

    # No violations → compute the actual plan from the REMAINDER.
    # Tier percentages are never used to compute amounts — the deferred
    # portion is always invoice_amount minus what the debtor pays now.
    deferred = invoice_paise - upfront_offered
    computed_plan = {
        "upfront_amount":  upfront_offered,
        "deferred_amount": deferred,
        "deferred_pct":    round(deferred / invoice_paise * 100, 1),
        "upfront_pct":     round(upfront_offered / invoice_paise * 100, 1),
    }
    _audit(session, "terms_validated",
           now_pct=now_pct, defer_pct=defer_pct,
           defer_days=defer_days, discount_pct=discount_pct,
           **computed_plan)

    return {"valid": True, "violations": [], "computed_plan": computed_plan}


def _handle_generate_payment_link(inputs: dict, session: dict) -> dict:
    """Create a Razorpay Order for the Checkout JS flow.

    Returns the order info dict the frontend needs to open the Checkout modal.
    Stores the order id + amount on the session and moves it to
    'awaiting_payment' until the payment.captured webhook confirms settlement.
    """
    from backend.razorpay_client import create_order

    amount_inr  = inputs["amount"]            # rupees
    invoice_id  = inputs.get("invoice_id", session["invoice_id"])

    order = create_order(
        amount_inr=amount_inr,
        invoice_id=invoice_id,
        session_id=session["session_id"],
        debtor_name=session["debtor_name"],
    )

    session["razorpay_order_id"] = order["id"]
    session["payment_amount"] = amount_inr
    session["status"] = "awaiting_payment"

    # RAZORPAY_KEY_ID is public — the frontend uses it to open Checkout JS.
    order_info = {
        "order_id":     order["id"],
        "amount":       amount_inr,
        "amount_display": _rupees(round(amount_inr * 100)),
        "key_id":       os.environ.get("RAZORPAY_KEY_ID", ""),
        "debtor_name":  session["debtor_name"],
        "invoice_id":   invoice_id,
        "session_id":   session["session_id"],
    }
    session["payment_order"] = order_info

    _audit(session, "razorpay_order_created",
           order_id=order["id"], amount=amount_inr)

    return order_info


_ESCALATION_MESSAGES = {
    "max_turns_reached": (
        "We've reached the limit of what I can negotiate in this session. "
        "Our team will follow up with you shortly. Thank you."
    ),
    "debtor_dispute": (
        "I understand you're disputing the invoice. I'll escalate this to our "
        "merchant team for review — someone will contact you within 24 hours."
    ),
    "debtor_requested_human": (
        "Absolutely understood. I'm escalating this to a human representative "
        "who will reach out to you shortly."
    ),
    "debtor_unresponsive": (
        "We haven't received a response. Our team will follow up through "
        "other channels."
    ),
}


def _handle_escalate(inputs: dict, session: dict) -> dict:
    reason = inputs["reason"]
    session["status"] = "disputed" if reason == "debtor_dispute" else "escalated"
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
        "escalate":                _handle_escalate,
    }
    handler = dispatch.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}
    return handler(tool_input, session)


# ---------------------------------------------------------------------------
# PART 4a: HELPERS FOR STRUCTURED FLOW (no LLM required)
# ---------------------------------------------------------------------------

import re as _re


def _build_opening_message(session: dict) -> str:
    """Return the fixed A/B choice opening text sent to every debtor."""
    amount_paise = session["invoice_amount_paise"]
    return (
        f"Hi {session['debtor_name']}, I'm reaching out regarding invoice "
        f"{session['invoice_id']} for {_rupees(amount_paise)} from {MERCHANT_NAME}, "
        f"which is {session['dpd']} days overdue.\n\n"
        f"How would you like to proceed?\n"
        f"  [A] Pay {_rupees(amount_paise)} in full\n"
        f"  [B] Pay partially \u2014 discuss a payment arrangement"
    )


def parse_amount(message: str) -> int | None:
    """
    Parse a rupee amount from a natural-language string.
    Returns amount in **paise** (int), or None if nothing parseable found.

    Handles:
      \u20b950,000  |  50000  |  50k  |  50K  |  50,000
      "I can pay 50000"  |  "around 60000"  |  "1.5 lakh"
    """
    msg = message.lower()
    # strip currency symbols and commas so digits are bare
    msg = msg.replace("\u20b9", "").replace(",", "").replace("rs.", "").replace("rs", "").replace("inr", "")

    # lakh / lac: e.g. "1.5 lakh" → 150000
    lakh = _re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lac)", msg)
    if lakh:
        return round(float(lakh.group(1)) * 100_000 * 100)

    # k suffix: e.g. "50k" → 50000
    k_match = _re.search(r"(\d+(?:\.\d+)?)\s*k\b", msg)
    if k_match:
        return round(float(k_match.group(1)) * 1_000 * 100)

    # bare integer or decimal (ignore values that look like percentages < 100)
    num = _re.search(r"(\d+(?:\.\d+)?)", msg)
    if num:
        val = float(num.group(1))
        if val >= 100:            # \u20b9100 minimum to avoid matching "50%"
            return round(val * 100)

    return None


def _handle_intent_response(session: dict, debtor_message: str, turn: int) -> tuple[str, dict]:
    """
    Handle the debtor's A / B intent choice.
    No LLM call — pure deterministic Python.
    """
    _audit(session, "debtor_turn", turn=turn, speaker="debtor", message=debtor_message)
    msg = debtor_message.strip().lower()

    # ---- Option A: pay in full ----
    if msg == "a" or any(w in msg for w in ("full", "in full", "pay all", "pay total")):
        session["awaiting_intent"] = False
        _audit(session, "intent_selected", intent="full")

        _handle_generate_payment_link(
            {"amount": session["invoice_amount"],
             "invoice_id": session["invoice_id"]},
            session,
        )
        reply = (
            f"Excellent! You've chosen to pay the full amount of "
            f"{_rupees(session['invoice_amount_paise'])}. "
            f"Please click the Pay Now button below to complete your payment securely."
        )
        session["messages"].append({"role": "assistant", "content": reply})
        _audit(session, "agent_turn", turn=turn, speaker="agent", message=reply)
        return reply, session

    # ---- Option B: partial payment ----
    if msg == "b" or any(w in msg for w in ("partial", "partially", "arrangement", "discuss", "some", "payment options", "options")):
        session["awaiting_intent"] = False
        session["awaiting_amount"] = True
        _audit(session, "intent_selected", intent="partial")

        max_days = session["tier_bounds"]["max_days"]
        reply = (
            f"Understood. How much are you able to pay right now? "
            f"(The remaining balance would be due within {max_days} days.)"
        )
        session["messages"].append({"role": "user", "content": debtor_message})
        session["messages"].append({"role": "assistant", "content": reply})
        _audit(session, "agent_turn", turn=turn, speaker="agent", message=reply)
        return reply, session

    # ---- Unrecognised response: re-prompt ----
    reply = (
        "Please choose one of the options:\n"
        "  [A] Pay the full amount\n"
        "  [B] Discuss a partial payment arrangement"
    )
    session["messages"].append({"role": "assistant", "content": reply})
    _audit(session, "agent_turn", turn=turn, speaker="agent", message=reply)
    return reply, session


def _handle_amount_response(session: dict, debtor_message: str, turn: int) -> tuple[str, dict]:
    """
    Handle the debtor's offered rupee amount.
    Parses with regex, validates against tier bounds, accepts or counters.
    No LLM call.
    """
    _audit(session, "debtor_turn", turn=turn, speaker="debtor", message=debtor_message)
    session["messages"].append({"role": "user", "content": debtor_message})

    # --- Detect dispute mid-flow ---
    msg_lower = debtor_message.lower()
    if any(w in msg_lower for w in ("dispute", "incorrect", "wrong amount", "don't agree", "agreed on less")):
        _handle_escalate({"reason": "debtor_dispute"}, session)
        reply = _ESCALATION_MESSAGES["debtor_dispute"]
        session["messages"].append({"role": "assistant", "content": reply})
        _audit(session, "agent_turn", turn=turn, speaker="agent", message=reply)
        return reply, session

    # --- Parse amount ---
    offered_paise = parse_amount(debtor_message)

    if offered_paise is None:
        reply = "Could you share the exact amount you're able to pay right now?"
        session["messages"].append({"role": "assistant", "content": reply})
        _audit(session, "agent_turn", turn=turn, speaker="agent", message=reply)
        # stay in awaiting_amount
        return reply, session

    _audit(session, "debtor_offered_amount", amount_paise=offered_paise)

    invoice_paise = session["invoice_amount_paise"]
    now_pct  = round(offered_paise / invoice_paise * 100, 2)
    defer_pct = round(100 - now_pct, 2)
    max_days  = session["tier_bounds"]["max_days"]

    # --- Validate against tier bounds ---
    validation = _handle_validate_proposed_terms(
        {"now_pct": now_pct, "defer_pct": defer_pct,
         "defer_days": max_days, "discount_pct": 0,
         "upfront_offered_paise": offered_paise},
        session,
    )
    _audit(session, "offer_valid", valid=validation["valid"],
           violations=validation["violations"])

    if validation["valid"]:
        # --- Build payment plan for confirmation (do NOT generate link yet) ---
        computed = validation["computed_plan"]

        # Pull exact amounts from computed_plan — do NOT recompute from tier %.
        upfront  = computed["upfront_amount"]
        deferred = computed["deferred_amount"]

        # Discount applies to the deferred portion only
        # (never reduce what the debtor is paying now).
        discount = 0
        if session["tier_bounds"]["max_discount_pct"] > 0 and deferred > 0:
            discount = deferred * session["tier_bounds"]["max_discount_pct"] // 100
        deferred_after_discount = deferred - discount

        deferred_days = max_days
        due_date_str  = (date.today() + timedelta(days=deferred_days)).isoformat()

        plan = {
            "upfront_amount":      upfront,
            "deferred_amount":     deferred_after_discount,   # after discount
            "deferred_amount_raw": deferred,                  # before discount
            "discount_amount":     discount,
            "deferred_pct":        computed["deferred_pct"],
            "upfront_pct":         computed["upfront_pct"],
            "deferred_days":       deferred_days,
            "deferred_due_date":   due_date_str,
            "total_payable":       upfront + deferred_after_discount,
            # Display strings
            "upfront_display":     _rupees(upfront),
            "deferred_display":    _rupees(deferred_after_discount),
            "discount_display":    _rupees(discount) if discount else "₹0",
            "total_display":       _rupees(upfront + deferred_after_discount),
            "due_date_display":    format_date(due_date_str),
        }

        # Sanity check — crash loud in dev if the plan doesn't add up.
        assert plan["upfront_amount"] + plan["deferred_amount_raw"] \
               == session["invoice_amount_paise"], \
            f"Plan amounts don't add up to invoice total"

        session["pending_plan"] = plan
        session["awaiting_amount"] = False
        session["awaiting_plan_confirmation"] = True
        _audit(session, "plan_presented", **plan)

        reply = (
            f"Here is your proposed payment arrangement. Please review and confirm."
        )
        _audit(session, "counter_offered", counter=False)

    else:
        # Offer too low — counter with the minimum (reveal it only now)
        min_paise  = session["min_now_paise"]
        min_pct    = session["tier_bounds"]["min_now_pct"]
        session["awaiting_amount"] = True   # stay in loop
        reply = (
            f"I appreciate you working with us. The minimum I\u2019m authorised to accept "
            f"upfront is {_rupees(min_paise)} ({min_pct}% of the invoice). "
            f"Would you be able to manage that?"
        )
        _audit(session, "counter_offered", counter=True,
               min_amount_paise=min_paise)

    session["messages"].append({"role": "assistant", "content": reply})
    _audit(session, "agent_turn", turn=turn, speaker="agent", message=reply)
    return reply, session


def _handle_plan_confirmation(session: dict, debtor_message: str, turn: int) -> tuple[str, dict]:
    """
    Handle the debtor's CONFIRM or RENEGOTIATE response to the payment plan card.
    No LLM call — pure deterministic Python.
    """
    _audit(session, "debtor_turn", turn=turn, speaker="debtor", message=debtor_message)
    session["messages"].append({"role": "user", "content": debtor_message})
    msg = debtor_message.strip().lower()

    CONFIRM_WORDS = ("confirm", "agree", "yes", "ok", "okay", "accept", "sure")
    REJECT_WORDS  = ("no", "change", "renegotiate", "different", "less", "more")

    if any(w in msg for w in CONFIRM_WORDS):
        plan = session["pending_plan"]
        session["awaiting_plan_confirmation"] = False
        session["agreed_terms"] = plan
        _audit(session, "plan_confirmed", **plan)

        is_full = plan["deferred_amount"] <= 0

        # Create a Razorpay Order for the upfront amount. This moves the
        # session to 'awaiting_payment'; the payment.captured webhook then
        # settles it (settled / partially_settled based on the deferred plan).
        _handle_generate_payment_link(
            {"amount": plan["upfront_amount"] / 100,
             "invoice_id": session["invoice_id"]},
            session,
        )

        if is_full:
            # Debtor pays the full invoice upfront \u2014 no deferred entry needed.
            reply = (
                f"[Confirmed] Full invoice settled \u2014 no deferred payment required.\n"
                f"Please click the Pay Now button below to pay "
                f"{plan['upfront_display']}."
            )
        else:
            _audit(session, "deferred_scheduled",
                   deferred_amount=plan["deferred_amount"],
                   due_date=plan["deferred_due_date"])
            reply = (
                f"[Confirmed] Your payment plan is confirmed!\n"
                f"Pay {plan['upfront_display']} now using the Pay Now button below.\n"
                f"The remaining {plan['deferred_display']} will be due by "
                f"{plan['deferred_due_date']}.\n"
                f"Reminder: Paying your deferred amount on time improves your account standing."
            )
        session["pending_plan"] = None

    elif any(w in msg for w in REJECT_WORDS):
        session["awaiting_plan_confirmation"] = False
        session["awaiting_amount"] = True
        session["pending_plan"] = None
        _audit(session, "plan_rejected")
        reply = "No problem. What amount would you like to offer instead?"

    else:
        # Ambiguous — re-prompt
        reply = (
            "Please reply \u2018CONFIRM\u2019 to accept this arrangement, "
            "or \u2018renegotiate\u2019 if you\u2019d like to propose a different amount."
        )

    session["messages"].append({"role": "assistant", "content": reply})
    _audit(session, "agent_turn", turn=turn, speaker="agent", message=reply)
    return reply, session


# ---------------------------------------------------------------------------
# PART 4: TURN FUNCTION
# ---------------------------------------------------------------------------

def _get_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def _call_llm(session: dict, client: OpenAI) -> str:
    """
    Call DeepSeek (OpenAI-compatible) API, handle tool-call loops, return final text.

    OpenAI tool-call protocol:
      1. API returns finish_reason="tool_calls" with message.tool_calls list.
      2. We append the raw assistant message, then one role="tool" message per call.
      3. Loop until finish_reason="stop" → extract message.content as text.

    System prompt is injected as the first message (role="system") on every
    request — OpenAI does not have a separate `system` parameter at the top level.
    """
    while True:
        # Build full message list: system prompt first, then conversation history
        messages_with_system = [
            {"role": "system", "content": session["system_prompt"]},
            *session["messages"],
        ]

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages_with_system,
            tools=TOOLS,
            tool_choice="auto",
        )

        choice = response.choices[0]
        msg = choice.message

        if choice.finish_reason == "stop" or not msg.tool_calls:
            # Pure text response — done
            return (msg.content or "").strip()

        if choice.finish_reason == "tool_calls":
            # 1. Append the raw assistant message (preserves tool_calls metadata)
            #    OpenAI SDK objects are not directly JSON-serialisable, so we
            #    build a plain dict manually.
            tool_call_dicts = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
            session["messages"].append({
                "role": "assistant",
                "content": msg.content,   # may be None or a preamble string
                "tool_calls": tool_call_dicts,
            })

            # 2. Execute each tool call; append one role="tool" message per call
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_input = json.loads(tc.function.arguments)
                result = _execute_tool(tool_name, tool_input, session)
                session["messages"].append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

            # 3. Loop: send updated history back for the model's next response
            continue

        # Unexpected finish_reason — return whatever text we have
        return (msg.content or "").strip()


def process_turn(session: dict, debtor_message: str) -> tuple[str, dict]:
    """
    Process one debtor message through the negotiation agent.

    Returns (agent_reply, updated_session).
    Empty debtor_message is treated as non-responsive.
    """
    if session["status"] != "active":
        return f"[Session {session['status']} — no further turns]", session

    session["turn_count"] += 1
    turn = session["turn_count"]

    # --- Non-responsive debtor (silent / empty message) ---
    if not debtor_message.strip():
        _audit(session, "debtor_turn", turn=turn, speaker="debtor", message="[silent]")
        if turn >= session["max_turns"]:
            _handle_escalate({"reason": "debtor_unresponsive"}, session)
            reply = _ESCALATION_MESSAGES["debtor_unresponsive"]
        else:
            reply = (
                "It looks like we haven't heard from you yet. "
                "Please reply when you're ready to discuss the outstanding invoice."
            )
        _audit(session, "agent_turn", turn=turn, speaker="agent", message=reply)
        return reply, session

    # --- Structured state: awaiting A/B intent choice ---
    if session.get("awaiting_intent"):
        return _handle_intent_response(session, debtor_message, turn)

    # --- Structured state: awaiting plan confirmation (CONFIRM / renegotiate) ---
    if session.get("awaiting_plan_confirmation"):
        return _handle_plan_confirmation(session, debtor_message, turn)

    # --- Structured state: awaiting debtor's offered amount ---
    if session.get("awaiting_amount"):
        return _handle_amount_response(session, debtor_message, turn)

    # --- Regular free-form LLM turn ---
    session["messages"].append({"role": "user", "content": debtor_message})
    _audit(session, "debtor_turn", turn=turn, speaker="debtor", message=debtor_message)

    # Hard turn limit (> not >= so turn 5 still gets a reply)
    if turn > session["max_turns"]:
        _handle_escalate({"reason": "max_turns_reached"}, session)
        reply = _ESCALATION_MESSAGES["max_turns_reached"]
        _audit(session, "agent_turn", turn=turn, speaker="agent", message=reply)
        return reply, session

    client = _get_client()
    agent_reply = _call_llm(session, client)

    # After tool-use loop, last message may already be assistant; add text reply
    last = session["messages"][-1] if session["messages"] else {}
    if last.get("role") != "assistant" or not isinstance(last.get("content"), str):
        session["messages"].append({"role": "assistant", "content": agent_reply})

    _audit(session, "agent_turn", turn=turn, speaker="agent", message=agent_reply)
    return agent_reply, session


def open_turn(session: dict) -> tuple[str, dict]:
    """
    Send the structured A/B-choice opening message.
    Pure Python — no LLM call needed for this fixed template.
    Sets session["awaiting_intent"] = True.
    """
    opening = _build_opening_message(session)
    session["awaiting_intent"] = True
    # Seed message history with just the opening so subsequent turns have context
    session["messages"] = [{"role": "assistant", "content": opening}]
    _audit(session, "agent_turn", turn=0, speaker="agent", message=opening)
    return opening, session


# ---------------------------------------------------------------------------
# PART 5: SIMULATED DEBTOR
# ---------------------------------------------------------------------------

def simulate_debtor_turn(
    session: dict,
    simulated_outcome: str | None = None,
    turn_override: int | None = None,
) -> str:
    """
    Return a simulated debtor message that matches the current session state.

    Three outcomes only. Simulator is state-aware:
      awaiting_intent            → ask about payment options
      awaiting_amount            → outcome-specific amount response
      awaiting_plan_confirmation → CONFIRM or renegotiate
    """
    outcome = simulated_outcome or session.get("simulated_outcome", "clean_settlement")

    # ----------------------------------------------------------------
    # State: A/B opening sent
    # ----------------------------------------------------------------
    if session.get("awaiting_intent"):
        return "What payment options do you have?"

    # ----------------------------------------------------------------
    # State: plan confirmation card shown
    # ----------------------------------------------------------------
    if session.get("awaiting_plan_confirmation"):
        if outcome == "clean_settlement":
            return "CONFIRM"
        if outcome == "repeat_extension":
            # First pass: renegotiate; subsequent passes: confirm
            if session.get("_plan_rejected_once"):
                return "CONFIRM"
            session["_plan_rejected_once"] = True
            return "renegotiate"
        if outcome == "dispute":
            return "renegotiate"   # goes back to amount loop → then disputes
        return "CONFIRM"  # fallback

    # ----------------------------------------------------------------
    # State: agent asked "how much can you pay?"
    # ----------------------------------------------------------------
    if session.get("awaiting_amount"):

        if outcome == "clean_settlement":
            min_pct    = session["tier_bounds"]["min_now_pct"]
            min_amount = int(session["invoice_amount"] * min_pct / 100)
            return f"I can pay ₹{min_amount:,} right now."

        if outcome == "dispute":
            return "I don't think this amount is correct, we agreed on less."

        if outcome == "repeat_extension":
            amount_turn = session.get("_amount_turn_count", 0)
            session["_amount_turn_count"] = amount_turn + 1
            scripts = [
                "Can I get more time to pay? Maybe 90 days?",
                "I really need at least 75 days, cash is tight.",
                "What's the minimum I have to pay right now?",
            ]
            if amount_turn < len(scripts):
                return scripts[amount_turn]
            # Offer minimum so the plan card appears
            min_pct    = session["tier_bounds"]["min_now_pct"]
            min_amount = int(session["invoice_amount"] * min_pct / 100)
            return f"I can pay ₹{min_amount:,} right now."

    # ----------------------------------------------------------------
    # Fallback
    # ----------------------------------------------------------------
    return ""
