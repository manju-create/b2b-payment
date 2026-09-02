"""
RecoverFlow — Negotiation Agent
=================================
Conducts real-time B2B payment recovery conversations as "Aria" — a warm,
human-sounding financial advisor.

Clean separation of concerns:
  * Python (NegotiationEngine + the state machine here) owns EVERY number and
    every decision — the tier, the floor, the ladder step, the dates, the plan,
    and the current remaining balance. All arithmetic happens in Python.
  * DeepSeek does exactly two things and nothing more: (1) it extracts intent +
    variables from the debtor's message as strict JSON (no math), and (2) it
    turns a state + instruction + numbers into the reply. It only READS the
    balance Python injects — it never calculates it.
  * The final state is forced via tool calling: when the debtor confirms the
    plan, DeepSeek is forced to call `finalize_agreement`, and Python intercepts
    it — the LLM writes no further text, and Python injects the payment payload.

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
import re
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
from backend.negotiation_engine import NegotiationEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MERCHANT_NAME = "RecoverFlow Demo Merchant"
AGENT_NAME = "Aria"
MODEL = "deepseek-reasoner"
# deepseek-reasoner emits a chain-of-thought (reasoning_content) BEFORE its
# final answer, and that reasoning counts against max_tokens. 1024 was too small
# — the model would burn its whole budget on reasoning and return an empty
# `content` (finish_reason=length). Give it room for both.
MAX_TOKENS = 8192
# The terminal "finalize" turn forces a function call. deepseek-reasoner has no
# function calling, so that one turn uses deepseek-chat (already used by the
# document verifier) instead of the conversational MODEL above.
FINALIZE_MODEL = "deepseek-chat"
DATA_DIR = REPO_ROOT / "data"

# Hard ceiling on how long a settlement plan may run. Enforced in Python,
# not just the prompt.
MAX_PLAN_DAYS = 34

# How many times we counter ABOVE an acceptable-but-low offer before we accept
# the debtor's number. Anchoring high first, then relaxing, recovers more than
# taking the first figure they volunteer — debtors often hold back their real max.
MAX_COUNTER_ATTEMPTS = 2
logger = logging.getLogger(__name__)

# Negotiation engines live here (keyed by session_id) so the session dict stays
# JSON-serializable — the session stores only engine.to_dict(), never the object.
_ENGINES: dict[str, NegotiationEngine] = {}

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


def _inr(amount: float) -> str:
    """Format a rupee amount (int/float) as ₹ with Indian grouping."""
    return _rupees(int(round(amount)) * 100)


def format_date(iso: str) -> str:
    """Format an ISO 'YYYY-MM-DD' date as a human-readable string (e.g. '26 Aug 2026')."""
    try:
        return datetime.strptime(iso, "%Y-%m-%d").strftime("%d %b %Y")
    except ValueError:
        return iso


def _first_name(name: str) -> str:
    return (name or "there").strip().split()[0]


def _parse_rupees(number_str: str) -> int:
    """Parse a digit string (possibly comma-grouped) into integer rupees."""
    try:
        return int(round(float(number_str.replace(",", ""))))
    except ValueError:
        return 0


_AMOUNT_UNIT_MULTIPLIERS = {
    "k": 1_000,
    "thousand": 1_000,
    "thousands": 1_000,
    "lakh": 100_000,
    "lakhs": 100_000,
    "lac": 100_000,
    "lacs": 100_000,
}


def _extract_amount_rupees(text: str) -> int | None:
    """Best-effort extract of a rupee amount the debtor committed to pay.

    This is a safety net, not the driver: `_handle_generate_payment_link` uses
    the debtor's stated amount as the ceiling for the payment link, so the link
    can never be inflated above what the debtor actually offered.
    """
    if not text:
        return None
    t = text.lower()
    found: list[tuple[int, int]] = []  # (character position, rupees)

    # ₹5,000 / ₹5000
    for m in re.finditer(r"₹\s*([\d,]+(?:\.\d+)?)", t):
        found.append((m.start(), _parse_rupees(m.group(1))))
    # rs 5000 / rupees 5000
    for m in re.finditer(r"\b(?:rs\.?|rupees?)\s+([\d,]+(?:\.\d+)?)", t):
        found.append((m.start(), _parse_rupees(m.group(1))))
    # 40k / 5 thousand / 2 lakh
    for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*(k|thousand|thousands|lakh|lakhs|lac|lacs)\b", t
    ):
        found.append((
            m.start(),
            int(round(float(m.group(1)) * _AMOUNT_UNIT_MULTIPLIERS[m.group(2).lower()])),
        ))
    # bare grouped/large number: 40,000 / 2,16,000 / 40000
    for m in re.finditer(r"\b(\d{1,3}(?:,\d{2,3})+|\d{4,})\b", t):
        found.append((m.start(), _parse_rupees(m.group(1))))

    if not found:
        return None

    # Prefer the amount tied to a "now/today" commitment; else the last one.
    now_pos = -1
    for tok in ("today itself", "now", "today"):
        pos = t.find(tok)
        if pos != -1 and (now_pos == -1 or pos < now_pos):
            now_pos = pos
    if now_pos >= 0:
        found.sort(key=lambda item: abs(item[0] - now_pos))
    else:
        found.sort(key=lambda item: item[0], reverse=True)

    return found[0][1]


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


def _trigger_escalation(session: dict, reason: str, status: str = "escalated") -> str | None:
    """Mark the session escalated, generate the L3 escalation PDF, and log it.

    Called from every terminal escalation path (all negotiation steps exhausted,
    a legal/RBI threat, no progress, a human-request, or the final ultimatum).
    The PDF is generated exactly once and cached on the session so the download
    endpoint never regenerates it.

    Returns the PDF path, or None if generation failed (e.g. reportlab missing).
    """
    session["status"] = status
    session["state"] = "escalated"
    timestamp = _ts()
    session["escalation_triggered_at"] = timestamp
    session["escalation_reason"] = reason

    pdf_path = session.get("escalation_pdf_path")
    pdf_generated = False
    if not pdf_path:
        try:
            from backend.pdf_generator import generate_escalation_pdf
            invoice = session.get("current_invoice") or {}
            pdf_path = generate_escalation_pdf(session, invoice)
            session["escalation_pdf_path"] = pdf_path
            pdf_generated = True
        except Exception:
            logger.exception("Failed to generate escalation PDF for %s",
                             session.get("invoice_id"))
            pdf_path = None

    _audit(session, "escalation_triggered", reason=reason,
           timestamp=timestamp, pdf_generated=pdf_generated, pdf_path=pdf_path)
    return pdf_path


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
    """Recompute the live trust score (display only — it does NOT drive the
    negotiation; the NegotiationEngine, fixed at session start, owns that)."""
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
        "stance":            stance,            # kept for legacy paths only
        "debtor_agreed_amount": None,           # today's agreed amount (rupees)
        "last_debtor_offer": None,              # the debtor's most recent counter-offer (rupees)
        "recent_agent_messages": [],            # last 3 agent replies (anti-repetition)
        # --- state machine (Python decides; DeepSeek only speaks) ---
        "state":                 "opening",     # opening|negotiating|collecting_dates|plan_ready|payment_pending|hardship|promise_to_pay|escalated|settled
        "negotiation_step":      1,             # 1-4 ask steps; 5 = hardship or escalate
        "counter_attempts":      0,             # times we've countered above an acceptable offer
        "current_ask":           None,          # exact amount to ask THIS turn (counter vs ladder)
        "first_counter_issued":  False,         # set once we've issued our first counter-offer
        "last_bot_offer":        None,          # the bot's most recent counter amount (rupees)
        "future_dates":          [],            # confirmed future payment dates (ISO)
        "installment_plan":      None,          # list of {date, amount, label, status}
        "plan_shown":            False,
        "finalize_requested":    False,         # set when the plan is confirmed → forced finalize tool call
        "reason_collected":      False,         # lock: the reason MCQ is asked at most once
        "final_ultimatum_requested": False,     # second rejection → terminal final ultimatum
        "negotiation_complete":  False,
        "hardship_verified":     False,         # set True when inability-to-pay proof is accepted
        "upload_requested":      False,         # ask for proof at most once
        "negotiation_engine":    {},            # serialized NegotiationEngine values
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
    _refresh_trust_score(session)  # computes the initial trust score

    # Instantiate the negotiation engine ONCE — every number and every decision
    # from here on comes from Python, never from the model.
    engine = NegotiationEngine(session["invoice_amount"], session["trust_score"])
    session["negotiation_engine"] = engine.to_dict()
    _ENGINES[session["session_id"]] = engine

    # Freeze the payment-history trust score for display. The live score
    # (session["trust_score"]) keeps moving with negotiation signals on every
    # turn, but the debtor card and the merchant dashboard must show the SAME
    # number — so both read this stable snapshot taken at session start.
    session["display_trust_score"] = session["trust_score"]
    session["display_trust_tier"]  = (session["trust_score_result"] or {}).get("tier", tier)

    _audit(session, "session_created",
           tier=tier, score=score,
           invoice_amount_paise=invoice_amount_paise,
           cold_start=score_result["cold_start"],
           negotiation_tier=engine.tier,
           min_today=engine.min_today,
           max_installments=engine.max_installments,
           gap_days=engine.gap_days,
           deadline=engine.deadline.isoformat())
    return session


# ---------------------------------------------------------------------------
# PART 2: SYSTEM PROMPT — a single intelligent prompt (no flow logic)
# ---------------------------------------------------------------------------

def _record_agent_message(session: dict, reply: str) -> None:
    """Keep the last 3 agent replies so the LLM can avoid repeating itself."""
    recent = session.setdefault("recent_agent_messages", [])
    recent.append((reply or "").strip())
    if len(recent) > 3:
        del recent[:-3]


def _render_recent_agent_messages(session: dict) -> str:
    """Render the agent's last few replies for the anti-repetition prompt block."""
    recent = [m for m in session.get("recent_agent_messages", []) if m]
    return "\n".join(f"- {m}" for m in recent) if recent else "(nothing yet)"


def build_system_prompt(session: dict, context: dict) -> str:
    """Build Aria's system prompt — conversational guidance plus the numbers.

    Python owns every number (the ask, the hard floor, the dates); the model
    never does money math. But the model's job is to actually READ what the
    debtor said and respond to it — acknowledging their offer, their reason, or
    their question — rather than sounding like a canned script.

    The conversation history is NOT embedded here — it is sent to the model as
    real user/assistant turns alongside this prompt (see `_call_llm`).
    """
    recent = _render_recent_agent_messages(session)

    state = context["state"]
    instruction = context["instruction"]
    nums = context["numbers"]
    invoice = _inr(nums["invoice_amount"])

    offer = nums.get("debtor_offer")
    ask = nums.get("step_ask")
    floor = nums.get("floor")
    balance = nums.get("current_remaining_balance")

    anchor_lines: list[str] = []
    if offer is not None:
        anchor_lines.append(
            f"The debtor just offered {_inr(offer)}. Acknowledge their number in your own words before you counter."
        )
    if ask is not None:
        anchor_lines.append(f"Your ask this turn is {_inr(ask)}.")
    if floor is not None:
        anchor_lines.append(
            f"Hard floor: never accept or offer less than {_inr(floor)} today."
        )
    if balance:
        anchor_lines.append(
            f"Current remaining balance after today's payment: {_inr(balance)}. "
            f"It is computed for you — read it, never recalculate it."
        )
    anchors = (
        "\n".join(f"- {line}" for line in anchor_lines)
        if anchor_lines else "- Follow the instruction below."
    )

    return f"""You are Aria, a warm and intelligent payment recovery specialist at {MERCHANT_NAME}. You are talking to {session['debtor_name']} about their overdue invoice of {invoice}.

YOUR ROLE:
You are negotiating in real time with a real person. Python has worked out the numbers for you, but you must actually READ and RESPOND to what the debtor says — never sound like you are following a script or repeating a canned line.

CURRENT STATE: {state}

NUMBERS FOR THIS TURN:
{anchors}

WHAT TO DO THIS TURN:
{instruction}

OUTPUT FORMAT:
Respond with ONLY a valid JSON object — no prose outside the JSON:
{{"thought_process": "<one short line of internal logic>", "action_type": "negotiate" | "trigger_reason_mcq" | "finalize_agreement", "reply_to_user": "<the exact text to send to the debtor>"}}
- Use "trigger_reason_mcq" when you must ask the debtor WHY they cannot meet the amount — Python renders the multiple-choice buttons; your reply_to_user is the hook that introduces the question.
- Use "finalize_agreement" only when the debtor has confirmed the plan.
- Otherwise use "negotiate".

WHAT YOU'VE ALREADY SAID — never repeat; rephrase if you're about to say something similar:
{recent}

VOICE & TONE:
- Natural, warm, human. 1-2 sentences. One question per message.
- Acknowledge the debtor's specific words first — their offer, their reason, or their question — then make your ask.
- Answer any question the debtor asks, directly and truthfully.
- You may go as low as the hard floor, but never a rupee below it, and never invent a number that isn't in the numbers above.
- Never do arithmetic. Every number you need — including the current remaining balance — is given to you above; read it, do not recalculate or invent it.
- Never mention trust score, tier, percentages, or the word "floor"/"minimum" to the debtor.
- Never use: "kindly", "as per", "please be advised", "I understand that", "I appreciate".
- Never send the same message twice."""


# ---------------------------------------------------------------------------
# PART 3: SIDE-EFFECT HANDLERS
# ---------------------------------------------------------------------------
#
# These set session state for the non-negotiation edge cases (dispute, document
# upload, payment link). They are called by Python in the state machine /
# process_turn — deepseek-reasoner has no function calling, so the model never
# invokes them.

# ---- Handlers ---------------------------------------------------------------

def _handle_generate_payment_link(inputs: dict, session: dict) -> dict:
    """Create a Razorpay Order for TODAY's agreed amount.

    Called by Python (the state machine), never by DeepSeek. Hard gates: the
    plan must have been shown, and the amount must be exactly the debtor's
    agreed amount — never more, never less.
    """
    from backend.razorpay_client import create_order

    if not session.get("plan_shown"):
        return {"error": "Show the plan first before generating the payment link."}

    agreed_inr = session.get("debtor_agreed_amount")
    if agreed_inr is None:
        return {"error": "No agreed amount recorded."}

    amount_inr = inputs.get("amount")
    if amount_inr is not None:
        amount_inr = float(amount_inr)

    # The link amount is the debtor's agreed amount, exactly.
    if amount_inr != float(agreed_inr):
        _audit(session, "payment_amount_forced_to_agreed",
               requested_amount=amount_inr, agreed_amount=agreed_inr)
        amount_inr = float(agreed_inr)

    amount_paise = round(amount_inr * 100)
    if amount_paise > session["invoice_amount_paise"]:
        return {"error": "Amount exceeds the invoice total."}

    invoice_id  = inputs.get("invoice_id", session["invoice_id"])

    try:
        order = create_order(
            amount_inr=amount_inr,
            invoice_id=invoice_id,
            session_id=session["session_id"],
            debtor_name=session["debtor_name"],
        )
    except Exception as exc:
        logger.exception("Failed to create Razorpay order")
        import uuid
        mock_id = f"order_demo_{uuid.uuid4().hex[:12]}"
        order = {"id": mock_id, "amount": amount_paise}

    session["razorpay_order_id"] = order["id"]
    session["payment_amount"]    = amount_inr
    session["status"]            = "awaiting_payment"

    # agreed_terms was built by set_installment_plan; normalize for safety.
    if not session.get("agreed_terms"):
        invoice_paise = session["invoice_amount_paise"]
        upfront_paise = amount_paise
        deferred_paise = max(0, invoice_paise - upfront_paise)
        due_date_str = (date.today() + timedelta(days=MAX_PLAN_DAYS)).isoformat()
        session["agreed_terms"] = _normalize_no_discount_plan(session, {
            "upfront_amount":      upfront_paise,
            "upfront_pct":         round(upfront_paise / invoice_paise * 100, 1),
            "deferred_amount_raw": deferred_paise,
            "deferred_pct":        round(deferred_paise / invoice_paise * 100, 1),
            "deferred_days":       MAX_PLAN_DAYS,
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


def create_full_payment_order(session: dict) -> dict:
    """Generate a Razorpay Order for the FULL invoice amount, bypassing negotiation.

    Used by the "Pay in full" button next to the debtor name — the debtor can
    settle the whole invoice immediately without going through the agent.
    """
    full = session["invoice_amount"]
    session["plan_shown"] = True
    session["debtor_agreed_amount"] = full
    session["last_debtor_offer"] = full
    return _handle_generate_payment_link({"amount": full}, session)


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
    return "✅ Unable-to-pay proof verified — hardship floor applied (20%)"


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
        else:  # CANNOT_PAY — verified hardship → lower the floor to 20% and continue
            merchant_flag = _flag_for_accept(situation, result)
            engine = _get_engine(session)
            new_min = engine.apply_hardship()
            session["hardship_verified"] = True
            session["negotiation_engine"] = engine.to_dict()
            session["state"] = "negotiating"
            session["negotiation_step"] = 3      # re-open at the hardship floor
            _audit(session, "hardship_verified", new_min_today=new_min)
            reply = (
                f"{debtor_friendly} Given your situation, we can come down to "
                f"{_inr(new_min)} today (20% of the invoice). Could you manage that?"
            ).strip()
    elif action == "REQUEST_BETTER_PROOF":
        # Warm, never accusatory. The upload card is shown again for one more
        # attempt (the server reads the final recommended_action).
        merchant_flag = "⚠️ Requesting better proof — document inconclusive"
        _specific_doc = {
            "CANNOT_PAY": "a clear bank statement (showing your name and recent balance) or a hardship letter",
            "ALREADY_PAID": "a payment receipt or bank statement showing the UTR/transaction ID",
            "DISPUTE": "a clear copy of the invoice or agreement showing the correct amount",
        }.get(situation)
        reply = debtor_friendly or (
            "Thanks — could you share a clearer copy so I can verify this properly?"
        )
        if _specific_doc and _specific_doc not in reply:
            reply = f"{reply.rstrip()} {_specific_doc.capitalize()} works best."
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

    # Record the upload itself as a debtor turn so the conversation stays
    # coherent — the assistant reply below otherwise floats with no user turn.
    session["messages"].append({
        "role": "user",
        "content": "[Debtor uploaded a document for verification]",
    })
    session["messages"].append({"role": "assistant", "content": reply})
    _record_agent_message(session, reply)
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


def _call_llm(session: dict, client: OpenAI) -> str:
    """Call DeepSeek (reasoner) and return Aria's final text.

    The full conversation is sent as real role-tagged turns (user/assistant) so
    the model remembers exactly what the debtor said and what Aria said, and can
    continue from where the chat left off. deepseek-reasoner has no function
    calling — every decision (dispute, document request, plan, link) is made by
    Python in the state machine, never by the model.
    """
    messages: list[dict] = [
        {"role": "system", "content": session["system_prompt"]},
    ]
    # Append the conversation so far as proper turns. The current turn's user
    # message is already the last entry in session["messages"] (appended before
    # this call), so the model sees the whole thread, in order.
    for m in session.get("messages", []):
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=messages,
    )
    return (response.choices[0].message.content or "").strip()


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


# ---------------------------------------------------------------------------
# Negotiation state machine — Python owns every number and every decision.
# DeepSeek is only told WHAT to say, never HOW to decide.
# ---------------------------------------------------------------------------

def _get_engine(session: dict) -> NegotiationEngine:
    """Return the session's negotiation engine (instantiated once at start)."""
    eng = _ENGINES.get(session["session_id"])
    if eng is None:
        eng = NegotiationEngine(session["invoice_amount"], session.get("trust_score", 0))
        _ENGINES[session["session_id"]] = eng
        session["negotiation_engine"] = eng.to_dict()
    return eng


def _extract_iso_dates(text: str) -> list[str]:
    """Return ISO dates (YYYY-MM-DD) mentioned in the debtor's message."""
    if not text:
        return []
    return re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", text)


_CONFIRM_WORDS = (
    "yes", "ya", "yeah", "yep", "yup", "ok", "okay", "sure", "confirm", "agree",
    "agreed", "sounds good", "works", "that works", "fine", "deal", "go ahead",
    "proceed", "perfect", "great", "correct", "done", "👍", "✅",
)


def _is_confirmation(text: str) -> bool:
    """True if the debtor is confirming the plan (vs renegotiating)."""
    m = (text or "").lower().strip()
    if m in ("no", "nope", "nah") or m.startswith("no "):
        return False
    return any(w in m for w in _CONFIRM_WORDS)


_BARE_REJECTIONS = {
    "no", "nope", "nah", "n", "no thanks", "no way",
    "can't", "cant", "cannot", "wont", "won't", "nothing",
}


def _looks_like_reason(text: str) -> bool:
    """True if the debtor gave an actual reason (not a bare 'no')."""
    m = (text or "").strip().lower()
    if m in _BARE_REJECTIONS or len(m) < 8:
        return False
    return True


_CANNOT_PAY_WORDS = (
    "no cash", "no money", "no funds", "no balance", "nothing to pay",
    "broke", "don't have", "dont have", "can't pay anything", "cant pay anything",
    "no income", "no sales", "no business", "have no money", "no way to pay",
)

_CEILING_WORDS = (
    "max", "maximum", "at most", "that's all", "thats all", "all i have",
    "can't afford", "cant afford", "can't go above", "cant go above", "only have",
    "my limit", "my budget",
)

_QUESTION_WORDS = (
    "what", "how", "why", "which", "where", "who", "when",
    "what's", "whats", "do i", "can i", "could i", "tell me", "explain",
)


def _signals_cannot_pay(message: str) -> bool:
    """True if the debtor signals they have no money at all right now."""
    m = (message or "").lower()
    return any(w in m for w in _CANNOT_PAY_WORDS)


def _signals_ceiling(message: str) -> bool:
    """True if the debtor states a hard upper limit on what they can pay."""
    m = (message or "").lower()
    return any(w in m for w in _CEILING_WORDS)


_ALREADY_PAID_WORDS = (
    "already paid", "paid already", "i've paid", "i have paid", "i paid",
    "sent the payment", "made the payment", "payment sent", "paid this",
    "paid it", "i did pay", "already sent", "utr",
)

_DISPUTE_WORDS = (
    "dispute", "wrong amount", "not right", "incorrect", "overcharged",
    "agreed on less", "wrong invoice", "not my invoice", "double charged",
    "billed twice", "this amount is wrong", "this is wrong",
)


def _signals_already_paid(message: str) -> bool:
    """True if the debtor claims they've already paid this invoice."""
    m = (message or "").lower()
    return any(w in m for w in _ALREADY_PAID_WORDS)


def _signals_dispute(message: str) -> bool:
    """True if the debtor disputes the invoice amount or charges."""
    m = (message or "").lower()
    return any(w in m for w in _DISPUTE_WORDS)


_FULL_PAYMENT_WORDS = (
    "pay full", "pay the full", "in full", "full amount", "full payment",
    "pay it all", "pay everything", "entire amount", "whole amount",
    "settle the full",
)


def _signals_pay_in_full(message: str) -> bool:
    """True if the debtor offers to pay the full invoice amount now.

    Negation flips the meaning ("can't pay the full amount" is a hardship
    signal, not a full-payment offer).
    """
    m = (message or "").lower()
    if any(w in m for w in (
        "can't pay", "cant pay", "cannot pay", "can not pay", "won't pay",
        "wont pay", "don't have", "dont have", "can't afford", "cant afford",
    )):
        return False
    return any(w in m for w in _FULL_PAYMENT_WORDS)


def _is_question(message: str) -> bool:
    """True if the debtor is asking a clarifying question (not offering/rejecting).

    A message with a rupee amount is treated as an offer, never a question, even
    if it happens to contain a question word (e.g. "can I pay 5k?").
    """
    m = (message or "").strip().lower()
    if not m or _extract_amount_rupees(m) is not None:
        return False
    if m.endswith("?"):
        return True
    return any(w in m for w in _QUESTION_WORDS)


# ---------------------------------------------------------------------------
# Intent extraction — the LLM returns structured JSON; Python owns the math.
# ---------------------------------------------------------------------------
#
# The model is a pure intent/variable extractor here. It returns a JSON object
# and does NO arithmetic — the remaining balance is computed later in Python
# (process_turn / _build_context) and injected back into the prompt as a
# read-only system variable.

_MONTH_NAMES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_EXTRACT_INTENT_SYSTEM = (
    "You extract structured payment intent from a debtor's message. "
    "Return ONLY a JSON object — no prose, no markdown, no code fences. "
    'Schema: {"intent": string, "upfront_amount": integer|null, "date": string|null}.\n'
    'intent is one of: "partial_payment", "full_payment", "cannot_pay", "dispute", '
    '"already_paid", "confirm", "reject", "question", "other".\n'
    "upfront_amount is the amount in whole rupees the debtor offers to pay now "
    '(interpret "lakh", "k", "thousand", and comma grouping). Use null when no amount is given.\n'
    'date is the payment date the debtor mentions (e.g. "Sept 1"). Use null when none is given.\n'
    "Do no arithmetic. Extract only what the debtor actually said."
)


def _safe_iso(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _normalize_intent_date(value) -> str | None:
    """Normalize an LLM-returned date ("Sept 1", "2026-09-01", …) to ISO.

    Returns ISO 'YYYY-MM-DD' (defaulting to the current year when the debtor
    doesn't give one), or None if the value can't be parsed.
    """
    if not value:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    iso = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if iso:
        return _safe_iso(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))

    year = date.today().year
    y = re.search(r"\b(20\d{2})\b", s)
    if y:
        year = int(y.group(1))
    s = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)\b", r"\1", s)
    s = re.sub(r"\b(?:20\d{2})\b", " ", s)
    s = s.replace("of", " ").strip()

    m = re.match(r"([a-z]{3,9})\.?\s+(\d{1,2})", s)
    if m:
        mon = _MONTH_NAMES.get(m.group(1))
        if mon:
            return _safe_iso(year, mon, int(m.group(2)))
    m = re.match(r"(\d{1,2})\s+([a-z]{3,9})", s)
    if m:
        mon = _MONTH_NAMES.get(m.group(2))
        if mon:
            return _safe_iso(year, mon, int(m.group(1)))
    return None


def _parse_intent_json(raw: str) -> dict | None:
    """Parse the LLM's JSON, tolerating fences/prose, and validate fields."""
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    intent = str(data.get("intent") or "other").strip().lower()

    upfront = data.get("upfront_amount")
    if isinstance(upfront, bool):
        upfront = None
    elif isinstance(upfront, (int, float)) and upfront > 0:
        upfront = int(upfront)
    else:
        upfront = None

    return {"intent": intent, "upfront_amount": upfront,
            "date": _normalize_intent_date(data.get("date"))}


def _parse_agent_json(raw: str) -> dict:
    """Parse the agent's JSON reply, falling back to treating raw text as the reply.

    The model is asked to return {"thought_process", "action_type", "reply_to_user"}.
    If it returns plain text (or malformed JSON), the whole string becomes
    reply_to_user and action_type defaults to "negotiate" — so a text-only model
    (or a test double) still works.
    """
    if raw:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
                if isinstance(data, dict) and data.get("reply_to_user"):
                    return {
                        "thought_process": str(data.get("thought_process") or "").strip(),
                        "action_type": str(data.get("action_type") or "negotiate").strip().lower(),
                        "reply_to_user": str(data["reply_to_user"]).strip(),
                    }
            except (json.JSONDecodeError, ValueError):
                pass
    return {"thought_process": "", "action_type": "negotiate",
            "reply_to_user": (raw or "").strip()}


# The four reasons shown as buttons when the agent must ask WHY the debtor can't
# pay. Python builds the buttons — the model never generates them.
MCQ_REASONS = [
    {"button_id": "client_not_paid", "label": "Client hasn't paid"},
    {"button_id": "cashflow",         "label": "Cash flow issues"},
    {"button_id": "dispute",          "label": "Dispute/Damaged goods"},
    {"button_id": "other",            "label": "Other"},
]


def _extract_intent_regex(message: str, invoice_amount: int | None = None) -> dict:
    """Deterministic fallback that mirrors the LLM's JSON schema.

    Used when the model is unavailable or returns non-JSON (e.g. in tests).
    """
    amount = _extract_amount_rupees(message)
    dates = _extract_iso_dates(message)
    if amount is not None:
        if invoice_amount and amount >= invoice_amount:
            intent = "full_payment"
        else:
            intent = "partial_payment"
    elif _signals_already_paid(message):
        intent = "already_paid"
    elif _signals_dispute(message):
        intent = "dispute"
    elif _signals_cannot_pay(message):
        intent = "cannot_pay"
    elif _is_confirmation(message):
        intent = "confirm"
    elif _is_question(message):
        intent = "question"
    else:
        intent = "other"
    return {
        "intent": intent,
        "upfront_amount": amount,
        "date": dates[0] if dates else None,
    }


def extract_intent(session: dict, message: str) -> dict:
    """Extract intent + variables as JSON. LLM first, regex fallback.

    The model is prompted STRICTLY to return JSON and to do no arithmetic. Any
    failure — missing API key, malformed JSON, or a non-JSON reply — falls back
    to deterministic regex extraction. The balance is computed later, in Python.
    """
    intent: dict | None = None
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=512,
            messages=[
                {"role": "system", "content": _EXTRACT_INTENT_SYSTEM},
                {"role": "user", "content": message},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        intent = _parse_intent_json(raw)
    except Exception:
        logger.debug("intent extraction via LLM failed; using regex fallback",
                     exc_info=True)

    if not intent:
        intent = _extract_intent_regex(message, session.get("invoice_amount"))
    return intent


def _round_to_100(amount: float) -> int:
    return int(round(amount / 100) * 100)


def _counter_above(offered: int, anchor: int, attempt: int) -> int:
    """A counter strictly between the debtor's offer and our opening ask.

    Pushes high on the first attempt and relaxes toward their number on later
    attempts, so we probe for the debtor's true maximum without overplaying.
    """
    gap = max(0, anchor - offered)
    fraction = max(0.30, 0.60 - 0.15 * attempt)
    counter = _round_to_100(offered + gap * fraction)
    # Clamp strictly between offered and anchor (rounded to ₹100).
    return max(offered + 100, min(counter, anchor - 100))


def _render_plan_text(session: dict, plan: list[dict], engine: NegotiationEngine) -> str:
    """Render the settlement plan block shown to the debtor before the link."""
    name = session["debtor_name"]
    inv = session["invoice_id"]
    total = _inr(engine.invoice_amount)
    lines = [
        f"Here's your payment plan, {name}:",
        "",
        f"📋 Settlement Plan — {inv}",
        "─────────────────────────────",
    ]
    for i in plan:
        if i["status"] == "pending_payment":
            lines.append(f"✅ Today ({format_date(i['date'])}):  {_inr(i['amount'])}  ← payment link below")
        else:
            lines.append(f"📅 {format_date(i['date'])}:  {_inr(i['amount'])}  ← reminder will be sent")
    lines += [
        "─────────────────────────────",
        f"Total:                 {total} (100% recovered)",
        "─────────────────────────────",
        f"All dates are within {MAX_PLAN_DAYS} days.",
        "Missing a payment will affect your trust score.",
    ]
    return "\n".join(lines)


def _set_plan_and_terms(session: dict, engine: NegotiationEngine, plan: list[dict]) -> None:
    """Persist the plan, mark it shown, and build agreed_terms for the dashboard."""
    session["installment_plan"] = plan
    session["plan_shown"] = True
    upfront = plan[0]["amount"]  # rupees
    invoice = engine.invoice_amount
    deferred = invoice - upfront
    last_date = plan[-1]["date"]
    session["agreed_terms"] = _normalize_no_discount_plan(session, {
        "upfront_amount":      round(upfront * 100),
        "upfront_pct":         round(upfront / invoice * 100, 1),
        "deferred_amount_raw": round(deferred * 100),
        "deferred_pct":        round(deferred / invoice * 100, 1),
        "deferred_days":       (date.fromisoformat(last_date) - engine.today).days,
        "deferred_due_date":   last_date,
        "upfront_display":     _inr(upfront),
        "due_date_display":    format_date(last_date),
    })
    for i in plan:
        _audit(session, "installment_scheduled",
               date=i["date"], amount=i["amount"], status=i["status"])
    _audit(session, "plan_ready", installments=plan)


def _accept_full_payment(session: dict, engine: NegotiationEngine) -> str:
    """Settle the full invoice today — no dates, no confirm step, send the link."""
    full = engine.invoice_amount
    session["debtor_agreed_amount"] = full
    session["last_debtor_offer"] = full
    session["future_dates"] = []
    plan, _status = engine.build_plan(full, [])
    _set_plan_and_terms(session, engine, plan)
    order = _handle_generate_payment_link({"amount": full}, session)
    session["negotiation_complete"] = True
    if isinstance(order, dict) and "error" in order:
        return (
            f"The debtor agreed to pay the full amount {_inr(full)}. Tell them "
            f"we're preparing their payment link and it will appear shortly."
        )
    from_state = session.get("state", "opening")
    session["state"] = "payment_pending"
    _audit(session, "state_transition", from_state=from_state,
           to_state="payment_pending", agreed_amount=full)
    return (
        f"The debtor agreed to pay the full amount {_inr(full)}. Tell them the "
        f"payment link for {_inr(full)} is ready and they can complete payment now."
    )


def _advance_negotiation(session: dict, engine: NegotiationEngine, msg: str,
                         intent: dict | None = None) -> str:
    """Run one step of the state machine and return the instruction for DeepSeek.

    `intent` (optional) is the LLM-extracted `{intent, upfront_amount, date}`
    JSON. When present, its `upfront_amount` and `date` are authoritative over
    the regex extraction — the model is the extractor, Python stays the decider.
    """
    state = session.get("state", "opening")
    offered = _extract_amount_rupees(msg)
    if intent and isinstance(intent.get("upfront_amount"), int):
        offered = intent["upfront_amount"]
    session["current_ask"] = None   # default; the counter path overrides this

    # Turn-tracking guard: once we've issued a counter-offer, a debtor who comes
    # back with a number still BELOW that counter is rejecting it. Stop the
    # arithmetic and ask WHY they can't meet the amount, rather than countering
    # again or silently accepting their lowball. This fires on the very next
    # turn after the first counter — independent of the min_upfront floor. The
    # reason MCQ is asked at most once (reason_collected).
    if (state == "negotiating"
            and session.get("first_counter_issued")
            and not session.get("reason_collected")
            and offered is not None
            and session.get("last_bot_offer") is not None
            and offered < session["last_bot_offer"]):
        session["last_debtor_offer"] = offered
        session["state"] = "hardship"
        session["upload_requested"] = False
        session["reason_collected"] = True
        session["reason_mcq_pending"] = True
        _audit(session, "state_transition", from_state=state,
               to_state="hardship", reason="offer_below_counter",
               offered=offered, previous_bot_offer=session["last_bot_offer"])
        return (
            f"The debtor offered {_inr(offered)}, below our previous counter of "
            f"{_inr(session['last_bot_offer'])}. Acknowledge their number warmly "
            f"and, in one question, ask what's making it hard to get to that amount."
        )

    # Full payment — the debtor settles the whole invoice today. No counter, no
    # dates, no confirm step: generate the payment link right away.
    if (offered is not None and offered >= engine.invoice_amount) or (
        offered is None and _signals_pay_in_full(msg)
    ):
        return _accept_full_payment(session, engine)

    if state == "opening":
        session["state"] = "negotiating"
        state = "negotiating"          # keep the local copy in sync for fall-through
        session["negotiation_step"] = 1
        _audit(session, "state_transition", from_state="opening",
               to_state="negotiating", step=1)
        if offered is not None:
            # The debtor answered the opening with an amount ("1k"). Treat it as
            # their first offer and fall through to the negotiating logic below,
            # which counters upward before settling.
            session["last_debtor_offer"] = offered
        else:
            return f"Ask for {_inr(engine.step1_amount)} today and one future payment."

    if state == "negotiating":
        # Record every offer (acceptable or not) so we can acknowledge it and,
        # if we've countered, accept their firm number.
        if offered is not None:
            session["last_debtor_offer"] = offered

        # 1) An acceptable offer. Don't accept the first number at face value —
        #    counter ABOVE it a few times to recover the debtor's true maximum,
        #    then settle on their best number.
        if offered is not None and engine.is_acceptable(offered):
            anchor = engine.step1_amount
            attempts = session.get("counter_attempts", 0)
            if (offered >= anchor
                    or _signals_ceiling(msg)
                    or attempts >= MAX_COUNTER_ATTEMPTS):
                # They've met our opening ask, stated a hard ceiling, or we've
                # already pushed enough — accept their number.
                session["debtor_agreed_amount"] = offered
                session["future_dates"] = []
                session["state"] = "collecting_dates"
                suggested = engine.suggest_dates(engine.max_installments - 1, offered)
                nxt = suggested[0] if suggested else engine.deadline.isoformat()
                _audit(session, "state_transition", from_state="negotiating",
                       to_state="collecting_dates", agreed_amount=offered)
                return (
                    f"The debtor agreed to {_inr(offered)} today. Ask what date works for the "
                    f"remaining {_inr(engine.invoice_amount - offered)}. Suggest {nxt} as an "
                    f"option. Remind them the latest possible date is {engine.deadline}."
                )
            # Raise the bar: counter above their offer before we settle.
            counter = _counter_above(offered, anchor, attempts)
            session["counter_attempts"] = attempts + 1
            session["current_ask"] = counter
            # Track the counter so the next turn can detect a rejection: a lower
            # follow-up offer triggers the reason MCQ instead of another counter.
            session["first_counter_issued"] = True
            session["last_bot_offer"] = counter
            _audit(session, "counter_offer", offered=offered, counter=counter,
                   attempt=session["counter_attempts"])
            return (
                f"The debtor offered {_inr(offered)}. Do not accept it yet — counter with "
                f"{_inr(counter)} and see if they will go higher. If they hold firm or raise "
                f"only a little, accept their best number."
            )

        # 2) They rejected our counter with no new number — they're holding firm
        #    at their last acceptable offer, so accept that now.
        prev = session.get("last_debtor_offer")
        if (offered is None and prev is not None
                and engine.is_acceptable(prev)
                and session.get("counter_attempts", 0) > 0
                and not _signals_cannot_pay(msg)
                and not _is_question(msg)):
            session["debtor_agreed_amount"] = prev
            session["future_dates"] = []
            session["state"] = "collecting_dates"
            suggested = engine.suggest_dates(engine.max_installments - 1, prev)
            nxt = suggested[0] if suggested else engine.deadline.isoformat()
            _audit(session, "state_transition", from_state="negotiating",
                   to_state="collecting_dates", agreed_amount=prev)
            return (
                f"The debtor is holding firm at {_inr(prev)}. Accept it now. Ask what date "
                f"works for the remaining {_inr(engine.invoice_amount - prev)}. Suggest {nxt} "
                f"as an option. Remind them the latest possible date is {engine.deadline}."
            )

        # 3) A clear inability-to-pay signal (with no concrete offer) → stop
        #    pushing the number and understand their situation first.
        if offered is None and _signals_cannot_pay(msg):
            if session.get("reason_collected"):
                return _trigger_final_ultimatum(session, engine)
            session["state"] = "hardship"
            session["upload_requested"] = False
            session["reason_collected"] = True
            session["reason_mcq_pending"] = True
            _audit(session, "state_transition", from_state="negotiating",
                   to_state="hardship")
            return (
                "The debtor says they have no money to pay right now. Acknowledge this "
                "warmly and, in one question, ask what's making it hard to pay."
            )

        # 4) A stated hard ceiling below our floor ("2k max") → stop countering
        #    and understand their situation, rather than pummeling the ladder.
        if offered is not None and _signals_ceiling(msg) and not engine.is_acceptable(offered):
            if session.get("reason_collected"):
                return _trigger_final_ultimatum(session, engine)
            session["state"] = "hardship"
            session["upload_requested"] = False
            session["reason_collected"] = True
            session["reason_mcq_pending"] = True
            _audit(session, "state_transition", from_state="negotiating",
                   to_state="hardship")
            return (
                f"The debtor set a hard ceiling of {_inr(offered)}, below our floor. "
                f"Acknowledge their limit warmly and ask, in one question, what's making "
                f"it hard to pay."
            )

        # 5) A clarifying question (no amount) → answer it; never step the ladder
        #    or escalate on a question.
        if offered is None and _is_question(msg):
            _audit(session, "debtor_question")
            return (
                "The debtor asked a question. Answer it directly and honestly, then restate "
                "what you can offer (your ask). Do not escalate and do not lower your ask."
            )

        # 6) Rejection (no amount, or an offer below the floor without a hard
        #    ceiling) → counter-anchor and step down the ladder toward the floor.
        step = session.get("negotiation_step", 1) + 1
        session["negotiation_step"] = step
        _audit(session, "negotiation_step", step=step)
        if step >= 5:
            if session.get("reason_collected") or session.get("hardship_verified"):
                # We already asked why once (or verified hardship) — a further
                # rejection is terminal. Trapdoor to the final ultimatum.
                return _trigger_final_ultimatum(session, engine, step=step)
            # Pivot to a hardship investigation instead of escalating and closing.
            session["state"] = "hardship"
            session["upload_requested"] = False
            session["reason_collected"] = True
            session["reason_mcq_pending"] = True
            _audit(session, "state_transition", from_state="negotiating",
                   to_state="hardship", step=step)
            return (
                "The debtor rejected every amount, including the minimum. Ask them, "
                "warmly and in one question, what's making it hard to pay."
            )
        if step == 2:
            return (
                f"The debtor pushed back. Acknowledge their reply, then ask for "
                f"{_inr(engine.step2_amount)} today instead."
            )
        if step == 3:
            return (
                f"The debtor pushed back again. Acknowledge their reply, then ask for "
                f"{_inr(engine.step3_amount)} today — explain this is the minimum needed."
            )
        return (
            f"The debtor rejected the minimum. Acknowledge their reply, make one final "
            f"appeal for {_inr(engine.min_today)}. Mention the escalation consequence."
        )

    if state == "hardship":
        # If they now offer any amount, accept it and defer the rest — never loop
        # back into asking for proof again.
        if offered is not None and offered > 0:
            session["last_debtor_offer"] = offered
            session["debtor_agreed_amount"] = offered
            session["future_dates"] = []
            session["state"] = "collecting_dates"
            suggested = engine.suggest_dates(engine.max_installments - 1, offered)
            nxt = suggested[0] if suggested else engine.deadline.isoformat()
            _audit(session, "state_transition", from_state="hardship",
                   to_state="collecting_dates", agreed_amount=offered)
            return (
                f"The debtor agreed to {_inr(offered)} today. Ask what date works for the "
                f"remaining {_inr(engine.invoice_amount - offered)}. Suggest {nxt} as an "
                f"option. Remind them the latest possible date is {engine.deadline}."
            )

        # A question (e.g. "what do you need from me?") → answer it, never escalate.
        if _is_question(msg):
            _audit(session, "debtor_question")
            if session.get("upload_requested"):
                return (
                    "The debtor asked what we need from them. Tell them exactly: a clear "
                    "bank statement or hardship document showing their financial situation "
                    "(name, recent balance, or a closure/medical letter). Reassure them it is "
                    "kept confidential and only used to verify their claim."
                )
            return (
                "The debtor asked a question. Answer it directly and warmly, and invite "
                "them to tell you what's going on."
            )

        # No amount offered. Ask for proof exactly once, and only when they give
        # a reason (otherwise they can always upload from the chat). Set the
        # upload card in Python — the model has no function calling.
        if not session.get("upload_requested") and _looks_like_reason(msg):
            session["upload_requested"] = True
            _handle_request_document_upload(
                {"document_type": "bank statement",
                 "reason": "debtor reports they cannot pay"},
                session,
            )
            _audit(session, "upload_requested")
            return (
                "The debtor explained why they can't pay. Ask them to upload a bank "
                "statement or hardship document so we can verify their situation and "
                "review a lower amount."
            )

        # No amount and either no reason or proof already requested — stop pushing.
        _audit(session, "state_transition", from_state="hardship",
               to_state="escalated", reason="negotiation_exhausted")
        _trigger_escalation(session, "negotiation_exhausted")
        return (
            "No agreement could be reached. Close warmly, explain next steps, and do not "
            "ask for payment again."
        )

    if state == "collecting_dates":
        dates = _extract_iso_dates(msg)
        if intent and intent.get("date") and intent["date"] not in dates:
            dates.insert(0, intent["date"])
        if not dates and _is_confirmation(msg):
            dates = engine.suggest_dates(engine.max_installments - 1)

        bad_date = None
        for d in dates:
            try:
                parsed = date.fromisoformat(d)
            except ValueError:
                continue
            if parsed > engine.deadline:
                bad_date = d
                break
            if parsed > engine.today and d not in session["future_dates"]:
                session["future_dates"].append(d)

        if bad_date:
            return (
                f"The debtor said {bad_date}, which is after our 34-day limit. Explain you "
                f"can only go up to {engine.deadline}. Ask if that works."
            )

        required = engine.max_installments - 1
        if len(session["future_dates"]) >= required:
            plan, status = engine.build_plan(session["debtor_agreed_amount"], session["future_dates"])
            if status != "ok":
                return f"Ask the debtor for a valid future date before {engine.deadline}."
            _set_plan_and_terms(session, engine, plan)
            session["state"] = "plan_ready"
            _audit(session, "state_transition", from_state="collecting_dates",
                   to_state="plan_ready")
            plan_text = _render_plan_text(session, plan, engine)
            return f"Show this exact plan and ask the debtor to confirm:\n{plan_text}"

        suggested = engine.suggest_dates(engine.max_installments - 1)
        remaining = required - len(session["future_dates"])
        if len(session["future_dates"]) < len(suggested):
            nxt = suggested[len(session["future_dates"])]
        else:
            nxt = engine.deadline.isoformat()
        return (
            f"Ask for {remaining} more future payment date(s). Suggest {nxt} as an option. "
            f"Remind them the latest possible date is {engine.deadline}."
        )

    if state == "plan_ready":
        if _is_confirmation(msg):
            # The debtor confirmed the plan. End the conversational phase: set the
            # flag that forces a finalize_agreement tool call. The order + final
            # message are produced by _finalize_agreement, never by the LLM.
            session["finalize_requested"] = True
            session["state"] = "finalizing"
            _audit(session, "state_transition", from_state="plan_ready",
                   to_state="finalizing")
            return "FINALIZE_AGREEMENT"
        session["state"] = "negotiating"
        session["negotiation_step"] = 1
        session["counter_attempts"] = 0
        session["future_dates"] = []
        session["plan_shown"] = False
        _audit(session, "state_transition", from_state="plan_ready",
               to_state="negotiating")
        return (
            f"The debtor wants to change the plan. Ask what amount works for them today "
            f"(starting from {_inr(engine.step1_amount)})."
        )

    if state == "payment_pending":
        return (
            f"The payment link for {_inr(session.get('debtor_agreed_amount', engine.min_today))} "
            f"is already shown. Acknowledge the debtor and gently remind them to complete payment."
        )

    return "This conversation is closed. Do not ask for payment."


# ---------------------------------------------------------------------------
# Finalization — a forced tool call ends the conversation
# ---------------------------------------------------------------------------
#
# Once the debtor confirms the plan ("ya"), the conversational phase must end.
# The model is FORCED (tool_choice) to call finalize_agreement; the backend
# intercepts that call, does NOT loop back to the model, and injects a
# deterministic final message carrying the payment payload. Python owns every
# number here — the model's tool arguments are echoed from the plan but never
# trusted for arithmetic.

FINALIZE_AGREEMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "finalize_agreement",
        "description": (
            "Finalize the agreed payment plan and produce the payment link. Call "
            "this exactly once, immediately after the debtor confirms the dates and "
            "amounts. Do not write any text when calling it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "upfront_amount": {
                    "type": "integer",
                    "description": "Rupees the debtor pays today.",
                },
                "deferred_amount": {
                    "type": "integer",
                    "description": "Rupees deferred to a later date.",
                },
                "deferred_date": {
                    "type": "string",
                    "description": "ISO date (YYYY-MM-DD) of the deferred payment.",
                },
            },
            "required": ["upfront_amount", "deferred_amount", "deferred_date"],
        },
    },
}


def _finalize_agreement(session: dict, engine: NegotiationEngine) -> str:
    """Terminal tool implementation — Python owns the final math + message.

    Computes the upfront/deferred split and the deferred date from the already-
    agreed plan, creates the Razorpay order, and returns a deterministic final
    message. The model never does this arithmetic.
    """
    upfront = session.get("debtor_agreed_amount") or 0
    invoice = engine.invoice_amount
    deferred = max(0, invoice - upfront)

    plan = session.get("installment_plan") or []
    future = [p for p in plan if p.get("status") == "scheduled"]
    deferred_date = future[-1]["date"] if future else engine.deadline.isoformat()

    order = _handle_generate_payment_link({"amount": upfront}, session)
    session["negotiation_complete"] = True
    if isinstance(order, dict) and "error" in order:
        return "Thanks — we're preparing your payment link and it will appear in a moment."

    session["state"] = "payment_pending"
    _audit(
        session,
        "finalize_agreement",
        upfront_amount=upfront,
        deferred_amount=deferred,
        deferred_date=deferred_date,
        order_id=session.get("razorpay_order_id"),
    )

    name = _first_name(session["debtor_name"])
    if deferred > 0:
        return (
            f"Thanks {name} — we're all set! Your payment link for {_inr(upfront)} "
            f"today is ready below, and the remaining {_inr(deferred)} is scheduled "
            f"for {format_date(deferred_date)}. The link is valid for 24 hours."
        )
    return (
        f"Thanks {name} — we're all set! Your payment link for {_inr(upfront)} "
        f"is ready below. It's valid for 24 hours."
    )


def _call_llm_finalize(
    session: dict, client: OpenAI, upfront: int, deferred: int, deferred_date: str
) -> list:
    """Force the model to emit the finalize_agreement tool call (no text)."""
    system = (
        "You are Aria, a payment recovery specialist. The debtor has confirmed the "
        "payment plan. Finalize it now by calling the finalize_agreement function "
        "with the agreed numbers. Do not write any text — only the function call.\n"
        f"Upfront amount: {_inr(upfront)}\n"
        f"Deferred amount: {_inr(deferred)}\n"
        f"Deferred date: {deferred_date}"
    )
    response = client.chat.completions.create(
        model=FINALIZE_MODEL,
        max_tokens=256,
        messages=[{"role": "system", "content": system}],
        tools=[FINALIZE_AGREEMENT_TOOL],
        tool_choice={"type": "function", "function": {"name": "finalize_agreement"}},
    )
    return response.choices[0].message.tool_calls or []


def _handle_finalize_turn(
    session: dict, engine: NegotiationEngine, turn: int
) -> tuple[str, dict]:
    """Intercept the finalize tool call and inject the deterministic final message.

    Best-effort: force the model to call finalize_agreement, then — regardless of
    whether the model obliged — finalize in Python and stop the LLM from writing
    any further text.
    """
    upfront = session.get("debtor_agreed_amount") or 0
    deferred = max(0, engine.invoice_amount - upfront)
    plan = session.get("installment_plan") or []
    future = [p for p in plan if p.get("status") == "scheduled"]
    deferred_date = future[-1]["date"] if future else engine.deadline.isoformat()

    try:
        client = _get_client()
        tool_calls = _call_llm_finalize(session, client, upfront, deferred, deferred_date)
        if tool_calls:
            _audit(session, "finalize_tool_called", tool_calls=[
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in tool_calls
            ])
    except Exception:
        logger.debug("finalize tool call failed; finalizing directly", exc_info=True)

    message = _finalize_agreement(session, engine)
    session["action_type"] = "finalize_agreement"
    session["messages"].append({"role": "assistant", "content": message})
    _record_agent_message(session, message)
    _audit(session, "agent_turn", turn=turn, speaker="agent", message=message,
           action_type="finalize_agreement")
    return _finalize_turn(session, message)


def _trigger_final_ultimatum(session: dict, engine: NegotiationEngine,
                             step: int | None = None) -> str:
    """Second rejection after the reason was collected → terminal escalation.

    Sets the terminal state and a flag so process_turn emits the deterministic
    final-ultimatum message (no LLM text), rather than re-asking the MCQ.
    """
    _audit(session, "state_transition", from_state="negotiating",
           to_state="escalated", reason="final_ultimatum", step=step)
    _trigger_escalation(session, "final_ultimatum")
    session["final_ultimatum_requested"] = True
    return "FINAL_ULTIMATUM"


def _final_ultimatum_message(session: dict, engine: NegotiationEngine) -> str:
    """Deterministic terminal message stating the absolute minimum."""
    name = _first_name(session["debtor_name"])
    return (
        f"{_inr(engine.min_today)} is the absolute minimum I can accept today, {name}. "
        f"If we can't agree on that, I'll need to pass this to our escalation team."
    )


def _handle_final_ultimatum(session: dict, engine: NegotiationEngine, turn: int):
    """Emit the final-ultimatum message and end the conversation — no LLM text."""
    message = _final_ultimatum_message(session, engine)
    session["action_type"] = "final_ultimatum"
    session["final_ultimatum_requested"] = False
    session["messages"].append({"role": "assistant", "content": message})
    _record_agent_message(session, message)
    _audit(session, "agent_turn", turn=turn, speaker="agent", message=message,
           action_type="final_ultimatum")
    return _finalize_turn(session, message)


def _handle_reason_mcq_answer(
    session: dict, engine: NegotiationEngine, button_id: str
) -> tuple[str, dict]:
    """Handle a debtor's reason-MCQ answer: lower the floor and concede.

    The button_id maps to a reason; Python lowers `min_today` via apply_hardship()
    (the arithmetic stays in the engine), reopens the negotiation at the hardship
    floor, and lets DeepSeek write the concession message.
    """
    session["turn_count"] = session.get("turn_count", 0) + 1
    turn = session["turn_count"]

    label = next((r["label"] for r in MCQ_REASONS if r["button_id"] == button_id), "Other")
    session["rejection_reason"] = label
    session["reason_mcq_pending"] = False

    new_min = engine.apply_hardship()
    
    highest_offer = session.get("highest_user_offer", 0)
    actual_floor = max(new_min, highest_offer)
    new_min = actual_floor
    engine.min_today = actual_floor
    engine.step3_amount = actual_floor
    
    session["hardship_verified"] = True
    session["negotiation_engine"] = engine.to_dict()
    session["state"] = "negotiating"
    session["negotiation_step"] = 3      # re-open at the hardship floor
    _audit(session, "reason_mcq_answered", button_id=button_id, reason=label,
           new_min_today=new_min)

    instruction = (
        f"The debtor explained why they can't pay: {label}. Acknowledge their reason "
        f"warmly, then explain we can come down to {_inr(new_min)} today and ask if "
        f"they can manage that.\n\n"
        f"The user's highest offer so far is {_inr(highest_offer)}.\n"
        f"CRITICAL GUARDRAIL: Never suggest an upfront payment lower than the user's highest offer. "
        f"If their offer meets or exceeds your hardship floor, accept their offer immediately."
    )
    context = _build_context(session, engine, instruction)
    session["system_prompt"] = build_system_prompt(session, context)
    # Mark the MCQ question as answered so the transcript keeps the question +
    # options visible, but re-renders them non-clickable on a chat preview.
    for m in reversed(session.get("messages", [])):
        if m.get("role") == "assistant" and m.get("mcq_options"):
            m["mcq_answered"] = True
            m["mcq_selected"] = label
            break
    session["messages"].append(
        {"role": "user", "content": f"Debtor selected reason: {label}"}
    )

    try:
        client = _get_client()
        reply = _parse_agent_json(_call_llm(session, client))["reply_to_user"]
    except EnvironmentError:
        reply = f"Given your situation, we can come down to {_inr(new_min)} today. Could you manage that?"
    except Exception:
        logger.exception("reason-MCQ concession turn failed")
        reply = f"Given your situation, we can come down to {_inr(new_min)} today. Could you manage that?"

    if not reply:
        reply = f"Given your situation, we can come down to {_inr(new_min)} today. Could you manage that?"

    session["action_type"] = "negotiate"
    session["messages"].append({"role": "assistant", "content": reply})
    _record_agent_message(session, reply)
    _audit(session, "agent_turn", turn=turn, speaker="agent", message=reply,
           action_type="negotiate")
    return _finalize_turn(session, reply)


def _build_context(session: dict, engine: NegotiationEngine, instruction: str) -> dict:
    """Assemble the numbers + instruction dict passed into the system prompt."""
    state = session.get("state", "opening")
    step = session.get("negotiation_step", 1)
    offered = session.get("debtor_agreed_amount")
    numbers = engine.get_context_for_agent(step, offered)
    current_ask = session.get("current_ask")   # set by the counter path, else None
    if state == "negotiating":
        step_ask = current_ask if current_ask is not None else {
            1: engine.step1_amount, 2: engine.step2_amount,
            3: engine.step3_amount, 4: engine.min_today,
        }.get(step, engine.min_today)
    else:
        step_ask = current_ask if current_ask is not None else (offered if offered else engine.min_today)
    numbers["step_ask"] = step_ask
    numbers["floor"] = engine.min_today
    numbers["debtor_offer"] = session.get("last_debtor_offer")
    # Remaining balance — computed in Python, never by the model. The debtor's
    # latest committed amount (offer or agreed) is subtracted from the invoice.
    committed = session.get("last_debtor_offer") or session.get("debtor_agreed_amount")
    numbers["current_remaining_balance"] = (
        max(0, engine.invoice_amount - committed) if committed else None
    )
    return {"state": state, "step": step, "instruction": instruction, "numbers": numbers}


def process_turn(session: dict, debtor_message: str) -> tuple[str, dict]:
    """
    Process one debtor message. The LLM first extracts intent + variables as
    JSON; Python computes the remaining balance and advances the state machine;
    DeepSeek then turns that instruction into a human message.
    """
    if session["status"] not in ("active", "promise_to_pay"):
        return f"[Session {session['status']} — no further turns]", session

    session["turn_count"] += 1
    turn = session["turn_count"]
    session["last_debtor_ts"] = _ts() if debtor_message.strip() else None
    session["reason_mcq_pending"] = False   # per-turn flag; set again by the state machine

    # --- Stopping rule: unresponsive (silent / empty message) ---
    if not debtor_message.strip():
        _audit(session, "debtor_turn", turn=turn, speaker="debtor", message="[silent]")
        if turn >= session["max_turns"]:
            _audit(session, "stopping_rule", status="escalated", reason="debtor_unresponsive")
            _trigger_escalation(session, "debtor_unresponsive")
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
        _trigger_escalation(session, "debtor_legal_threat", status="legal_hold")
        session["messages"].append({"role": "user", "content": debtor_message})
        session["messages"].append({"role": "assistant", "content": reply})
        _record_agent_message(session, reply)
        _audit(session, "agent_turn", turn=turn, speaker="agent", message=reply)
        return _finalize_turn(session, reply)

    # --- Stopping rule: debtor asks for a human ---
    if _requests_human(debtor_message):
        reply = "No problem — I'll connect you with a real person on our team who'll reach out shortly."
        _audit(session, "stopping_rule", status="escalated", reason="debtor_requested_human")
        _trigger_escalation(session, "debtor_requested_human")
        session["messages"].append({"role": "user", "content": debtor_message})
        session["messages"].append({"role": "assistant", "content": reply})
        _record_agent_message(session, reply)
        _audit(session, "agent_turn", turn=turn, speaker="agent", message=reply)
        return _finalize_turn(session, reply)

    # --- Python decides; DeepSeek only speaks ---
    engine = _get_engine(session)

    # Extract intent + variables as JSON (LLM, with regex fallback). The model
    # does NOT do arithmetic here — it only reports what the debtor said.
    intent = extract_intent(session, debtor_message)
    session["last_intent"] = intent
    _audit(session, "intent_extracted", intent=intent)

    # Special cases (already-paid / dispute) are decided in Python now that the
    # model has no function calling — they override the normal state machine.
    if _signals_already_paid(debtor_message):
        _handle_request_document_upload(
            {"document_type": "payment receipt",
             "reason": "debtor claims the invoice was already paid"},
            session,
        )
        instruction = (
            "The debtor says they already paid this invoice. Warmly ask them to "
            "upload their payment receipt or bank statement (UTR/transaction ID) "
            "so we can verify it — and reassure them we'll stop the reminders if "
            "it's confirmed."
        )
    elif _signals_dispute(debtor_message):
        _handle_flag_dispute({"reason": debtor_message}, session)
        instruction = (
            "The debtor disputes the invoice. Acknowledge their concern warmly, "
            "tell them we've paused payment requests and flagged it for review."
        )
    else:
        instruction = _advance_negotiation(session, engine, debtor_message, intent=intent)

    # Terminal: the debtor confirmed the plan. Force the finalize_agreement tool
    # call and inject the deterministic final message — no further LLM text.
    if session.get("finalize_requested"):
        session["messages"].append({"role": "user", "content": debtor_message})
        return _handle_finalize_turn(session, engine, turn)

    # Terminal: second rejection after the reason was collected — final ultimatum.
    if session.get("final_ultimatum_requested"):
        session["messages"].append({"role": "user", "content": debtor_message})
        return _handle_final_ultimatum(session, engine, turn)

    context = _build_context(session, engine, instruction)
    session["current_remaining_balance"] = context["numbers"]["current_remaining_balance"]
    session["system_prompt"] = build_system_prompt(session, context)
    session["messages"].append({"role": "user", "content": debtor_message})

    try:
        client = _get_client()
        raw_reply = _call_llm(session, client)
    except EnvironmentError:
        raw_reply = _no_key_reply(session)
    except Exception:
        logger.exception("LLM turn failed")
        raw_reply = "Sorry, I hit a little snag — could you say that again?"

    parsed = _parse_agent_json(raw_reply)
    reply = parsed["reply_to_user"]
    if not reply:
        reply = "Thanks — let me get that sorted for you."

    # Python decides the authoritative action_type; the LLM's is advisory only.
    if session.get("reason_mcq_pending"):
        action_type = "trigger_reason_mcq"
        session["mcq_options"] = MCQ_REASONS
        session["reason_mcq_pending"] = False
    else:
        action_type = "negotiate"
    session["action_type"] = action_type
    session["agent_thought_process"] = parsed["thought_process"]
    session["agent_suggested_action"] = parsed["action_type"]

    assistant_entry = {"role": "assistant", "content": reply}
    if action_type == "trigger_reason_mcq":
        # Persist the question + options on the message so the transcript can be
        # re-rendered (and the buttons made non-clickable once answered).
        assistant_entry["mcq_options"] = session["mcq_options"]
        assistant_entry["mcq_answered"] = False
    session["messages"].append(assistant_entry)
    _record_agent_message(session, reply)
    _audit(session, "agent_turn", turn=turn, speaker="agent", message=reply,
           action_type=action_type, thought_process=parsed["thought_process"])

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
    _record_agent_message(session, opening)
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
