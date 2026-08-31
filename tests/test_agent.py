"""
Unit tests for the negotiation agent (backend.agent) and its state machine.
=============================================================================

Core invariants (new architecture — Python decides, DeepSeek only speaks):
1. The NegotiationEngine owns every number; the agent never calculates.
2. The state machine transitions opening → negotiating → collecting_dates
   → plan_ready → payment_pending purely in Python.
3. The ladder: reject below the minimum → step up; step > 4 → escalate.
4. Dates beyond the 34-day deadline are rejected.
5. The payment link amount is forced to debtor_agreed_amount, never more.
6. The system prompt contains state + numbers + instruction — NOT the ladder,
   percentages, or tools.
7. project_score_change / stance invariants (legacy scoring) still hold.

Run with:
    python -m pytest tests/test_agent.py -v
"""

import json
from datetime import date, timedelta
from types import SimpleNamespace

from backend import agent
from backend.scoring import (
    get_negotiation_stance,
    project_score_change,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(invoice_amount: int, score: int = 65) -> dict:
    """Build a minimal session dict for tool-handler tests."""
    stance = get_negotiation_stance(score)
    paise = invoice_amount * 100
    return {
        "session_id":           "sess-test",
        "invoice_id":           "INV-TEST",
        "debtor_id":            "D-TEST",
        "debtor_name":          "Test Debtor",
        "company_name":         "Test Co",
        "invoice_amount_paise": paise,
        "invoice_amount":       invoice_amount,
        "score":                score,
        "tier":                 "B",         # display only
        "stance":               stance,
        "debtor_agreed_amount": None,
        "recent_agent_messages": [],
        "state":                "opening",
        "negotiation_step":     1,
        "future_dates":         [],
        "installment_plan":     None,
        "plan_shown":           False,
        "negotiation_complete": False,
        "negotiation_engine":   {},
        "score_projections":    {"full_upfront": 78, "partial_deferred": 73, "escalated": 50},
        "system_prompt":        "[test system prompt]",
        "status":               "active",
        "messages":             [],
        "audit_log":            [],
    }


def _real_session(invoice_id="INV-0001"):
    """Create + open a real session, returning (session, engine)."""
    s = agent.create_session(invoice_id)
    agent.open_turn(s)
    return s, agent._get_engine(s)


# ---------------------------------------------------------------------------
# Negotiation stance + score projection invariants (legacy scoring)
# ---------------------------------------------------------------------------

SCORE_SAMPLES = [90, 72, 47, 15]


def test_stance_opening_above_target_above_floor():
    for score in SCORE_SAMPLES:
        stance = get_negotiation_stance(score)
        assert stance["opening"] > stance["target"]
        assert stance["target"] > 20
        assert stance["floor"] == 20


def test_stance_floor_is_always_20():
    for score in range(0, 101, 5):
        assert get_negotiation_stance(score)["floor"] == 20


def test_project_score_never_exceeds_100():
    for score in range(90, 101):
        for stype in ("full_upfront", "partial_deferred", "escalated", "ghosted"):
            assert project_score_change(score, stype) <= 100


def test_project_score_never_below_0():
    for score in range(0, 11):
        for stype in ("full_upfront", "partial_deferred", "escalated", "ghosted"):
            assert project_score_change(score, stype) >= 0


def test_project_full_upfront_beats_partial():
    for score in range(0, 101, 10):
        full = project_score_change(score, "full_upfront")
        partial = project_score_change(score, "partial_deferred")
        if full < 100 or partial < 100:
            assert full > partial
        else:
            assert full >= partial


def test_plan_amounts_sum_to_invoice():
    cases = [(400000, 25), (100000, 20), (80000, 50), (219000, 30)]
    for invoice, pct in cases:
        paise = invoice * 100
        upfront = round(paise * pct / 100)
        deferred = paise - upfront
        assert upfront + deferred == paise


# ---------------------------------------------------------------------------
# System prompt — "what to say", never "how to decide"
# ---------------------------------------------------------------------------

def _build_prompt(session, state="negotiating", step=1, instruction="Ask for ₹X today."):
    engine = agent._get_engine(session)
    # set the state/step BEFORE building the context, mirroring process_turn
    session["state"] = state
    session["negotiation_step"] = step
    ctx = agent._build_context(session, engine, instruction)
    return agent.build_system_prompt(session, ctx)


def test_system_prompt_has_state_numbers_instruction():
    s, _ = _real_session()
    p = _build_prompt(s, instruction="Ask for ₹72,500 today and one future payment.")
    assert "Aria" in p
    assert "RecoverFlow Demo Merchant" in p
    assert "CURRENT STATE" in p
    assert "WHAT TO DO THIS TURN" in p
    assert "Your ask this turn is ₹72,500" in p
    assert "Ask for ₹72,500 today and one future payment." in p
    assert "₹1,45,000" in p          # invoice amount


def test_system_prompt_hides_decision_logic():
    """The model sees a bounded range (ask + hard floor) but never the ladder
    steps, tier, percentages, or tools."""
    s, _ = _real_session()
    p = _build_prompt(s)
    assert "NEGOTIATION LADDER" not in p
    assert "HARD RULES" not in p
    assert "set_installment_plan" not in p
    assert "generate_payment_link" not in p
    assert "%" not in p
    # the floor is given as a bound the model must not go below — phrased as a
    # number, never exposed as an internal "minimum"/"min_today" label
    assert "Hard floor" in p
    assert "Minimum today" not in p
    assert "min_today" not in p
    # the tier VALUE is never revealed
    assert "Tier A" not in p and "Tier B" not in p and "Tier C" not in p and "Tier D" not in p


def test_system_prompt_does_not_embed_history():
    """History is sent as real message turns to the LLM, not flattened into the
    system prompt — so the prompt itself carries no transcript."""
    s, _ = _real_session()
    s["messages"].append({"role": "user", "content": "I can pay 40000 today"})
    p = _build_prompt(s)
    assert "CONVERSATION HISTORY" not in p
    assert "Debtor: I can pay 40000 today" not in p


# ---------------------------------------------------------------------------
# State machine (Python decides)
# ---------------------------------------------------------------------------

def test_opening_moves_to_negotiating():
    s, eng = _real_session()
    inst = agent._advance_negotiation(s, eng, "business is slow")
    assert s["state"] == "negotiating"
    assert s["negotiation_step"] == 1
    assert "₹72,500" in inst   # step 1 ask = 50% of ₹1,45,000


def test_opening_captures_first_offer():
    """A debtor who answers the opening with an amount ("1k") must be heard,
    not ignored in favour of the full 50% ask."""
    s, eng = _real_session()
    inst = agent._advance_negotiation(s, eng, "1k")
    assert s["state"] == "negotiating"
    assert s["last_debtor_offer"] == 1000
    assert "Acknowledge" in inst
    # the offer is surfaced to the LLM via the prompt's debtor-offer anchor
    ctx = agent._build_context(s, eng, inst)
    assert ctx["numbers"]["debtor_offer"] == 1000


def test_ceiling_below_floor_routes_to_hardship():
    """A hard ceiling below the floor ("no 2k max") stops the ladder and opens
    an understanding conversation instead of pummeling the same number."""
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")   # -> negotiating
    inst = agent._advance_negotiation(s, eng, "no 2k max")
    assert s["state"] == "hardship"
    assert s["last_debtor_offer"] == 2000
    assert "hard to pay" in inst


def test_no_cash_signal_routes_to_hardship():
    """'no cash now at all' is heard immediately, not after five rejections."""
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")   # -> negotiating
    inst = agent._advance_negotiation(s, eng, "no cash now at all")
    assert s["state"] == "hardship"
    assert "hard to pay" in inst


def test_below_floor_offer_is_acknowledged_not_ignored():
    """A plain lowball (no ceiling word) counters and records the offer."""
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")   # -> negotiating step 1
    inst = agent._advance_negotiation(s, eng, "can I pay 5000")
    assert s["negotiation_step"] == 2
    assert s["last_debtor_offer"] == 5000
    assert "Acknowledge" in inst


def test_question_does_not_escalate_in_hardship():
    """A clarifying question ("what do u need") is answered, never escalated."""
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")
    for _ in range(4):
        agent._advance_negotiation(s, eng, "no")
    assert s["state"] == "hardship"
    s["upload_requested"] = True
    inst = agent._advance_negotiation(s, eng, "what do u need")
    assert s["state"] == "hardship"
    assert s["status"] == "active"
    assert "bank statement" in inst.lower()


def test_acceptable_offer_is_countered_upward():
    """An acceptable first offer is not accepted at face value — we counter
    higher to recover the debtor's true maximum."""
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")
    inst = agent._advance_negotiation(s, eng, "I can pay 40000 today")
    assert s["state"] == "negotiating"          # still negotiating, not accepted
    assert s["last_debtor_offer"] == 40000
    assert s["counter_attempts"] == 1
    assert s["current_ask"] > 40000             # counter is above their offer
    assert "counter with" in inst


def test_acceptable_offer_with_ceiling_accepts_immediately():
    """A hard ceiling at an acceptable amount is honoured right away."""
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")
    inst = agent._advance_negotiation(s, eng, "I can pay 40000 max")
    assert s["state"] == "collecting_dates"
    assert s["debtor_agreed_amount"] == 40000


def test_hold_firm_after_counter_accepts_last_offer():
    """If the debtor rejects our counter with no new number, we accept their
    last offer rather than pushing further."""
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")
    agent._advance_negotiation(s, eng, "I can pay 40000 today")   # -> counter
    inst = agent._advance_negotiation(s, eng, "no")               # hold firm
    assert s["state"] == "collecting_dates"
    assert s["debtor_agreed_amount"] == 40000


def test_counter_exhausted_then_accepts():
    """After MAX_COUNTER_ATTEMPTS, a raised offer is accepted (not re-countered)."""
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")
    agent._advance_negotiation(s, eng, "I can pay 40000 today")   # counter 1 (59500)
    agent._advance_negotiation(s, eng, "I can pay 60000 today")   # counter 2 (65600)
    inst = agent._advance_negotiation(s, eng, "I can pay 68000 today")  # accept
    assert s["state"] == "collecting_dates"
    assert s["debtor_agreed_amount"] == 68000


def test_offer_below_counter_triggers_reason_mcq():
    """After the first counter, a follow-up offer still below the counter stops
    the negotiation and asks WHY the debtor can't meet the amount — rather than
    countering again or relying on the min_upfront floor."""
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")
    agent._advance_negotiation(s, eng, "I can pay 40000 today")   # -> counter 59500
    assert s["first_counter_issued"] is True
    assert s["last_bot_offer"] == 59500

    inst = agent._advance_negotiation(s, eng, "I can pay 45000 today")  # below counter

    assert s["state"] == "hardship"
    assert s["reason_collected"] is True
    assert s["reason_mcq_pending"] is True
    assert s["last_debtor_offer"] == 45000
    assert "hard to" in inst


def test_offer_above_counter_still_counters_up():
    """An offer above our counter (but below the opening ask) is not a rejection
    — we counter higher, probing for the debtor's true maximum."""
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")
    agent._advance_negotiation(s, eng, "I can pay 40000 today")   # -> counter 59500
    inst = agent._advance_negotiation(s, eng, "I can pay 60000 today")
    assert s["reason_collected"] is False
    assert s["counter_attempts"] == 2
    assert s["current_ask"] > 60000
    assert "counter with" in inst


def test_rejection_steps_down_then_pivots_to_hardship():
    """Rejecting the minimum no longer escalates — it opens a hardship path."""
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")   # -> step 1 (50%)
    agent._advance_negotiation(s, eng, "no")           # -> step 2 (30%)
    assert s["negotiation_step"] == 2
    agent._advance_negotiation(s, eng, "no")           # -> step 3 (min)
    assert s["negotiation_step"] == 3
    agent._advance_negotiation(s, eng, "no")           # -> step 4
    assert s["negotiation_step"] == 4
    inst = agent._advance_negotiation(s, eng, "no")    # -> step 5 = hardship, not escalate
    assert s["state"] == "hardship"
    assert s["status"] == "active"                     # conversation stays open
    assert s["negotiation_step"] == 5
    assert "hard to pay" in inst                       # asks for the reason first


def test_hardship_accepts_any_offer(monkeypatch):
    """In hardship, a debtor offering an amount is accepted — no proof loop."""
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")
    for _ in range(4):
        agent._advance_negotiation(s, eng, "no")
    assert s["state"] == "hardship"
    inst = agent._advance_negotiation(s, eng, "i can do 3k")   # 3000 > 0 → accept
    assert s["state"] == "collecting_dates"
    assert s["debtor_agreed_amount"] == 3000


def test_hardship_asks_proof_once_on_reason_then_escalates():
    """Proof is requested once (on a reason); a bare 'no' after that escalates."""
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")
    for _ in range(4):
        agent._advance_negotiation(s, eng, "no")
    assert s["state"] == "hardship"
    # a reason → ask for proof once
    inst = agent._advance_negotiation(s, eng, "my business has no sales")
    assert s["state"] == "hardship"
    assert s["upload_requested"] is True
    assert s["pending_upload"] is not None
    assert s["pending_upload"]["situation"] == "CANNOT_PAY"
    # bare rejection after proof was requested → escalate (no more pushing)
    inst = agent._advance_negotiation(s, eng, "no")
    assert s["state"] == "escalated"


def test_hardship_verified_then_rejection_escalates():
    """Once hardship is verified, a further rejection does escalate."""
    s, eng = _real_session()
    s["hardship_verified"] = True
    agent._advance_negotiation(s, eng, "slow month")
    agent._advance_negotiation(s, eng, "no")   # step 2
    agent._advance_negotiation(s, eng, "no")   # step 3
    agent._advance_negotiation(s, eng, "no")   # step 4
    agent._advance_negotiation(s, eng, "no")   # step 5 -> escalate
    assert s["state"] == "escalated"
    assert s["status"] == "escalated"


def test_verified_hardship_reopens_negotiation_at_lower_floor():
    """handle_document_verdict (CANNOT_PAY accepted) lowers the floor + reopens."""
    s, eng = _real_session()
    # drive to the hardship state
    agent._advance_negotiation(s, eng, "slow month")
    for _ in range(4):
        agent._advance_negotiation(s, eng, "no")
    assert s["state"] == "hardship"

    result = {
        "recommended_action": "ACCEPT_CLAIM",
        "verdict": "VALID",
        "confidence": 0.9,
        "debtor_friendly_response": "Thanks for sharing that document.",
    }
    reply, s = agent.handle_document_verdict(s, "CANNOT_PAY", result)
    assert s["hardship_verified"] is True
    assert eng.hardship_verified is True
    assert s["state"] == "negotiating"
    assert s["negotiation_step"] == 3
    assert "20%" in reply


def test_date_beyond_deadline_rejected():
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")
    agent._advance_negotiation(s, eng, "40000")    # -> counter
    agent._advance_negotiation(s, eng, "no")       # hold firm -> accept 40000
    far = (date.today() + timedelta(days=40)).isoformat()
    inst = agent._advance_negotiation(s, eng, f"how about {far}")
    assert s["state"] == "collecting_dates"
    assert s["future_dates"] == []            # not accepted
    assert "34-day limit" in inst


def test_full_flow_reaches_payment_pending(monkeypatch):
    monkeypatch.setattr(
        "backend.razorpay_client.create_order",
        lambda **k: {"id": "order_x", "amount": int(k["amount_inr"] * 100), "currency": "INR"},
    )
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")          # -> negotiating
    agent._advance_negotiation(s, eng, "40000")               # -> counter
    agent._advance_negotiation(s, eng, "no")                  # hold firm -> collecting_dates
    future = (date.today() + timedelta(days=5)).isoformat()
    agent._advance_negotiation(s, eng, f"how about {future}")  # -> plan_ready
    assert s["state"] == "plan_ready"
    assert s["plan_shown"] is True
    assert s["installment_plan"] is not None
    agent._advance_negotiation(s, eng, "yes, confirm")         # -> finalizing (forced finalize)
    assert s["state"] == "finalizing"
    assert s["finalize_requested"] is True
    assert s["payment_order"] is None                          # order moved to _finalize_agreement
    agent._finalize_agreement(s, eng)                          # terminal tool
    assert s["state"] == "payment_pending"
    assert s["payment_order"] is not None
    assert s["status"] == "awaiting_payment"
    assert s["payment_amount"] == 40000


def test_full_payment_sends_link_directly(monkeypatch):
    """Paying the whole invoice skips dates + confirmation and sends the link."""
    s, eng = _real_session()
    monkeypatch.setattr(
        "backend.razorpay_client.create_order",
        lambda **k: {"id": "order_x", "amount": int(k["amount_inr"] * 100), "currency": "INR"},
    )
    agent._advance_negotiation(s, eng, "I will pay 145000")
    assert s["state"] == "payment_pending"
    assert s["debtor_agreed_amount"] == 145000
    assert s["future_dates"] == []
    assert s["payment_order"] is not None
    assert s["status"] == "awaiting_payment"
    assert s["payment_amount"] == 145000


def test_pay_in_full_phrase_sends_link_directly(monkeypatch):
    """Full-payment phrases (no number) settle and send the link directly."""
    for phrase in ("I'll pay in full", "pay full", "I'll pay the full amount"):
        s, eng = _real_session()
        monkeypatch.setattr(
            "backend.razorpay_client.create_order",
            lambda **k: {"id": "order_x", "amount": int(k["amount_inr"] * 100), "currency": "INR"},
        )
        agent._advance_negotiation(s, eng, phrase)
        assert s["state"] == "payment_pending", phrase
        assert s["payment_order"] is not None, phrase
        assert s["payment_amount"] == s["invoice_amount"], phrase


def test_cannot_pay_full_is_not_full_payment(monkeypatch):
    """Negation is respected: 'can't pay the full amount' is a hardship signal."""
    s, eng = _real_session()
    inst = agent._advance_negotiation(s, eng, "I can't pay the full amount")
    assert s["state"] != "payment_pending"
    assert s["payment_order"] is None
    assert s["state"] == "negotiating"


# ---------------------------------------------------------------------------
# Tool handlers — payment link gating + dispute/upload
# ---------------------------------------------------------------------------

def test_generate_payment_link_sets_order(monkeypatch):
    monkeypatch.setattr(
        "backend.razorpay_client.create_order",
        lambda **k: {"id": "order_x", "amount": 4000000, "currency": "INR"},
    )
    s = _make_session(145000)
    s["plan_shown"] = True
    s["debtor_agreed_amount"] = 40000
    res = agent._handle_generate_payment_link({"amount": 40000}, s)
    assert res["order_id"] == "order_x"
    assert s["payment_order"] is not None
    assert s["status"] == "awaiting_payment"


def test_generate_payment_link_requires_plan_shown(monkeypatch):
    monkeypatch.setattr(
        "backend.razorpay_client.create_order",
        lambda **k: {"id": "order_x", "amount": 2900000, "currency": "INR"},
    )
    s = _make_session(145000)
    res = agent._handle_generate_payment_link({"amount": 29000}, s)
    assert "error" in res
    assert s.get("payment_order") is None


def test_generate_payment_link_never_exceeds_agreed_amount(monkeypatch):
    monkeypatch.setattr(
        "backend.razorpay_client.create_order",
        lambda **k: {"id": "order_agreed", "amount": 4000000, "currency": "INR"},
    )
    s = _make_session(145000)
    s["plan_shown"] = True
    s["debtor_agreed_amount"] = 40000
    res = agent._handle_generate_payment_link({"amount": 43200}, s)
    assert s["payment_amount"] == 40000
    assert res["amount"] == 40000
    assert res["order_id"] == "order_agreed"


def test_flag_dispute():
    s = _make_session(145000)
    agent._handle_flag_dispute({"reason": "wrong amount"}, s)
    assert s["status"] == "disputed"
    assert s["identified_situation"] == "DISPUTE"


def test_request_document_upload_derives_situation():
    s = _make_session(145000)
    agent._handle_request_document_upload({"document_type": "payment receipt"}, s)
    assert s["pending_upload"]["situation"] == "ALREADY_PAID"
    agent._handle_request_document_upload({"document_type": "bank statement"}, s)
    assert s["pending_upload"]["situation"] == "CANNOT_PAY"


# ---------------------------------------------------------------------------
# process_turn — LLM is only told what to say
# ---------------------------------------------------------------------------

def _assistant_msg(content=""):
    return SimpleNamespace(content=content, tool_calls=None)


def _resp(msg):
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


def _start_session(monkeypatch, invoice_id="INV-0001"):
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_key")
    monkeypatch.setattr(
        "backend.razorpay_client.create_order",
        lambda amount_inr, invoice_id, session_id, debtor_name: {
            "id": f"order_{invoice_id}", "amount": int(amount_inr * 100), "currency": "INR",
        },
    )
    s = agent.create_session(invoice_id)
    agent.open_turn(s)
    return s


def test_process_turn_speaks_only(monkeypatch):
    """The model gets an instruction and just returns text — no tools."""
    s = _start_session(monkeypatch)
    monkeypatch.setattr(agent, "_get_client", lambda: _FakeClient([
        _resp(_assistant_msg("Could you do ₹72,500 today?")),
    ]))
    reply, s = agent.process_turn(s, "I can't pay the full amount")
    assert reply == "Could you do ₹72,500 today?"
    assert s["state"] == "negotiating"
    assert s["negotiation_step"] == 1


def test_process_turn_dispute_flags_without_tools(monkeypatch):
    s = _start_session(monkeypatch)
    monkeypatch.setattr(agent, "_get_client", lambda: _FakeClient([
        _resp(_assistant_msg("I've flagged that for review.")),
    ]))
    reply, s = agent.process_turn(s, "this amount is wrong")
    assert s["status"] == "disputed"
    assert s["identified_situation"] == "DISPUTE"
    assert reply == "I've flagged that for review."


def test_process_turn_already_paid_requests_receipt(monkeypatch):
    s = _start_session(monkeypatch)
    monkeypatch.setattr(agent, "_get_client", lambda: _FakeClient([
        _resp(_assistant_msg("Could you share your payment receipt so I can verify?")),
    ]))
    reply, s = agent.process_turn(s, "I already paid this invoice")
    assert s["pending_upload"] is not None
    assert s["pending_upload"]["situation"] == "ALREADY_PAID"
    assert reply == "Could you share your payment receipt so I can verify?"


def test_process_turn_no_key_fallback(monkeypatch):
    s = _start_session(monkeypatch)
    monkeypatch.setattr(agent, "_get_client", lambda: (_ for _ in ()).throw(EnvironmentError("no key")))
    reply, s = agent.process_turn(s, "hello")
    assert s["status"] == "active"
    assert reply


def test_call_llm_sends_full_history_as_message_turns(monkeypatch):
    """The LLM receives the whole conversation as real user/assistant turns so
    it remembers prior messages (the core anti-forgetting behaviour)."""
    s = _start_session(monkeypatch)
    s["messages"].append({"role": "user", "content": "I can pay 40000 today"})
    s["messages"].append({"role": "assistant", "content": "Could you do ₹60,000?"})

    captured = {}

    class _CapturingCompletions:
        def create(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return _resp(_assistant_msg("That works, thanks."))

    class _CapturingClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=_CapturingCompletions())

    monkeypatch.setattr(agent, "_get_client", lambda: _CapturingClient())
    agent.process_turn(s, "I can do 50000")

    msgs = captured["messages"]
    assert msgs[0]["role"] == "system"
    body = msgs[1:]
    # Prior exchange AND the current turn are all present, in order.
    assert {"role": "user", "content": "I can pay 40000 today"} in body
    assert {"role": "assistant", "content": "Could you do ₹60,000?"} in body
    assert {"role": "user", "content": "I can do 50000"} in body


def test_legal_threat_sets_legal_hold(monkeypatch):
    s = _start_session(monkeypatch)
    _, s = agent.process_turn(s, "I'm taking this to my lawyer")
    assert s["status"] == "legal_hold"


def test_human_request_escalates(monkeypatch):
    s = _start_session(monkeypatch)
    _, s = agent.process_turn(s, "can I talk to a real person")
    assert s["status"] == "escalated"


def test_opening_message_voice_rules():
    s = agent.create_session("INV-0001")
    opening, _ = agent.open_turn(s)
    assert "Aria" in opening
    assert "regarding" not in opening.lower()
    assert "kindly" not in opening.lower()


# ---------------------------------------------------------------------------
# Intent extraction (LLM returns JSON, Python computes the balance)
# ---------------------------------------------------------------------------

def test_normalize_intent_date_parses_variants():
    year = date.today().year
    assert agent._normalize_intent_date("Sept 1") == f"{year}-09-01"
    assert agent._normalize_intent_date("September 1st") == f"{year}-09-01"
    assert agent._normalize_intent_date("1 Sep") == f"{year}-09-01"
    assert agent._normalize_intent_date("2026-09-01") == "2026-09-01"
    assert agent._normalize_intent_date("Sept 1 2026") == "2026-09-01"
    assert agent._normalize_intent_date("nonsense") is None
    assert agent._normalize_intent_date(None) is None


def test_extract_intent_regex_returns_schema():
    partial = agent._extract_intent_regex("I can pay 40000 today", 145000)
    assert partial == {"intent": "partial_payment", "upfront_amount": 40000, "date": None}

    full = agent._extract_intent_regex("I can pay 145000 today", 145000)
    assert full["intent"] == "full_payment"

    dispute = agent._extract_intent_regex("this amount is wrong", 145000)
    assert dispute["intent"] == "dispute"
    assert dispute["upfront_amount"] is None


def test_extract_intent_uses_llm_json(monkeypatch):
    """A well-formed LLM JSON reply is authoritative over regex extraction."""
    s, _ = _real_session()
    payload = json.dumps({"intent": "partial_payment",
                          "upfront_amount": 100000, "date": "Sept 1"})
    monkeypatch.setattr(agent, "_get_client", lambda: _FakeClient([
        _resp(_assistant_msg(payload)),
    ]))
    intent = agent.extract_intent(s, "I can pay 1 lakh on Sept 1")
    assert intent["intent"] == "partial_payment"
    assert intent["upfront_amount"] == 100000
    assert intent["date"] == f"{date.today().year}-09-01"


def test_extract_intent_falls_back_to_regex_on_non_json(monkeypatch):
    """A non-JSON reply (e.g. a test double) falls back to regex extraction."""
    s, _ = _real_session()
    monkeypatch.setattr(agent, "_get_client", lambda: _FakeClient([
        _resp(_assistant_msg("Sure, here is your plan.")),
    ]))
    intent = agent.extract_intent(s, "I can pay 40000 today")
    assert intent["upfront_amount"] == 40000
    assert intent["intent"] == "partial_payment"


def test_build_context_includes_remaining_balance():
    s, eng = _real_session()   # INV-0001 → ₹1,45,000
    s["last_debtor_offer"] = 40000
    ctx = agent._build_context(s, eng, "ask for a date")
    assert ctx["numbers"]["current_remaining_balance"] == 105000


def test_system_prompt_reads_balance_never_calculates():
    s, _ = _real_session()
    s["last_debtor_offer"] = 40000
    p = _build_prompt(s)
    assert "Current remaining balance" in p
    assert "₹1,05,000" in p
    assert "never recalculate" in p
    assert "Never do arithmetic" in p


def test_system_prompt_omits_balance_before_any_offer():
    """No remaining balance is shown before the debtor has committed an amount."""
    s, _ = _real_session()
    p = _build_prompt(s)
    assert "Current remaining balance" not in p


# ---------------------------------------------------------------------------
# Forced finalization via the finalize_agreement tool call
# ---------------------------------------------------------------------------

def _drive_to_plan_ready(s, eng, amount=40000):
    """Advance a fresh session to plan_ready with `amount` agreed upfront."""
    agent._advance_negotiation(s, eng, "slow month")     # -> negotiating
    agent._advance_negotiation(s, eng, str(amount))      # -> counter
    agent._advance_negotiation(s, eng, "no")             # hold firm -> collecting_dates
    future = (date.today() + timedelta(days=5)).isoformat()
    agent._advance_negotiation(s, eng, f"how about {future}")  # -> plan_ready
    assert s["state"] == "plan_ready"
    return future


def test_confirm_requests_finalize_not_conversation():
    """'ya' in plan_ready ends the conversational phase — no order, no LLM text."""
    s, eng = _real_session()
    _drive_to_plan_ready(s, eng)
    inst = agent._advance_negotiation(s, eng, "ya")
    assert inst == "FINALIZE_AGREEMENT"
    assert s["finalize_requested"] is True
    assert s["state"] == "finalizing"
    assert s["payment_order"] is None          # order moved to _finalize_agreement
    assert s["negotiation_complete"] is False


def test_finalize_agreement_injects_deterministic_message(monkeypatch):
    monkeypatch.setattr(
        "backend.razorpay_client.create_order",
        lambda **k: {"id": "order_x", "amount": int(k["amount_inr"] * 100), "currency": "INR"},
    )
    s, eng = _real_session()
    _drive_to_plan_ready(s, eng)
    msg = agent._finalize_agreement(s, eng)
    assert "payment link" in msg
    assert "₹40,000" in msg
    assert "valid for 24 hours" in msg
    assert s["state"] == "payment_pending"
    assert s["payment_order"] is not None
    assert s["payment_amount"] == 40000
    events = [e["event"] for e in s["audit_log"]]
    assert "finalize_agreement" in events


class _ToolCallCompletions:
    def create(self, **kwargs):
        tc = SimpleNamespace(
            id="call_1", type="function",
            function=SimpleNamespace(
                name="finalize_agreement",
                arguments=json.dumps({
                    "upfront_amount": 40000,
                    "deferred_amount": 105000,
                    "deferred_date": "2026-09-05",
                }),
            ),
        )
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=None, tool_calls=[tc]))])


class _ToolCallClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_ToolCallCompletions())


def test_process_turn_finalize_via_tool_call(monkeypatch):
    """The LLM's forced finalize_agreement call is intercepted; the final message
    is deterministic (not LLM text) and the payment order is set."""
    monkeypatch.setattr(
        "backend.razorpay_client.create_order",
        lambda **k: {"id": "order_x", "amount": int(k["amount_inr"] * 100), "currency": "INR"},
    )
    s, eng = _real_session()
    _drive_to_plan_ready(s, eng)
    monkeypatch.setattr(agent, "_get_client", lambda: _ToolCallClient())

    reply, s = agent.process_turn(s, "ya")

    assert reply.startswith("Thanks")
    assert "payment link" in reply
    assert "valid for 24 hours" in reply
    assert s["state"] == "payment_pending"
    assert s["payment_order"] is not None
    assert s["payment_amount"] == 40000
    events = [e["event"] for e in s["audit_log"]]
    assert "finalize_tool_called" in events
    assert "finalize_agreement" in events


def test_process_turn_finalize_falls_back_without_tool_call(monkeypatch):
    """No tool call from the model → finalize deterministically in Python anyway."""
    monkeypatch.setattr(
        "backend.razorpay_client.create_order",
        lambda **k: {"id": "order_x", "amount": int(k["amount_inr"] * 100), "currency": "INR"},
    )
    s, eng = _real_session()
    _drive_to_plan_ready(s, eng)
    # A plain-text reply (no tool_calls) exercises the fallback path.
    monkeypatch.setattr(agent, "_get_client", lambda: _FakeClient([
        _resp(_assistant_msg("Here is your link.")),
    ]))

    reply, s = agent.process_turn(s, "ya")

    assert "payment link" in reply
    assert s["state"] == "payment_pending"
    assert s["payment_order"] is not None


# ---------------------------------------------------------------------------
# Structured JSON output + reason MCQ
# ---------------------------------------------------------------------------

def test_parse_agent_json():
    p = agent._parse_agent_json(
        '{"thought_process":"counter","action_type":"negotiate","reply_to_user":"Hi"}'
    )
    assert p["reply_to_user"] == "Hi"
    assert p["action_type"] == "negotiate"

    fenced = agent._parse_agent_json('```json\n{"reply_to_user":"Hey"}\n```')
    assert fenced["reply_to_user"] == "Hey"

    plain = agent._parse_agent_json("Could you do ₹72,500 today?")
    assert plain["reply_to_user"] == "Could you do ₹72,500 today?"
    assert plain["action_type"] == "negotiate"

    assert agent._parse_agent_json("")["reply_to_user"] == ""


def test_reason_mcq_pending_on_rejection():
    """Rejecting every amount (the ladder to step 5) flags the reason MCQ."""
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")
    for _ in range(4):
        agent._advance_negotiation(s, eng, "no")
    assert s["state"] == "hardship"
    assert s["reason_mcq_pending"] is True


def test_reason_mcq_answer_lowers_floor_and_reopens(monkeypatch):
    monkeypatch.setattr(agent, "_get_client", lambda: _FakeClient([
        _resp(_assistant_msg('{"reply_to_user":"We can come down to ₹29,000 today."}')),
    ]))
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")
    for _ in range(4):
        agent._advance_negotiation(s, eng, "no")
    assert s["state"] == "hardship"

    reply, s = agent._handle_reason_mcq_answer(s, eng, "cashflow")

    assert s["state"] == "negotiating"
    assert s["negotiation_step"] == 3
    assert s["hardship_verified"] is True
    assert s["rejection_reason"] == "Cash flow issues"
    assert eng.min_today == eng.hardship_min   # floor applied
    assert reply == "We can come down to ₹29,000 today."
    events = [e["event"] for e in s["audit_log"]]
    assert "reason_mcq_answered" in events


def test_process_turn_surfaces_trigger_reason_mcq(monkeypatch):
    monkeypatch.setattr(agent, "_get_client", lambda: _FakeClient([
        _resp(_assistant_msg(
            '{"thought_process":"ask why","action_type":"negotiate",'
            '"reply_to_user":"What is making it hard to pay?"}'
        )),
    ]))
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")   # -> negotiating

    reply, s = agent.process_turn(s, "no cash now at all")

    assert s["action_type"] == "trigger_reason_mcq"
    assert s["mcq_options"] == agent.MCQ_REASONS
    assert reply == "What is making it hard to pay?"


def test_reason_mcq_asked_only_once_then_final_ultimatum():
    """The reason MCQ is collected once; a second rejection is terminal."""
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")
    for _ in range(4):
        agent._advance_negotiation(s, eng, "no")
    assert s["state"] == "hardship"
    assert s["reason_collected"] is True

    # Simulate the debtor answering the MCQ → reopen at the hardship floor.
    eng.apply_hardship()
    s["hardship_verified"] = True
    s["state"] = "negotiating"
    s["negotiation_step"] = 3

    # Reject the reduced amount → step 4, then step 5 → final ultimatum (not MCQ).
    agent._advance_negotiation(s, eng, "no")
    assert s["negotiation_step"] == 4
    inst = agent._advance_negotiation(s, eng, "no")
    assert inst == "FINAL_ULTIMATUM"
    assert s["final_ultimatum_requested"] is True
    assert s["state"] == "escalated"
    assert s["status"] == "escalated"


def test_final_ultimatum_message():
    s, eng = _real_session()
    msg = agent._final_ultimatum_message(s, eng)
    assert "absolute minimum" in msg
    assert "₹" in msg


def test_process_turn_final_ultimatum(monkeypatch):
    monkeypatch.setattr(agent, "_get_client", lambda: _FakeClient([
        _resp(_assistant_msg('{"reply_to_user":"anything"}')),
    ]))
    s, eng = _real_session()
    agent._advance_negotiation(s, eng, "slow month")
    for _ in range(4):
        agent._advance_negotiation(s, eng, "no")
    assert s["reason_collected"] is True
    eng.apply_hardship()
    s["hardship_verified"] = True
    s["state"] = "negotiating"
    s["negotiation_step"] = 3

    agent.process_turn(s, "no")                    # step 4 (normal negotiate)
    reply, s = agent.process_turn(s, "no")         # step 5 → final ultimatum
    assert s["action_type"] == "final_ultimatum"
    assert "absolute minimum" in reply
    assert s["status"] == "escalated"
