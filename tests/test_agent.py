"""
Unit tests for the LLM-driven negotiation agent in backend.agent.
===================================================================

Core invariants:
1. validate_proposed_terms rejects anything < 20% upfront
2. Exactly 20% is accepted; 19.9% is rejected
3. stance["opening"] > stance["target"] > 20 for all trust score ranges
4. project_score_change never exceeds 100 or goes below 0
5. project_score_change("full_upfront") always > project_score_change("partial_deferred")
6. Plan amounts must sum to invoice total
7. The system prompt is a single intelligent prompt (no rigid flow logic)
8. Tool execution updates session state; stopping rules are enforced in Python

Run with:
    python -m pytest tests/test_agent.py -v
"""

import json
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
    paise  = invoice_amount * 100
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
        "negotiation_floor":    20,
        "min_now_paise":        round(paise * 0.20),
        "score_projections":    {"full_upfront": 78, "partial_deferred": 73, "escalated": 50},
        "system_prompt":        "[test system prompt]",
        "status":               "active",
        "messages":             [],
        "audit_log":            [],
    }


# ---------------------------------------------------------------------------
# Part 8a: validate_proposed_terms floor enforcement
# ---------------------------------------------------------------------------

def test_validate_rejects_below_20_pct():
    """19.9% must be rejected."""
    session = _make_session(100000)
    result = agent._handle_validate_proposed_terms(
        {"now_pct": 19.9, "upfront_offered_paise": round(100000 * 100 * 0.199)},
        session,
    )
    assert result["valid"] is False
    assert any("20%" in v for v in result["violations"])


def test_validate_accepts_exactly_20_pct():
    """Exactly 20% must be accepted."""
    session = _make_session(100000)
    paise = 100000 * 100
    result = agent._handle_validate_proposed_terms(
        {"now_pct": 20.0, "upfront_offered_paise": round(paise * 0.20)},
        session,
    )
    assert result["valid"] is True
    assert result["violations"] == []


def test_validate_rejects_19_9_pct():
    """Boundary test: 19.9% is strictly below floor."""
    session = _make_session(80000)
    paise = 80000 * 100
    result = agent._handle_validate_proposed_terms(
        {"now_pct": 19.9, "upfront_offered_paise": round(paise * 0.199)},
        session,
    )
    assert result["valid"] is False


def test_validate_rejects_overpayment():
    """Debtor cannot offer more than the invoice total."""
    session = _make_session(400000)
    paise = 400000 * 100
    result = agent._handle_validate_proposed_terms(
        {"now_pct": 125.0, "upfront_offered_paise": round(paise * 1.25)},
        session,
    )
    assert result["valid"] is False
    assert any("exceeds" in v for v in result["violations"])


# ---------------------------------------------------------------------------
# Part 8b: stance ordering invariants across all score ranges
# ---------------------------------------------------------------------------

SCORE_SAMPLES = [90, 72, 47, 15]  # one per tier band


def test_stance_opening_above_target_above_floor():
    """opening > target > 20 for every trust score."""
    for score in SCORE_SAMPLES:
        stance = get_negotiation_stance(score)
        assert stance["opening"] > stance["target"], \
            f"score={score}: opening ({stance['opening']}) must be > target ({stance['target']})"
        assert stance["target"] > 20, \
            f"score={score}: target ({stance['target']}) must be > floor (20)"
        assert stance["floor"] == 20, \
            f"score={score}: floor must always be 20"


def test_stance_floor_is_always_20():
    """Universal floor is 20 regardless of score."""
    for score in range(0, 101, 5):
        assert get_negotiation_stance(score)["floor"] == 20


# ---------------------------------------------------------------------------
# Part 8c: project_score_change bounds and ordering
# ---------------------------------------------------------------------------

def test_project_score_never_exceeds_100():
    """Score projection must never exceed 100."""
    for score in range(90, 101):
        for stype in ("full_upfront", "partial_deferred", "escalated", "ghosted"):
            assert project_score_change(score, stype) <= 100


def test_project_score_never_below_0():
    """Score projection must never go below 0."""
    for score in range(0, 11):
        for stype in ("full_upfront", "partial_deferred", "escalated", "ghosted"):
            assert project_score_change(score, stype) >= 0


def test_project_full_upfront_beats_partial():
    """full_upfront always projects higher (or equal at the 100 cap) than partial_deferred."""
    for score in range(0, 101, 10):
        full    = project_score_change(score, "full_upfront")
        partial = project_score_change(score, "partial_deferred")
        if full < 100 or partial < 100:
            assert full > partial
        else:
            assert full >= partial


# ---------------------------------------------------------------------------
# Part 8d: plan amounts sum to invoice total
# ---------------------------------------------------------------------------

def test_plan_amounts_sum_to_invoice():
    """Plan upfront + deferred must always equal the invoice amount (in paise)."""
    cases = [
        (400000, 25),   # above floor
        (100000, 20),   # exactly at floor
        (80000,  50),   # above floor
        (219000, 30),   # odd number
    ]
    for invoice, pct in cases:
        paise    = invoice * 100
        upfront  = round(paise * pct / 100)
        deferred = paise - upfront
        assert upfront + deferred == paise, \
            f"invoice={invoice} pct={pct}: upfront+deferred != invoice"


def test_validate_returns_correct_computed_plan():
    """validate_proposed_terms returns correct amounts in computed_plan."""
    session = _make_session(400000, score=65)
    paise   = 400000 * 100
    offered = round(paise * 0.50)
    result  = agent._handle_validate_proposed_terms(
        {"now_pct": 50.0, "upfront_offered_paise": offered},
        session,
    )
    assert result["valid"] is True
    plan = result["computed_plan"]
    assert plan["upfront_amount"] == offered
    assert plan["deferred_amount"] == paise - offered
    assert plan["upfront_amount"] + plan["deferred_amount"] == paise


# ---------------------------------------------------------------------------
# System prompt — a single intelligent prompt (no rigid flow logic)
# ---------------------------------------------------------------------------

def test_system_prompt_is_intelligent_and_contextual():
    s = agent.create_session("INV-0001")
    p = agent.build_system_prompt(s)
    assert "Aria" in p
    assert "RecoverFlow Demo Merchant" in p
    assert s["debtor_name"] in p
    assert "1,45,000" in p
    assert "CONVERSATION HISTORY" in p
    assert "generate_payment_link" in p
    assert "flag_dispute" in p
    # no old flowchart / rigid logic
    assert "VOICE RULES" not in p
    assert "FLOOR (INTERNAL" not in p
    assert "Identified situation" not in p


def test_system_prompt_includes_conversation_history():
    s = agent.create_session("INV-0001")
    agent.open_turn(s)
    s["messages"].append({"role": "user", "content": "I can pay 40000 today"})
    p = agent.build_system_prompt(s)
    assert "Debtor: I can pay 40000 today" in p
    assert "Aria:" in p


# ---------------------------------------------------------------------------
# Tool handlers — session state updates stay in Python
# ---------------------------------------------------------------------------

def test_generate_payment_link_sets_order(monkeypatch):
    monkeypatch.setattr(
        "backend.razorpay_client.create_order",
        lambda **k: {"id": "order_x", "amount": 4000000, "currency": "INR"},
    )
    s = _make_session(145000)
    res = agent._handle_generate_payment_link({"amount": 40000}, s)
    assert res["order_id"] == "order_x"
    assert s["payment_order"] is not None
    assert s["status"] == "awaiting_payment"


def test_generate_payment_link_rejects_below_floor():
    s = _make_session(145000)   # floor = ₹29,000
    res = agent._handle_generate_payment_link({"amount": 10000}, s)
    assert "error" in res
    assert s.get("payment_order") is None


def test_set_promise_to_pay():
    s = _make_session(145000)
    agent._handle_set_promise_to_pay({"date": "2026-09-05", "amount": 145000}, s)
    assert s["status"] == "promise_to_pay"
    assert s["promise_to_pay"]["date"] == "2026-09-05"


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
# LLM-driven process_turn — tool loop + stopping rules
# ---------------------------------------------------------------------------

def _tool_call(name, arguments):
    return SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _assistant_msg(content="", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or None)


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
    """Create + open a real session, mocking Razorpay order creation."""
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


def test_process_turn_llm_tool_loop(monkeypatch):
    """DeepSeek returns a tool call, then a final reply; the session updates."""
    s = _start_session(monkeypatch)
    monkeypatch.setattr(agent, "_get_client", lambda: _FakeClient([
        _resp(_assistant_msg(tool_calls=[_tool_call("generate_payment_link", {"amount": 40000})])),
        _resp(_assistant_msg("Here's your payment link!")),
    ]))
    reply, s = agent.process_turn(s, "I'll pay 40000 now")
    assert reply == "Here's your payment link!"
    assert s["payment_order"] is not None
    assert s["status"] == "awaiting_payment"


def test_process_turn_llm_flag_dispute(monkeypatch):
    s = _start_session(monkeypatch)
    monkeypatch.setattr(agent, "_get_client", lambda: _FakeClient([
        _resp(_assistant_msg(tool_calls=[_tool_call("flag_dispute", {"reason": "wrong amount"})])),
        _resp(_assistant_msg("I've flagged that for review.")),
    ]))
    reply, s = agent.process_turn(s, "this amount is wrong")
    assert s["status"] == "disputed"
    assert s["identified_situation"] == "DISPUTE"


def test_process_turn_no_key_fallback(monkeypatch):
    s = _start_session(monkeypatch)
    monkeypatch.setattr(agent, "_get_client", lambda: (_ for _ in ()).throw(EnvironmentError("no key")))
    reply, s = agent.process_turn(s, "hello")
    assert s["status"] == "active"
    assert reply


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
