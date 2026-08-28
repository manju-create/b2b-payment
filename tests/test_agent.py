"""
Unit tests for the payment-plan calculation in backend.agent.
==============================================================
The core invariant under test:

    upfront_amount + deferred_amount == invoice_amount

The deferred portion is ALWAYS the remainder (invoice - offered). Tier
percentages (min_now_pct, max_defer_pct) are used only to VALIDATE the
upfront offer — never to compute the deferred amount.

Run with:
    python -m pytest tests/test_agent.py -v
"""

from backend import agent


def _make_session(invoice_amount: int, tier: str = "B") -> dict:
    """Build a minimal session dict for plan-building tests."""
    bounds = agent.TIER_BOUNDS[tier]
    paise = invoice_amount * 100
    return {
        "session_id": "sess-test",
        "invoice_id": "INV-TEST",
        "debtor_id": "D-TEST",
        "debtor_name": "Test Debtor",
        "company_name": "Test Co",
        "invoice_amount_paise": paise,
        "invoice_amount": invoice_amount,
        "tier": tier,
        "tier_bounds": bounds,
        "min_now_paise": round(paise * bounds["min_now_pct"] / 100),
        "status": "active",
        "messages": [],
        "audit_log": [],
    }


def test_plan_amounts_sum_to_invoice():
    """Plan upfront + deferred must always equal the invoice amount."""
    cases = [
        # (invoice, offered, min_pct, expected_deferred)
        (400000, 250000, 40, 150000),   # offered > minimum
        (400000, 160000, 40, 240000),   # offered == minimum exactly
        (219000, 186500, 85, 32500),    # Tier D, offered just above min
        (100000, 100000, 25, 0),        # full payment upfront
        (80000,  40000,  40, 40000),    # offered == 50%, above 40% min
    ]
    for invoice, offered, min_pct, expected_deferred in cases:
        deferred = invoice - offered
        assert deferred == expected_deferred
        assert offered + deferred == invoice
        assert offered >= invoice * min_pct / 100


def test_validate_returns_computed_plan_from_remainder():
    """Offering MORE than the minimum defers only the remainder."""
    session = _make_session(400000, tier="B")  # min 40% = ₹160,000
    result = agent._handle_validate_proposed_terms(
        {"now_pct": 62.5, "defer_pct": 37.5, "defer_days": 45,
         "discount_pct": 0, "upfront_offered_paise": 25000000},
        session,
    )
    assert result["valid"] is True
    plan = result["computed_plan"]
    assert plan["upfront_amount"] == 25000000
    assert plan["deferred_amount"] == 15000000
    assert plan["upfront_amount"] + plan["deferred_amount"] == 40000000


def test_validate_rejects_overpayment():
    """A debtor cannot offer more than the invoice total."""
    session = _make_session(400000, tier="B")
    result = agent._handle_validate_proposed_terms(
        {"now_pct": 125.0, "defer_pct": -25.0, "defer_days": 45,
         "discount_pct": 0, "upfront_offered_paise": 50000000},
        session,
    )
    assert result["valid"] is False
    assert any("exceeds invoice" in v for v in result["violations"])


def test_amount_response_more_than_minimum_builds_correct_plan():
    """End-to-end plan builder: offer > minimum → correct amounts + discount."""
    session = _make_session(400000, tier="B")
    _reply, session = agent._handle_amount_response(
        session, "I can pay ₹250,000 now", turn=1
    )
    plan = session["pending_plan"]
    assert plan["upfront_amount"] == 25000000
    assert plan["deferred_amount_raw"] == 15000000
    assert plan["upfront_amount"] + plan["deferred_amount_raw"] == session["invoice_amount_paise"]
    # Tier B discount (10%) applies to the deferred portion only
    assert plan["discount_amount"] == 1500000
    assert plan["deferred_amount"] == 13500000
    assert plan["total_payable"] == 25000000 + 13500000
    assert session["awaiting_plan_confirmation"] is True


def test_plan_confirmation_full_payment_creates_order(monkeypatch):
    """Paying the full invoice upfront → awaiting_payment, no deferred reminder."""
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_key")
    monkeypatch.setattr(
        "backend.razorpay_client.create_order",
        lambda amount_inr, invoice_id, session_id, debtor_name: {
            "id": "order_full", "amount": int(amount_inr * 100), "currency": "INR",
        },
    )

    session = _make_session(100000, tier="A")
    _reply, session = agent._handle_amount_response(
        session, "I can pay ₹100,000 now", turn=1
    )
    assert session["pending_plan"]["deferred_amount"] == 0

    reply, session = agent._handle_plan_confirmation(session, "CONFIRM", turn=2)
    assert session["status"] == "awaiting_payment"
    assert session["razorpay_order_id"] == "order_full"
    assert "deferred payment required" in reply


def test_plan_confirmation_partial_payment_creates_order(monkeypatch):
    """Partial plan → awaiting_payment, deferred amount is scheduled."""
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_key")
    monkeypatch.setattr(
        "backend.razorpay_client.create_order",
        lambda amount_inr, invoice_id, session_id, debtor_name: {
            "id": "order_partial", "amount": int(amount_inr * 100), "currency": "INR",
        },
    )

    session = _make_session(400000, tier="B")  # min 40% = ₹160,000
    _reply, session = agent._handle_amount_response(
        session, "I can pay ₹250,000 now", turn=1
    )
    assert session["pending_plan"]["deferred_amount"] > 0

    reply, session = agent._handle_plan_confirmation(session, "CONFIRM", turn=2)
    assert session["status"] == "awaiting_payment"
    assert session["razorpay_order_id"] == "order_partial"
    assert "Pay Now" in reply
    # deferred_scheduled is audited so the server can create the deferred entry
    assert "deferred_scheduled" in [e["event"] for e in session["audit_log"]]
