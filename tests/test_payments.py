import hashlib
import hmac
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend import agent
from backend import server


def _signed_headers(secret: str, body: bytes) -> dict[str, str]:
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return {
        "Content-Type": "application/json",
        "X-Razorpay-Signature": signature,
    }


def setup_function():
    server.sessions.clear()
    server.batch_results.clear()
    server.deferred_schedule.clear()
    server.processed_webhooks.clear()
    server.webhook_log.clear()


def test_generate_payment_link_creates_order(monkeypatch):
    """The agent tool now creates a Razorpay Order, not a Payment Link."""
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_key")
    monkeypatch.setattr(
        "backend.razorpay_client.create_order",
        lambda amount_inr, invoice_id, session_id, debtor_name: {
            "id": "order_test123",
            "amount": int(amount_inr * 100),
            "currency": "INR",
        },
    )

    session = {
        "invoice_id": "INV-TEST",
        "session_id": "sess-test",
        "debtor_name": "Test Debtor",
        "invoice_amount_paise": 40000,
        "audit_log": [],
        "status": "active",
        "plan_shown": True,
        "debtor_agreed_amount": 400,
    }

    result = agent._handle_generate_payment_link(
        {"amount": 400, "invoice_id": "INV-TEST"}, session
    )

    assert result["order_id"] == "order_test123"
    assert result["amount"] == 400
    assert result["key_id"] == "rzp_test_key"
    assert result["invoice_id"] == "INV-TEST"
    assert result["debtor_name"] == "Test Debtor"
    assert session["razorpay_order_id"] == "order_test123"
    assert session["status"] == "awaiting_payment"
    assert session["payment_order"]["order_id"] == "order_test123"


def test_pay_full_endpoint_creates_order_for_full_amount(monkeypatch):
    """The pay-full endpoint creates an order for the whole invoice amount."""
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_key")
    monkeypatch.setattr(
        "backend.razorpay_client.create_order",
        lambda amount_inr, invoice_id, session_id, debtor_name: {
            "id": "order_full123",
            "amount": int(amount_inr * 100),
            "currency": "INR",
        },
    )

    session = {
        "session_id": "sess-full",
        "invoice_id": "INV-FULL",
        "debtor_id": "DEBTOR-FULL",
        "debtor_name": "Full Debtor",
        "invoice_amount": 400.0,
        "invoice_amount_paise": 40000,
        "audit_log": [],
        "status": "active",
        "plan_shown": False,
        "debtor_agreed_amount": None,
        "last_debtor_offer": None,
        "agreed_terms": None,
        "razorpay_order_id": None,
        "payment_order": None,
        "payment_amount": None,
        "recovered_paise": 0,
    }
    server.sessions["sess-full"] = session
    server.batch_results["INV-FULL"] = session

    client = TestClient(server.app)
    resp = client.post("/api/negotiate/pay-full/sess-full")

    assert resp.status_code == 200
    data = resp.json()
    assert data["order_id"] == "order_full123"
    assert data["amount"] == 400.0
    assert data["key_id"] == "rzp_test_key"
    assert data["invoice_id"] == "INV-FULL"
    assert data["debtor_name"] == "Full Debtor"

    stored = server.batch_results["INV-FULL"]
    assert stored["status"] == "awaiting_payment"
    assert stored["razorpay_order_id"] == "order_full123"
    assert stored["payment_amount"] == 400.0
    assert stored["agreed_terms"]["deferred_amount"] == 0


def test_payment_captured_webhook_maps_invoice_id_from_notes(monkeypatch):
    """payment.captured maps back to MongoDB via notes.invoice_id (no in-memory lookup)."""
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "whsec_test")

    calls = []

    def fake_apply_payment(invoice_id, payment_id, amount_paise):
        calls.append((invoice_id, payment_id, amount_paise))
        return {"action": "payment_applied", "invoice_id": invoice_id, "new_status": "settled"}

    monkeypatch.setattr(server, "apply_payment", fake_apply_payment)

    payload = {
        "event": "payment.captured",
        "payload": {
            "payment": {"entity": {
                "id": "pay_123",
                "order_id": "order_1001",
                "amount": 40000,
                "notes": {"invoice_id": "INV-1001"},
            }},
        },
    }
    body = json.dumps(payload).encode()
    client = TestClient(server.app)

    response = client.post(
        "/webhooks/razorpay",
        data=body,
        headers=_signed_headers("whsec_test", body),
    )

    assert response.status_code == 200
    assert response.json()["action"] == "payment_applied"
    assert calls == [("INV-1001", "pay_123", 40000)]

    # Duplicate delivery is deduped by processed_webhooks — no second call.
    duplicate = client.post(
        "/webhooks/razorpay",
        data=body,
        headers=_signed_headers("whsec_test", body),
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["status"] == "already_processed"
    assert len(calls) == 1


def test_payment_captured_webhook_acks_when_invoice_id_missing(monkeypatch):
    """Without notes.invoice_id the webhook acks 200 and applies nothing."""
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "whsec_test")
    monkeypatch.setattr(server, "apply_payment", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")))

    payload = {
        "event": "payment.captured",
        "payload": {
            "payment": {"entity": {"id": "pay_noid", "order_id": "order_x", "amount": 40000}},
        },
    }
    body = json.dumps(payload).encode()
    client = TestClient(server.app)

    response = client.post(
        "/webhooks/razorpay",
        data=body,
        headers=_signed_headers("whsec_test", body),
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_webhook_rejects_invalid_signature(monkeypatch):
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "whsec_test")
    body = b'{"event":"payment.captured","payload":{"payment":{"entity":{"id":"pay_bad","amount":40000}}}}'
    client = TestClient(server.app)

    response = client.post(
        "/webhooks/razorpay",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Razorpay-Signature": "bad-signature",
        },
    )

    assert response.status_code == 400
    assert server.webhook_log[-1]["action"] == "signature_rejected"


def test_simulated_webhook_uses_same_payment_application():
    server.batch_results["INV-2002"] = {
        "invoice_id": "INV-2002",
        "status": "active",
        "recovered_paise": 0,
        "audit_log": [],
    }
    client = TestClient(server.app)

    response = client.post(
        "/api/simulate-webhook/INV-2002",
        json={"amount": 80000},
    )

    assert response.status_code == 200
    assert response.json()["action"] == "payment_applied"
    assert server.batch_results["INV-2002"]["status"] == "settled"
    assert server.batch_results["INV-2002"]["recovered_paise"] == 80000
    assert server.batch_results["INV-2002"]["audit_log"][-1]["event"] == "simulated_webhook"


def test_reason_mcq_endpoint_lowers_floor(monkeypatch):
    """A reason-MCQ button click lowers the floor and returns a concession."""
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_key")
    monkeypatch.setattr(
        "backend.razorpay_client.create_order",
        lambda **k: {"id": "order_x", "amount": int(k["amount_inr"] * 100), "currency": "INR"},
    )

    s = agent.create_session("INV-0001")
    agent.open_turn(s)
    eng = agent._get_engine(s)
    agent._advance_negotiation(s, eng, "slow month")
    for _ in range(4):
        agent._advance_negotiation(s, eng, "no")
    assert s["state"] == "hardship"

    server.sessions[s["session_id"]] = s
    server.batch_results[s["invoice_id"]] = s

    class _Completions:
        def create(self, **kwargs):
            msg = SimpleNamespace(
                content='{"reply_to_user":"We can come down to ₹29,000 today."}',
                tool_calls=None,
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    class _Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr(agent, "_get_client", lambda: _Client())

    client = TestClient(server.app)
    resp = client.post(
        f"/api/reason-mcq/{s['session_id']}",
        json={"button_id": "cashflow"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["agent_reply"] == "We can come down to ₹29,000 today."
    assert data["action_type"] == "negotiate"

    stored = server.batch_results["INV-0001"]
    assert stored["rejection_reason"] == "Cash flow issues"
    assert stored["hardship_verified"] is True
