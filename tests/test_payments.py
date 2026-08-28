import hashlib
import hmac
import json

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


def test_payment_captured_webhook_updates_session_and_dedupes(monkeypatch):
    """payment.captured maps back to the session via order_id and dedupes."""
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "whsec_test")

    session = {
        "session_id": "sess-1001",
        "invoice_id": "INV-1001",
        "razorpay_order_id": "order_1001",
        "debtor_id": "DEBTOR-1001",
        "status": "awaiting_payment",
        "recovered_paise": 0,
        "audit_log": [],
    }
    server.sessions["sess-1001"] = session
    server.batch_results["INV-1001"] = session
    server.deferred_schedule["INV-1001"] = {"status": "pending"}

    payload = {
        "event": "payment.captured",
        "payload": {
            "payment": {"entity": {"id": "pay_123", "order_id": "order_1001", "amount": 40000}},
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
    assert server.batch_results["INV-1001"]["status"] == "partially_settled"
    assert server.batch_results["INV-1001"]["recovered_paise"] == 40000
    assert server.batch_results["INV-1001"]["payment_captured"] is True

    duplicate = client.post(
        "/webhooks/razorpay",
        data=body,
        headers=_signed_headers("whsec_test", body),
    )

    assert duplicate.status_code == 200
    assert duplicate.json()["status"] == "already_processed"


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
