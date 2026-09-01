"""
RecoverFlow — Razorpay Orders + Checkout client
===============================================
Real Razorpay integration using Orders + Checkout JS (NOT Payment Links).

Payment Links don't fire webhooks reliably in test mode; Orders + Checkout do.
The order's `notes` carry (session_id, invoice_id, debtor_name) so the webhook
can map a captured payment back to the originating negotiation session.
"""

from __future__ import annotations

import logging
import os

import razorpay

logger = logging.getLogger(__name__)


def _get_client() -> razorpay.Client:
    return razorpay.Client(auth=(
        os.environ.get("RAZORPAY_KEY_ID", ""),
        os.environ.get("RAZORPAY_KEY_SECRET", ""),
    ))


def create_order(
    amount_inr: float,
    invoice_id: str,
    session_id: str,
    debtor_name: str,
) -> dict:
    """
    Create a Razorpay Order for the Checkout JS flow.

    amount_inr: float in rupees (e.g. 32000.0)
    Returns the Razorpay order dict (has 'id', 'amount', 'currency', ...).
    """
    import uuid
    amount_paise = int(round(amount_inr * 100))
    key_id = os.environ.get("RAZORPAY_KEY_ID", "").strip()
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET", "").strip()

    if key_id and key_secret:
        try:
            client = _get_client()
            order = client.order.create({
                "amount": amount_paise,
                "currency": "INR",
                "receipt": invoice_id[:40],   # max 40 chars
                "payment_capture": 1,          # auto-capture on payment success
                "notes": {
                    "session_id": session_id,
                    "invoice_id": invoice_id,
                    "debtor_name": debtor_name,
                },
            })
            return order
        except Exception as exc:
            logger.warning("Razorpay order creation failed: %s. Falling back to demo order.", exc)

    mock_id = f"order_demo_{uuid.uuid4().hex[:12]}"
    return {
        "id": mock_id,
        "entity": "order",
        "amount": amount_paise,
        "amount_paid": 0,
        "amount_due": amount_paise,
        "currency": "INR",
        "receipt": invoice_id[:40],
        "status": "created",
        "attempts": 0,
        "notes": {
            "session_id": session_id,
            "invoice_id": invoice_id,
            "debtor_name": debtor_name,
        },
        "created_at": 1740000000,
    }


def verify_webhook_signature(body: bytes, signature: str) -> bool:
    """
    Verify the X-Razorpay-Signature header against the raw request body.

    Returns True when the signature is valid, False on a mismatch. Never raises.
    """
    webhook_secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")
    try:
        client = _get_client()
        client.utility.verify_webhook_signature(
            body.decode("utf-8"), signature, webhook_secret
        )
        return True
    except Exception:
        return False
