"""
RecoverFlow — Razorpay Orders + Checkout client
===============================================
Real Razorpay integration using Orders + Checkout JS (NOT Payment Links).

Payment Links don't fire webhooks reliably in test mode; Orders + Checkout do.
The order's `notes` carry (session_id, invoice_id, debtor_name) so the webhook
can map a captured payment back to the originating negotiation session.
"""

from __future__ import annotations

import os

import razorpay

# The Razorpay KEY_ID is public and is sent to the frontend for Checkout JS.
# The KEY_SECRET never leaves the backend. Use .get() so a missing env var
# degrades to an auth error at request time rather than an import crash.
client = razorpay.Client(auth=(
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
    amount_paise = int(round(amount_inr * 100))
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


def verify_webhook_signature(body: bytes, signature: str) -> bool:
    """
    Verify the X-Razorpay-Signature header against the raw request body.

    Returns True when the signature is valid, False on a mismatch. Never raises.
    """
    webhook_secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")
    try:
        client.utility.verify_webhook_signature(
            body.decode("utf-8"), signature, webhook_secret
        )
        return True
    except razorpay.errors.SignatureVerificationError:
        return False
