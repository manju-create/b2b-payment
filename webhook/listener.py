"""
RecoverFlow — Razorpay Webhook Listener
========================================
Receives payment.captured events from Razorpay, verifies the
HMAC-SHA256 signature, and updates invoice status in the JSON store.

Critical constraints (from CLAUDE.md):
  - Signature MUST be verified before processing
  - Duplicate webhook events MUST NOT double-count payments (idempotency)
  - Webhook secret lives in env: RAZORPAY_WEBHOOK_SECRET

Run:
    uvicorn webhook.listener:app --port 8001 --reload

Razorpay dashboard → Webhooks → URL: https://<your-ngrok>.ngrok.io/webhook/razorpay
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Auto-load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

DATA_DIR = REPO_ROOT / "data"

# ---------------------------------------------------------------------------
# In-memory idempotency store
# Keyed by payment_id — prevents double-counting if Razorpay retries.
# In production this would be a DB unique constraint on payment_id.
# ---------------------------------------------------------------------------

_PROCESSED_PAYMENT_IDS: set[str] = set()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="RecoverFlow Webhook", version="1.0.0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_invoices() -> list[dict]:
    path = DATA_DIR / "invoices.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _save_invoices(invoices: list[dict]) -> None:
    (DATA_DIR / "invoices.json").write_text(
        json.dumps(invoices, indent=2, ensure_ascii=False)
    )


def _append_webhook_log(entry: dict) -> None:
    """Append to data/webhook_log.json (create if absent)."""
    path = DATA_DIR / "webhook_log.json"
    log: list[dict] = []
    if path.exists():
        try:
            log = json.loads(path.read_text())
        except json.JSONDecodeError:
            log = []
    log.append(entry)
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False))


def _verify_signature(body: bytes, received_sig: str, secret: str) -> bool:
    """
    Razorpay sends X-Razorpay-Signature: HMAC-SHA256(body, webhook_secret).
    We recompute and compare using hmac.compare_digest (timing-safe).
    """
    expected = hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, received_sig)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "recoverflow-webhook", "ts": _ts()}


@app.post("/webhook/razorpay", status_code=status.HTTP_200_OK)
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: str | None = Header(default=None),
) -> JSONResponse:
    """
    Handle incoming Razorpay webhook events.

    Razorpay guarantees at-least-once delivery. We use payment_id as the
    idempotency key so duplicate events are silently acknowledged (200 OK)
    without double-counting.
    """
    body = await request.body()
    secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")

    # ── 1. Signature verification ──────────────────────────────────────────
    if secret:
        if not x_razorpay_signature:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing X-Razorpay-Signature header",
            )
        if not _verify_signature(body, x_razorpay_signature, secret):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Signature verification failed",
            )
    else:
        # No secret configured — allow in dev, warn loudly
        print(
            "⚠️  WARNING: RAZORPAY_WEBHOOK_SECRET not set. "
            "Signature verification DISABLED. Do not use in production.",
            flush=True,
        )

    # ── 2. Parse payload ───────────────────────────────────────────────────
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload",
        )

    event = payload.get("event", "")
    log_entry: dict = {
        "received_at": _ts(),
        "event": event,
        "payload_summary": {},
        "action": "ignored",
    }

    # ── 3. Only handle payment.captured ───────────────────────────────────
    if event != "payment.captured":
        _append_webhook_log(log_entry)
        return JSONResponse({"status": "ignored", "event": event})

    # ── 4. Extract payment details ─────────────────────────────────────────
    try:
        payment = payload["payload"]["payment"]["entity"]
        payment_id   = payment["id"]                          # pay_XXXX
        amount_paise = int(payment["amount"])                 # in paise
        description  = payment.get("description", "")
        notes        = payment.get("notes", {})               # key-value store
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Malformed payment payload: {exc}",
        )

    log_entry["payload_summary"] = {
        "payment_id": payment_id,
        "amount_paise": amount_paise,
        "description": description,
        "notes": notes,
    }

    # ── 5. Idempotency check ───────────────────────────────────────────────
    if payment_id in _PROCESSED_PAYMENT_IDS:
        log_entry["action"] = "duplicate_ignored"
        _append_webhook_log(log_entry)
        print(f"⏭️  Duplicate payment.captured for {payment_id} — ignored", flush=True)
        return JSONResponse({"status": "duplicate_ignored", "payment_id": payment_id})

    # ── 6. Resolve invoice ─────────────────────────────────────────────────
    # Payment links created by generate_payment_link() carry the invoice_id
    # in the description ("INV-XXXX") or in notes["invoice_id"].
    invoice_id: str | None = notes.get("invoice_id")
    if not invoice_id:
        # Fallback: scan description for INV-\d+ pattern
        import re
        m = re.search(r"INV-\d+", description)
        invoice_id = m.group(0) if m else None

    # ── 7. Update invoice status ───────────────────────────────────────────
    updated = False
    invoices = _load_invoices()

    for inv in invoices:
        if inv["invoice_id"] == invoice_id:
            if inv.get("status") == "paid":
                # Already marked paid — still idempotent
                log_entry["action"] = "already_paid"
                break

            inv["status"] = "paid"
            inv["paid_at"]        = _ts()
            inv["payment_id"]     = payment_id
            inv["amount_paid_paise"] = amount_paise
            updated = True
            log_entry["action"] = "invoice_marked_paid"
            log_entry["invoice_id"] = invoice_id
            print(
                f"✅  payment.captured → {invoice_id} marked PAID "
                f"| amount=₹{amount_paise//100:,} | payment_id={payment_id}",
                flush=True,
            )
            break
    else:
        # invoice_id not found in store
        log_entry["action"] = "invoice_not_found"
        log_entry["invoice_id"] = invoice_id
        print(f"⚠️  payment.captured for unknown invoice_id={invoice_id!r}", flush=True)

    if updated:
        _save_invoices(invoices)

    # ── 8. Mark payment_id as processed ───────────────────────────────────
    _PROCESSED_PAYMENT_IDS.add(payment_id)
    _append_webhook_log(log_entry)

    return JSONResponse({
        "status": "ok",
        "payment_id": payment_id,
        "invoice_id": invoice_id,
        "action": log_entry["action"],
    })


# ---------------------------------------------------------------------------
# Stand-alone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webhook.listener:app", host="0.0.0.0", port=8001, reload=True)
