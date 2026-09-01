"""
RecoverFlow — FastAPI Web Server
=================================
Run:  uvicorn backend.server:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from backend.agent import (  # noqa: E402
    create_session,
    open_turn,
    process_turn,
    handle_document_verdict,
    create_full_payment_order,
    MERCHANT_NAME,
)
from backend.message_handler import (  # noqa: E402
    handle_incoming_message,
    handle_reason_mcq_answer,
    start_session,
    get_chat_history,
    clear_chat_history,
    pay_full as mongo_pay_full,
    apply_payment,
    handle_document_upload,
    mongo_available,
    mongo_last_error,
)
from backend.document_verifier import verify_document                # noqa: E402
from backend.scoring import update_trust_score, get_score_status, get_score_breakdown  # noqa: E402
from backend.razorpay_client import verify_webhook_signature          # noqa: E402

DATA_DIR     = REPO_ROOT / "data"
FRONTEND_DIR = REPO_ROOT / "frontend"

# Document upload limits (see document_verifier.py)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024            # 10 MB
ALLOWED_UPLOAD_TYPES = {                        # mime type → verifier file_type
    "application/pdf": "pdf",
    "image/jpeg": "image",
    "image/png": "image",
}
# Situations that trigger document upload. CASHFLOW is the agent's internal
# label; the verifier expects CANNOT_PAY.
_UPLOAD_SITUATIONS = {"DISPUTE", "ALREADY_PAID", "CASHFLOW"}
_SITUATION_NORMALISE = {"CASHFLOW": "CANNOT_PAY"}

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

sessions:           dict[str, dict] = {}
batch_results:      dict[str, dict] = {}
deferred_schedule:  dict[str, dict] = {}
tier_events:        list[dict]      = []
processed_webhooks: set[str]        = set()   # idempotency: payment_id → seen
webhook_log:        list[dict]      = []       # every inbound event, for audit

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="RecoverFlow", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve uploaded documents from the persistent volume — the `url` recorded in
# the Mongo `documents` array points here (e.g. /uploads/INV-0016_....pdf).
try:
    from fastapi.staticfiles import StaticFiles
    from backend.storage import upload_dir as _upload_dir
    app.mount("/uploads", StaticFiles(directory=str(_upload_dir())), name="uploads")
except Exception:  # noqa: BLE001 — storage optional; metadata url is still recorded
    pass


def _load_json(filename: str) -> Any:
    p = DATA_DIR / filename
    return json.loads(p.read_text()) if p.exists() else []


def _rupees_fmt(paise: int) -> str:
    rupees = paise // 100
    s = str(rupees)
    if len(s) <= 3:
        return f"₹{s}"
    result, s = s[-3:], s[:-3]
    while len(s) > 2:
        result, s = s[-2:] + "," + result, s[:-2]
    return f"₹{s},{result}"


def _should_show_upload_card(s: dict) -> bool:
    """True when the chat should surface the document-upload card."""
    if s.get("document_verification"):       # already verified / escalated
        return False
    if s.get("payment_order"):               # already committed to payment
        return False
    if s.get("pending_upload"):              # LLM requested a document via tool
        return True
    return s.get("identified_situation") in _UPLOAD_SITUATIONS


def _last_debtor_message(s: dict) -> str:
    """Return the most recent debtor message, used as the claim fallback."""
    for m in reversed(s.get("messages", [])):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    p = FRONTEND_DIR / "merchant" / "index.html"
    if not p.exists():
        raise HTTPException(404, "Dashboard not found")
    return HTMLResponse(p.read_text())


def _chat_uses_mongo() -> bool:
    """Chat UI flow selector.

    ``MONGO_ENABLED`` is an optional override:
      * "true"  → force the MongoDB-backed flow
      * "false" → force the in-memory demo flow (fast local dev)
      * unset   → auto: use Mongo when it's reachable, else in-memory
    """
    v = os.environ.get("MONGO_ENABLED", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return mongo_available()


@app.get("/chat/{invoice_id}", response_class=HTMLResponse)
async def chat(invoice_id: str):
    p = FRONTEND_DIR / "debtor" / "index.html"
    if not p.exists():
        raise HTTPException(404, "Chat interface not found")
    html = p.read_text()
    html = html.replace("__INVOICE_ID__", invoice_id)
    html = html.replace("__MONGO_ENABLED__", "true" if _chat_uses_mongo() else "false")
    return HTMLResponse(html)


@app.get("/api/health/mongo")
async def mongo_health():
    """Report whether chat persistence is actually live, and why if not.

    The chat silently degrades to an in-memory demo flow when Mongo is
    unreachable, which loses history on every reload. This endpoint surfaces the
    exact connection error so that failure is no longer invisible.
    """
    from backend.message_handler import _redact_uri
    ok = mongo_available()
    return JSONResponse({
        "mongo_enabled_forced": os.environ.get("MONGO_ENABLED", "").strip().lower() in ("1", "true", "yes", "on"),
        "mongo_reachable": ok,
        "chat_will_persist": ok,
        "mongo_uri": _redact_uri(os.environ.get("MONGO_URI", "mongodb://localhost:27017")),
        "error": mongo_last_error(),
    })


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

@app.post("/api/batch/start")
async def batch_start():
    invoices_list = _load_json("invoices.json")
    if not invoices_list:
        raise HTTPException(500, "data/invoices.json missing")
    tier_breakdown = {"A": 0, "B": 0, "C": 0, "D": 0}
    total = 0
    errors = []
    for inv in invoices_list:
        iid = inv["invoice_id"]
        try:
            s = create_session(iid)
            sessions[s["session_id"]] = s
            batch_results[iid] = s
            tier_breakdown[s["tier"]] = tier_breakdown.get(s["tier"], 0) + 1
            total += s["invoice_amount_paise"]
        except Exception as exc:
            errors.append({"invoice_id": iid, "error": str(exc)})
    return {"total": len(batch_results), "tier_breakdown": tier_breakdown,
            "total_outstanding_paise": total, "total_outstanding_display": _rupees_fmt(total),
            "errors": errors}


@app.get("/api/batch/status")
async def batch_status():
    counts = {"settled": 0, "partially_settled": 0, "awaiting_payment": 0,
              "disputed": 0, "escalated": 0, "active": 0}
    total_recovered = 0
    invoices_out = []
    for iid, s in batch_results.items():
        status = s.get("status", "active")
        counts[status if status in counts else "active"] += 1
        total_recovered += s.get("recovered_paise", 0)
        plan = s.get("agreed_terms")
        plan_summary = None
        if plan and plan.get("deferred_amount", 0) > 0:
            plan_summary = f"{plan['upfront_display']} now + {plan['deferred_display']} by {plan['deferred_due_date']}"
        elif status == "settled":
            plan_summary = "Paid in full"
        # Dashboard mirrors the stable trust score the debtor sees in chat
        # (additive model, frozen at session start), not the legacy weighted
        # "score" or the live-updating trust score.
        trust_score = s.get("display_trust_score", s.get("trust_score", s.get("score", 0)))
        trust_tier  = s.get("display_trust_tier") or (s.get("trust_score_result") or {}).get("tier") or s.get("tier", "")
        invoices_out.append({
            "invoice_id": iid, "debtor_name": s.get("debtor_name", ""),
            "company_name": s.get("company_name", ""), "tier": trust_tier,
            "score": trust_score, "invoice_amount_paise": s.get("invoice_amount_paise", 0),
            "dpd": s.get("dpd", 0), "status": status,
            "recovered_paise": s.get("recovered_paise", 0),
            "razorpay_order_id": s.get("razorpay_order_id"), "session_id": s.get("session_id"),
            "turn_count": s.get("turn_count", 0), "plan_summary": plan_summary,
            "identified_situation": s.get("identified_situation"),
            "document": s.get("document_verification"),
        })
    return {**counts, "total_recovered_paise": total_recovered,
            "total_invoices": len(batch_results), "invoices": invoices_out}


# ---------------------------------------------------------------------------
# Negotiate
# ---------------------------------------------------------------------------

@app.post("/api/negotiate/start/{invoice_id}")
async def negotiate_start(invoice_id: str):
    try:
        s = create_session(invoice_id)
        opening, s = open_turn(s)
        sessions[s["session_id"]] = s
        batch_results[invoice_id] = s
        # Build score breakdown for the debtor UI trust score card
        from backend.scoring import get_score_breakdown as _gsb
        import json as _j
        debtors_path = DATA_DIR / "debtors.json"
        debtors_list = _j.loads(debtors_path.read_text()) if debtors_path.exists() else []
        debtor_rec = next((d for d in debtors_list if d.get("debtor_id") == s["debtor_id"]), {})
        breakdown = _gsb({"invoices": debtor_rec.get("historical_invoices", [])})
        return {"session_id": s["session_id"], "opening_message": opening,
                "invoice_amount_paise": s["invoice_amount_paise"],
                "invoice_amount": s["invoice_amount"],
                "debtor_name": s["debtor_name"], "company_name": s["company_name"],
                "dpd": s["dpd"], "tier": s["tier"],
                "score": s["score"],
                "trust_score": s.get("display_trust_score", s.get("trust_score", s.get("score", 0))),
                "score_delta": s.get("trust_score_delta", 0),
                "score_reason": s.get("trust_score_reason", "initial assessment"),
                "score_projections": s["score_projections"],
                "score_breakdown": breakdown,
                "identified_situation": s.get("identified_situation")}
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


class TurnRequest(BaseModel):
    message: str


@app.post("/api/negotiate/turn/{session_id}")
async def negotiate_turn(session_id: str, body: TurnRequest):
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(404, f"Session {session_id!r} not found")
    try:
        agent_reply, s = process_turn(s, body.message)
        sessions[session_id] = s
        batch_results[s["invoice_id"]] = s
        # Register deferred entry on plan confirmation.
        # A deferred_amount of 0 means the debtor paid the full invoice
        # upfront — no deferred entry is created and the session is settled.
        iid = s["invoice_id"]
        if (s.get("agreed_terms") and iid not in deferred_schedule
                and s["agreed_terms"].get("deferred_amount", 0) > 0):
            plan = s["agreed_terms"]
            deferred_schedule[iid] = {
                "invoice_id":      iid,
                "debtor_id":       s["debtor_id"],
                "debtor_name":     s["debtor_name"],
                "company_name":    s.get("company_name", ""),
                "session_id":      session_id,
                "deferred_amount": plan["deferred_amount"],
                "deferred_display": plan["deferred_display"],
                "due_date":        plan["deferred_due_date"],
                "status":          "pending",
                "scheduled_at":    datetime.now(timezone.utc).isoformat(),
            }
        # Sync messages to MongoDB if available
        if mongo_available():
            try:
                from backend.message_handler import _get_collection, _sync_to_mongo
                col = _get_collection()
                _sync_to_mongo(col, iid, s, col.find_one({"invoice_id": iid}))
            except Exception:
                pass

        import json as _json
        safe = {
            "agent_reply":     agent_reply,
            "session_status":  s["status"],
            "trust_score":     s.get("trust_score", s.get("score", 0)),
            "score_delta":     s.get("trust_score_delta", 0),
            "score_reason":    s.get("trust_score_reason", "initial assessment"),
            "payment_order":   s.get("payment_order"),
            "razorpay_order_id": s.get("razorpay_order_id"),
            "recovered_paise": s.get("recovered_paise", 0),
            "turn_count":      s["turn_count"],
            "identified_situation": s.get("identified_situation"),
            "agreed_terms":    s.get("agreed_terms"),
            "show_upload_card": _should_show_upload_card(s),
            "action_type":      s.get("action_type"),
            "mcq_options":      s.get("mcq_options"),
        }
        body = _json.dumps(safe, ensure_ascii=False)
        from fastapi.responses import Response
        return Response(content=body, media_type="application/json")
    except Exception as exc:
        raise HTTPException(500, str(exc))


class ReasonMcqBody(BaseModel):
    button_id: str


@app.post("/api/reason-mcq/{session_id}")
async def reason_mcq(session_id: str, body: ReasonMcqBody):
    """Receive a reason-MCQ button click: lower the floor and concede."""
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(404, f"Session {session_id!r} not found")
    from backend.agent import _get_engine, _handle_reason_mcq_answer
    try:
        engine = _get_engine(s)
        agent_reply, s = _handle_reason_mcq_answer(s, engine, body.button_id)
        sessions[session_id] = s
        batch_results[s["invoice_id"]] = s
        return {
            "agent_reply": agent_reply,
            "session_status": s["status"],
            "action_type": s.get("action_type"),
            "trust_score": s.get("trust_score", s.get("score", 0)),
        }
    except Exception as exc:
        raise HTTPException(500, str(exc))


class IncomingMessageBody(BaseModel):
    invoice_id: str
    user_text: str = ""
    user_offer_amount: int | None = None


@app.post("/api/message/incoming")
async def message_incoming(body: IncomingMessageBody):
    """
    Inbound debtor-message webhook (MongoDB-backed).

    Runs the negotiation trapdoors before the DeepSeek call, then falls through
    to the existing agent when no trapdoor fires.
    """
    try:
        return handle_incoming_message(
            body.invoice_id, body.user_text, body.user_offer_amount
        )
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


class MessageReasonMcqBody(BaseModel):
    invoice_id: str
    button_id: str


@app.post("/api/message/reason-mcq")
async def message_reason_mcq(body: MessageReasonMcqBody):
    """
    Reason-MCQ button click (MongoDB-backed). Lowers the floor, records the
    selection in chat_history, and returns DeepSeek's sympathetic reply.
    """
    try:
        return handle_reason_mcq_answer(body.invoice_id, body.button_id)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/message/start/{invoice_id}")
async def message_start(invoice_id: str):
    """Idempotent Mongo start — seeds the session + opening message if needed."""
    try:
        return start_session(invoice_id)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/message/history/{invoice_id}")
async def message_history(invoice_id: str):
    """Full transcript for a chat preview / page reload."""
    try:
        return get_chat_history(invoice_id)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/message/clear/{invoice_id}")
async def message_clear(invoice_id: str):
    """Clear chat history and reset negotiation state in MongoDB."""
    try:
        return clear_chat_history(invoice_id)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/negotiate/clear/{invoice_id}")
async def negotiate_clear(invoice_id: str):
    """Clear chat history in-memory and in MongoDB if available."""
    try:
        res = clear_chat_history(invoice_id)
        if invoice_id in sessions:
            sessions.pop(invoice_id, None)
        return res
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/message/pay-full/{invoice_id}")
async def message_pay_full(invoice_id: str):
    """Full-amount Razorpay order, bypassing negotiation (Mongo flow)."""
    try:
        return mongo_pay_full(invoice_id)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/negotiate/pay-full/{session_id}")
async def pay_full(session_id: str):
    """
    Generate a Razorpay Order for the FULL invoice amount, bypassing negotiation.

    Backs the "Pay in full" button next to the debtor name — the debtor can
    settle the whole invoice immediately without going through the agent.
    """
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(404, f"Session {session_id!r} not found")
    try:
        order = create_full_payment_order(s)
        sessions[session_id] = s
        batch_results[s["invoice_id"]] = s
        return order
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/negotiate/session/{session_id}")
async def get_session(session_id: str):
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(404, f"Session {session_id!r} not found")
    # Build score breakdown from debtor's historical invoices
    from backend.scoring import get_score_breakdown as _gsb
    import json as _j
    from pathlib import Path as _P
    debtors_path = _P(__file__).resolve().parent.parent / "data" / "debtors.json"
    debtors_list = _j.loads(debtors_path.read_text()) if debtors_path.exists() else []
    debtor = next((d for d in debtors_list if d.get("debtor_id") == s.get("debtor_id")), {})
    hist_invoices = debtor.get("historical_invoices", [])
    breakdown = _gsb({"invoices": hist_invoices})
    enriched = dict(s)
    enriched["score_breakdown"]   = breakdown
    enriched["score_projections"] = s.get("score_projections", {})
    return JSONResponse(enriched)


@app.post("/api/upload-document")
async def upload_document(
    invoice_id: str = Form(""),
    session_id: str = Form(""),
    situation: str = Form(""),
    file: UploadFile = File(...),
):
    """
    Receive a debtor-uploaded document and verify it.

    Two routing modes:
      * ``invoice_id`` → MongoDB flow (production): store externally, record
        metadata, and freeze disputes to ``escalated_to_human``.
      * ``session_id`` → in-memory demo flow.
    """
    # Normalise the situation label (frontend may send the agent's CASHFLOW).
    situation = (situation or "").strip().upper()
    situation = _SITUATION_NORMALISE.get(situation, situation)
    if not situation:
        situation = "GENERAL"
    if situation not in ("DISPUTE", "ALREADY_PAID", "CANNOT_PAY", "GENERAL"):
        raise HTTPException(400, f"Invalid situation: {situation!r}")

    content = await file.read()
    mime = (file.content_type or "").lower()

    if not content:
        raise HTTPException(400, "Empty file")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "File exceeds the 10MB limit")
    if mime not in ALLOWED_UPLOAD_TYPES:
        raise HTTPException(400, "Only PDF, JPG and PNG are accepted")
    file_type = ALLOWED_UPLOAD_TYPES[mime]

    # MongoDB flow (production) — route by invoice_id.
    if invoice_id:
        try:
            result = handle_document_upload(
                invoice_id, situation, content, file_type, file.filename
            )
        except EnvironmentError as exc:
            raise HTTPException(503, str(exc))
        except Exception as exc:
            raise HTTPException(500, str(exc))

        if result.get("error"):
            raise HTTPException(404, result["error"])

        return {
            "verdict": result.get("verdict"),
            "agent_reply": result.get("agent_reply"),
            "session_status": result.get("status"),
            "show_upload_again": result.get("show_upload_again"),
            "situation": result.get("situation"),
        }

    # In-memory demo flow — route by session_id.
    if session_id:
        s = sessions.get(session_id)
        if not s:
            raise HTTPException(404, f"Session {session_id!r} not found")

        invoice = {
            "invoice_id": s["invoice_id"],
            "amount": s.get("invoice_amount"),
            "due_date": (s.get("current_invoice") or {}).get("due_date"),
            "merchant_name": MERCHANT_NAME,
        }
        debtor_claim = s.get("situation_claim") or _last_debtor_message(s)

        s["upload_attempts"] = s.get("upload_attempts", 0) + 1

        result = verify_document(content, file_type, situation, invoice, debtor_claim)
        agent_reply, s = handle_document_verdict(s, situation, result)
        s["pending_upload"] = None   # the upload was just fulfilled

        sessions[session_id] = s
        batch_results[s["invoice_id"]] = s

        final_action = (s.get("document_verification") or {}).get("recommended_action")
        return {
            "verdict": s.get("document_verification"),
            "agent_reply": agent_reply,
            "session_status": s.get("status"),
            "show_upload_again": final_action == "REQUEST_BETTER_PROOF",
            "situation": situation,
        }

    raise HTTPException(400, "invoice_id or session_id is required")


def _apply_payment(invoice_id: str, payment_id: str,
                   amount_paise: int, event_name: str) -> dict:
    """
    Core payment application logic — shared by the real webhook, the
    /api/payment-confirmed endpoint, and the simulate-webhook safety net.
    """
    s = batch_results.get(invoice_id)
    if not s:
        return {"action": "invoice_not_found", "invoice_id": invoice_id}

    # Determine new status: if there is a deferred plan still pending,
    # this upfront payment makes it partially_settled; otherwise settled.
    has_deferred = invoice_id in deferred_schedule and \
                   deferred_schedule[invoice_id].get("status") == "pending"
    new_status = "partially_settled" if has_deferred else "settled"

    ts = datetime.now(timezone.utc).isoformat()

    s["status"]                = new_status
    s["recovered_paise"]       = amount_paise
    s["payment_id"]            = payment_id
    s["payment_captured"]      = True
    s["payment_confirmed_at"]  = ts
    s["settled_at"]            = ts

    # Append to session audit log
    s.setdefault("audit_log", []).append({
        "event":      event_name,
        "timestamp":  ts,
        "invoice_id": invoice_id,
        "payment_id": payment_id,
        "amount_paise": amount_paise,
    })
    return {"action": "payment_applied", "invoice_id": invoice_id,
            "new_status": new_status, "amount_paise": amount_paise}


def find_session_by_order_id(order_id: str) -> dict | None:
    """Locate the negotiation session that created a given Razorpay order."""
    for s in sessions.values():
        if s.get("razorpay_order_id") == order_id:
            return s
    return None


@app.post("/webhooks/razorpay")
async def webhook_razorpay(request: Request):
    """
    Real Razorpay webhook handler — Orders + Checkout flow.

    Security:
      - X-Razorpay-Signature verified via the Razorpay SDK utility.
      - If RAZORPAY_WEBHOOK_SECRET is not configured, the webhook is rejected.

    Idempotency:
      - payment_id tracked in the processed_webhooks set.
      - Duplicate deliveries are acknowledged with 200 and skipped.

    Always returns 200 for valid, signed events — even when no matching
    session is found — so Razorpay does not retry. Invalid signatures → 400.
    """
    raw_body = await request.body()
    ts       = datetime.now(timezone.utc).isoformat()

    # ── 1. Signature verification ─────────────────────────────────────────
    if not os.environ.get("RAZORPAY_WEBHOOK_SECRET"):
        webhook_log.append({"ts": ts, "action": "configuration_error",
                            "detail": "RAZORPAY_WEBHOOK_SECRET is not set"})
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": "Webhook secret not configured"},
        )

    signature = request.headers.get("X-Razorpay-Signature", "")
    if not verify_webhook_signature(raw_body, signature):
        webhook_log.append({"ts": ts, "action": "signature_rejected"})
        raise HTTPException(status_code=400, detail="Invalid signature")

    # ── 2. Parse payload ──────────────────────────────────────────────────
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        webhook_log.append({"ts": ts, "action": "parse_error"})
        return {"status": "ok"}   # malformed — ack and move on

    event = payload.get("event", "")
    log_entry: dict = {"ts": ts, "event": event, "action": "ignored"}

    if event != "payment.captured":
        webhook_log.append({**log_entry, "action": "unsupported_event"})
        return {"status": "ok"}

    # ── 3. Extract payment identifiers + invoice_id from notes ────────────
    # The order's `notes.invoice_id` (injected at order creation) is the only
    # reliable way to map an external payment back to the internal database.
    payment = payload.get("payload", {}).get("payment", {}).get("entity", {})
    order_id     = payment.get("order_id", "")
    payment_id   = payment.get("id", "")
    amount_paise = int(payment.get("amount", 0))
    invoice_id   = (payment.get("notes") or {}).get("invoice_id", "")
    log_entry.update({"payment_id": payment_id, "order_id": order_id,
                      "amount_paise": amount_paise, "invoice_id": invoice_id})

    # ── 4. Idempotency ────────────────────────────────────────────────────
    if payment_id and payment_id in processed_webhooks:
        webhook_log.append({**log_entry, "action": "duplicate_ignored"})
        return {"status": "already_processed", "payment_id": payment_id}

    # ── 5. Apply payment to MongoDB by notes.invoice_id (no in-memory lookup)
    if not invoice_id:
        webhook_log.append({**log_entry, "action": "invoice_id_missing"})
        return {"status": "ok"}   # ack 200 so Razorpay does not retry

    result = apply_payment(invoice_id, payment_id, amount_paise)
    log_entry["action"] = result["action"]

    if payment_id:
        processed_webhooks.add(payment_id)
    webhook_log.append(log_entry)
    return {"status": "ok", **result}


@app.get("/api/audit/{invoice_id}")
async def get_audit(invoice_id: str):
    s = batch_results.get(invoice_id)
    if not s:
        raise HTTPException(404, f"Invoice {invoice_id!r} not in batch")
    return {"invoice_id": invoice_id, "audit_log": s.get("audit_log", [])}


@app.get("/api/webhook-log")
async def get_webhook_log():
    """Returns the last 50 inbound webhook events (all, including ignored)."""
    return {"events": webhook_log[-50:]}


# ---------------------------------------------------------------------------
# Simulate-webhook: demo safety net (no signature required)
# ---------------------------------------------------------------------------

class SimWebhookBody(BaseModel):
    amount: int   # paise
    payment_id: str = ""   # optional; auto-generated if blank


@app.post("/api/simulate-webhook/{invoice_id}")
async def simulate_webhook(invoice_id: str, body: SimWebhookBody):
    """
    Demo safety net: fires the same internal payment-application logic
    as a real webhook but skips signature verification.

    Use this if ngrok drops or Razorpay sandbox is slow during a live demo.

    Audit log records event as 'simulated_webhook' so it is clearly
    distinguishable from a real Razorpay event.
    """
    import uuid
    payment_id = body.payment_id or f"sim_{uuid.uuid4().hex[:12]}"
    ts         = datetime.now(timezone.utc).isoformat()

    result = _apply_payment(invoice_id, payment_id, body.amount, "simulated_webhook")

    webhook_log.append({
        "ts":         ts,
        "event":      "simulated_webhook",
        "action":     result["action"],
        "invoice_id": invoice_id,
        "payment_id": payment_id,
        "amount_paise": body.amount,
    })
    return {"status": "ok", "payment_id": payment_id, **result}


# ---------------------------------------------------------------------------
# Payment confirmation (Checkout JS handler callback)
# ---------------------------------------------------------------------------

class PaymentConfirmBody(BaseModel):
    payment_id: str
    order_id: str
    session_id: str = ""


@app.post("/api/payment-confirmed")
async def payment_confirmed(body: PaymentConfirmBody):
    """
    Client-side confirmation from the Checkout JS `handler` callback.

    The payment.captured webhook is the source of truth; this endpoint is a
    safety net so the session updates even if the webhook is delayed. It
    reuses the same idempotent payment-application logic keyed on the order.
    """
    session = sessions.get(body.session_id) or find_session_by_order_id(body.order_id)
    if not session:
        raise HTTPException(404, "Session not found")

    # Skip if already settled by a webhook (idempotency).
    if session.get("payment_captured"):
        return {"status": "already_confirmed", "payment_id": body.payment_id}

    amount_paise = round((session.get("payment_amount") or 0) * 100)
    result = _apply_payment(session["invoice_id"], body.payment_id,
                            amount_paise, "payment_client_confirmed")
    return {"status": "ok", **result}


# ---------------------------------------------------------------------------
# Deferred payments
# ---------------------------------------------------------------------------

@app.get("/api/deferred/status")
async def deferred_status():
    return {"deferred_payments": [
        {"invoice_id": k, **v} for k, v in deferred_schedule.items()
    ]}


class SimPayBody(BaseModel):
    on_time: bool = True


@app.post("/api/deferred/simulate-payment/{invoice_id}")
async def simulate_deferred(invoice_id: str, body: SimPayBody):
    entry = deferred_schedule.get(invoice_id)
    if not entry:
        raise HTTPException(404, f"No deferred payment for {invoice_id!r}")
    entry["status"] = "paid_on_time" if body.on_time else "paid_late"
    result = update_trust_score(entry["debtor_id"], on_time=body.on_time)
    if result["tier_changed"] and invoice_id in batch_results:
        batch_results[invoice_id]["tier"] = result["new_tier"]
    if result["tier_changed"]:
        icon = "✅" if result["direction"] == "upgraded" else "⬇️"
        tier_events.append({
            "icon": icon,
            "text": (f"{result['company_name']}: Tier {result['old_tier']} → "
                     f"{result['new_tier']} "
                     f"({'on time' if body.on_time else 'late'})"),
            **result,
        })
    return {**result, "deferred_entry": entry}


@app.get("/api/score/{debtor_id}")
async def score_endpoint(debtor_id: str):
    try:
        return get_score_status(debtor_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@app.get("/api/tier-events")
async def get_tier_events():
    return {"events": tier_events[-20:]}


@app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(sessions), "batch": len(batch_results)}
