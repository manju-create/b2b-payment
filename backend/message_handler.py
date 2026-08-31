"""
RecoverFlow — Inbound Debtor-Message Handler (MongoDB-backed)
=============================================================
Handles incoming debtor messages against the durable negotiation state stored
in MongoDB (Railway). The Mongo ``chat_history`` array is the single source of
truth for the conversation:

  * the debtor's message is recorded first,
  * the agent's reply is recorded after,
  * and DeepSeek is always fed the *entire* chat_history.

The two negotiation TRAPDOORS run *before* any DeepSeek call, so a debtor who
keeps lowballing below the floor is stopped deterministically by Python — never
by the model:

  * TRAPDOOR 1 (Hard Stop):    ``reason_collected`` True + offer < floor → escalate.
  * TRAPDOOR 2 (First Reject): ``first_counter_issued`` True + no reason yet
                               + offer < floor → ask WHY via the MCQ buttons.

Public API
----------
start_session(invoice_id)             -> seed Mongo + opening message (idempotent)
handle_incoming_message(invoice_id, user_text, user_offer_amount) -> dict
handle_reason_mcq_answer(invoice_id, button_id)  -> dict
get_chat_history(invoice_id)          -> the transcript (for chat preview)

The Mongo collection is injectable (``collection=``) so everything is testable
without a live database.
"""

from __future__ import annotations

import os
from typing import Any

from pymongo import MongoClient

from backend.agent import (
    create_session,
    open_turn,
    process_turn,
    _get_engine,
    _extract_amount_rupees,
    _handle_reason_mcq_answer,
    handle_document_verdict,
    create_full_payment_order,
    MCQ_REASONS,
    MERCHANT_NAME,
)

# button_id → human label, mirrors the app's MCQ_REASONS (see backend.agent).
REASON_LABELS = {r["button_id"]: r["label"] for r in MCQ_REASONS}
REASON_OPTIONS = [r["label"] for r in MCQ_REASONS]

# Reconstructed agent sessions live here (keyed by invoice_id) so the full
# context (debtor history, engine numbers) survives between webhook calls in a
# long-running process. Mongo is still the source of truth for the locks, floor,
# chat history and status — this cache just avoids re-reading the JSON data files
# on every message.
_AGENT_SESSIONS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Mongo connection (lazy — only connects on the first message)
# ---------------------------------------------------------------------------

_client = None
_collection = None


def _get_collection():
    """Return the ``sessions`` collection, connecting lazily on first use."""
    global _client, _collection
    if _collection is not None:
        return _collection
    uri = os.environ.get("MONGO_URI")
    if not uri:
        raise EnvironmentError("MONGO_URI environment variable is not set.")
    # Fail fast (~3s) instead of hanging for the default 30s when the DB is
    # unreachable (e.g. the Railway-internal URI hit from a local machine).
    _client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    _collection = _client["recoverflow_db"]["sessions"]
    return _collection


# Result of the reachability probe (None = not probed yet).
_MONGO_AVAILABLE: bool | None = None


def mongo_available() -> bool:
    """Return True if MongoDB responds to a ping. The result is cached.

    When the DB is reachable (e.g. on Railway) this returns quickly; when it is
    unreachable it pays the ~3s fast-fail timeout once, then remembers.
    """
    global _MONGO_AVAILABLE
    if _MONGO_AVAILABLE is None:
        try:
            _get_collection().database.client.admin.command("ping")
            _MONGO_AVAILABLE = True
        except Exception:
            _MONGO_AVAILABLE = False
    return _MONGO_AVAILABLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_offer(value: Any) -> int | None:
    """Normalise the caller-provided offer amount to an int (or None)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().replace(",", "")
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def _append_history(col, invoice_id: str, role: str, content: str, **extra) -> None:
    """Append one message to the Mongo ``chat_history`` array."""
    entry = {"role": role, "content": content}
    if extra:
        entry.update(extra)
    col.update_one({"invoice_id": invoice_id}, {"$push": {"chat_history": entry}})


def _rehydrate(session: dict, doc: dict) -> None:
    """Layer the durable Mongo state (locks, floor, chat history) onto a session."""
    locks = doc.get("state_locks") or {}
    session["first_counter_issued"] = bool(locks.get("first_counter_issued"))
    session["reason_collected"] = bool(locks.get("reason_collected"))

    bounds = doc.get("financial_bounds") or {}
    floor = bounds.get("current_floor")
    if isinstance(floor, (int, float)):
        engine = _get_engine(session)
        if int(floor) < engine.min_today:
            engine.hardship_verified = True
        engine.min_today = int(floor)
        engine.step3_amount = int(floor)
        session["negotiation_engine"] = engine.to_dict()

    history = doc.get("chat_history") or []
    if history:
        session["messages"] = [m for m in history if isinstance(m, dict)]
        session["turn_count"] = sum(
            1 for m in session["messages"] if m.get("role") == "user"
        )


def _build_agent_session(invoice_id: str, doc: dict | None) -> dict:
    """Return the full agent session for this invoice, rehydrated from Mongo.

    The agent (``backend.agent``) needs the complete session dict it builds in
    ``create_session`` — debtor history, invoice, the frozen negotiation engine.
    That is reconstructed once, then the durable bits from the Mongo doc are
    layered back on top so the conversation continues correctly across restarts.
    """
    session = _AGENT_SESSIONS.get(invoice_id)
    if session is not None:
        return session

    session = create_session(invoice_id)
    _AGENT_SESSIONS[invoice_id] = session
    if doc:
        _rehydrate(session, doc)
    return session


def _sync_to_mongo(col, invoice_id: str, session: dict, doc: dict | None) -> None:
    """Persist the durable negotiation state back to the Mongo session doc."""
    engine = _get_engine(session)
    col.update_one(
        {"invoice_id": invoice_id},
        {"$set": {
            "status": session.get("status", (doc or {}).get("status", "active")),
            "trust_score": session.get("trust_score", (doc or {}).get("trust_score", 0)),
            "state_locks.first_counter_issued": bool(session.get("first_counter_issued")),
            "state_locks.reason_collected": bool(session.get("reason_collected")),
            "financial_bounds.current_floor": engine.min_today,
            "chat_history": session.get("messages", []),
        }},
    )


def _should_show_upload_card(session: dict) -> bool:
    """Mirror of server._should_show_upload_card, for the Mongo flow."""
    if session.get("document_verification"):
        return False
    if session.get("payment_order"):
        return False
    if session.get("pending_upload"):
        return True
    return session.get("identified_situation") in ("DISPUTE", "ALREADY_PAID", "CASHFLOW")


def _delegate_to_agent(invoice_id: str, user_text: str, doc: dict, col) -> dict:
    """No trapdoor fired — hand the turn to the existing DeepSeek agent."""
    session = _build_agent_session(invoice_id, doc)

    try:
        agent_reply, session = process_turn(session, user_text)
    except ValueError as exc:
        return {"action_type": "error", "message": str(exc)}

    _sync_to_mongo(col, invoice_id, session, doc)

    result: dict[str, Any] = {
        "action_type": session.get("action_type", "negotiate"),
        "message": agent_reply,
        "status": session.get("status"),
    }
    if session.get("mcq_options"):
        result["mcq_options"] = session["mcq_options"]
    if session.get("agreed_terms"):
        result["agreed_terms"] = session["agreed_terms"]
    if session.get("payment_order"):
        result["payment_order"] = session["payment_order"]
    if session.get("identified_situation"):
        result["identified_situation"] = session["identified_situation"]
    if _should_show_upload_card(session):
        result["show_upload_card"] = True
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_session(invoice_id: str, collection: Any = None) -> dict:
    """Idempotent start: seed the Mongo doc + opening message, else return state.

    Returns the transcript and enough invoice context for the frontend to render
    the chat from scratch (so a page reload shows the full conversation).
    """
    col = collection or _get_collection()
    doc = col.find_one({"invoice_id": invoice_id})

    if doc is not None:
        # Already started — reconstruct the session so the response always
        # carries the invoice/debtor context the frontend needs (a reload would
        # otherwise get NaN/undefined in the invoice card).
        session = _build_agent_session(invoice_id, doc)
        return {
            "invoice_id": invoice_id,
            "status": doc.get("status"),
            "current_floor": (doc.get("financial_bounds") or {}).get("current_floor"),
            "history": doc.get("chat_history") or [],
            "invoice_amount": session.get("invoice_amount"),
            "invoice_amount_paise": session.get("invoice_amount_paise"),
            "debtor_name": session.get("debtor_name"),
            "company_name": session.get("company_name"),
            "dpd": session.get("dpd"),
            "tier": session.get("tier"),
            "trust_score": session.get("display_trust_score", session.get("trust_score")),
        }

    # First contact — build the full session, seed Mongo, and open the chat.
    session = _build_agent_session(invoice_id, None)
    opening, session = open_turn(session)
    engine = _get_engine(session)

    doc = {
        "invoice_id": invoice_id,
        "status": session["status"],
        "trust_score": session.get("trust_score", 0),
        "financial_bounds": {
            "principal": session["invoice_amount"],
            "current_floor": engine.min_today,
            "max_allowed_date": str(engine.deadline),
        },
        "state_locks": {
            "first_counter_issued": False,
            "reason_collected": False,
        },
        "chat_history": session.get("messages", []),
    }
    col.insert_one(doc)

    return {
        "invoice_id": invoice_id,
        "status": doc["status"],
        "current_floor": engine.min_today,
        "history": doc["chat_history"],
        "invoice_amount": session.get("invoice_amount"),
        "invoice_amount_paise": session.get("invoice_amount_paise"),
        "debtor_name": session.get("debtor_name"),
        "company_name": session.get("company_name"),
        "dpd": session.get("dpd"),
        "tier": session.get("tier"),
        "trust_score": session.get("display_trust_score", session.get("trust_score")),
    }


def handle_incoming_message(
    invoice_id: str,
    user_text: str,
    user_offer_amount: Any = None,
    collection: Any = None,
) -> dict:
    """Process one incoming debtor message against the Mongo session state.

    The trapdoors run first (no DeepSeek call when they trigger); otherwise the
    turn falls through to the existing DeepSeek agent.
    """
    col = collection or _get_collection()

    # 1. Pull the live session from MongoDB.
    doc = col.find_one({"invoice_id": invoice_id})
    if not doc:
        return {"action_type": "error", "message": f"No session found for {invoice_id}"}

    # 2. Record the debtor's message in the transcript immediately (before any
    #    trapdoor or DeepSeek call), so the chat_history is always complete.
    _append_history(col, invoice_id, "user", user_text)

    locks = doc.get("state_locks") or {}
    bounds = doc.get("financial_bounds") or {}
    current_floor = bounds.get("current_floor")

    offer = _coerce_offer(user_offer_amount)
    if offer is None and user_text:
        offer = _extract_amount_rupees(user_text)

    # 3. TRAPDOOR 1 — the hard stop. The reason was already collected and the
    #    debtor still offers below the floor → escalate.
    if locks.get("reason_collected") is True:
        if offer is not None and current_floor is not None and offer < current_floor:
            message = (
                f"₹{current_floor} is the absolute minimum. "
                "Negotiation closed. Escalating to legal."
            )
            col.update_one({"invoice_id": invoice_id}, {"$set": {"status": "escalated"}})
            _append_history(col, invoice_id, "assistant", message)
            return {"action_type": "final_ultimatum", "message": message}

    # 4. TRAPDOOR 2 — the first rejection. A counter was issued, we haven't
    #    collected the reason yet, and the offer is below the floor → ask WHY
    #    (MCQ) and flip the lock so we never ask twice.
    if locks.get("first_counter_issued") is True and not locks.get("reason_collected"):
        if offer is not None and current_floor is not None and offer < current_floor:
            col.update_one(
                {"invoice_id": invoice_id},
                {"$set": {"state_locks.reason_collected": True}},
            )
            message = "What is making it hard to meet this amount?"
            _append_history(
                col, invoice_id, "assistant", message,
                mcq_options=MCQ_REASONS, mcq_answered=False,
            )
            return {
                "action_type": "trigger_reason_mcq",
                "message": message,
                "options": list(REASON_OPTIONS),
                "mcq_options": MCQ_REASONS,
            }

    # 5. No trapdoor triggered — proceed to DeepSeek.
    return _delegate_to_agent(invoice_id, user_text, doc, col)


def handle_reason_mcq_answer(
    invoice_id: str, button_id: str, collection: Any = None
) -> dict:
    """Handle a reason-MCQ button click directly against the database.

    Three things happen, in order:
      1. the ``current_floor`` is lowered (a valid reason → the 20% hardship floor),
      2. the selection is recorded in ``chat_history`` as a user message,
      3. the updated chat_history is handed to DeepSeek for a sympathetic reply.
    """
    col = collection or _get_collection()

    doc = col.find_one({"invoice_id": invoice_id})
    if not doc:
        return {"action_type": "error", "message": f"No session found for {invoice_id}"}

    # Rebuild the agent session from the current Mongo state.
    session = _build_agent_session(invoice_id, doc)
    engine = _get_engine(session)

    # Reuse the agent's concession logic: apply the hardship floor, record the
    # "Debtor selected reason: …" user message, and let DeepSeek write the reply.
    reply, session = _handle_reason_mcq_answer(session, engine, button_id)

    _sync_to_mongo(col, invoice_id, session, doc)

    result: dict[str, Any] = {
        "action_type": session.get("action_type", "negotiate"),
        "message": reply,
        "status": session.get("status"),
    }
    if session.get("payment_order"):
        result["payment_order"] = session["payment_order"]
    return result


def get_chat_history(invoice_id: str, collection: Any = None) -> dict:
    """Return the full transcript (for chat preview / page reload)."""
    col = collection or _get_collection()
    doc = col.find_one({"invoice_id": invoice_id})
    if not doc:
        return {"invoice_id": invoice_id, "history": []}
    return {
        "invoice_id": invoice_id,
        "status": doc.get("status"),
        "history": doc.get("chat_history") or [],
    }


def pay_full(invoice_id: str, collection: Any = None) -> dict:
    """Generate a Razorpay order for the FULL invoice amount (bypass negotiation)."""
    col = collection or _get_collection()
    doc = col.find_one({"invoice_id": invoice_id})
    if not doc:
        return {"error": f"No session found for {invoice_id}"}

    session = _build_agent_session(invoice_id, doc)
    order = create_full_payment_order(session)
    _sync_to_mongo(col, invoice_id, session, doc)
    return order


def apply_payment(
    invoice_id: str, payment_id: str, amount_paise: int, collection: Any = None
) -> dict:
    """Apply a captured payment to the Mongo session (Razorpay webhook).

    Uses ``$set`` to mark ``status: "settled"`` and record the
    ``razorpay_payment_id``, then ``$push`` a terminal message into
    ``chat_history`` so the frontend reflects the closed state on next poll/reload.
    """
    col = collection or _get_collection()
    doc = col.find_one({"invoice_id": invoice_id})
    if not doc:
        return {"action": "invoice_not_found", "invoice_id": invoice_id}
    if doc.get("status") == "settled":
        return {"action": "already_settled", "invoice_id": invoice_id}

    col.update_one(
        {"invoice_id": invoice_id},
        {
            "$set": {
                "status": "settled",
                "razorpay_payment_id": payment_id,
                "recovered_paise": amount_paise,
                "payment_captured": True,
            },
            "$push": {
                "chat_history": {
                    "role": "assistant",
                    "content": "Payment received successfully. This invoice is now closed.",
                },
            },
        },
    )
    return {"action": "payment_applied", "invoice_id": invoice_id, "new_status": "settled"}


def _last_debtor_message(session: dict) -> str:
    """Return the most recent debtor message, used as the claim fallback."""
    for m in reversed(session.get("messages", [])):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def handle_document_upload(
    invoice_id: str,
    situation: str,
    content: bytes,
    file_type: str,
    file_name: str,
    collection: Any = None,
) -> dict:
    """Verify an uploaded document and persist it against the Mongo session.

    - Save the file to persistent storage (never raw bytes in Mongo).
    - ``$push`` its metadata into the ``documents`` array.
    - Freeze a dispute: any dispute evidence sets ``status`` to
      ``escalated_to_human`` so the AI agent stops demanding payment.
    """
    from backend.document_verifier import verify_document  # heavy (fitz) — lazy
    from backend.storage import save_upload

    col = collection or _get_collection()
    doc = col.find_one({"invoice_id": invoice_id})
    if not doc:
        return {"error": f"No session found for {invoice_id}"}

    session = _build_agent_session(invoice_id, doc)

    invoice = {
        "invoice_id": session["invoice_id"],
        "amount": session.get("invoice_amount"),
        "due_date": (session.get("current_invoice") or {}).get("due_date"),
        "merchant_name": MERCHANT_NAME,
    }
    debtor_claim = session.get("situation_claim") or _last_debtor_message(session)

    session["upload_attempts"] = session.get("upload_attempts", 0) + 1
    result = verify_document(content, file_type, situation, invoice, debtor_claim)
    agent_reply, session = handle_document_verdict(session, situation, result)
    session["pending_upload"] = None

    # External storage + metadata sync (raw bytes never touch Mongo).
    metadata = save_upload(invoice_id, file_name, content)
    col.update_one({"invoice_id": invoice_id}, {"$push": {"documents": metadata}})

    # Dispute freeze — a human must review before the AI asks for payment again.
    if situation == "DISPUTE":
        session["status"] = "escalated_to_human"

    _sync_to_mongo(col, invoice_id, session, doc)

    final_action = (session.get("document_verification") or {}).get("recommended_action")
    return {
        "verdict": session.get("document_verification"),
        "agent_reply": agent_reply,
        "status": session.get("status"),
        "show_upload_again": final_action == "REQUEST_BETTER_PROOF",
        "situation": situation,
    }
