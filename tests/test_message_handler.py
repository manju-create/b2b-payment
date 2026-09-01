"""
Tests for the MongoDB-backed inbound message handler.

The two trapdoors must fire deterministically (before any DeepSeek call):

  * Trapdoor 1 (Hard Stop):   reason_collected=True  + offer < floor  → escalate
  * Trapdoor 2 (First Reject): first_counter_issued=True + reason_collected=False
                              + offer < floor → trigger the reason MCQ (once)

Every inbound message is recorded in Mongo ``chat_history`` — the user message
first, then the assistant reply. The reason-MCQ answer lowers the floor, records
the selection, and returns DeepSeek's sympathetic reply.
"""

import pytest

import backend.agent as agent_mod
import backend.message_handler as mh
from backend.message_handler import (
    handle_incoming_message,
    handle_reason_mcq_answer,
    apply_payment,
    handle_document_upload,
)


@pytest.fixture(autouse=True)
def _clear_state():
    # The handler caches agent sessions and engines keyed by invoice/session id;
    # clear them so each test starts from a fresh reconstruction.
    mh._AGENT_SESSIONS.clear()
    agent_mod._ENGINES.clear()
    yield


class FakeCollection:
    """Minimal stand-in for the Mongo ``sessions`` collection."""

    def __init__(self, doc):
        self.doc = doc

    def find_one(self, query):
        if query.get("invoice_id") == self.doc.get("invoice_id"):
            return dict(self.doc)
        return None

    def update_one(self, query, update):
        assert query.get("invoice_id") == self.doc.get("invoice_id")
        for key, value in (update.get("$set") or {}).items():
            if "." in key:
                top, rest = key.split(".", 1)
                self.doc.setdefault(top, {})[rest] = value
            else:
                self.doc[key] = value
        for key, value in (update.get("$push") or {}).items():
            self.doc.setdefault(key, []).append(value)


def _doc(**overrides):
    base = {
        "invoice_id": "INV-0016",
        "status": "negotiating",
        "financial_bounds": {"principal": 147000, "current_floor": 40200},
        "state_locks": {"first_counter_issued": False, "reason_collected": False},
        "chat_history": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Trapdoors
# ---------------------------------------------------------------------------

def test_trapdoor_hard_stop_escalates():
    doc = _doc(state_locks={"first_counter_issued": True, "reason_collected": True})
    col = FakeCollection(doc)

    result = handle_incoming_message(
        "INV-0016", "I still can only do 30000", 30000, collection=col
    )

    assert result["action_type"] == "final_ultimatum"
    assert "₹40200" in result["message"]
    assert col.doc["status"] == "escalated"


def test_trapdoor_first_rejection_triggers_reason_mcq():
    doc = _doc(state_locks={"first_counter_issued": True, "reason_collected": False})
    col = FakeCollection(doc)

    result = handle_incoming_message(
        "INV-0016", "I can only do 30000", 30000, collection=col
    )

    assert result["action_type"] == "trigger_reason_mcq"
    assert result["options"] == ["Client hasn't paid", "Cash flow issues", "Dispute/Damaged goods", "Other"]
    # The lock flips so the bot never asks for a reason twice.
    assert col.doc["state_locks"]["reason_collected"] is True
    assert col.doc["status"] == "negotiating"   # not escalated yet


def test_no_trapdoor_when_offer_meets_floor(monkeypatch):
    # Offer above the floor → neither trapdoor fires; falls through to the agent.
    doc = _doc(state_locks={"first_counter_issued": True, "reason_collected": False})
    col = FakeCollection(doc)

    monkeypatch.setattr(
        agent_mod, "_get_client",
        lambda: (_ for _ in ()).throw(EnvironmentError("no key in test")),
    )

    result = handle_incoming_message(
        "INV-0016", "I can pay 45000 today", 45000, collection=col
    )

    assert result["action_type"] in ("negotiate", "finalize_agreement", "error")
    assert result.get("message")


# ---------------------------------------------------------------------------
# chat_history recording + reason-MCQ answer
# ---------------------------------------------------------------------------

def test_incoming_records_user_and_mcq_question_in_history():
    doc = _doc(state_locks={"first_counter_issued": True, "reason_collected": False})
    col = FakeCollection(doc)

    result = handle_incoming_message(
        "INV-0016", "I can only do 30000", 30000, collection=col
    )

    assert result["action_type"] == "trigger_reason_mcq"
    history = col.doc["chat_history"]
    assert history[0] == {"role": "user", "content": "I can only do 30000"}
    assert history[1]["role"] == "assistant"
    assert history[1]["mcq_options"]
    assert history[1]["mcq_answered"] is False


def test_reason_mcq_answer_lowers_floor_and_records_reason(monkeypatch):
    monkeypatch.setattr(
        agent_mod, "_get_client",
        lambda: (_ for _ in ()).throw(EnvironmentError("no key in test")),
    )
    doc = {
        "invoice_id": "INV-0016",
        "status": "negotiating",
        "financial_bounds": {"principal": 147000, "current_floor": 40200},
        "state_locks": {"first_counter_issued": True, "reason_collected": True},
        "chat_history": [
            {
                "role": "assistant",
                "content": "What is making it hard to meet this amount?",
                "mcq_options": [{"button_id": "cashflow", "label": "Cash flow issues"}],
                "mcq_answered": False,
            },
        ],
    }
    col = FakeCollection(doc)

    result = handle_reason_mcq_answer("INV-0016", "cashflow", collection=col)

    assert result["action_type"] == "negotiate"
    assert result["message"]
    # Floor was lowered (valid reason → 20% hardship floor).
    assert col.doc["financial_bounds"]["current_floor"] < 40200

    history = col.doc["chat_history"]
    assert any(
        m.get("role") == "user" and "Cash flow issues" in m["content"]
        for m in history
    )
    question = next(m for m in history if m.get("mcq_options"))
    assert question["mcq_answered"] is True
    assert question["mcq_selected"] == "Cash flow issues"


# ---------------------------------------------------------------------------
# Payment application + document upload
# ---------------------------------------------------------------------------

def test_apply_payment_sets_settled_and_pushes_terminal_message():
    doc = _doc()
    col = FakeCollection(doc)

    result = apply_payment("INV-0016", "pay_123", 14700000, collection=col)

    assert result["action"] == "payment_applied"
    assert col.doc["status"] == "settled"
    assert col.doc["razorpay_payment_id"] == "pay_123"
    assert col.doc["recovered_paise"] == 14700000
    last = col.doc["chat_history"][-1]
    assert last["role"] == "assistant"
    assert "Payment received successfully" in last["content"]


def test_apply_payment_is_idempotent_for_settled_invoice():
    doc = _doc(status="settled", razorpay_payment_id="pay_old")
    col = FakeCollection(doc)

    result = apply_payment("INV-0016", "pay_new", 14700000, collection=col)

    assert result["action"] == "already_settled"
    assert col.doc["razorpay_payment_id"] == "pay_old"


def test_document_upload_dispute_freezes_agent(monkeypatch):
    import backend.document_verifier as dv
    import backend.storage as storage

    monkeypatch.setattr(dv, "verify_document", lambda *a, **k: {
        "verdict": "VALID",
        "recommended_action": "ACCEPT_CLAIM",
        "debtor_friendly_response": "Thanks, noted.",
    })
    monkeypatch.setattr(storage, "save_upload", lambda invoice_id, fn, c: {
        "file_name": fn, "url": "/uploads/x.pdf", "uploaded_at": "2026-08-31T00:00:00Z",
    })

    doc = _doc()
    doc["chat_history"] = [{"role": "user", "content": "The amount is wrong"}]
    col = FakeCollection(doc)

    result = handle_document_upload(
        "INV-0016", "DISPUTE", b"%PDF-1.4 fake", "pdf", "proof.pdf", collection=col
    )

    assert result["status"] == "escalated_to_human"
    assert col.doc["status"] == "escalated_to_human"
    assert col.doc["documents"][0]["file_name"] == "proof.pdf"
    assert col.doc["documents"][0]["url"] == "/uploads/x.pdf"


def test_clear_chat_history_removes_messages_and_resets_doc():
    doc = _doc()
    doc["chat_history"] = [
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "I offer 1000"},
        {"role": "assistant", "content": "That is too low."}
    ]
    col = FakeCollection(doc)

    res = mh.clear_chat_history("INV-0016", collection=col)

    assert res["status"] == "cleared"
    assert len(res["history"]) == 1
    assert res["history"][0]["role"] == "assistant"
    assert col.doc["status"] == "negotiating"
    assert len(col.doc["chat_history"]) == 1
    assert col.doc["chat_history"][0]["role"] == "assistant"

