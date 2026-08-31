"""
RecoverFlow — Document Verifier
================================
Verifies a debtor-uploaded document (payment proof, invoice copy, cashflow
proof, etc.) against their claim using DeepSeek (OpenAI-compatible API).

Files are processed **in memory only** — never written to disk.

Supported situations:
  DISPUTE      — debtor disputes the invoice amount
  ALREADY_PAID — debtor claims they already paid
  CANNOT_PAY   — debtor provides a reason they cannot pay (cashflow proof,
                 business closure letter, etc.)

Content extraction:
  PDF         → text via pymupdf (fitz)
  image       → text via pytesseract OCR (DeepSeek has no vision input)

The DeepSeek call returns JSON only. `verify_document` returns the parsed dict
with a stable shape (verdict / confidence / checks / red_flags /
recommended_action / friendly summaries).
"""

from __future__ import annotations

import io
import json
import logging
import os
from typing import Any

import fitz  # pymupdf

from openai import OpenAI

logger = logging.getLogger(__name__)

SITUATION_DISPUTE = "DISPUTE"
SITUATION_ALREADY_PAID = "ALREADY_PAID"
SITUATION_CANNOT_PAY = "CANNOT_PAY"
SITUATION_GENERAL = "GENERAL"   # no specific claim identified yet

MODEL = "deepseek-chat"
MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from a PDF using pymupdf (fitz)."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text())
        return "".join(parts).strip()
    finally:
        doc.close()


def extract_text_from_image(file_bytes: bytes) -> str:
    """OCR an image (JPG/PNG) using pytesseract. Returns "" if unavailable.

    tesseract is optional — if the binary or the python binding is missing,
    this degrades to "" and the caller returns an INCONCLUSIVE verdict rather
    than crashing the flow.
    """
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        logger.warning("pytesseract/PIL not installed — image OCR unavailable")
        return ""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return (text or "").strip()
    except Exception as exc:  # noqa: BLE001 — OCR is best-effort
        logger.warning("Image OCR failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# DeepSeek client + prompt construction
# ---------------------------------------------------------------------------

def _get_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


_SYSTEM_PROMPT = (
    "You are a document verification specialist for a B2B payment recovery "
    "system. Analyse the document provided and verify the debtor's claim. Be "
    "thorough but fair. Always respond in JSON only — no preamble, no markdown."
)


def _invoice_lines(invoice: dict, *, include_merchant: bool) -> str:
    lines = [
        f"- Invoice ID: {invoice.get('invoice_id', '')}",
        f"- Amount: ₹{invoice.get('amount', '')}",
        f"- Due date: {invoice.get('due_date', '')}",
    ]
    if include_merchant:
        lines.append(f"- Merchant: {invoice.get('merchant_name', '')}")
    return "\n".join(lines)


def _build_user_prompt(
    situation: str, invoice: dict, debtor_claim: str, content: str
) -> str:
    """Build the situation-specific USER prompt. `content` is extracted text."""
    if situation == SITUATION_ALREADY_PAID:
        return f"""Invoice details:
{_invoice_lines(invoice, include_merchant=True)}

Debtor's claim: "{debtor_claim}"

Document provided:
{content}

Verify the following and respond with JSON:
{{
  "verdict": "VALID" | "INVALID" | "INCONCLUSIVE",
  "confidence": "high" | "medium" | "low",
  "checks": {{
    "amount_matches": true | false | null,
    "utr_or_transaction_id_found": true | false | null,
    "date_before_due_date": true | false | null,
    "document_looks_genuine": true | false | null
  }},
  "red_flags": ["list any suspicious elements or inconsistencies"],
  "extracted_utr": "UTR number if found, else null",
  "extracted_amount": "amount found in document, else null",
  "extracted_date": "payment date if found, else null",
  "merchant_friendly_summary": "2 sentences for merchant dashboard",
  "debtor_friendly_response": "1 warm sentence to show debtor in chat — never accusatory even if invalid",
  "recommended_action": "ACCEPT_CLAIM" | "REQUEST_BETTER_PROOF" | "ESCALATE_TO_MERCHANT"
}}"""

    if situation == SITUATION_DISPUTE:
        return f"""Invoice details:
{_invoice_lines(invoice, include_merchant=True)}

Debtor's claim: "{debtor_claim}"

Document provided:
{content}

Verify the following and respond with JSON:
{{
  "verdict": "VALID" | "INVALID" | "INCONCLUSIVE",
  "confidence": "high" | "medium" | "low",
  "checks": {{
    "amount_matches_invoice": true | false | null,
    "signature_or_approval_found": true | false | null,
    "document_looks_genuine": true | false | null
  }},
  "red_flags": ["list any suspicious elements or inconsistencies"],
  "extracted_amount": "amount on their document if found",
  "amount_discrepancy": "difference between invoice and document amount",
  "merchant_friendly_summary": "2 sentences for merchant",
  "debtor_friendly_response": "1 warm sentence for chat",
  "recommended_action": "ACCEPT_CLAIM" | "REQUEST_BETTER_PROOF" | "ESCALATE_TO_MERCHANT"
}}"""

    if situation == SITUATION_GENERAL:
        return f"""Invoice details:
{_invoice_lines(invoice, include_merchant=True)}

Debtor's claim: "{debtor_claim}"

Document provided:
{content}

Analyse the document and respond with JSON:
{{
  "verdict": "VALID" | "INVALID" | "INCONCLUSIVE",
  "confidence": "high" | "medium" | "low",
  "checks": {{
    "document_looks_genuine": true | false | null
  }},
  "red_flags": ["list any suspicious elements or inconsistencies"],
  "merchant_friendly_summary": "2 sentences summarising what the document shows",
  "debtor_friendly_response": "1 warm sentence acknowledging receipt",
  "recommended_action": "ACCEPT_CLAIM" | "REQUEST_BETTER_PROOF" | "ESCALATE_TO_MERCHANT"
}}"""

    # CANNOT_PAY
    return f"""Invoice details:
{_invoice_lines(invoice, include_merchant=False)}

Debtor's claim: "{debtor_claim}"

Document provided:
{content}

Assess whether the document genuinely supports their inability to pay (e.g. bank
statement showing low balance, business closure letter, medical emergency proof).
Respond with JSON:
{{
  "verdict": "VALID" | "INVALID" | "INCONCLUSIVE",
  "confidence": "high" | "medium" | "low",
  "checks": {{
    "document_supports_claim": true | false | null,
    "document_looks_genuine": true | false | null,
    "date_is_recent": true | false | null
  }},
  "red_flags": ["list any suspicious elements or inconsistencies"],
  "merchant_friendly_summary": "2 sentences for merchant",
  "debtor_friendly_response": "1 warm sentence for chat",
  "recommended_action": "ACCEPT_CLAIM" | "REQUEST_BETTER_PROOF" | "ESCALATE_TO_MERCHANT"
}}"""


def _inconclusive(message: str) -> dict:
    """Fallback result used when content cannot be read or the LLM call fails."""
    return {
        "verdict": "INCONCLUSIVE",
        "confidence": "low",
        "checks": {},
        "red_flags": [],
        "extracted_utr": None,
        "extracted_amount": None,
        "extracted_date": None,
        "merchant_friendly_summary": "Document could not be read automatically.",
        "debtor_friendly_response": message,
        "recommended_action": "REQUEST_BETTER_PROOF",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_document(
    file_bytes: bytes,
    file_type: str,           # "pdf" or "image"
    situation: str,           # "DISPUTE" | "ALREADY_PAID" | "CANNOT_PAY"
    invoice: dict,            # {invoice_id, amount, due_date, merchant_name}
    debtor_claim: str,        # what debtor said in their own words
) -> dict[str, Any]:
    """
    Extract content from an uploaded document and verify it against the debtor's
    claim. Returns the parsed verification JSON dict (never raises — falls back
    to an INCONCLUSIVE result on unreadable content or API failure).
    """
    # ---- Step 1: extract content -------------------------------------------
    if file_type == "pdf":
        try:
            content = extract_text_from_pdf(file_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PDF text extraction failed: %s", exc)
            content = ""
        if not content:
            return _inconclusive(
                "I had trouble reading that document — could you try uploading "
                "a clearer version or a different format?"
            )
    elif file_type == "image":
        content = extract_text_from_image(file_bytes)
        if not content:
            return _inconclusive(
                "I couldn't read the text in that image. A clear PDF, or a "
                "straight-on photo of the document (bank statement, receipt, or "
                "letter), works best."
            )
    else:
        raise ValueError(f"Unsupported file_type: {file_type!r}")

    # ---- Step 2: call DeepSeek ---------------------------------------------
    user_prompt = _build_user_prompt(situation, invoice, debtor_claim, content)
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001 — degrade, never crash the chat
        logger.warning("DeepSeek verification call failed: %s", exc)
        return _inconclusive(
            "I couldn't fully check that document just now — could you try once more?"
        )

    # ---- Step 3: parse JSON ------------------------------------------------
    # Defensively strip markdown fences some models add despite instructions.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("DeepSeek returned non-JSON: %s", exc)
        return _inconclusive(
            "I couldn't fully check that document just now — could you try once more?"
        )

    if not isinstance(result, dict):
        return _inconclusive(
            "I couldn't fully check that document just now — could you try once more?"
        )

    return result
