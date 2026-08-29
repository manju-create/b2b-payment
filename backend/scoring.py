# Tier thresholds validated against IBM Finance Factoring Dataset (n=2,466)
# Real distribution: on-time 64.4%, mild delay 28.5%, at-risk 7.1%
# Source: WA_Fn-UseC_-Accounts-Receivable.csv
"""
RecoverFlow — Debtor Scoring Engine
====================================
Pure, side-effect-free scoring function.

Scoring Model (source of truth: PRD.md §4 / CLAUDE.md)
------------------------------------------------------
Signal                      Weight  Range after normalisation
on_time_rate                35 %    0.0 – 1.0  (higher = better)
avg_days_late               25 %    0 – ∞      (lower = better)
invoice_vs_typical_ratio    15 %    0.0 – ∞    (1.0 = typical; >1 = large = riskier)
current_dpd                 15 %    0 – ∞      (lower = better)
dispute_count               10 %    0 – ∞      (lower = better)

Score 0–100 → Tier:
  A  85–100  reliable payer
  B  60–84   generally good, occasionally late
  C  35–59   inconsistent, needs structured terms
  D   0–34   high risk / serial late payer

Cold-start: no prior history → default Tier C (score 47), cold_start=True.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# In-memory trust score overrides (resets on restart — fine for demo)
# ---------------------------------------------------------------------------

score_overrides: dict[str, dict] = {}
# key: debtor_id
# value: { adjustment: float, reason: str, timestamp: str, events: list }

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHTS = {
    "on_time_rate": 0.35,
    "avg_days_late": 0.25,
    "invoice_vs_typical_ratio": 0.15,
    "current_dpd": 0.15,
    "dispute_count": 0.10,
}

# Tier boundaries (inclusive lower bound)
TIER_THRESHOLDS = [
    ("A", 85),
    ("B", 60),
    ("C", 35),
    ("D", 0),
]

# Cold-start defaults
COLD_START_SCORE = 47  # mid-point of Tier C (35–59)
COLD_START_TIER = "C"

# Normalisation caps / scaling parameters
# avg_days_late: cap at 90 days → score_component = 1 - min(days, 90)/90
AVG_DAYS_LATE_CAP = 90.0

# invoice_vs_typical_ratio: ratio=1.0 is neutral (score=1.0); cap penalty at ratio=3.0
# score_component = max(0, 1 - (ratio - 1) / 2)
RATIO_PENALTY_SCALE = 2.0

# current_dpd: cap at 120 days
CURRENT_DPD_CAP = 120.0

# dispute_count: cap penalty at 5 disputes
DISPUTE_CAP = 5.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_on_time_rate(rate: float) -> float:
    """Return a 0-1 score for on_time_rate. Higher is better."""
    return float(max(0.0, min(1.0, rate)))


def _normalise_avg_days_late(days: float) -> float:
    """Return a 0-1 score for avg_days_late. Lower days = higher score."""
    return 1.0 - min(days, AVG_DAYS_LATE_CAP) / AVG_DAYS_LATE_CAP


def _normalise_invoice_ratio(ratio: float) -> float:
    """
    Return a 0-1 score for invoice_vs_typical_ratio.
    ratio=1.0 → score=1.0 (perfectly typical)
    ratio=3.0 → score=0.0 (very large relative to history → risk)
    ratio<1.0 → score capped at 1.0 (smaller than typical = not a risk signal)
    """
    if ratio <= 1.0:
        return 1.0
    return float(max(0.0, 1.0 - (ratio - 1.0) / RATIO_PENALTY_SCALE))


def _normalise_current_dpd(dpd: float) -> float:
    """Return a 0-1 score for current_dpd. Lower DPD = higher score."""
    return 1.0 - min(dpd, CURRENT_DPD_CAP) / CURRENT_DPD_CAP


def _normalise_dispute_count(count: int) -> float:
    """Return a 0-1 score for dispute_count. Fewer disputes = higher score."""
    return 1.0 - min(count, DISPUTE_CAP) / DISPUTE_CAP


def _tier_from_score(score: float) -> str:
    """Map a 0-100 score to tier letter."""
    for tier, floor in TIER_THRESHOLDS:
        if score >= floor:
            return tier
    return "D"  # unreachable but safe


def _derive_history_signals(debtor_history: dict[str, Any]) -> dict[str, float]:
    """
    Extract or compute the three history-derived signals from debtor_history.

    Expected keys (all optional — missing triggers cold-start):
      historical_invoices : list of invoice dicts (each may have 'paid_date',
                            'due_date', 'status', 'days_late', 'amount')
      on_time_rate        : float (overrides computed value if present)
      avg_days_late       : float (overrides computed value if present)
      dispute_count       : int   (overrides computed value if present)
    """
    historical = debtor_history.get("historical_invoices", [])

    # --- on_time_rate ---
    if "on_time_rate" in debtor_history:
        on_time_rate = float(debtor_history["on_time_rate"])
    elif historical:
        paid_on_time = sum(
            1 for inv in historical if inv.get("status") == "paid_on_time"
        )
        on_time_rate = paid_on_time / len(historical)
    else:
        return {}  # cold-start signal: no data

    # --- avg_days_late ---
    if "avg_days_late" in debtor_history:
        avg_days_late = float(debtor_history["avg_days_late"])
    elif historical:
        late_days = [
            float(inv.get("days_late", 0))
            for inv in historical
            if inv.get("status") in ("paid_late", "disputed")
            and inv.get("days_late") is not None
        ]
        avg_days_late = sum(late_days) / len(late_days) if late_days else 0.0
    else:
        avg_days_late = 0.0

    # --- dispute_count ---
    if "dispute_count" in debtor_history:
        dispute_count = int(debtor_history["dispute_count"])
    elif historical:
        dispute_count = sum(
            1 for inv in historical if inv.get("status") == "disputed"
        )
    else:
        dispute_count = 0

    return {
        "on_time_rate": on_time_rate,
        "avg_days_late": avg_days_late,
        "dispute_count": dispute_count,
    }


def _typical_invoice_amount(debtor_history: dict[str, Any]) -> float | None:
    """
    Compute typical (median) invoice amount from historical invoices.
    Returns None if no history.
    """
    historical = debtor_history.get("historical_invoices", [])
    amounts = [float(inv["amount"]) for inv in historical if "amount" in inv]
    if not amounts:
        return None
    amounts.sort()
    mid = len(amounts) // 2
    if len(amounts) % 2 == 0:
        return (amounts[mid - 1] + amounts[mid]) / 2.0
    return amounts[mid]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_debtor(
    debtor_history: dict[str, Any],
    current_invoice: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute a debtor risk score and tier from history + current invoice.

    Parameters
    ----------
    debtor_history : dict
        Debtor's payment history. Expected keys:
          - debtor_id         : str
          - historical_invoices : list[dict]  (optional)
          - on_time_rate      : float  (optional override)
          - avg_days_late     : float  (optional override)
          - dispute_count     : int    (optional override)

    current_invoice : dict
        The invoice being worked. Expected keys:
          - invoice_id        : str
          - amount            : float
          - dpd               : int    (days past due)

    Returns
    -------
    dict with keys:
      score               : float   — 0 to 100
      tier                : str     — "A" | "B" | "C" | "D"
      cold_start          : bool
      signals             : dict    — raw signal values before normalisation
      normalised_signals  : dict    — per-signal 0-1 scores
      weighted_components : dict    — per-signal contribution to final score
    """
    # ---- Step 1: extract history-derived signals -------------------------
    history_signals = _derive_history_signals(debtor_history)
    is_cold_start = not history_signals  # empty dict = no history

    if is_cold_start:
        return {
            "score": float(COLD_START_SCORE),
            "tier": COLD_START_TIER,
            "cold_start": True,
            "signals": {
                "on_time_rate": None,
                "avg_days_late": None,
                "invoice_vs_typical_ratio": None,
                "current_dpd": float(current_invoice.get("dpd", 0)),
                "dispute_count": None,
            },
            "normalised_signals": {},
            "weighted_components": {},
        }

    on_time_rate: float = history_signals["on_time_rate"]
    avg_days_late: float = history_signals["avg_days_late"]
    dispute_count: int = history_signals["dispute_count"]

    # ---- Step 2: invoice_vs_typical_ratio --------------------------------
    current_amount = float(current_invoice.get("amount", 0))
    typical = _typical_invoice_amount(debtor_history)

    if typical and typical > 0:
        invoice_ratio = current_amount / typical
    else:
        # No history to derive typical — treat as neutral (ratio = 1.0)
        invoice_ratio = 1.0

    # ---- Step 3: current DPD --------------------------------------------
    current_dpd = float(current_invoice.get("dpd", 0))

    # ---- Step 4: normalise each signal to [0, 1] -------------------------
    norm = {
        "on_time_rate": _normalise_on_time_rate(on_time_rate),
        "avg_days_late": _normalise_avg_days_late(avg_days_late),
        "invoice_vs_typical_ratio": _normalise_invoice_ratio(invoice_ratio),
        "current_dpd": _normalise_current_dpd(current_dpd),
        "dispute_count": _normalise_dispute_count(dispute_count),
    }

    # ---- Step 5: weighted sum → 0-100 ------------------------------------
    weighted = {signal: norm[signal] * WEIGHTS[signal] for signal in WEIGHTS}
    raw_score = sum(weighted.values())  # 0.0 – 1.0
    score = round(raw_score * 100, 2)

    # ---- Step 5b: apply any trust-score adjustment (in-memory) -----------
    override = score_overrides.get(debtor_history.get("debtor_id", ""))
    if override:
        score = round(min(100.0, max(0.0, score + override["adjustment"])), 2)

    # ---- Step 6: tier mapping --------------------------------------------
    tier = _tier_from_score(score)

    return {
        "score": score,
        "tier": tier,
        "cold_start": False,
        "signals": {
            "on_time_rate": on_time_rate,
            "avg_days_late": avg_days_late,
            "invoice_vs_typical_ratio": round(invoice_ratio, 4),
            "current_dpd": current_dpd,
            "dispute_count": dispute_count,
        },
        "normalised_signals": {k: round(v, 4) for k, v in norm.items()},
        "weighted_components": {k: round(v * 100, 4) for k, v in weighted.items()},
    }


# ---------------------------------------------------------------------------
# Negotiation stance (replaces TIER_BOUNDS hard-coded guardrails)
# ---------------------------------------------------------------------------

def get_negotiation_stance(trust_score: int) -> dict:
    """
    Return negotiation parameters derived from the debtor's trust score.
    The 20% floor is universal and never changes regardless of score.
    """
    floor = 20  # universal, never changes
    if trust_score >= 85:
        return {
            "opening": 30, "target": 25, "floor": floor,
            "max_days": 60, "max_discount": 0,
            "stance": "cooperative",
        }
    elif trust_score >= 60:
        return {
            "opening": 50, "target": 35, "floor": floor,
            "max_days": 45, "max_discount": 0,
            "stance": "firm_but_flexible",
        }
    elif trust_score >= 35:
        return {
            "opening": 65, "target": 40, "floor": floor,
            "max_days": 30, "max_discount": 0,
            "stance": "skeptical",
        }
    else:
        return {
            "opening": 80, "target": 50, "floor": floor,
            "max_days": 15, "max_discount": 0,
            "stance": "firm",
        }


def project_score_change(current_score: int, settlement_type: str) -> int:
    """
    Projects what the debtor's score will become after this invoice.
    settlement_type: "full_upfront" | "partial_deferred" |
                     "escalated" | "ghosted"
    """
    deltas = {
        "full_upfront":     +13,
        "partial_deferred": +8,
        "escalated":        -15,
        "ghosted":          -20,
    }
    delta = deltas.get(settlement_type, 0)
    return max(0, min(100, int(current_score) + delta))


def get_score_breakdown(debtor_history: dict) -> list:
    """
    Returns human-readable breakdown of what's affecting the score.
    Does NOT expose signal weights — only outcomes.
    """
    breakdown = []

    total = len(debtor_history.get("invoices", []))
    if total == 0:
        return [{"label": "New account — no history yet", "impact": "neutral"}]

    on_time = sum(
        1 for i in debtor_history["invoices"]
        if i["status"] == "paid_on_time"
    )
    late = sum(
        1 for i in debtor_history["invoices"]
        if i["status"] == "paid_late"
    )
    disputed = sum(
        1 for i in debtor_history["invoices"]
        if i["status"] == "disputed"
    )
    written_off = sum(
        1 for i in debtor_history["invoices"]
        if i["status"] == "written_off"
    )

    if on_time > 0:
        breakdown.append({
            "label": f"{on_time} invoices paid on time",
            "impact": "positive",
        })
    if late > 0:
        breakdown.append({
            "label": f"{late} invoice{'s' if late > 1 else ''} paid late",
            "impact": "negative",
        })
    if disputed > 0:
        breakdown.append({
            "label": f"{disputed} invoice{'s' if disputed > 1 else ''} disputed",
            "impact": "negative",
        })
    if written_off > 0:
        breakdown.append({
            "label": f"{written_off} invoice{'s' if written_off > 1 else ''} written off",
            "impact": "negative",
        })

    return breakdown


# ---------------------------------------------------------------------------
# Trust score feedback loop
# ---------------------------------------------------------------------------

ADJUSTMENT_ON_TIME = +8.0   # deferred payment paid on time
ADJUSTMENT_LATE    = -5.0   # deferred payment paid late

TIER_FLOORS = {"A": 85, "B": 60, "C": 35, "D": 0}
NEXT_TIER   = {"D": "C", "C": "B", "B": "A", "A": None}


def update_trust_score(debtor_id: str, on_time: bool) -> dict:
    """
    Apply a trust-score adjustment after a deferred payment event.

    Loads the debtor's current score, applies +8 (on-time) or -5 (late),
    persists the adjustment in score_overrides, recomputes tier, and returns
    a full change summary.
    """
    # Load debtor + invoice from data files
    debtors_path  = REPO_ROOT / "data" / "debtors.json"
    invoices_path = REPO_ROOT / "data" / "invoices.json"

    if not debtors_path.exists() or not invoices_path.exists():
        raise FileNotFoundError("data/debtors.json or data/invoices.json not found")

    debtors_list = json.loads(debtors_path.read_text())
    invoices_list = json.loads(invoices_path.read_text())

    debtor = next((d for d in debtors_list if d["debtor_id"] == debtor_id), None)
    if debtor is None:
        raise ValueError(f"Debtor {debtor_id!r} not found")

    invoice = next((i for i in invoices_list if i["debtor_id"] == debtor_id), None)
    if invoice is None:
        raise ValueError(f"No invoice found for debtor {debtor_id!r}")

    # Current score (includes any prior adjustments via score_overrides)
    current_result = score_debtor(debtor, invoice)
    old_score = current_result["score"]
    old_tier  = current_result["tier"]

    # Determine adjustment
    adjustment = ADJUSTMENT_ON_TIME if on_time else ADJUSTMENT_LATE
    reason     = "deferred_payment_on_time" if on_time else "deferred_payment_late"

    # Accumulate adjustment on top of any existing override
    existing = score_overrides.get(debtor_id, {"adjustment": 0.0, "events": []})
    new_total_adj = existing["adjustment"] + adjustment

    ts = datetime.now(timezone.utc).isoformat()
    event = {"reason": reason, "adjustment": adjustment, "timestamp": ts}
    events = existing.get("events", []) + [event]

    score_overrides[debtor_id] = {
        "adjustment": new_total_adj,
        "reason":     reason,
        "timestamp":  ts,
        "events":     events,
    }

    # Recompute score with new override
    new_result = score_debtor(debtor, invoice)
    new_score  = new_result["score"]
    new_tier   = new_result["tier"]

    tier_changed = new_tier != old_tier
    if tier_changed:
        direction = "upgraded" if TIER_FLOORS[new_tier] > TIER_FLOORS[old_tier] else "downgraded"
    else:
        direction = "unchanged"

    # Points needed for next tier
    next_tier = NEXT_TIER.get(new_tier)
    points_to_next = None
    if next_tier:
        points_to_next = round(TIER_FLOORS[next_tier] - new_score, 2)

    return {
        "debtor_id":       debtor_id,
        "debtor_name":     debtor.get("contact_name", ""),
        "company_name":    debtor.get("company_name", ""),
        "old_score":       old_score,
        "new_score":       new_score,
        "old_tier":        old_tier,
        "new_tier":        new_tier,
        "tier_changed":    tier_changed,
        "direction":       direction,
        "adjustment":      adjustment,
        "total_adjustment": new_total_adj,
        "reason":          reason,
        "points_to_next":  points_to_next,
        "next_tier":       next_tier,
    }


def get_score_status(debtor_id: str) -> dict:
    """
    Return current score, tier, adjustment history, and distance to next tier.
    """
    debtors_path  = REPO_ROOT / "data" / "debtors.json"
    invoices_path = REPO_ROOT / "data" / "invoices.json"

    debtors_list = json.loads(debtors_path.read_text())
    invoices_list = json.loads(invoices_path.read_text())

    debtor = next((d for d in debtors_list if d["debtor_id"] == debtor_id), None)
    if debtor is None:
        raise ValueError(f"Debtor {debtor_id!r} not found")

    invoice = next((i for i in invoices_list if i["debtor_id"] == debtor_id), None)
    if invoice is None:
        raise ValueError(f"No invoice for debtor {debtor_id!r}")

    result = score_debtor(debtor, invoice)
    score  = result["score"]
    tier   = result["tier"]

    override   = score_overrides.get(debtor_id, {})
    adjustment = override.get("adjustment", 0.0)
    events     = override.get("events", [])

    next_tier = NEXT_TIER.get(tier)
    points_to_next = round(TIER_FLOORS[next_tier] - score, 2) if next_tier else None
    next_tier_label = f"{round(points_to_next, 1)} points from Tier {next_tier}" if next_tier else "Already Tier A"

    return {
        "debtor_id":      debtor_id,
        "debtor_name":    debtor.get("contact_name", ""),
        "company_name":   debtor.get("company_name", ""),
        "current_score":  score,
        "current_tier":   tier,
        "base_adjustment": adjustment,
        "next_tier":      next_tier,
        "points_to_next": points_to_next,
        "next_tier_label": next_tier_label,
        "adjustment_history": events,
    }


# ---------------------------------------------------------------------------
# Trust Score Engine — calculate_trust_score
# ---------------------------------------------------------------------------
# Additive points model: every signal contributes points, total clamped 0-100.
# This is the debtor-facing / agent-facing trust score (replaces the legacy
# weighted score_debtor for the live negotiation). Tier is INTERNAL only.

COLD_START_TRUST_SCORE = 50   # new debtor with no history starts here

# (tier, score_floor, min_acceptance_pct, tone) — tier is INTERNAL only.
TRUST_TIERS = [
    ("A", 75, 0.85, "collegial"),
    ("B", 50, 0.70, "professional"),
    ("C", 25, 0.60, "formal"),
    ("D", 0,  1.00, "legal"),
]


def _invoice_days_late(inv: dict[str, Any]) -> float | None:
    """Return days-late for one historical invoice (positive = paid late)."""
    if inv.get("days_late") is not None:
        return float(inv["days_late"])
    due, paid = inv.get("due_date"), inv.get("paid_date")
    if due and paid:
        try:
            return float((date.fromisoformat(paid) - date.fromisoformat(due)).days)
        except (ValueError, TypeError):
            pass
    if inv.get("status") == "paid_on_time":
        return 0.0
    return None


def _parse_ts(ts: Any) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts))
    except (ValueError, TypeError):
        return None


def _historical_trust_signals(
    debtor_history: dict[str, Any], current_invoice: dict[str, Any]
) -> dict[str, int]:
    """Historical signals 1-5. Returns {} when there is no history (cold start)."""
    historical = debtor_history.get("historical_invoices") or []
    signals: dict[str, int] = {}
    if not historical:
        return signals

    # 1. on_time_rate — % of past invoices paid on time (days_late <= 0)
    days_late_vals: list[float] = []
    on_time = 0
    for inv in historical:
        dl = _invoice_days_late(inv)
        if dl is not None:
            days_late_vals.append(dl)
            if dl <= 0:
                on_time += 1
        elif inv.get("status") == "paid_on_time":
            on_time += 1
    total = len(historical)
    rate = on_time / total
    if rate >= 0.80:
        signals["on_time_rate"] = 30
    elif rate >= 0.60:
        signals["on_time_rate"] = 20
    elif rate >= 0.40:
        signals["on_time_rate"] = 10
    else:
        signals["on_time_rate"] = 0

    # 2. avg_days_late (IBM baseline: 3.4 days average)
    if days_late_vals:
        avg = sum(days_late_vals) / len(days_late_vals)
        if avg <= 0:
            signals["avg_days_late"] = 20
        elif avg <= 3:
            signals["avg_days_late"] = 15
        elif avg <= 15:
            signals["avg_days_late"] = 5
        elif avg <= 45:
            signals["avg_days_late"] = -15
        else:
            signals["avg_days_late"] = -25

    # 3. dispute_history
    if "dispute_count" in debtor_history:
        dispute_count = int(debtor_history["dispute_count"])
    else:
        dispute_count = sum(
            1 for inv in historical if inv.get("status") == "disputed"
        )
    if dispute_count == 0:
        signals["dispute_history"] = 15
    elif dispute_count == 1:
        signals["dispute_history"] = -10
    else:
        signals["dispute_history"] = -20

    # 4. repeat_customer
    signals["repeat_customer"] = 5 if total >= 5 else 0

    # 5. invoice_size_vs_typical (vs. debtor average invoice amount)
    amounts = [
        float(inv["amount"]) for inv in historical if inv.get("amount") is not None
    ]
    if amounts:
        typical = sum(amounts) / len(amounts)
        current_amount = float(current_invoice.get("amount") or 0)
        if typical > 0:
            ratio = current_amount / typical
            if ratio <= 1:
                signals["invoice_size_vs_typical"] = 10
            elif ratio <= 2:
                signals["invoice_size_vs_typical"] = 0
            else:
                signals["invoice_size_vs_typical"] = -10

    return signals


def _live_trust_signals(
    session: dict[str, Any], current_invoice: dict[str, Any]
) -> dict[str, int]:
    """Live session signals 6-9, recalculated every turn."""
    signals: dict[str, int] = {}

    # 6. response_engagement — how quickly the debtor responded
    agent_ts = _parse_ts(session.get("last_agent_ts"))
    debtor_ts = _parse_ts(session.get("last_debtor_ts"))
    if agent_ts and debtor_ts:
        elapsed_min = (debtor_ts - agent_ts).total_seconds() / 60.0
        if elapsed_min >= 0:
            if elapsed_min <= 2:
                signals["response_engagement"] = 10
            elif elapsed_min <= 10:
                signals["response_engagement"] = 5
    if "response_engagement" not in signals:
        signals["response_engagement"] = 0

    # 7. voluntary_partial_offer
    if session.get("voluntary_partial_offered"):
        signals["voluntary_partial_offer"] = 10
    elif session.get("partial_after_suggested"):
        signals["voluntary_partial_offer"] = 5
    else:
        signals["voluntary_partial_offer"] = 0

    # 8. negotiation_behaviour
    if session.get("accepted_first_offer"):
        signals["negotiation_behaviour"] = 5
    elif session.get("offers_rejected", 0) >= 2:
        signals["negotiation_behaviour"] = -10
    elif session.get("negotiated_down"):
        signals["negotiation_behaviour"] = -5
    else:
        signals["negotiation_behaviour"] = 0

    # 9. current_dpd
    dpd = int(current_invoice.get("dpd") or 0)
    if dpd <= 7:
        signals["current_dpd"] = 0
    elif dpd <= 30:
        signals["current_dpd"] = -10
    elif dpd <= 60:
        signals["current_dpd"] = -20
    else:
        signals["current_dpd"] = -25

    return signals


def _trust_tier(score: int) -> dict[str, Any]:
    for tier, floor, min_acc, tone in TRUST_TIERS:
        if score >= floor:
            return {"tier": tier, "min_acceptance_pct": min_acc, "tone": tone}
    return {"tier": "D", "min_acceptance_pct": 1.00, "tone": "legal"}


def calculate_trust_score(
    debtor_history: dict, current_invoice: dict, session: dict
) -> dict:
    """
    Additive trust-score engine. Returns:
    {
        "score": int (0-100),
        "tier": "A" | "B" | "C" | "D",   # used internally by agent only
        "signals": { signal_name: points_applied },
        "negotiation_flex": { "min_acceptance_pct": float, "tone": str },
    }
    """
    signals = _historical_trust_signals(debtor_history, current_invoice)
    has_history = bool(signals)
    base = 0 if has_history else COLD_START_TRUST_SCORE
    signals.update(_live_trust_signals(session, current_invoice))

    raw = base + sum(signals.values())
    score = int(max(0, min(100, raw)))

    tier_info = _trust_tier(score)
    return {
        "score": score,
        "tier": tier_info["tier"],
        "signals": signals,
        "negotiation_flex": {
            "min_acceptance_pct": tier_info["min_acceptance_pct"],
            "tone": tier_info["tone"],
        },
    }
