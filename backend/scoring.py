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
from datetime import datetime, timezone
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
