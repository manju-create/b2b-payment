#!/usr/bin/env python3
"""
RecoverFlow — Synthetic Data Generator
========================================
Produces data/debtors.json  and  data/invoices.json.

Strategy
--------
Each tier has a "recipe" — target ranges for the three history-derived
signals (on_time_rate, avg_days_late, dispute_count) and the current-invoice
signals (dpd, invoice_amount_vs_typical).

We sample from those ranges, build the full historical_invoices list so it is
internally consistent, then run the real score_debtor() to verify the tier
matches.  If it doesn't, we re-sample up to MAX_RETRIES times.  The scoring
formula is deterministic, so with correctly tuned ranges this always converges.

Scoring formula (for reference):
  on_time_rate        * 35  → norm = rate
  avg_days_late       * 25  → norm = 1 - min(days,90)/90
  invoice_ratio       * 15  → norm = max(0, 1-(ratio-1)/2)   [1.0 when ratio≤1]
  current_dpd         * 15  → norm = 1 - min(dpd,120)/120
  dispute_count       * 10  → norm = 1 - min(cnt,5)/5

  score = 100 * sum(norm_i * weight_i)

Tier floors: A≥85, B≥60, C≥35, D<35
"""

from __future__ import annotations

import json
import random
import sys
import os
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

# ---------------------------------------------------------------------------
# Path setup — allow running from any CWD
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.scoring import score_debtor  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
fake = Faker("en_IN")
fake.seed_instance(SEED)

TODAY = date(2025, 8, 26)   # anchored so JSON is deterministic

DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

MAX_RETRIES = 500   # re-sample limit per debtor

# ---------------------------------------------------------------------------
# Tier recipes
# Ranges are designed so the weighted sum lands in the tier band
# with high probability after clamping.
#
# Score formula targets (normalised, 0–1 per signal):
#   Tier A: need score ≥ 85/100 = 0.85
#   Tier B: need 0.60 ≤ score < 0.85
#   Tier C: need 0.35 ≤ score < 0.60
#   Tier D: need score < 0.35
#
# We derive target *raw* signal ranges from those goals.
# ---------------------------------------------------------------------------

TIER_RECIPES: dict[str, dict] = {
    "A": {
        "on_time_rate":   (0.90, 1.00),   # norm: 0.90–1.00  × 35 = 31.5–35
        "avg_days_late":  (0,    5),       # norm: 0.94–1.00  × 25 = 23.5–25
        "dispute_count":  (0,    0),       # norm: 1.00       × 10 = 10
        "dpd":            (30,   50),      # norm: 0.58–0.75  × 15 = 8.75–11.25
        # ratio target: keep ≤ 1.5 (light penalty) → norm 0.75–1.0 × 15 = 11.25–15
        "invoice_ratio":  (0.6,  1.4),     # norm: ~0.80–1.00
        # min composite ≈ 31.5+23.5+11.25+8.75+10 = 85 ✓
    },
    "B": {
        "on_time_rate":   (0.65, 0.85),   # norm 0.65–0.85 × 35 = 22.75–29.75
        "avg_days_late":  (8,    20),      # norm 0.78–0.91 × 25 = 19.5–22.75
        "dispute_count":  (0,    1),       # norm 0.80–1.00 × 10 = 8–10
        "dpd":            (35,   65),      # norm 0.46–0.71 × 15 = 6.9–10.6
        "invoice_ratio":  (0.7,  1.8),     # norm 0.60–1.00 × 15 = 9–15
        # mid-range composite ≈ 26+21+9+8.7+9 = 73.7  → B band ✓
    },
    "C": {
        "on_time_rate":   (0.35, 0.60),   # norm 0.35–0.60 × 35 = 12.25–21
        "avg_days_late":  (25,   55),      # norm 0.39–0.72 × 25 = 9.75–18
        "dispute_count":  (1,    3),       # norm 0.40–0.80 × 10 = 4–8
        "dpd":            (50,   90),      # norm 0.25–0.58 × 15 = 3.75–8.75
        "invoice_ratio":  (0.8,  2.2),     # norm 0.40–1.00 × 15 = 6–15
        # mid-range ≈ 16+13+6+6+10 = 51 → C band ✓
    },
    "D": {
        "on_time_rate":   (0.00, 0.30),   # norm 0.00–0.30 × 35 = 0–10.5
        "avg_days_late":  (55,   88),      # norm 0.02–0.39 × 25 = 0.5–9.75
        "dispute_count":  (2,    5),       # norm 0.00–0.60 × 10 = 0–6
        "dpd":            (70,   115),     # norm 0.04–0.42 × 15 = 0.6–6.3
        "invoice_ratio":  (1.0,  2.8),     # norm 0.00–1.00 × 15 = 0–15
        # mid-range ≈ 5+5+3+3+7 = 23 → D band ✓
    },
}

# ---------------------------------------------------------------------------
# Indian B2B company name pool (faker's en_IN company names look realistic)
# We pre-generate 50 unique company names.
# ---------------------------------------------------------------------------

COMPANY_SUFFIXES = [
    "Pvt Ltd", "Traders", "Distributors", "Enterprises", "Industries",
    "& Sons", "& Co", "Corporation", "Exports", "Imports",
]

SECTORS = [
    "Sharma", "Mehta", "Kapoor", "Gupta", "Singh", "Patel", "Jain",
    "Agarwal", "Reddy", "Nair", "Kumar", "Verma", "Rao", "Malhotra",
    "Bhat", "Iyer", "Pillai", "Chopra", "Sinha", "Desai",
    "Tiwari", "Pandey", "Mishra", "Dubey", "Chauhan", "Rajput",
    "Saxena", "Shukla", "Tripathi", "Bhatt", "Rastogi", "Srivastava",
    "Bansal", "Goel", "Arora", "Khanna", "Bajaj", "Birla", "Tata", "Ambani",
    "Oberoi", "Modi", "Naidu", "Swamy", "Anand", "Bose", "Roy", "Das", "Mitra",
]

# ---------------------------------------------------------------------------
# Failure mode pool: 50 outcomes in exact proportions
# ---------------------------------------------------------------------------

OUTCOME_POOL: list[str] = (
    ["clean_settlement"] * 30
    + ["dispute"]         * 12
    + ["repeat_extension"] * 8
)
assert len(OUTCOME_POOL) == 50
random.shuffle(OUTCOME_POOL)


# ---------------------------------------------------------------------------
# Helper: build a realistic historical_invoices list
# ---------------------------------------------------------------------------

def _build_historical_invoices(
    debtor_id: str,
    n: int,
    on_time_rate: float,
    avg_days_late: float,
    dispute_count: int,
    typical_amount: int,
) -> list[dict]:
    """
    Build n historical invoice dicts that are consistent with the supplied
    aggregate signals.  We reverse-engineer from the desired on_time_rate,
    avg_days_late, and dispute_count.
    """
    invoices = []

    # Decide per-invoice statuses
    n_disputes = min(dispute_count, n)
    n_paid_on_time = round(on_time_rate * n)
    n_paid_on_time = max(0, min(n - n_disputes, n_paid_on_time))
    n_paid_late = n - n_paid_on_time - n_disputes

    statuses = (
        ["paid_on_time"] * n_paid_on_time
        + ["paid_late"] * n_paid_late
        + ["disputed"] * n_disputes
    )
    random.shuffle(statuses)

    # Compute late_days values that average to avg_days_late
    late_status_count = n_paid_late + n_disputes
    if late_status_count > 0 and avg_days_late > 0:
        # Spread around the target average with ±20% noise
        base_days = avg_days_late
        late_days_pool = []
        for _ in range(late_status_count):
            jitter = random.uniform(0.8, 1.2)
            d = max(1, round(base_days * jitter))
            late_days_pool.append(d)
        # Adjust last element so the mean matches avg_days_late exactly
        total_needed = round(avg_days_late * late_status_count)
        diff = total_needed - sum(late_days_pool)
        late_days_pool[-1] = max(1, late_days_pool[-1] + diff)
    else:
        late_days_pool = [0] * late_status_count

    late_idx = 0
    ref_date = TODAY - timedelta(days=400)

    for i, status in enumerate(statuses):
        inv_id = f"{debtor_id}-H{i+1:03d}"
        # Stagger due dates ~30 days apart going backwards
        due_date = ref_date + timedelta(days=30 * i)
        # Amount: jitter around typical ±35%
        amount = max(20_000, round(
            typical_amount * random.uniform(0.65, 1.35) / 1000
        ) * 1000)

        inv: dict = {
            "invoice_id": inv_id,
            "amount": amount,
            "due_date": due_date.isoformat(),
        }

        if status == "paid_on_time":
            paid_date = due_date - timedelta(days=random.randint(0, 5))
            inv["paid_date"] = paid_date.isoformat()
            inv["status"] = "paid_on_time"
        elif status == "paid_late":
            days_late = late_days_pool[late_idx]
            late_idx += 1
            paid_date = due_date + timedelta(days=days_late)
            inv["paid_date"] = paid_date.isoformat()
            inv["status"] = "paid_late"
            inv["days_late"] = days_late
        else:  # disputed
            days_late = late_days_pool[late_idx]
            late_idx += 1
            inv["paid_date"] = None
            inv["status"] = "disputed"
            inv["days_late"] = days_late

        invoices.append(inv)

    return invoices


# ---------------------------------------------------------------------------
# Helper: compute the invoice amount for the current (overdue) invoice
# given a desired ratio and a typical amount.
# ---------------------------------------------------------------------------

def _current_invoice_amount(typical: int, ratio: float) -> int:
    raw = typical * ratio
    return max(20_000, min(400_000, round(raw / 1000) * 1000))


# ---------------------------------------------------------------------------
# Core debtor generator — guaranteed-tier version
# ---------------------------------------------------------------------------

def _generate_debtor(
    debtor_index: int,
    tier: str,
    outcome: str,
) -> tuple[dict, dict]:
    """
    Generate one (debtor_history, current_invoice) pair that scores into `tier`.
    Retries up to MAX_RETRIES times if scoring disagrees.
    Returns (debtor_history_dict, current_invoice_dict).
    """
    recipe = TIER_RECIPES[tier]
    debtor_id = f"DEBTOR-{debtor_index:03d}"

    # Company identity (stable across retries)
    surname = random.choice(SECTORS)
    suffix = random.choice(COMPANY_SUFFIXES)
    company_name = f"{surname} {suffix}"
    contact_name = fake.name()
    contact_email = f"{surname.lower()}{debtor_index}@{surname.lower().replace(' ', '')}.in"

    for attempt in range(MAX_RETRIES):
        # Sample signals from recipe ranges
        on_time_rate = round(random.uniform(*recipe["on_time_rate"]), 3)
        avg_days_late = round(random.uniform(*recipe["avg_days_late"]), 1)
        dispute_count = random.randint(*recipe["dispute_count"])
        dpd = random.randint(*recipe["dpd"])
        ratio = round(random.uniform(*recipe["invoice_ratio"]), 3)

        # Typical invoice amount for this debtor (used for ratio calc + history)
        typical_amount = random.randint(40, 200) * 1000  # ₹40k–₹2L
        n_hist = random.randint(8, 20)

        historical = _build_historical_invoices(
            debtor_id=debtor_id,
            n=n_hist,
            on_time_rate=on_time_rate,
            avg_days_late=avg_days_late,
            dispute_count=dispute_count,
            typical_amount=typical_amount,
        )

        # Current invoice
        current_amount = _current_invoice_amount(typical_amount, ratio)
        issue_date = TODAY - timedelta(days=dpd + 30)
        due_date = issue_date + timedelta(days=30)

        inv_id = f"INV-{debtor_index:04d}"

        debtor_history: dict = {
            "debtor_id": debtor_id,
            "company_name": company_name,
            "contact_name": contact_name,
            "contact_email": contact_email,
            "historical_invoices": historical,
        }

        current_invoice: dict = {
            "invoice_id": inv_id,
            "debtor_id": debtor_id,
            "amount": current_amount,
            "issue_date": issue_date.isoformat(),
            "due_date": due_date.isoformat(),
            "dpd": dpd,
            "status": "overdue",
            "tier": None,
            "score": None,
            "recovered": 0,
            "negotiation_status": "pending",
            "simulated_outcome": outcome,
        }

        # Verify with the real scoring function
        result = score_debtor(debtor_history, current_invoice)
        if result["tier"] == tier:
            return debtor_history, current_invoice, result

    raise RuntimeError(
        f"Could not generate a {tier}-tier debtor after {MAX_RETRIES} attempts "
        f"(debtor_index={debtor_index}). Check tier recipe bounds."
    )


# ---------------------------------------------------------------------------
# Main: build all 50 debtors
# ---------------------------------------------------------------------------

def generate_all() -> tuple[list[dict], list[dict], dict]:
    """
    Returns (debtors, invoices, batch_summary).
    """
    tier_plan: list[str] = (
        ["A"] * 10 + ["B"] * 15 + ["C"] * 15 + ["D"] * 10
    )
    assert len(tier_plan) == 50

    outcomes = OUTCOME_POOL[:]  # already shuffled

    debtors: list[dict] = []
    invoices: list[dict] = []

    # Verification table rows
    table_rows: list[dict] = []

    print("\n" + "=" * 70)
    print(f"{'DEBTOR_ID':<14} {'INTENDED':>8} {'ACTUAL':>8} {'SCORE':>8}  STATUS")
    print("=" * 70)

    for i, (tier, outcome) in enumerate(zip(tier_plan, outcomes), start=1):
        debtor_history, current_invoice, score_result = _generate_debtor(
            debtor_index=i,
            tier=tier,
            outcome=outcome,
        )
        actual_tier = score_result["tier"]
        score_val = score_result["score"]

        status = "✓" if actual_tier == tier else "✗ MISMATCH"
        print(
            f"{debtor_history['debtor_id']:<14} "
            f"{tier:>8} "
            f"{actual_tier:>8} "
            f"{score_val:>8.2f}  {status}"
        )

        table_rows.append({
            "debtor_id": debtor_history["debtor_id"],
            "intended_tier": tier,
            "actual_tier": actual_tier,
            "score": score_val,
            "match": actual_tier == tier,
        })

        debtors.append(debtor_history)
        invoices.append(current_invoice)

    print("=" * 70)

    mismatches = [r for r in table_rows if not r["match"]]
    if mismatches:
        print(f"\n❌  {len(mismatches)} tier mismatches detected!")
        for r in mismatches:
            print(f"   {r['debtor_id']}: intended {r['intended_tier']}, got {r['actual_tier']} (score={r['score']})")
        sys.exit(1)
    else:
        print(f"\n✅  All 50 debtors verified — tiers match intended distribution.\n")

    # ----- batch_summary -----
    total_outstanding = sum(inv["amount"] for inv in invoices)
    tier_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    for inv in invoices:
        t = next(
            r["actual_tier"] for r in table_rows if r["debtor_id"] == inv["debtor_id"]
        )
        tier_counts[t] = tier_counts.get(t, 0) + 1
        o = inv["simulated_outcome"]
        outcome_counts[o] = outcome_counts.get(o, 0) + 1

    batch_summary = {
        "total_invoices": 50,
        "total_outstanding_amount": total_outstanding,
        "tier_breakdown": {
            "A": tier_counts.get("A", 0),
            "B": tier_counts.get("B", 0),
            "C": tier_counts.get("C", 0),
            "D": tier_counts.get("D", 0),
        },
        "outcome_distribution": {
            "clean_settlement": outcome_counts.get("clean_settlement", 0),
            "dispute":          outcome_counts.get("dispute",          0),
            "repeat_extension": outcome_counts.get("repeat_extension", 0),
        },
        "generated_at": TODAY.isoformat(),
    }

    return debtors, invoices, batch_summary


# ---------------------------------------------------------------------------
# Write JSON helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    kb = path.stat().st_size / 1024
    print(f"  Written → {path}  ({kb:.1f} KB, {len(obj) if isinstance(obj, list) else 1} records)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("RecoverFlow — Synthetic Data Generator")
    print(f"  Repo root : {REPO_ROOT}")
    print(f"  Output dir: {DATA_DIR}")
    print(f"  RNG seed  : {SEED}")

    debtors, invoices, batch_summary = generate_all()

    _write_json(DATA_DIR / "debtors.json", debtors)
    _write_json(DATA_DIR / "invoices.json", invoices)
    _write_json(DATA_DIR / "batch_summary.json", batch_summary)

    print("\nBatch Summary:")
    print(json.dumps(batch_summary, indent=2))
