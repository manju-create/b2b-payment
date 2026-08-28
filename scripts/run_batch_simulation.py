#!/usr/bin/env python3
"""
RecoverFlow — Batch Simulation Runner
========================================
Runs simulate mode across all 50 invoices and prints a results table.

Usage:
  python scripts/run_batch_simulation.py
  python scripts/run_batch_simulation.py --max 10          # first N invoices only
  python scripts/run_batch_simulation.py --verbose         # show full transcripts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.agent import (  # noqa: E402
    create_session,
    open_turn,
    process_turn,
    simulate_debtor_turn,
    _rupees,
)

DATA_DIR = REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# Result dataclass (plain dict)
# ---------------------------------------------------------------------------

def _run_one(invoice_id: str, verbose: bool) -> dict:
    """
    Run a full simulated negotiation for one invoice.
    Returns a result dict.
    """
    try:
        session = create_session(invoice_id)
    except Exception as exc:
        return {
            "invoice_id": invoice_id,
            "tier": "?",
            "simulated_outcome": "?",
            "turns": 0,
            "status": f"ERROR: {exc}",
            "payment_link": None,
            "amount_recovered_paise": 0,
            "error": str(exc),
        }

    outcome = session["simulated_outcome"]

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  {invoice_id}  |  Tier {session['tier']}  |  {outcome}")
        print(f"{'─'*60}")

    # Opening turn
    try:
        agent_reply, session = open_turn(session)
    except Exception as exc:
        return {
            "invoice_id": invoice_id,
            "tier": session["tier"],
            "simulated_outcome": outcome,
            "turns": 0,
            "status": f"API_ERROR",
            "payment_link": None,
            "amount_recovered_paise": 0,
            "error": str(exc),
        }

    if verbose:
        print(f"  🤖 {agent_reply[:120]}{'...' if len(agent_reply)>120 else ''}")

    for _ in range(session["max_turns"] + 1):
        if session["status"] != "active":
            break

        debtor_msg = simulate_debtor_turn(session)
        if verbose:
            label = "[silent]" if not debtor_msg.strip() else debtor_msg[:100]
            print(f"  👤 {label}")

        try:
            agent_reply, session = process_turn(session, debtor_msg)
        except Exception as exc:
            session["status"] = "escalated"
            session["audit_log"].append({"event": "api_error", "error": str(exc)})
            break

        if verbose:
            print(f"  🤖 {agent_reply[:120]}{'...' if len(agent_reply)>120 else ''}")

    # Calculate recovered amount
    amount_recovered_paise = 0
    if session["status"] in ("settled", "partially_settled") and session.get("agreed_terms"):
        amount_recovered_paise = session["agreed_terms"].get("upfront_amount", 0)
    elif session["status"] == "settled":
        amount_recovered_paise = session["invoice_amount_paise"]

    return {
        "invoice_id": invoice_id,
        "tier": session["tier"],
        "simulated_outcome": outcome,
        "turns": session["turn_count"],
        "status": session["status"],
        "payment_link": session.get("payment_link"),
        "amount_recovered_paise": amount_recovered_paise,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _print_results_table(results: list[dict]) -> None:
    # Header
    print(f"\n{'='*100}")
    header = (
        f"{'Invoice':<12} {'Tier':>5} {'Outcome':<20} "
        f"{'Turns':>6} {'Status':<15} {'Recovered':>14} {'Link'}"
    )
    print(header)
    print('─' * 100)

    total_recovered_paise = 0
    status_counts: dict[str, int] = {}

    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
        total_recovered_paise += r["amount_recovered_paise"]

        recovered_str = _rupees(r["amount_recovered_paise"]) if r["amount_recovered_paise"] else "—"
        link_str = r["payment_link"] or "—"
        # Truncate link for display
        if len(link_str) > 35:
            link_str = link_str[:32] + "..."

        status_icon = {
            "settled":           "✅",
            "partially_settled": "🟡",
            "escalated":         "⬆️ ",
            "disputed":          "⚠️ ",
            "active":            "🔄",
        }.get(r["status"], "❓")

        print(
            f"{r['invoice_id']:<12} "
            f"{r['tier']:>5} "
            f"{r['simulated_outcome']:<20} "
            f"{r['turns']:>6} "
            f"{status_icon} {r['status']:<13} "
            f"{recovered_str:>14}  "
            f"{link_str}"
        )

    print('─' * 100)

    # Summary
    n = len(results)
    settled   = status_counts.get("settled",   0)
    escalated = status_counts.get("escalated", 0)
    disputed  = status_counts.get("disputed",  0)
    active    = status_counts.get("active",    0)
    errors    = sum(1 for r in results if r.get("error"))

    print(f"\n  RESULTS  ({n} invoices)")
    print(f"  ✅  Settled        : {settled:>3}   ({_rupees(total_recovered_paise)} recovered)")
    print(f"  ⬆️   Escalated      : {escalated:>3}")
    print(f"  ⚠️   Disputed       : {disputed:>3}")
    print(f"  🔄  Still active   : {active:>3}")
    if errors:
        print(f"  ❌  Errors         : {errors:>3}")
    print(f"\n  Total recovered   : {_rupees(total_recovered_paise)}")
    if settled:
        avg_paise = total_recovered_paise // settled
        print(f"  Avg per settlement: {_rupees(avg_paise)}")
    print(f"{'='*100}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RecoverFlow — batch simulation across all 50 invoices"
    )
    parser.add_argument("--max", type=int, default=None,
                        help="Max number of invoices to process (default: all 50)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full transcript for each invoice")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="Seconds to wait between invoices (rate limiting)")
    args = parser.parse_args()

    # Load invoice list
    invoices_path = DATA_DIR / "invoices.json"
    if not invoices_path.exists():
        print("ERROR: data/invoices.json not found. Run scripts/seed_data.py first.")
        sys.exit(1)

    all_invoices = json.loads(invoices_path.read_text())
    invoice_ids = [inv["invoice_id"] for inv in all_invoices]

    if args.max:
        invoice_ids = invoice_ids[: args.max]

    print(f"\nRecoverFlow — Batch Simulation")
    print(f"Processing {len(invoice_ids)} invoices...")
    print("(Each negotiation makes real DeepSeek API calls — this may take a few minutes)\n")

    results: list[dict] = []
    for i, inv_id in enumerate(invoice_ids, 1):
        print(f"  [{i:02d}/{len(invoice_ids)}] {inv_id} ...", end="", flush=True)
        t0 = time.time()
        result = _run_one(inv_id, verbose=args.verbose)
        elapsed = time.time() - t0

        status_label = result["status"]
        recovered = (
            f"  {_rupees(result['amount_recovered_paise'])}"
            if result["amount_recovered_paise"]
            else ""
        )
        print(f" {status_label.upper():<12}{recovered}  ({elapsed:.1f}s)")

        results.append(result)

        if i < len(invoice_ids) and args.delay > 0:
            time.sleep(args.delay)

    _print_results_table(results)

    # Write results to data/
    out_path = DATA_DIR / "batch_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
