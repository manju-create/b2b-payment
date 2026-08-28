#!/usr/bin/env python3
"""
RecoverFlow — Single-Invoice Negotiation Runner
================================================
Usage:
  python scripts/run_negotiation.py --invoice INV-0001 --mode interactive
  python scripts/run_negotiation.py --invoice INV-0001 --mode simulate

Modes:
  interactive  — Agent opens, then reads debtor replies from stdin
  simulate     — Uses simulate_debtor_turn() for automated testing;
                 prints full transcript + audit log at the end
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.agent import (  # noqa: E402
    create_session,
    open_turn,
    process_turn,
    simulate_debtor_turn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIVIDER = "─" * 60


def _print_agent(text: str) -> None:
    print(f"\n🤖  AGENT\n{DIVIDER}")
    for line in text.strip().splitlines():
        print(f"  {line}")
    print(DIVIDER)


def _print_debtor(text: str) -> None:
    if not text.strip():
        print(f"\n👤  DEBTOR  [silent / no response]")
    else:
        print(f"\n👤  DEBTOR\n{DIVIDER}")
        for line in text.strip().splitlines():
            print(f"  {line}")
        print(DIVIDER)


def _print_status(session: dict) -> None:
    print(f"\n  ℹ  Turn {session['turn_count']}/{session['max_turns']}  |  "
          f"Status: {session['status'].upper()}  |  "
          f"Tier: {session['tier']}  |  Score: {session['score']:.1f}")


def _print_audit(session: dict) -> None:
    print(f"\n{'=' * 60}")
    print("AUDIT LOG")
    print('=' * 60)
    print(json.dumps(session["audit_log"], indent=2))
    if session.get("payment_link"):
        print(f"\n💳  Payment link: {session['payment_link']}")
    if session.get("agreed_terms"):
        print(f"\n📋  Agreed terms: {json.dumps(session['agreed_terms'], indent=2)}")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(invoice_id: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  RecoverFlow — Interactive Negotiation")
    print(f"  Invoice: {invoice_id}")
    print(f"{'=' * 60}")
    print("  Type your debtor replies. Press Ctrl+C or type 'quit' to exit.")
    print(f"{'=' * 60}\n")

    session = create_session(invoice_id)
    print(f"  Tier: {session['tier']}  |  Score: {session['score']:.1f}  |  "
          f"Invoice: ₹{session['invoice_amount_paise']//100:,}")

    # Agent opens
    agent_reply, session = open_turn(session)
    _print_agent(agent_reply)

    while session["status"] == "active":
        try:
            debtor_input = input("\n👤  You (debtor): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[Session ended by user]")
            break

        if debtor_input.lower() in ("quit", "exit", "q"):
            print("\n[Exiting]")
            break

        agent_reply, session = process_turn(session, debtor_input)
        _print_agent(agent_reply)
        _print_status(session)

    print(f"\n\n  Final status: {session['status'].upper()}")
    _print_audit(session)


# ---------------------------------------------------------------------------
# Simulate mode
# ---------------------------------------------------------------------------

def run_simulate(invoice_id: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  RecoverFlow — Simulated Negotiation")
    print(f"  Invoice: {invoice_id}")
    print(f"{'=' * 60}\n")

    session = create_session(invoice_id)
    outcome = session["simulated_outcome"]
    print(f"  Tier: {session['tier']}  |  Score: {session['score']:.1f}  |  "
          f"Invoice: ₹{session['invoice_amount_paise']//100:,}  |  "
          f"Simulated outcome: {outcome}")
    print()

    # Agent opens
    agent_reply, session = open_turn(session)
    _print_agent(agent_reply)

    for _ in range(session["max_turns"] + 1):
        if session["status"] != "active":
            break

        debtor_msg = simulate_debtor_turn(session)
        _print_debtor(debtor_msg)

        agent_reply, session = process_turn(session, debtor_msg)
        _print_agent(agent_reply)
        _print_status(session)

    print(f"\n\n{'=' * 60}")
    print(f"  NEGOTIATION COMPLETE")
    print(f"  Final status : {session['status'].upper()}")
    print(f"  Turns used   : {session['turn_count']}")
    if session.get("payment_link"):
        print(f"  Payment link : {session['payment_link']}")
    if session.get("agreed_terms"):
        print(f"  Agreed terms : {json.dumps(session['agreed_terms'])}")
    print(f"{'=' * 60}")

    _print_audit(session)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RecoverFlow — single invoice negotiation runner"
    )
    parser.add_argument("--invoice", required=True,
                        help="Invoice ID to negotiate (e.g. INV-0001)")
    parser.add_argument("--mode", choices=["interactive", "simulate"],
                        default="interactive",
                        help="interactive = human debtor; simulate = automated")
    args = parser.parse_args()

    if args.mode == "interactive":
        run_interactive(args.invoice)
    else:
        run_simulate(args.invoice)


if __name__ == "__main__":
    main()
