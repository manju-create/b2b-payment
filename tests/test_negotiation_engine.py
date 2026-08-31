"""
Unit tests for the NegotiationEngine — the single source of truth for numbers.
================================================================================
Invariants:
1. Tier mapping from trust score (A/B/C/D) sets min_pct / installments / gap.
2. All amounts are pre-calculated and rounded to the nearest 100.
3. is_acceptable enforces the min_today floor (20% / 30%).
4. build_plan splits the balance across future dates and rejects dates past
   the 34-day deadline.
5. suggest_dates stays within the deadline.
"""

from datetime import date, timedelta

from backend.negotiation_engine import NegotiationEngine


def test_tier_mapping():
    a = NegotiationEngine(100000, 80)
    assert a.tier == "A"
    assert a.min_pct == 0.20
    assert a.max_installments == 3
    assert a.gap_days == 14

    b = NegotiationEngine(100000, 60)
    assert b.tier == "B"
    assert b.max_installments == 2
    assert b.gap_days == 10

    c = NegotiationEngine(100000, 30)
    assert c.tier == "C"
    assert c.max_installments == 2
    assert c.gap_days == 7

    d = NegotiationEngine(100000, 10)
    assert d.tier == "D"
    assert d.min_pct == 0.30
    assert d.max_installments == 2
    assert d.gap_days == 5


def test_amounts_are_precalculated_and_rounded():
    e = NegotiationEngine(145000, 60)   # Tier B
    assert e.min_today == 29000          # 20% of 145000
    assert e.step1_amount == 72500       # 50%
    assert e.step2_amount == 43500       # 30%
    assert e.step3_amount == 29000       # == min_today
    # rounding to nearest 100
    assert e.min_today % 100 == 0
    assert e.step1_amount % 100 == 0


def test_is_acceptable_enforces_floor():
    e = NegotiationEngine(145000, 60)
    assert e.is_acceptable(29000) is True
    assert e.is_acceptable(50000) is True
    assert e.is_acceptable(28999) is False


def test_tier_d_floor_is_30_percent():
    e = NegotiationEngine(100000, 10)
    assert e.min_today == 30000


def test_apply_hardship_lowers_floor_to_20_percent():
    e = NegotiationEngine(100000, 10)   # Tier D: 30% floor
    assert e.min_today == 30000
    assert e.hardship_min == 20000
    e.apply_hardship()
    assert e.hardship_verified is True
    assert e.min_today == 20000          # dropped to the 20% hardship floor
    assert e.step3_amount == 20000


def test_apply_hardship_noop_when_already_20_percent():
    e = NegotiationEngine(100000, 60)   # Tier B: already 20%
    assert e.min_today == 20000
    e.apply_hardship()
    assert e.min_today == 20000


def test_build_plan_splits_balance():
    e = NegotiationEngine(145000, 60)   # 2 installments (1 future date)
    future = (e.today + timedelta(days=10)).isoformat()
    plan, status = e.build_plan(50000, [future])
    assert status == "ok"
    assert len(plan) == 2
    assert plan[0]["status"] == "pending_payment"
    assert plan[0]["amount"] == 50000
    assert plan[1]["status"] == "scheduled"
    assert plan[1]["amount"] == 95000   # 145000 - 50000
    assert sum(i["amount"] for i in plan) == 145000


def test_build_plan_rejects_date_past_deadline():
    e = NegotiationEngine(145000, 60)
    far = (e.today + timedelta(days=40)).isoformat()
    plan, status = e.build_plan(50000, [far])
    assert plan is None
    assert status.startswith("date_exceeds_deadline")


def test_build_plan_needs_dates():
    e = NegotiationEngine(145000, 60)
    plan, status = e.build_plan(50000, [])
    assert plan is None
    assert status == "need_dates"


def test_build_plan_full_payment_no_dates():
    e = NegotiationEngine(145000, 60)
    plan, status = e.build_plan(145000, [])
    assert status == "ok"
    assert len(plan) == 1
    assert plan[0]["amount"] == 145000
    assert plan[0]["status"] == "pending_payment"


def test_suggest_dates_within_deadline():
    e = NegotiationEngine(100000, 80)   # Tier A: 2 future dates, gap 14
    dates = e.suggest_dates(2)
    assert len(dates) == 2
    for d in dates:
        assert date.fromisoformat(d) <= e.deadline


def test_context_for_agent_has_all_numbers():
    e = NegotiationEngine(145000, 60)
    ctx = e.get_context_for_agent(2, today_offered=40000)
    assert ctx["invoice_amount"] == 145000
    assert ctx["min_today"] == 29000
    assert ctx["step2_ask"] == 43500
    assert ctx["max_installments"] == 2
    assert ctx["gap_days"] == 10
    assert ctx["balance"] == 105000          # 145000 - 40000
    assert ctx["is_acceptable"] is True
    assert ctx["suggested_next_date"] is not None


def test_to_dict_is_json_safe():
    import json
    e = NegotiationEngine(145000, 60)
    json.dumps(e.to_dict())
