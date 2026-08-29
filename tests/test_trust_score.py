"""
Unit tests for the additive trust-score engine (backend.scoring.calculate_trust_score).
=======================================================================================

Invariants:
1. Cold start (no history) starts at 50
2. Score is always clamped to 0-100
3. Tier mapping: A 75-100, B 50-74, C 25-49, D 0-24
4. negotiation_flex.min_acceptance_pct / tone follow the tier
5. Historical signal points match the spec bands

Run with:
    python -m pytest tests/test_trust_score.py -v
"""

from backend.scoring import calculate_trust_score, _trust_tier


def _hist(on_time=0, late=0, disputed=0, amount=1000):
    """Build a debtor_history with the given mix of historical invoices."""
    invoices = []
    for _ in range(on_time):
        invoices.append({
            "status": "paid_on_time", "amount": amount,
            "due_date": "2024-01-01", "paid_date": "2024-01-01",
        })
    for _ in range(late):
        invoices.append({
            "status": "paid_late", "amount": amount, "days_late": 10,
        })
    for _ in range(disputed):
        invoices.append({
            "status": "disputed", "amount": amount, "days_late": 30,
        })
    return {"historical_invoices": invoices}


def test_cold_start_is_50():
    result = calculate_trust_score({}, {"amount": 1000, "dpd": 0}, {})
    assert result["score"] == 50


def test_cold_start_dpd_penalty():
    result = calculate_trust_score({}, {"amount": 1000, "dpd": 50}, {})
    assert result["score"] == 30  # 50 base - 20 (31-60 DPD)


def test_score_never_exceeds_100():
    hist = _hist(on_time=10)  # excellent payer
    result = calculate_trust_score(hist, {"amount": 1000, "dpd": 0}, {})
    assert 0 <= result["score"] <= 100


def test_score_never_below_0():
    # serial late payer, disputed, huge invoice, very overdue, rejected offers
    hist = _hist(late=5, disputed=3, amount=1000)
    session = {"offers_rejected": 3, "negotiated_down": True}
    result = calculate_trust_score(hist, {"amount": 5000, "dpd": 90}, session)
    assert result["score"] == 0


def test_tier_boundaries():
    assert _trust_tier(100)["tier"] == "A"
    assert _trust_tier(75)["tier"] == "A"
    assert _trust_tier(74)["tier"] == "B"
    assert _trust_tier(50)["tier"] == "B"
    assert _trust_tier(49)["tier"] == "C"
    assert _trust_tier(25)["tier"] == "C"
    assert _trust_tier(24)["tier"] == "D"
    assert _trust_tier(0)["tier"] == "D"


def test_negotiation_flex_matches_tier():
    assert _trust_tier(80) == {"tier": "A", "min_acceptance_pct": 0.85, "tone": "collegial"}
    assert _trust_tier(60) == {"tier": "B", "min_acceptance_pct": 0.70, "tone": "professional"}
    assert _trust_tier(30) == {"tier": "C", "min_acceptance_pct": 0.60, "tone": "formal"}
    assert _trust_tier(5) == {"tier": "D", "min_acceptance_pct": 1.00, "tone": "legal"}


def test_on_time_rate_bands():
    assert calculate_trust_score(_hist(on_time=1), {"amount": 1000, "dpd": 0}, {})["signals"]["on_time_rate"] == 30   # 100%
    # 60% on-time → +20
    hist = _hist(on_time=3, late=2)
    assert calculate_trust_score(hist, {"amount": 1000, "dpd": 0}, {})["signals"]["on_time_rate"] == 20
    # 40% on-time → +10
    hist = _hist(on_time=2, late=3)
    assert calculate_trust_score(hist, {"amount": 1000, "dpd": 0}, {})["signals"]["on_time_rate"] == 10
    # < 40% → +0
    hist = _hist(on_time=1, late=4)
    assert calculate_trust_score(hist, {"amount": 1000, "dpd": 0}, {})["signals"]["on_time_rate"] == 0


def test_dispute_history_bands():
    assert calculate_trust_score(_hist(on_time=5), {"amount": 1000, "dpd": 0}, {})["signals"]["dispute_history"] == 15
    assert calculate_trust_score(_hist(on_time=4, disputed=1), {"amount": 1000, "dpd": 0}, {})["signals"]["dispute_history"] == -10
    assert calculate_trust_score(_hist(on_time=3, disputed=2), {"amount": 1000, "dpd": 0}, {})["signals"]["dispute_history"] == -20


def test_live_signal_points():
    # voluntary partial offer (no agent asking) → +10
    assert calculate_trust_score(
        _hist(on_time=5), {"amount": 1000, "dpd": 0}, {"voluntary_partial_offered": True}
    )["signals"]["voluntary_partial_offer"] == 10

    # partial accepted after agent suggested → +5
    assert calculate_trust_score(
        _hist(on_time=5), {"amount": 1000, "dpd": 0}, {"partial_after_suggested": True}
    )["signals"]["voluntary_partial_offer"] == 5

    # accepted first offer → +5
    assert calculate_trust_score(
        _hist(on_time=5), {"amount": 1000, "dpd": 0}, {"accepted_first_offer": True}
    )["signals"]["negotiation_behaviour"] == 5

    # rejected 2+ offers → -10
    assert calculate_trust_score(
        _hist(on_time=5), {"amount": 1000, "dpd": 0}, {"offers_rejected": 2}
    )["signals"]["negotiation_behaviour"] == -10
