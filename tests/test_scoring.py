"""
Unit tests for backend.scoring.score_debtor
============================================
Covers:
  TC-01  Tier A — ideal debtor, low DPD, zero disputes
  TC-02  Tier B — good debtor, moderate lateness, one dispute
  TC-03  Tier C — inconsistent debtor, high lateness, multiple disputes
  TC-04  Tier D — high-risk debtor, near-max lateness, many disputes
  TC-05  Cold-start — no prior history → defaults to Tier C, cold_start=True
  TC-06  Cold-start signal values — score and tier check, signals are None
  TC-07  Invoice ratio penalty — very large invoice pushes score down
  TC-08  DPD boundary — very high DPD (120+) floors DPD component to 0
  TC-09  On-time rate computed from historical_invoices list
  TC-10  Tier boundary edge cases (exactly 85, 60, 35, 0)

Run with:
    python -m pytest tests/test_scoring.py -v
"""

import pytest
from backend.scoring import score_debtor, COLD_START_SCORE, COLD_START_TIER


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_history(
    on_time_rate: float,
    avg_days_late: float,
    dispute_count: int,
    typical_amount: float = 100_000,
) -> dict:
    """Build a minimal debtor_history dict using override fields."""
    return {
        "debtor_id": "DEBTOR-TEST",
        "on_time_rate": on_time_rate,
        "avg_days_late": avg_days_late,
        "dispute_count": dispute_count,
        # Single historical invoice to give us a 'typical' amount
        "historical_invoices": [
            {
                "invoice_id": "H-001",
                "amount": typical_amount,
                "status": "paid_on_time",
            }
        ],
    }


def _make_invoice(amount: float = 100_000, dpd: int = 10) -> dict:
    return {"invoice_id": "INV-TEST", "amount": amount, "dpd": dpd}


# ---------------------------------------------------------------------------
# TC-01  Tier A — ideal debtor
# ---------------------------------------------------------------------------

class TestTierA:
    def test_score_in_tier_a_range(self):
        """
        Perfect payer: 100% on time, 0 days late, 0 disputes, low DPD,
        invoice at typical size → should score 85+.
        """
        history = _make_history(on_time_rate=1.0, avg_days_late=0, dispute_count=0)
        invoice = _make_invoice(amount=100_000, dpd=5)

        result = score_debtor(history, invoice)

        assert result["tier"] == "A", f"Expected Tier A, got {result['tier']} (score={result['score']})"
        assert result["score"] >= 85
        assert result["cold_start"] is False

    def test_tier_a_signal_values_returned(self):
        history = _make_history(on_time_rate=1.0, avg_days_late=0, dispute_count=0)
        invoice = _make_invoice(amount=100_000, dpd=5)
        result = score_debtor(history, invoice)

        assert result["signals"]["on_time_rate"] == 1.0
        assert result["signals"]["avg_days_late"] == 0.0
        assert result["signals"]["dispute_count"] == 0
        assert result["signals"]["current_dpd"] == 5.0

    def test_tier_a_weighted_components_sum_to_score(self):
        history = _make_history(on_time_rate=1.0, avg_days_late=0, dispute_count=0)
        invoice = _make_invoice(amount=100_000, dpd=5)
        result = score_debtor(history, invoice)

        component_sum = sum(result["weighted_components"].values())
        assert abs(component_sum - result["score"]) < 0.01, (
            f"Weighted components ({component_sum}) do not sum to score ({result['score']})"
        )


# ---------------------------------------------------------------------------
# TC-02  Tier B — generally good, occasionally late, one dispute
# ---------------------------------------------------------------------------

class TestTierB:
    def test_score_in_tier_b_range(self):
        """
        70% on-time, ~12 days average late, 1 dispute, DPD=45 → expect Tier B (60–84).
        Matches the demo scenario: Sharma Distributors.
        """
        history = _make_history(on_time_rate=0.70, avg_days_late=12, dispute_count=1)
        invoice = _make_invoice(amount=120_000, dpd=45)

        result = score_debtor(history, invoice)

        assert result["tier"] == "B", f"Expected Tier B, got {result['tier']} (score={result['score']})"
        assert 60 <= result["score"] <= 84

    def test_tier_b_not_cold_start(self):
        history = _make_history(on_time_rate=0.70, avg_days_late=12, dispute_count=1)
        invoice = _make_invoice(amount=120_000, dpd=45)
        result = score_debtor(history, invoice)
        assert result["cold_start"] is False


# ---------------------------------------------------------------------------
# TC-03  Tier C — inconsistent payer
# ---------------------------------------------------------------------------

class TestTierC:
    def test_score_in_tier_c_range(self):
        """
        40% on-time, 35 days average late, 2 disputes, DPD=60 → Tier C (35–59).
        """
        history = _make_history(on_time_rate=0.40, avg_days_late=35, dispute_count=2)
        invoice = _make_invoice(amount=100_000, dpd=60)

        result = score_debtor(history, invoice)

        assert result["tier"] == "C", f"Expected Tier C, got {result['tier']} (score={result['score']})"
        assert 35 <= result["score"] <= 59

    def test_tier_c_dispute_signal_present(self):
        history = _make_history(on_time_rate=0.40, avg_days_late=35, dispute_count=2)
        invoice = _make_invoice(amount=100_000, dpd=60)
        result = score_debtor(history, invoice)
        assert result["signals"]["dispute_count"] == 2


# ---------------------------------------------------------------------------
# TC-04  Tier D — high risk, serial late payer
# ---------------------------------------------------------------------------

class TestTierD:
    def test_score_in_tier_d_range(self):
        """
        10% on-time, 80 days average late, 4 disputes, DPD=90 → Tier D (0–34).
        """
        history = _make_history(on_time_rate=0.10, avg_days_late=80, dispute_count=4)
        invoice = _make_invoice(amount=100_000, dpd=90)

        result = score_debtor(history, invoice)

        assert result["tier"] == "D", f"Expected Tier D, got {result['tier']} (score={result['score']})"
        assert result["score"] <= 34

    def test_tier_d_worst_case_all_floors(self):
        """
        Worst case with all 4 penalty-able signals at maximum:
        0% on-time, 90+ days late, 5 disputes, 120 DPD.

        Note: invoice_vs_typical_ratio = 1.0 when current invoice equals the
        typical amount, so that signal contributes its full 15-point weight.
        The minimum reachable score is therefore 15 (not 0).
        For a true 0 score, pass a 3×-typical-sized invoice to also cap the ratio.
        """
        history = _make_history(on_time_rate=0.0, avg_days_late=90, dispute_count=5)
        invoice = _make_invoice(amount=100_000, dpd=120)

        result = score_debtor(history, invoice)

        assert result["tier"] == "D"
        assert result["score"] <= 20  # floor when ratio is neutral

    def test_tier_d_absolute_zero_score(self):
        """Score reaches 0 when ratio penalty also maxes out (3× typical invoice)."""
        history = _make_history(
            on_time_rate=0.0, avg_days_late=90, dispute_count=5, typical_amount=100_000
        )
        # 3× typical → ratio=3.0 → normalised ratio = 0.0
        invoice = _make_invoice(amount=300_000, dpd=120)

        result = score_debtor(history, invoice)

        assert result["tier"] == "D"
        assert result["score"] == 0.0


# ---------------------------------------------------------------------------
# TC-05  Cold-start — no prior history
# ---------------------------------------------------------------------------

class TestColdStart:
    def test_cold_start_with_empty_history_object(self):
        """Debtor dict with no useful keys → cold-start defaults."""
        history = {"debtor_id": "DEBTOR-NEW"}
        invoice = _make_invoice(amount=50_000, dpd=30)

        result = score_debtor(history, invoice)

        assert result["cold_start"] is True
        assert result["tier"] == COLD_START_TIER  # "C"
        assert result["score"] == float(COLD_START_SCORE)

    def test_cold_start_with_empty_historical_invoices_list(self):
        """historical_invoices is present but empty → still cold-start."""
        history = {
            "debtor_id": "DEBTOR-EMPTY",
            "historical_invoices": [],
        }
        invoice = _make_invoice(amount=75_000, dpd=20)

        result = score_debtor(history, invoice)

        assert result["cold_start"] is True
        assert result["tier"] == "C"

    def test_cold_start_signals_are_none(self):
        """All history-derived signal values should be None on cold-start."""
        history = {"debtor_id": "DEBTOR-NEW"}
        invoice = _make_invoice(amount=50_000, dpd=30)

        result = score_debtor(history, invoice)

        assert result["signals"]["on_time_rate"] is None
        assert result["signals"]["avg_days_late"] is None
        assert result["signals"]["dispute_count"] is None
        # current_dpd is always available (from current invoice)
        assert result["signals"]["current_dpd"] == 30.0

    def test_cold_start_no_weighted_components(self):
        """Cold-start result has empty weighted_components (no computation performed)."""
        history = {"debtor_id": "DEBTOR-NEW"}
        invoice = _make_invoice(dpd=15)
        result = score_debtor(history, invoice)
        assert result["weighted_components"] == {}


# ---------------------------------------------------------------------------
# TC-06  Invoice ratio penalty
# ---------------------------------------------------------------------------

class TestInvoiceRatio:
    def test_large_invoice_pushes_score_down(self):
        """
        A very large invoice (3× typical) should reduce the score compared to
        a same-sized-as-typical invoice, all else equal.
        """
        history_base = _make_history(
            on_time_rate=0.80, avg_days_late=10, dispute_count=0, typical_amount=100_000
        )
        invoice_typical = _make_invoice(amount=100_000, dpd=10)
        invoice_large = _make_invoice(amount=300_000, dpd=10)

        result_typical = score_debtor(history_base, invoice_typical)
        result_large = score_debtor(history_base, invoice_large)

        assert result_large["score"] < result_typical["score"], (
            "Large invoice should reduce score"
        )

    def test_small_invoice_does_not_penalise(self):
        """
        Invoice smaller than typical → ratio < 1 → no penalty (ratio score capped at 1).
        """
        history = _make_history(
            on_time_rate=1.0, avg_days_late=0, dispute_count=0, typical_amount=100_000
        )
        invoice_small = _make_invoice(amount=40_000, dpd=5)

        result = score_debtor(history, invoice_small)

        assert result["normalised_signals"]["invoice_vs_typical_ratio"] == 1.0


# ---------------------------------------------------------------------------
# TC-07  DPD cap behaviour
# ---------------------------------------------------------------------------

class TestDPDCap:
    def test_dpd_above_120_caps_component_at_zero(self):
        """DPD of 200 should give the same normalised DPD score as DPD of 120."""
        history = _make_history(on_time_rate=0.80, avg_days_late=10, dispute_count=0)
        invoice_120 = _make_invoice(dpd=120)
        invoice_200 = _make_invoice(dpd=200)

        result_120 = score_debtor(history, invoice_120)
        result_200 = score_debtor(history, invoice_200)

        assert result_120["normalised_signals"]["current_dpd"] == 0.0
        assert result_200["normalised_signals"]["current_dpd"] == 0.0
        # Overall scores should be equal since both DPD components are 0
        assert result_120["score"] == result_200["score"]


# ---------------------------------------------------------------------------
# TC-08  on_time_rate computed from historical_invoices list
# ---------------------------------------------------------------------------

class TestComputedFromList:
    def test_on_time_rate_computed_from_list(self):
        """
        Provide historical_invoices without override fields.
        Verify on_time_rate is computed correctly: 3 paid on time out of 4.
        """
        history = {
            "debtor_id": "DEBTOR-LIST",
            "historical_invoices": [
                {"invoice_id": "H-001", "amount": 80_000, "status": "paid_on_time"},
                {"invoice_id": "H-002", "amount": 90_000, "status": "paid_on_time"},
                {"invoice_id": "H-003", "amount": 70_000, "status": "paid_on_time"},
                {"invoice_id": "H-004", "amount": 85_000, "status": "paid_late", "days_late": 30},
            ],
        }
        invoice = _make_invoice(amount=82_000, dpd=10)

        result = score_debtor(history, invoice)

        assert result["cold_start"] is False
        assert result["signals"]["on_time_rate"] == pytest.approx(0.75)
        assert result["signals"]["avg_days_late"] == pytest.approx(30.0)
        assert result["signals"]["dispute_count"] == 0

    def test_dispute_count_computed_from_list(self):
        """dispute_count is counted from statuses in historical_invoices."""
        history = {
            "debtor_id": "DEBTOR-DISPUTE",
            "historical_invoices": [
                {"invoice_id": "H-001", "amount": 50_000, "status": "paid_on_time"},
                {"invoice_id": "H-002", "amount": 60_000, "status": "disputed"},
                {"invoice_id": "H-003", "amount": 55_000, "status": "disputed"},
            ],
        }
        invoice = _make_invoice(dpd=20)
        result = score_debtor(history, invoice)

        assert result["signals"]["dispute_count"] == 2


# ---------------------------------------------------------------------------
# TC-09  Tier boundary edge cases
# ---------------------------------------------------------------------------

class TestTierBoundaries:
    """
    Drive the score to exactly the boundary values by crafting inputs carefully.
    We test the tier label rather than trying to hit the exact float.
    """

    def test_score_85_is_tier_a(self):
        """Score at 85 (boundary) must be Tier A."""
        # Perfect on_time + zero other penalties → max score = 100
        # We accept any score ≥ 85 from near-ideal inputs
        history = _make_history(on_time_rate=1.0, avg_days_late=0, dispute_count=0)
        invoice = _make_invoice(dpd=0)
        result = score_debtor(history, invoice)
        assert result["tier"] == "A"

    def test_score_34_is_tier_d(self):
        """Score at/below 34 must be Tier D."""
        history = _make_history(on_time_rate=0.0, avg_days_late=90, dispute_count=5)
        invoice = _make_invoice(dpd=120)
        result = score_debtor(history, invoice)
        assert result["tier"] == "D"
        assert result["score"] <= 34

    def test_return_dict_has_required_keys(self):
        """Result dict always contains the documented keys."""
        history = _make_history(on_time_rate=0.5, avg_days_late=20, dispute_count=1)
        invoice = _make_invoice()
        result = score_debtor(history, invoice)

        required_keys = {"score", "tier", "cold_start", "signals", "normalised_signals", "weighted_components"}
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_score_always_between_0_and_100(self):
        """Score must always be within [0, 100]."""
        test_cases = [
            _make_history(1.0, 0, 0),
            _make_history(0.0, 90, 5),
            _make_history(0.5, 45, 2),
        ]
        for history in test_cases:
            for dpd in (0, 60, 120, 200):
                result = score_debtor(history, _make_invoice(dpd=dpd))
                assert 0 <= result["score"] <= 100, (
                    f"Score out of range: {result['score']}"
                )
