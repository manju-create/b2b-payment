"""
NegotiationEngine — the single source of truth for all negotiation numbers.
================================================================================
The agent (DeepSeek) NEVER calculates anything. Python owns every number and
every decision; the engine pre-computes the amounts, the installment structure,
and the deadline, and hands them to the agent as plain values.

Amounts are in RUPEES throughout this module (not paise).
"""

from __future__ import annotations

from datetime import date, timedelta


class NegotiationEngine:
    def __init__(self, invoice_amount, trust_score):
        self.invoice_amount = invoice_amount
        self.trust_score = trust_score
        self.today = date.today()
        self.deadline = self.today + timedelta(days=34)

        # Calculate tier from trust score
        if trust_score >= 75:
            self.tier = "A"
            self.max_installments = 3
            self.gap_days = 14
            self.min_pct = 0.20
        elif trust_score >= 50:
            self.tier = "B"
            self.max_installments = 2
            self.gap_days = 10
            self.min_pct = 0.20
        elif trust_score >= 25:
            self.tier = "C"
            self.max_installments = 2
            self.gap_days = 7
            self.min_pct = 0.20
        else:
            self.tier = "D"
            self.max_installments = 2
            self.gap_days = 5
            self.min_pct = 0.30

        # Pre-calculate all amounts — round to nearest 100
        self.min_today = self._round(invoice_amount * self.min_pct)
        self.step1_amount = self._round(invoice_amount * 0.50)
        self.step2_amount = self._round(invoice_amount * 0.30)
        self.step3_amount = self.min_today

        # Hardship floor (20% for everyone) — applied only after the debtor's
        # inability-to-pay proof is verified. Until then it is unused.
        self.hardship_min = self._round(invoice_amount * 0.20)
        self.hardship_verified = False

    def _round(self, amount):
        return round(amount / 100) * 100

    def apply_hardship(self):
        """Lower the floor to the hardship minimum (20%) after verified proof."""
        self.hardship_verified = True
        self.min_today = self.hardship_min
        self.step3_amount = self.min_today
        return self.min_today

    def is_acceptable(self, offered_amount):
        """Is this amount acceptable for today's payment?"""
        return offered_amount >= self.min_today

    def build_plan(self, today_amount, future_dates):
        """
        Given today's amount and list of future dates the debtor agreed to,
        build the installment plan.

        future_dates: list of date strings the debtor confirmed
        Returns: (installments, "ok") or (None, error) if a date exceeds deadline
        """
        balance = self.invoice_amount - today_amount
        installments = [
            {
                "date": str(self.today),
                "amount": today_amount,
                "label": "Today",
                "status": "pending_payment",
            }
        ]

        # Full payment (or overpayment) settles the invoice today — no dates.
        if balance <= 0:
            return installments, "ok"

        if not future_dates:
            return None, "need_dates"

        remaining_installments = self.max_installments - 1

        if len(future_dates) > remaining_installments:
            future_dates = future_dates[:remaining_installments]

        # Validate no date exceeds deadline
        for d in future_dates:
            parsed = date.fromisoformat(d)
            if parsed > self.deadline:
                return None, f"date_exceeds_deadline_{self.deadline}"

        # Split balance equally across future dates
        per_installment = self._round(balance / len(future_dates))

        for i, d in enumerate(future_dates):
            amount = per_installment if i < len(future_dates) - 1 \
                else balance - (per_installment * (len(future_dates) - 1))
            installments.append({
                "date": d,
                "amount": amount,
                "label": f"Payment {i + 2}",
                "status": "scheduled",
            })

        return installments, "ok"

    def suggest_dates(self, num_payments, today_amount=None):
        """
        Auto-suggest future payment dates based on gap_days.
        Used when the debtor hasn't given specific dates yet.
        """
        dates = []
        base = self.today
        for _ in range(num_payments):
            next_date = base + timedelta(days=self.gap_days)
            if next_date > self.deadline:
                next_date = self.deadline
            dates.append(str(next_date))
            base = next_date
        return dates

    def get_context_for_agent(self, step, today_offered=None):
        """
        Returns a clean dict of numbers for the agent prompt.
        The agent reads these — never calculates.
        """
        suggested_dates = self.suggest_dates(
            self.max_installments - 1,
            today_offered or self.min_today,
        )

        return {
            "invoice_amount": self.invoice_amount,
            "min_today": self.min_today,
            "step1_ask": self.step1_amount,
            "step2_ask": self.step2_amount,
            "step3_ask": self.step3_amount,
            "current_step": step,
            "max_installments": self.max_installments,
            "gap_days": self.gap_days,
            "deadline": str(self.deadline),
            "suggested_next_date": suggested_dates[0] if suggested_dates else None,
            "suggested_final_date": suggested_dates[-1] if suggested_dates else None,
            "today_offered": today_offered,
            "balance": self.invoice_amount - (today_offered or 0),
            "is_acceptable": self.is_acceptable(today_offered) if today_offered else False,
        }

    def to_dict(self) -> dict:
        """Serialized snapshot for session storage (session must stay JSON-safe)."""
        return {
            "invoice_amount": self.invoice_amount,
            "trust_score": self.trust_score,
            "tier": self.tier,
            "min_pct": self.min_pct,
            "min_today": self.min_today,
            "step1_amount": self.step1_amount,
            "step2_amount": self.step2_amount,
            "step3_amount": self.step3_amount,
            "max_installments": self.max_installments,
            "gap_days": self.gap_days,
            "today": str(self.today),
            "deadline": str(self.deadline),
            "hardship_min": self.hardship_min,
            "hardship_verified": self.hardship_verified,
        }
