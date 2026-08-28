# RecoverFlow — Claude Code Context

## What this project is
AI-mediated B2B debt recovery agent. Debtors receive a chat link, 
negotiate payment terms with an AI, and pay via Razorpay payment link 
generated in the chat. Merchant sees results on a dashboard.

Full PRD is in PRD.md — read it before making architectural decisions.

## Tech stack
- Backend: FastAPI (Python)
- Database: PostgreSQL via Supabase
- LLM: Anthropic Claude Sonnet via API (tool-calling)
- Payments: Razorpay Python SDK
- Frontend: React + Tailwind (merchant dashboard), plain HTML (debtor chat)
- PDF: reportlab
- Synthetic data: faker + custom distribution logic

## Core data models (source of truth)
See PRD.md Section 9 for full schemas.
Key tables: invoices, debtors, debtor_history, audit_log, negotiation_turns

## Architecture
Invoice ingested → Scoring engine (5-signal weighted model → Tier A/B/C/D) 
→ Negotiation agent (system prompt parameterised by tier) 
→ Razorpay payment link generated in chat 
→ Webhook listener confirms payment.captured 
→ Dashboard updates in real-time

## Scoring model (do not change without reading PRD)
5 signals, weighted:
- on_time_rate: 35%
- avg_days_late: 25%
- invoice_vs_typical_ratio: 15%
- current_dpd: 15%
- dispute_count: 10%
Score 0-100 → Tier A (85-100), B (60-84), C (35-59), D (0-34)

## Tier guardrails (hard rules — agent cannot breach these)
| Tier | Pay now (min) | Deferred (max) | Timeline | Discount |
|------|--------------|----------------|----------|----------|
| A    | 25%          | 75%            | 60 days  | 15%      |
| B    | 40%          | 60%            | 45 days  | 10%      |
| C    | 60%          | 40%            | 30 days  | 5%       |
| D    | 85%          | 15%            | 15 days  | 0%       |
Hard floor: every debtor pays minimum 25% now. No exceptions.

## What NOT to build
- No ERP integrations (Tally, SAP, Oracle)
- No WhatsApp integration
- No real ML model (weighted scoring only)
- No voice calling
- No mobile app
- No multi-language support

## Build sequence (follow this order strictly)
1. Scoring function (pure, unit-testable)
2. Synthetic data generator (50 invoices, realistic failure modes)
3. Negotiation agent + tier guardrail enforcement
4. Razorpay payment link generation
5. Webhook listener (payment.captured → invoice status update)
6. Merchant dashboard
7. L3 PDF generator
8. Audit log

## Critical constraints
- Webhook loop MUST be built before dashboard work starts
- Synthetic data MUST include: 40% non-responders, 20% ghosters, 15% disputers
- Agent system prompt is parameterised by tier at session init — not hardcoded
- Webhook signature must be verified (Razorpay sends X-Razorpay-Signature header)
- Idempotency: duplicate webhook events must not double-count payments

## Demo scenario (everything should serve this moment)
Judge plays debtor with a ₹80,000 Tier B invoice. 
They lowball. Agent holds tier line. Settlement agreed. 
Razorpay link appears in chat. Judge pays. 
Merchant dashboard updates live: "₹32,000 confirmed."
That is the demo. Build toward it.