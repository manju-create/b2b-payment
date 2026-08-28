# RecoverFlow — Full Product Document
### Razorpay AI Buildathon · Track 03: AI Revenue Recovery

---

## 0. The One-Line Pitch

> **RecoverFlow** is an AI agent that negotiates payment settlements with overdue B2B debtors in real-time — offering personalised terms based on their payment history rating, generating a Razorpay payment link inside the chat, and confirming recovered money on the merchant's dashboard.

This is not dunning. Dunning talks *at* debtors. RecoverFlow talks *with* them.

---

## 1. The Problem

### What is happening in Indian B2B trade

Every Indian SMB that sells on credit — distributor, wholesaler, SaaS company, manufacturer, agency — has a receivables problem. They issue invoices with 30–60 day payment terms. A significant percentage of those invoices go overdue. Cash does not arrive. The business borrows to cover the gap, strains supplier relationships, and eventually writes off debt it should have collected.

**The numbers are brutal:**
- Indian SMB DSO (Days Sales Outstanding) averages 60–90 days vs a healthy 30–45
- Approximately ₹10.7 lakh crore in B2B invoices are overdue in India at any given time
- Collections agencies charge 15–30% of the recovered amount
- Manual calling is relationship-damaging, embarrassing, and unscalable

### Why the current tools fail

Every existing solution is a **one-way dunning sequence**:
- Send email reminder on Day 1
- Send SMS on Day 7
- Send escalation email on Day 14
- Transfer to collections agency

None of these allow the debtor to respond, negotiate, or reach a settlement in the same session. The debtor gets a demand. They feel no urgency. They defer.

**The core gap:** There is no product on the market — not Kolleno, not Credflow, not HighRadius — that allows a debtor to have a real-time conversation with an AI, receive a personalised payment offer based on their history, negotiate the terms, and pay immediately in the same session.

**That gap is RecoverFlow.**

---

## 2. The Core Insight

The bottleneck in B2B collections is not *reaching* the debtor.
The bottleneck is the **negotiation back-and-forth** — which currently takes 2–4 weeks of calls, emails, and follow-ups.

If an AI can compress that negotiation into a 10-minute chat session, with a payment link at the end, you convert "outstanding receivable" into "confirmed cash" in under 15 minutes.

The second insight: **not all debtors are the same risk.**

A debtor who has paid 18 of 20 historical invoices on time deserves different terms than one who has paid 3 of 10 and always late. Fixed guardrails treat them identically. That is commercially stupid. Dynamic, score-derived terms are commercially intelligent.

---

## 3. Product Overview

RecoverFlow has four components that work in sequence:

```
[1] INVOICE INGESTION + SCORING]
         ↓
[2] AI NEGOTIATION AGENT]
         ↓
[3] RAZORPAY PAYMENT LINK + WEBHOOK CONFIRMATION]
         ↓
[4] MERCHANT DASHBOARD]
```

---

## 4. Component 1 — Invoice Ingestion & Debtor Scoring

### What it does

When a merchant uploads (or auto-syncs) a batch of overdue invoices, the system does not immediately contact debtors. First, it runs a **silent scoring pass** on every debtor.

The scoring model evaluates each debtor and assigns:
- A **score from 0–100**
- A **Tier (A / B / C / D)**

The tier determines what settlement terms the AI is allowed to offer in the negotiation. The AI cannot offer terms better than the tier allows. This is the guardrail — not a fixed config, but a score-derived, per-debtor boundary.

### The Scoring Model

Five signals, weighted, produce the score. This is intentionally a **weighted model, not a black-box ML model** — because you can explain it to a judge in 30 seconds.

| Signal | Weight | What It Measures |
|--------|--------|-----------------|
| Historical on-time payment rate | 35% | % of prior invoices paid before due date |
| Average days late on past invoices | 25% | When they pay late, how late on average |
| Invoice amount vs their typical order size | 15% | Is this invoice unusually large for this debtor? |
| Days past due on current invoice | 15% | How overdue is the specific invoice right now |
| Number of previous disputes | 10% | Do they habitually dispute invoices |

**Score → Tier mapping:**

| Score | Tier | Interpretation |
|-------|------|----------------|
| 85–100 | A | Reliable payer, likely a cash flow timing issue |
| 60–84 | B | Generally good, occasionally late |
| 35–59 | C | Inconsistent, needs structured terms |
| 0–34 | D | High risk, serial late payer or disputer |

### The Tier-to-Terms Mapping

This is the guardrail table. The AI **cannot** breach these bounds.

| Tier | Pay Now (min) | Pay Later (max) | Timeline | Discount Eligible |
|------|--------------|-----------------|----------|-------------------|
| A | 25% | 75% | 60 days | Up to 15% |
| B | 40% | 60% | 45 days | Up to 10% |
| C | 60% | 40% | 30 days | Up to 5% |
| D | 85% | 15% | 15 days | None |

**Hard floor rule:** Every debtor pays at least 25% now. No tier offers 100% deferral. This is a firm business rule — a demo where money is fully deferred recovered zero money today.

### Cold-Start Handling

First-time debtor with no payment history? The system does not crash or guess randomly.

- Default assignment: **Tier C**
- Rationale: Middle ground — not generous enough to be exploited, not aggressive enough to cause disengagement
- Audit log records explicitly: `"cold_start": true, "reason": "no prior payment history", "defaulted_to": "Tier C"`

This intellectual honesty about cold starts is important. Judges notice when systems pretend they have data they don't.

### Data Source Options

**Option A (hackathon default):** Merchant uploads historical invoice CSV during onboarding. System parses and scores from that.

**Option B (ideal):** Pull from Razorpay's transaction history API. If the merchant has been processing payments through Razorpay, that history already exists. Score from real payment data.

**Option C (fallback for unknown debtors):** Proxy score using industry, company age, invoice amount, and DPD as signals. Score is marked as `estimated` in the audit log.

For the hackathon, **Option A with well-structured synthetic data** is the build path. Option B is the pitch for what production looks like.

### Score Transparency

The debtor is **never told their raw score**. The AI does not say "your score is 34, therefore Tier D."

Instead, it says: *"Based on your account history with [Merchant Name], the best terms we can offer at this time are..."*

This protects the merchant (not exposing scoring methodology to debtors), manages debtor emotion (raw scores create conflict), and is better aligned with privacy-by-design principles for automated decision systems.

---

## 5. Component 2 — The AI Negotiation Agent

### What the agent is

A conversational AI agent with:
- A **system prompt** that is parameterised by the debtor's tier at session initialisation
- **Tool-calling** to fetch invoice details, check tier bounds, and generate payment links
- **Session memory** for the duration of the negotiation
- **Stopping rules** that escalate automatically when conditions are met

### The Agent's Persona

The agent presents as a professional collections assistant representing the merchant. It is:
- Firm but not aggressive
- Empathetic about cash flow difficulties
- Transparent about what it can and cannot offer
- Clear about consequences of non-payment

It does **not** pretend to be human if asked directly. It does not threaten illegally. It does not offer terms outside its tier.

### System Prompt Architecture

The system prompt has two parts:

**Static section (same for all debtors):**
```
You are a professional payment recovery assistant for [Merchant Name]. 
Your goal is to reach a payment settlement on the outstanding invoice below.
You are empathetic but firm. You can discuss terms, accept reasonable 
explanations, and adjust offers — but only within the bounds defined below.
You cannot make commitments you are not authorised to make.
If asked if you are an AI, confirm that you are an automated assistant.
```

**Dynamic section (parameterised by tier):**
```
INVOICE: ₹[amount] — [invoice number] — [days past due] days overdue
DEBTOR TIER: [A/B/C/D]
AUTHORISED TERMS:
  - Minimum immediate payment: [tier minimum]%
  - Maximum deferred amount: [tier maximum]%
  - Maximum deferral period: [tier timeline]
  - Maximum discount you can offer: [tier discount]%
HARD RULE: You cannot offer terms better than the above. 
If the debtor demands better terms, explain that you are not authorised 
to go beyond these bounds and offer to escalate.
```

### The Negotiation Flow

**Turn 1 (Agent opens):**
"Hi [Debtor Name], I'm reaching out regarding invoice #[X] for ₹[amount] from [Merchant], which has been outstanding for [N] days. I'd like to work with you to resolve this today. Would you like to discuss a payment arrangement?"

**Debtor responds** — they may:
- Agree to pay in full → Agent generates full payment link immediately
- Claim they can't pay anything → Agent asks for explanation, tries to understand blocker
- Offer a low amount → Agent evaluates against tier bounds
- Dispute the invoice → Agent flags for L2 (dispute review) and logs
- Ask for more time → Agent evaluates against tier timeline, offers the maximum allowed
- Ignore / close tab → Session saved, system logs non-response, re-contact queued

**Negotiation turns** — the agent can:
- Adjust the upfront percentage within tier bounds
- Adjust the timeline within tier bounds
- Offer a discount within tier bounds (to incentivise immediate settlement)
- Explain consequences of non-payment (escalation to legal, credit flag)
- Acknowledge cash flow difficulties without conceding terms

**What the agent cannot do:**
- Offer better terms than the tier allows
- Make legal threats that are not backed by actual process
- Accept verbal promises without a confirmed payment link click
- Negotiate indefinitely — maximum 5 turns before escalation

### Stopping Rules

The agent automatically stops and escalates when:

| Condition | Action |
|-----------|--------|
| Debtor refuses all offers after 5 turns | Escalate to L3 |
| Debtor explicitly disputes invoice validity | Escalate to L2 (dispute queue) |
| Debtor asks to speak to a human | Escalate to L2, flag for merchant review |
| Payment link confirmed via webhook | Close session, mark resolved |
| Debtor has not responded in 48h | Mark non-responsive, queue re-contact |
| Debtor attempts to negotiate below hard floor | Hold firm, offer escalation |

### Escalation Levels

**L1 — AI negotiation** (described above)

**L2 — Merchant review queue**
- Invoice flagged in merchant dashboard with full transcript
- Merchant can review, override tier, add notes, re-open AI negotiation with adjusted params
- Or manually call the debtor with full context already surfaced

**L3 — Legal escalation**
- System generates a **PDF legal notice** with: merchant details, debtor details, invoice details, negotiation transcript, escalation date, payment demand, and consequence statement
- This is a **visible, tangible artifact** — not a string in a database
- Merchant downloads the PDF. They can send it, or a lawyer can send it.
- Audit log records: `L3 triggered`, `PDF generated`, `timestamp`, `invoice ID`

---

## 6. Component 3 — Razorpay Payment Link & Webhook

### Payment Link Generation

When a settlement is agreed in chat:

1. Agent calls Razorpay's `/v1/payment_links` API
2. Link is created for the **agreed upfront amount** (e.g., ₹40,000 of ₹1L invoice)
3. Link has a 24-hour expiry
4. Link is sent directly inside the chat: "Great — here's your payment link: [link]. This is valid for 24 hours."
5. A second payment link for the deferred amount is scheduled for creation at the agreed future date

### Webhook Listener

This is the **most critical technical component** and the most common gap in similar projects.

RecoverFlow runs a webhook listener on `/webhooks/razorpay` that:

1. Receives `payment.captured` and `payment_link.paid` events from Razorpay
2. Validates the webhook signature (Razorpay sends a signature header — validate it or you're a security hole)
3. Handles idempotency (same event received twice should not double-count)
4. On confirmed payment:
   - Updates invoice status to `PARTIALLY_SETTLED` or `SETTLED`
   - Records: `amount_paid`, `timestamp`, `payment_id`, `settlement_type`
   - Updates merchant dashboard in real-time
   - Logs to audit trail

**Without this webhook loop, your dashboard shows "link sent" as the terminal state.
That is not revenue recovery. That is link generation. Completely different metric.**

### The Payment States

```
OVERDUE
  → CONTACTED (agent initiated chat)
  → NEGOTIATING (debtor engaged)
  → SETTLED (full payment confirmed via webhook)
  → PARTIALLY_SETTLED (upfront portion confirmed, remainder scheduled)
  → DISPUTED (debtor contested invoice, in L2 queue)
  → NON_RESPONSIVE (no reply in 48h)
  → ESCALATED_L3 (legal notice generated)
  → WRITTEN_OFF (merchant decision)
```

---

## 7. Component 4 — Merchant Dashboard

### Purpose

The merchant is the actual customer. They need:
1. Visibility into what the AI is doing on their behalf
2. Confidence that the AI is not making commitments they didn't authorise
3. Override capability for edge cases
4. A clear, single metric: how much money was recovered

### Dashboard Views

**Batch Overview (main screen):**

| Invoice # | Debtor | Amount | DPD | Tier | Status | Recovered | Action |
|-----------|--------|--------|-----|------|--------|-----------|--------|
| INV-0042 | Sharma Distributors | ₹1,20,000 | 45 | B | Negotiating | — | View Chat |
| INV-0018 | Mehta Traders | ₹85,000 | 67 | D | Settled | ₹72,250 | View |
| INV-0055 | Kapoor & Sons | ₹2,50,000 | 30 | A | Contacted | — | View Chat |
| INV-0031 | RK Logistics | ₹45,000 | 91 | D | Escalated L3 | — | Download Notice |

**Metrics bar (top of dashboard):**
- **Total overdue:** ₹14,70,000
- **Total recovered:** ₹3,80,000 (26%)
- **Invoices settled:** 8 of 50
- **In negotiation:** 12
- **Non-responsive:** 11
- **Escalated:** 6
- **Avg time to settlement:** 11 minutes

**Invoice detail view:**
- Full negotiation transcript
- Debtor tier + scoring signals (merchant sees the score and why — debtor does not)
- Payment confirmation timestamp + Razorpay payment ID
- Audit log for this invoice
- Override controls: adjust tier, re-open negotiation, mark for manual follow-up

### Merchant Config Screen

Before running a batch, merchants set:
- Which invoice statuses to include (30+ DPD, 60+ DPD, etc.)
- Whether to allow discounts (toggle)
- Maximum discount percentage cap (overrides tier defaults if lower)
- Whether L3 escalation is automatic or requires merchant approval
- Notification preferences (email when payment confirmed, email when invoice escalated)

This screen exists because **no CFO will deploy a fully autonomous agent they cannot configure**. This is a trust prerequisite, not a nice-to-have.

---

## 8. Audit Trail

Every action in the system is logged to an immutable audit trail. The audit log records:

For each debtor:
```json
{
  "invoice_id": "INV-0042",
  "debtor_id": "DEBTOR-007",
  "score": 71,
  "scoring_signals": {
    "on_time_rate": 0.68,
    "avg_days_late": 12,
    "invoice_vs_typical_ratio": 1.1,
    "current_dpd": 45,
    "dispute_count": 0
  },
  "tier": "B",
  "cold_start": false,
  "tier_assigned_at": "2025-08-10T09:14:22Z",
  "negotiation_turns": [
    {
      "turn": 1,
      "speaker": "agent",
      "message": "Hi Rahul, I'm reaching out regarding...",
      "timestamp": "2025-08-10T09:15:01Z"
    },
    {
      "turn": 2,
      "speaker": "debtor",
      "message": "I can pay 30% now, rest in 60 days",
      "timestamp": "2025-08-10T09:16:34Z"
    },
    {
      "turn": 3,
      "speaker": "agent",
      "message": "I can offer 40% now with the remaining 60% in 45 days...",
      "timestamp": "2025-08-10T09:17:12Z"
    }
  ],
  "settlement": {
    "agreed_upfront_pct": 40,
    "agreed_deferred_pct": 60,
    "deferred_days": 45,
    "discount_applied": 0,
    "payment_link_id": "plink_XXXXX",
    "payment_confirmed_at": "2025-08-10T09:23:41Z",
    "payment_id": "pay_XXXXX",
    "amount_recovered": 48000
  },
  "escalation": null,
  "stopping_rule_triggered": "payment_confirmed"
}
```

This audit trail is what makes the brief's "compliant escalation" requirement real. It's not a string. It's a structured, queryable, exportable record.

---

## 9. Synthetic Data Design

### Why synthetic data needs failure modes

Your seed data **must not** show clean resolution on every invoice. Real collections looks like this:

| Outcome | % of debtors |
|---------|-------------|
| Never respond at all | 40% |
| Respond, then ghost mid-negotiation | 20% |
| Dispute the invoice amount | 15% |
| Ask for extensions repeatedly before paying | 10% |
| Resolve cleanly in one session | 15% |

If your demo shows 80% clean resolution, judges with any collections experience know it's rigged.

### Debtor History Schema (what you generate per debtor)

```python
{
  "debtor_id": "DEBTOR-007",
  "company_name": "Sharma Distributors Pvt Ltd",
  "contact_name": "Rahul Sharma",
  "contact_email": "rahul@sharmadist.in",
  "historical_invoices": [
    { "invoice_id": "H-001", "amount": 85000, "due_date": "2024-11-15", "paid_date": "2024-11-12", "status": "paid_on_time" },
    { "invoice_id": "H-002", "amount": 120000, "due_date": "2024-12-31", "paid_date": "2025-01-18", "status": "paid_late", "days_late": 18 },
    { "invoice_id": "H-003", "amount": 45000, "due_date": "2025-02-28", "paid_date": "2025-03-20", "status": "paid_late", "days_late": 20 },
    { "invoice_id": "H-004", "amount": 200000, "due_date": "2025-05-15", "paid_date": null, "status": "disputed" }
  ],
  "dispute_count": 1,
  "avg_days_late": 12.67,
  "on_time_rate": 0.5
}
```

### Invoice Schema (current batch)

```python
{
  "invoice_id": "INV-0042",
  "debtor_id": "DEBTOR-007",
  "amount": 120000,
  "issue_date": "2025-06-15",
  "due_date": "2025-07-15",
  "dpd": 45,
  "status": "overdue",
  "tier": null,        # populated after scoring pass
  "score": null,       # populated after scoring pass
  "recovered": 0,
  "negotiation_status": "pending"
}
```

### Generating 50 invoices with realistic distribution

Build your synthetic batch so it produces:
- 10 Tier A debtors (reliable, small overdue, likely to resolve quickly)
- 15 Tier B debtors (mostly good, one or two issues)
- 15 Tier C debtors (inconsistent, cold-start candidates go here)
- 10 Tier D debtors (serial late payers, some will need L3)

Mix invoice amounts: ₹20,000 – ₹4,00,000 range. Include a few very large ones (₹2L–₹4L) to make the recovery numbers meaningful.

---

## 10. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MERCHANT INTERFACE                        │
│   Config Screen → Batch Upload → Dashboard → Invoice Detail     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                    ┌───────▼───────┐
                    │ SCORING ENGINE │
                    │  (5 signals,   │
                    │  weighted,     │
                    │  → Tier A-D)  │
                    └───────┬───────┘
                            │
                    ┌───────▼───────────────────┐
                    │   NEGOTIATION AGENT        │
                    │  (LLM + tool calling)      │
                    │  System prompt param'd     │
                    │  by tier at init           │
                    │  Max 5 turns               │
                    │  Stopping rules enforced   │
                    └───────┬───────────────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
     ┌────────▼───┐  ┌──────▼──────┐  ┌──▼──────────┐
     │ RAZORPAY   │  │  ESCALATION  │  │ AUDIT LOG   │
     │ PAYMENT    │  │  HANDLER     │  │ (structured,│
     │ LINK API   │  │  L2/L3 PDF  │  │  immutable) │
     └────────┬───┘  └─────────────┘  └─────────────┘
              │
     ┌────────▼───────────┐
     │  WEBHOOK LISTENER  │
     │  /webhooks/razorpay│
     │  payment.captured  │
     │  payment_link.paid │
     │  → update invoice  │
     │  → update dashboard│
     └────────────────────┘
```

### Tech Stack (recommended for 10-day build)

**Backend:** Python (FastAPI) or Node.js (Express)
**LLM:** Claude Sonnet (via Anthropic API) or GPT-4o — tool-calling support required
**Database:** PostgreSQL or Supabase (free tier, fast setup)
**Frontend:** React + Tailwind (merchant dashboard), plain HTML/CSS (debtor chat — keep it simple)
**Payment:** Razorpay Node.js / Python SDK
**Webhook:** ngrok for local dev, Render or Railway for deployment
**PDF generation:** `reportlab` (Python) or `pdfkit` (Node) for L3 notices
**Synthetic data:** Python script using `faker` + custom distribution logic

---

## 11. 10-Day Build Sequence

### Day 1–2: Foundation
- [ ] Define invoice + debtor schemas in database
- [ ] Write scoring function (pure function, unit-testable)
- [ ] Build synthetic data generator with realistic failure modes
- [ ] Score all 50 synthetic invoices, verify tier distribution looks realistic
- [ ] Razorpay sandbox setup — test payment link creation manually

### Day 3–4: Agent
- [ ] Build negotiation agent with parameterised system prompt
- [ ] Implement tool: `get_invoice_details`
- [ ] Implement tool: `check_tier_bounds` (validates proposed terms against tier)
- [ ] Implement tool: `generate_payment_link` (calls Razorpay API)
- [ ] Implement stopping rules (turn counter, escalation conditions)
- [ ] Test with manual debtor responses — does the agent hold the tier line?

### Day 5–6: Payment Confirmation
- [ ] Build webhook listener endpoint
- [ ] Implement signature verification
- [ ] Implement idempotency handling
- [ ] Test full loop: negotiate → link → pay in sandbox → webhook fires → status updates
- [ ] **This loop must work before Day 7. It is the most critical path.**

### Day 7–8: Merchant Dashboard
- [ ] Batch overview table (invoice, debtor, tier, status, recovered)
- [ ] Metrics bar (total recovered, % resolved, avg time to settlement)
- [ ] Invoice detail view with transcript and audit log
- [ ] Merchant config screen (minimum viable: discount toggle, DPD threshold)
- [ ] Real-time dashboard update on webhook receipt

### Day 9: Escalation & Polish
- [ ] L3 PDF generator (legal notice with full invoice + transcript)
- [ ] L2 merchant review queue
- [ ] Cold-start tier defaulting with explicit audit log entry
- [ ] End-to-end test of entire flow: upload batch → score → negotiate → pay → dashboard
- [ ] Fix all broken paths

### Day 10: Demo Preparation
- [ ] Script the "judge plays debtor" live demo scenario (see Section 12)
- [ ] Prepare the specific invoice to use in demo (₹80,000, Tier B debtor)
- [ ] Verify webhook fires fast enough for live demo (< 3 seconds on dashboard)
- [ ] Prepare pitch narrative
- [ ] Record a backup video demo in case of live failure

---

## 12. The Demo Moment

**This is the most important section in this document.**

Everything you build should serve a single demo moment. Here is what that moment looks like:

---

**Setup (before the demo):**
- Merchant dashboard is open on a large screen visible to judges
- 50 synthetic invoices are pre-loaded, all scored and tiered
- Dashboard shows: "Total outstanding: ₹14,70,000 | Recovered: ₹0 | Invoices settled: 0"

**Step 1:** You explain the product in 30 seconds:
"Indian SMBs lose crores chasing overdue invoices with WhatsApp messages and awkward phone calls. RecoverFlow lets an AI negotiate payment settlements with debtors in real time — personalised terms based on their history, payment link in the chat, money confirmed on this dashboard."

**Step 2:** Ask a judge to be the debtor.
Hand them a phone or open a second browser window. Show them the chat interface.
"You are Rahul Sharma from Sharma Distributors. You owe ₹80,000. You're a Tier B debtor — mostly reliable but tight on cash this month. Try to get the best deal you can."

**Step 3:** The AI opens the negotiation.
The judge lowballs: "I can only pay ₹20,000 now, rest later."
The AI holds the line: "I understand cash flow can be challenging. I'm authorised to offer you 40% now (₹32,000) with the remaining 60% in 45 days. That's the best arrangement I can offer on this account."
The judge pushes: "Can you do 25% now?"
The AI: "I'm not able to go below 40% on this account — but I can extend the timeline to 45 days and apply a 5% discount on the deferred portion. That brings your total to ₹76,000 — saving you ₹4,000."
The judge accepts.

**Step 4:** Payment link drops in chat.
"Here's your payment link for ₹32,000: [link]. Valid for 24 hours."
Judge clicks it. Completes sandbox payment.

**Step 5 (the moment):**
The merchant dashboard on the big screen updates live:
**"INV-0042 — Sharma Distributors — SETTLED — ₹32,000 confirmed"**
**"Total Recovered: ₹32,000 | Invoices Settled: 1 of 50"**

That is the moment. A judge who has seen 100 projects will remember this one.

---

## 13. Pitch Narrative

**Opening:** "Every Indian SMB with B2B customers has a problem they don't talk about in public: they're owed crores they can't collect."

**Problem:** "The current solution is embarrassing phone calls, WhatsApp messages, and 30-day email sequences that debtors ignore. When that fails, collections agencies take 20% of whatever they recover."

**Insight:** "The bottleneck isn't reaching the debtor. It's the negotiation. Two weeks of back-and-forth to agree on terms that could have been settled in 10 minutes."

**Product:** "RecoverFlow compresses that negotiation into a real-time AI chat. The AI scores each debtor on their payment history, assigns them to a tier, and negotiates settlement terms within that tier — never too generous, never aggressive enough to cause disengagement."

**Live demo:** (run it)

**Results:** "In our test batch of 50 invoices, ₹3.2L was recovered — an average of 11 minutes per settled invoice, at zero collections commission."

**Differentiation:** "Every other tool talks *at* debtors. RecoverFlow talks *with* them. And it does it natively on Razorpay's infrastructure — so the payment link, the confirmation, the audit trail are all in one system."

**Vision:** "The endgame is a Razorpay-native receivables intelligence layer — where every merchant's AR book is automatically scored, prioritised, and worked by AI. Collections agencies take 20% and 90 days. RecoverFlow takes 3% and 11 minutes."

---

## 14. Business Model

**Primary:** Performance fee — **3% of recovered amount**
- Aligned incentives: RecoverFlow only earns when money moves
- Easy to sell: "We take 3%. Agencies take 20–30%."
- Easy to calculate ROI: If we recover ₹5L, we charge ₹15,000. Equivalent collections agency fee: ₹1–1.5L.

**Secondary:** SaaS floor — **₹10,000–₹20,000/month**
- Covers base platform access for merchants with high volume
- Predictable revenue for RecoverFlow

**Pricing comparison:**

| Method | Cost | Time |
|--------|------|------|
| Manual calls (in-house) | ₹0 cash, but 10hr/week staff time | 2–8 weeks |
| Collections agency | 15–30% of recovered | 60–90 days |
| Legal action | ₹30,000+ legal fees | 6–18 months |
| **RecoverFlow** | **3% of recovered** | **~11 minutes** |

**Distribution:**
- Razorpay partnership / marketplace (post-hackathon win)
- Direct outreach to Razorpay merchants in manufacturing, distribution, wholesale
- Content: "How to reduce your DSO by 30 days" — finance managers are the audience

**TAM:** 8M+ Razorpay merchants. Conservative addressable segment: 200,000 merchants with serious B2B AR exposure. At ₹10,000/month average revenue per merchant, that's a ₹24,000 Cr ARR ceiling. Realistic 3-year target: 5,000 merchants = ₹60 Cr ARR.

---

## 15. Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Debtors don't engage with AI chat | 🔴 Critical | Design chat UX to feel personal, not robotic. Subject line A/B test. WhatsApp as future channel. |
| No debtor history at cold start | 🔴 Critical | Default Tier C + explicit log. Option B (Razorpay API) in production. |
| Webhook loop not built | 🔴 Critical | Build Days 5–6, test before dashboard work begins |
| Razorpay API sandbox limitations | 🟠 Serious | Test on Day 1. Know the limits before you depend on them. |
| Synthetic data looks too clean | 🟠 Serious | Force 40% non-responders, 20% ghosters into generator |
| Merchants don't trust autonomous AI | 🟠 Serious | Merchant config screen + override capability |
| Agent offers terms outside tier | 🟡 Manageable | Tool validation layer checks every proposed term before utterance |
| Session persistence (debtor closes tab) | 🟡 Manageable | Save session state to DB, re-send link resumes conversation |
| Razorpay builds this natively | 🟡 Manageable | Post-hackathon risk, not Day-10 risk |
| DPDP Act / data privacy | 🟢 Low | B2B context, don't expose raw score to debtor, audit trail covers you |

---

## 16. What This Is Not

To be clear about scope:

- **Not an ERP integration** — you are not connecting to Tally, SAP, or Oracle
- **Not a WhatsApp bot** — chat interface only (for the hackathon)
- **Not a credit bureau** — your score is internal to RecoverFlow, not a financial credit score
- **Not a legal service** — your L3 PDF is a generated notice, not legal advice
- **Not a replacement for complex dispute resolution** — disputes go to L2 for human review
- **Not real ML** — weighted scoring model, explainable, not a neural network

These are not weaknesses. They are scope decisions. The hackathon rewards a bounded, working prototype with real measurable outcomes — not an enterprise platform.

---

## 17. The Single Most Important Metric

At the end of the demo, one number should be visible on your screen:

> **₹[X] recovered from [N] invoices in [Y] minutes average.**

Everything else — the scoring model, the tiers, the webhook, the audit log — is infrastructure that makes that number real and defensible.

CRITICAL: You MUST call validate_proposed_terms before accepting 
any debtor offer. Never skip this tool call. If the tool returns 
violations, you cannot accept those terms.

Build toward that number. Not toward feature count.

---

*RecoverFlow — Dialogue, not dunning. Dynamic terms, not fixed templates. Razorpay-native, not generic.*
