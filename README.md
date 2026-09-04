# RecoverFlow

**Autonomous B2B Debt Negotiation Agent**

RecoverFlow is an enterprise-grade AI agent designed to recover working capital trapped in overdue B2B invoices. It replaces manual sales team follow-ups and predatory third-party collection agencies by autonomously negotiating debt settlements, enforcing strict mathematical guardrails, and securing capital via integrated Razorpay payment links.

## The Problem

* **Manual Inefficiency:** Chasing payments drains sales team productivity and damages client relationships with awkward follow-ups.
* **Predatory Fees:** Traditional debt collection agencies charge exorbitant cuts of 15% to 30% on recovered funds.
* **Standard AI Limitations:** General-purpose LLMs are unsafe for financial negotiations. They suffer from context amnesia, hallucinate basic arithmetic, spontaneously offer unsolicited concessions, and frequently negotiate against the merchant's own pricing floors.

## The Solution

RecoverFlow utilizes a deterministic, state-aware AI architecture to solve LLM non-determinism. By decoupling the reasoning engine from text generation and actively tracking the negotiation state in a database, the agent holds firm on dynamically calculated pricing floors and safely processes hardship exceptions.

## Key Features

* **Two-Step Agentic Pipeline:** Messages are processed by a Classifier AI to evaluate debtor linguistic cues (Aggressive, Evasive, Collaborative) before a separate Generator AI drafts the response, allowing for dynamic psychological persona shifting.
* **Deterministic State-Tracking:** A MongoDB backend actively stores the `highest_user_offer`. Mathematical constraints are injected into the LLM context window on every turn, mathematically preventing the AI from negotiating against itself or forgetting locked values.
* **X-Ray Judge Mode (UI):** The Next.js frontend exposes the AI's internal logic in real-time via dynamic UI badges, proving the agent's tone classification, stance shifts, and floor-locking mechanisms.
* **JSON-Driven UI Hydration:** The AI bypasses natural language for the final closing action, outputting strict JSON payloads with ISO-8601 dates to dynamically render interactive Razorpay payment cards in the chat UI.

## Tech Stack

* **Frontend:** Next.js, React, Tailwind CSS
* **Backend:** FastAPI (Python), Uvicorn
* **AI / LLM:** DeepSeek API
* **Database:** MongoDB
* **Payments:** Razorpay API & Webhooks
* **Deployment:** Railway (Unified private network for DNS resolution)

## System Architecture

1. **Incoming Message:** Next.js sends user text to the FastAPI backend.
2. **State Retrieval:** Backend fetches the active `highest_user_offer` and `invoice_total` from MongoDB.
3. **Tone Classification:** LLM Step 1 analyzes the text and outputs a JSON classification of the user's tone.
4. **Guardrail Injection:** Backend calculates the updated dynamic floor and injects rigid arithmetic rules into the context window.
5. **Response Generation:** LLM Step 2 generates the negotiated response based on the strict behavioral constraints.
6. **Serialization:** The updated chat history, exact JSON UI payloads, and AI reasoning metadata are saved to MongoDB.
7. **Client Rendering:** The frontend re-hydrates the interactive UI components directly from the database payload.




Create a `.env.local` file in the `frontend` directory with your `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_RAZORPAY_KEY`. Run the client: `npm run dev`
