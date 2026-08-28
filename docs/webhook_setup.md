# Webhook Setup for Local Development

## Prerequisites

- RecoverFlow server running on port 8000
- ngrok installed (for tunnelling to Razorpay)
- Razorpay account with sandbox access

---

## Step 1 — Set environment variables

Add these to your `.env` file:

```env
RAZORPAY_KEY_ID=rzp_test_xxxxxxxxxxxx
RAZORPAY_KEY_SECRET=your_razorpay_key_secret
RAZORPAY_WEBHOOK_SECRET=any_string_you_choose
WEBHOOK_BASE_URL=https://your-ngrok-url.ngrok.io
```

---

## Step 2 — Start the RecoverFlow server

```bash
cd /path/to/b2b-payment
uvicorn backend.server:app --port 8000
```

---

## Step 3 — Start ngrok tunnel

```bash
ngrok http 8000
```

Copy the HTTPS URL shown, e.g. `https://abc123.ngrok.io`

Set it in your shell:

```bash
export WEBHOOK_BASE_URL=https://abc123.ngrok.io
```

Or add it to `.env` and restart the server.

---

## Step 4 — Configure webhook in Razorpay Dashboard

1. Log in to [dashboard.razorpay.com](https://dashboard.razorpay.com)
2. Go to **Settings → Webhooks → Add New Webhook**
3. Set:
   - **Webhook URL:** `https://abc123.ngrok.io/webhooks/razorpay`
   - **Secret:** the value you used for `RAZORPAY_WEBHOOK_SECRET`
   - **Active Events:** check both:
     - `payment_link.paid`
     - `payment.captured`
4. Click **Create Webhook**

---

## Step 5 — Test the flow

1. Open the merchant dashboard: `http://localhost:8000/dashboard`
2. Click **Start Batch** to load invoices
3. Click **View Chat** on any invoice
4. Go through the negotiation and reach the payment link step
5. Open the Razorpay payment link → complete payment in **Test Mode**
6. Watch the merchant dashboard update the invoice status in real time

---

## Demo safety net

If ngrok drops during a live demo, use the simulate endpoint instead:

```bash
curl -X POST http://localhost:8000/api/simulate-webhook/INV-0042 \
  -H "Content-Type: application/json" \
  -d '{"amount": 40000}'
```

This fires the same internal update logic without requiring a real webhook.

---

## Verifying signature verification

Send a test webhook with the correct signature:

```python
import hmac, hashlib, json, requests

secret = "your_RAZORPAY_WEBHOOK_SECRET"
payload = json.dumps({"event": "payment.captured", "payload": {"payment": {"entity": {"id": "pay_test123", "amount": 40000, "description": "INV-0042"}}}})
sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

requests.post(
    "http://localhost:8000/webhooks/razorpay",
    data=payload,
    headers={"Content-Type": "application/json", "X-Razorpay-Signature": sig}
)
```
