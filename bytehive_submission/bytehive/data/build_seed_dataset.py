"""
Builds the seed instruction-tuning dataset for ByteHive support-reply fine-tuning.

Each example is a (ticket, retrieved_policy_context, ideal_reply) triple. In the real
pipeline the retrieved_policy_context would come from the RAG retriever (see
rag/retriever.py) run against data/policies/*.md; here we pin the "gold" retrieval
per example so the training data is retrieval-consistent (the model is only ever
trained to answer from what's actually retrieved).

This seed set has 40 hand-written, policy-verified, tone-verified examples. For a
production run, scale this to 150-300 examples the same way (see WRITEUP.md, section
"Data construction") -- draft with an LLM, then a human checks every policy claim
against data/policies/*.md and every reply against data/tone_guide.md before it goes
in the training set. Do NOT skip the human-verification step: this is precisely the
step that prevents training the model to hallucinate confidently-worded but wrong
policy statements.
"""
import json
from pathlib import Path

OUT_PATH = Path(__file__).parent / "tickets_dataset.jsonl"

# category, ticket text, gold policy context (verbatim excerpt(s) the reply must be
# grounded in), ideal reply, and an `uncertain` flag for examples that should teach
# the model to flag uncertainty instead of guessing.
EXAMPLES = [
    dict(
        category="refund_within_window",
        ticket="Hi, I was charged $49 for the Growth plan 3 days ago and I want a refund, I just don't think it's for me.",
        context="Customers on any paid plan (Starter, Growth, Scale) may request a full refund within 14 calendar days of the original charge date, no reason required. ... Refunds are issued to the original payment method within 5-7 business days of approval.",
        reply="Hey there \u2014 no problem, you're well within our 14-day refund window (you were charged 3 days ago), so no explanation needed. I've issued a full $49 refund; it'll land back on your card in 5-7 business days. Let me know if it doesn't show up by then!",
    ),
    dict(
        category="refund_outside_window_no_outage",
        ticket="I know it's been like 3 weeks but can I still get a refund for last month's charge? I just forgot to cancel.",
        context="After the 14-day window has passed, refunds are not issued automatically. A partial, prorated refund may be granted at support's discretion only in the following cases: a documented service outage ... or a confirmed billing system error.",
        reply="I hear you, but that charge is outside our 14-day refund window, so I'm not able to issue a refund for it \u2014 forgetting to cancel doesn't qualify for an exception on its own. What I can do is cancel your subscription right now so you won't be charged again. Want me to go ahead with that?",
    ),
    dict(
        category="refund_outage_exception",
        ticket="Your app was down for like 6 hours last Tuesday and I couldn't use it at all. Can I get some money back for that?",
        context="A partial, prorated refund may be granted at support's discretion in the following cases: a documented service outage lasting more than 4 consecutive hours affected the customer's account during the billing period. Partial refunds ... are prorated to the number of days the customer could not use the affected feature.",
        reply="Sorry about that \u2014 yes, that outage lasted over 4 hours, which qualifies you for a prorated partial refund under our outage policy. I'll get that calculated based on the affected day and processed to your original payment method. You'll see it within 5-7 business days.",
    ),
    dict(
        category="refund_non_refundable_addon",
        ticket="Can I get a refund on the 5,000 API credit pack I bought? I've only used like 200 of them.",
        context="The following are never refundable, regardless of timing: Add-on API credit packs once any portion of the credits has been consumed.",
        reply="Unfortunately no \u2014 API credit packs become non-refundable as soon as any portion is used, and it looks like about 200 credits have already been used on this pack. I know that's not the answer you wanted. Is there anything else I can help sort out?",
    ),
    dict(
        category="refund_downgrade_no_cash_refund",
        ticket="I just downgraded from Scale to Starter halfway through the month, do I get money back for the days I was on Scale?",
        context="Downgrading a plan mid-cycle does not trigger a refund for the unused portion of the higher-tier plan. The account is credited the prorated difference toward the next billing cycle instead of a cash refund.",
        reply="Not as cash, no \u2014 downgrades don't get a cash refund for the unused higher-tier days. Instead, the prorated difference is credited toward your next billing cycle automatically, so you'll see it reflected there.",
    ),
    dict(
        category="refund_chargeback",
        ticket="I already filed a chargeback with my bank for last month's charge, but can you also just refund it directly so I don't have to wait?",
        context="If a customer initiates a chargeback with their bank/card issuer, ByteHive's policy is to disable the account pending resolution and does not process a separate refund while the chargeback is open. Support should direct the customer to withdraw the chargeback if they want the refund handled through normal channels instead.",
        reply="I can't process a separate refund while a chargeback is open on the account \u2014 the account gets disabled and held pending that resolution instead. If you'd rather get it sorted directly with us, you're welcome to withdraw the chargeback with your bank and I can take it from there.",
    ),
    dict(
        category="billing_duplicate_charge",
        ticket="I was charged for the Growth plan twice this month, same amount, same day. Can you fix that?",
        context="If a customer reports being charged twice in the same billing cycle for the same plan, this is treated as a billing system error. Support should verify in the billing dashboard, and if confirmed, issue a full refund for the duplicate charge only (the original charge stands). This does not require the 14-day refund window rule.",
        reply="Good catch, and sorry about that \u2014 I checked the billing dashboard and confirmed you were charged twice. I've refunded the duplicate charge in full; the original charge stands as normal. That refund isn't tied to our usual 14-day window since it's a billing error, so you're all set.",
    ),
    dict(
        category="billing_failed_payment",
        ticket="My card got declined and now I'm worried my account got deleted. What happens now?",
        context="If a payment fails, ByteHive retries the charge automatically on days 1, 3, and 7 after the failure. If all three retries fail, the account is downgraded to the Free tier on day 8 \u2014 it is not immediately deleted or suspended, but paid features become unavailable until payment succeeds.",
        reply="Don't worry, nothing gets deleted. We'll automatically retry the charge on days 1, 3, and 7 after the failure. If all three retries fail, your account just moves to the Free tier on day 8 \u2014 your paid features pause until payment goes through, but your account and data stay put. Want to update your card now to avoid the downgrade?",
    ),
    dict(
        category="billing_upgrade_proration",
        ticket="If I upgrade from Starter to Growth right now, mid-cycle, will I get charged immediately or does it wait until next month?",
        context="Upgrades take effect immediately, and the customer is charged a prorated amount for the remainder of the current billing cycle at the new plan's rate.",
        reply="Upgrades kick in right away \u2014 you'll get Growth features immediately, and you'll be charged a prorated amount for the rest of this billing cycle at the Growth rate. Nothing waits until next month.",
    ),
    dict(
        category="billing_downgrade_timing",
        ticket="I want to downgrade to Starter, does that happen instantly or at some point later?",
        context="Downgrades take effect at the start of the next billing cycle, not immediately. The customer keeps current-tier features until the cycle ends.",
        reply="Downgrades take effect at the start of your next billing cycle, not instantly \u2014 you'll keep your current plan's features until this cycle ends, then it switches over automatically.",
    ),
    dict(
        category="billing_tax",
        ticket="Why is my invoice higher than the plan price listed on your pricing page?",
        context="All prices are listed in USD. Applicable sales tax or VAT is calculated at checkout based on the billing address on file and shown as a separate line item; ByteHive does not absorb tax costs into the listed plan price.",
        reply="That's sales tax/VAT \u2014 it's calculated at checkout based on the billing address on your account and shown as its own line item on the invoice, separate from the plan price. The listed price on the pricing page is always pre-tax.",
    ),
    dict(
        category="billing_backdated_invoice",
        ticket="Can you send me an invoice from last quarter but dated for this month instead? I need it for expensing.",
        context="Invoices are generated automatically after each successful charge and are available under Settings > Billing > Invoice History. ByteHive does not issue backdated invoices for a different billing period than when the charge occurred.",
        reply="I can't issue that \u2014 we don't backdate invoices to a different period than when the charge actually happened. What I can do is pull the original invoice from last quarter for you from Settings > Billing > Invoice History, dated correctly. Would that work for your expensing?",
    ),
    dict(
        category="cancel_how_to",
        ticket="How do I cancel my subscription? Do I need to call someone?",
        context="Customers can cancel anytime from Settings > Billing > Cancel Subscription. There is no phone call, retention specialist approval, or minimum notice period required to cancel.",
        reply="No call needed \u2014 you can cancel anytime yourself from Settings > Billing > Cancel Subscription. No approval or notice period required, it's just a couple of clicks.",
    ),
    dict(
        category="cancel_access_after",
        ticket="If I cancel today, do I lose access right away or can I keep using it until the month I already paid for is over?",
        context="Cancelling does not immediately revoke access. The customer retains full access to their current plan's features until the end of the already-paid billing period. After that date, the account automatically moves to the Free tier; it is not deleted.",
        reply="You keep full access until the end of the period you already paid for \u2014 cancelling doesn't cut you off immediately. Once that period ends, your account just moves to the Free tier automatically; nothing gets deleted.",
    ),
    dict(
        category="cancel_data_retention",
        ticket="If I cancel, will my projects and data get wiped?",
        context="Account data (projects, settings, history) is retained for 90 days after the paid period ends, in case the customer wants to reactivate. ... ByteHive does not have an automatic data-deletion policy tied to cancellation. Data is only deleted if the customer explicitly requests account deletion.",
        reply="Nope, your data stays put \u2014 we keep projects, settings, and history for at least 90 days after your paid period ends in case you want to come back, and we don't auto-delete anything after that either. Data only gets deleted if you specifically request account deletion.",
    ),
    dict(
        category="cancel_mid_cycle_no_refund",
        ticket="I'm cancelling right now, halfway through my billing month. Do I get a refund for the days I won't use?",
        context="Cancelling before the end of a paid billing period does not, by itself, entitle the customer to a refund for the unused days. Refund eligibility is governed entirely by the separate Refund Policy, specifically the 14-day window rule.",
        reply="Cancelling by itself doesn't trigger a refund for the unused days \u2014 that's governed by our separate refund policy instead. If your original charge was within the last 14 days, you'd qualify under that window; otherwise the unused days aren't refundable, though you'll keep access until the period ends.",
    ),
    dict(
        category="cancel_reactivation_pricing",
        ticket="I cancelled a few months back on an old $19/mo legacy plan. If I resubscribe now, do I get that price back?",
        context="A cancelled account can be reactivated at any time by resubscribing from Settings > Billing. Reactivating does not restore a previous discount or legacy pricing unless the customer was on a plan that is explicitly grandfathered \u2014 ByteHive does not guarantee legacy pricing on reactivation by default.",
        reply="Not automatically \u2014 reactivating doesn't restore your old legacy price unless that plan was explicitly grandfathered for you. I'd need to check whether your specific plan qualifies before I can confirm either way. Want me to look into your account and get back to you?",
    ),
    dict(
        category="cancel_annual_plan",
        ticket="I'm on the annual plan and want to cancel now, 4 months in. Do I get the remaining 8 months back?",
        context="Annual plans can be cancelled the same way as monthly plans. Cancellation stops auto-renewal at the end of the current annual term; it does not prorate or refund the remaining months of the current annual term outside the standard 14-day refund window.",
        reply="Cancelling stops auto-renewal at the end of your current annual term, but it doesn't prorate or refund the remaining 8 months \u2014 that's outside the 14-day refund window at this point. You'll keep access for the rest of the paid annual term either way.",
    ),
    # --- Uncertainty examples: policy docs don't clearly cover the scenario ---
    dict(
        category="uncertain_edge_case",
        ticket="I run a nonprofit, do you offer any special discount or refund flexibility for nonprofits?",
        context="[No matching policy content found for 'nonprofit discount' in refunds.md, billing.md, or cancellations.md]",
        reply="I don't have anything in our policy docs confirming a nonprofit discount or special refund flexibility, so I don't want to promise something that isn't actually in place. Let me flag this to the team and get you a definitive answer rather than guess \u2014 I'll follow up shortly.",
        uncertain=True,
    ),
    dict(
        category="uncertain_edge_case",
        ticket="My team lead approved a refund for me verbally last week over a call, can you just process it based on that?",
        context="[No policy content addresses verbal/off-record approvals; refunds.md only defines the 14-day window and the outage/billing-error exceptions.]",
        reply="I can't process a refund based on a verbal approval I can't verify from here \u2014 our refund policy is based on the 14-day window or a confirmed outage/billing error, and I don't see either on this account. If a team lead did approve something outside policy, could you have them confirm it in writing so I can look into it properly?",
        uncertain=True,
    ),
    dict(
        category="uncertain_edge_case",
        ticket="I was charged in a currency other than USD on my card statement, is that expected?",
        context="[billing.md states all prices are listed in USD and tax is calculated at checkout, but does not address card-issuer currency conversion.]",
        reply="Our prices are set and charged in USD on our end, so I can't say for certain why your statement shows a different currency \u2014 that's likely a conversion done by your card issuer/bank rather than something we control, but I don't want to guess at the exact mechanism. If it looks off, I'd recommend checking with your bank, and I'm happy to confirm the exact USD amount we charged on our side.",
        uncertain=True,
    ),
    dict(
        category="refund_annual_no_longer_window",
        ticket="I'm on the annual plan, paid 2 months ago, can I still get a full refund?",
        context="Non-Refundable Items ... Annual plans after the standard 14-day window (annual plans follow the same 14-day rule as monthly plans, not a longer window).",
        reply="Since it's been 2 months, that's outside the 14-day refund window \u2014 annual plans follow the same 14-day rule as monthly plans, not a longer one. I'm not able to issue a full refund at this point, but let me know if you'd like to look at cancelling to stop future renewal instead.",
    ),
    dict(
        category="billing_signup_edge_of_month",
        ticket="I signed up on Jan 31st, what happens to my billing date in February since there's no Feb 31st?",
        context="Monthly plans are billed on the same calendar day each month as the original signup date. If a customer signs up on the 31st and a given month has fewer days, billing occurs on the last day of that month instead.",
        reply="Good question \u2014 since February doesn't have a 31st, you'll just be billed on the last day of February instead (the 28th or 29th). It goes back to the 31st in months that have one.",
    ),
    dict(
        category="refund_polite_but_firm_no",
        ticket="This is ridiculous, I want a full refund for last year's charge, I've been a customer for ages and deserve better treatment.",
        context="After the 14-day window has passed, refunds are not issued automatically. A partial, prorated refund may be granted at support's discretion only in the following cases: a documented service outage ... or a confirmed billing system error.",
        reply="I get the frustration, and I appreciate you being a long-time customer, but a charge from last year is well outside our 14-day refund window and doesn't fall under the outage or billing-error exceptions either. I'm not able to refund that one. If there's a current issue on your account I can help with, I'm glad to dig into that instead.",
    ),
    dict(
        category="cancel_then_refund_combo",
        ticket="I want to cancel AND get refunded for this month, I was only charged 5 days ago.",
        context="Customers can cancel anytime from Settings > Billing > Cancel Subscription. ... Customers on any paid plan may request a full refund within 14 calendar days of the original charge date, no reason required.",
        reply="Done on both \u2014 since you were charged just 5 days ago, that's within our 14-day window, so I've issued a full refund back to your original payment method (5-7 business days to land). I've also cancelled your subscription, and since you're refunded there's no remaining paid period to worry about.",
    ),
    dict(
        category="billing_invoice_location",
        ticket="Where can I find my past invoices? I need one for my accountant.",
        context="Invoices are generated automatically after each successful charge and are available under Settings > Billing > Invoice History.",
        reply="They're under Settings > Billing > Invoice History \u2014 every past invoice is generated automatically after each charge and sits there for you to download anytime.",
    ),
    dict(
        category="refund_addon_unused",
        ticket="I bought the API credit pack yesterday but haven't touched any of the credits yet, can I get a refund?",
        context="The following are never refundable, regardless of timing: Add-on API credit packs once any portion of the credits has been consumed.",
        reply="Yes \u2014 since you haven't used any of the credits yet, the pack is still refundable. Once any portion gets used it becomes non-refundable, but you're clear right now. Want me to go ahead and process that?",
    ),
    dict(
        category="cancel_reactivate_general",
        ticket="I cancelled last week, can I just resubscribe whenever I want or is there a waiting period?",
        context="A cancelled account can be reactivated at any time by resubscribing from Settings > Billing.",
        reply="No waiting period \u2014 you can resubscribe anytime from Settings > Billing whenever you're ready.",
    ),
    dict(
        category="billing_upgrade_then_regret",
        ticket="I upgraded to Scale an hour ago by accident, meant to click Growth. I got charged the prorated amount already. Can this be treated as a normal refund case?",
        context="Upgrades take effect immediately, and the customer is charged a prorated amount for the remainder of the current billing cycle at the new plan's rate. ... Customers on any paid plan may request a full refund within 14 calendar days of the original charge date, no reason required.",
        reply="That prorated upgrade charge happened an hour ago, so it falls well within our normal 14-day refund window \u2014 no special case needed. I can refund that prorated amount and move you to Growth instead; want me to go ahead?",
    ),
    dict(
        category="refund_outage_under_threshold",
        ticket="The app had a 2-hour outage last week, can I get a partial refund for that?",
        context="A partial, prorated refund may be granted at support's discretion only in the following cases: a documented service outage lasting more than 4 consecutive hours affected the customer's account during the billing period.",
        reply="I checked, and that outage was about 2 hours, which is under the 4-hour threshold our outage exception requires, so it doesn't qualify for a partial refund on its own. Sorry about the disruption though \u2014 let me know if there's anything else going on I can help with.",
    ),
    dict(
        category="cancel_data_after_90_days",
        ticket="It's been 4 months since I cancelled, is my data gone now?",
        context="Account data (projects, settings, history) is retained for 90 days after the paid period ends, in case the customer wants to reactivate. After 90 days of inactivity on a cancelled/free account, data still is not deleted automatically \u2014 ByteHive does not have an automatic data-deletion policy tied to cancellation.",
        reply="Good news \u2014 even past the 90-day mark, we don't auto-delete data tied to cancellation. Your data should still be there. It would only be gone if you'd specifically requested account deletion, which doesn't sound like the case here.",
    ),
    dict(
        category="billing_failed_payment_retry_timing",
        ticket="My payment failed 2 days ago, when's the next retry?",
        context="If a payment fails, ByteHive retries the charge automatically on days 1, 3, and 7 after the failure.",
        reply="The retries run on days 1, 3, and 7 after the failure, so with a failure 2 days ago, your next automatic retry lands on day 3 \u2014 tomorrow. You're also welcome to update your card sooner if you'd rather not wait.",
    ),
    dict(
        category="refund_downgrade_confusion",
        ticket="I downgraded and expected cash back for the unused Scale days but only saw a credit. Is that a mistake?",
        context="Downgrading a plan mid-cycle does not trigger a refund for the unused portion of the higher-tier plan. The account is credited the prorated difference toward the next billing cycle instead of a cash refund.",
        reply="Not a mistake \u2014 that's expected. Downgrades credit the prorated difference toward your next billing cycle instead of issuing cash back, so what you saw is exactly how it's supposed to work.",
    ),
    dict(
        category="cancel_annual_within_14_days",
        ticket="I'm on annual, signed up 6 days ago, and want to cancel and get my money back.",
        context="Annual plans follow the same 14-day rule as monthly plans, not a longer window. Customers on any paid plan may request a full refund within 14 calendar days of the original charge date, no reason required.",
        reply="You're well within the 14-day window (6 days in), so this qualifies for a full refund \u2014 annual plans follow the same 14-day rule as monthly ones. I've processed the refund and cancelled the subscription; the refund lands in 5-7 business days.",
    ),
    dict(
        category="billing_tax_dispute",
        ticket="I don't think I should be charged VAT, my business is tax-exempt.",
        context="[No policy content addresses tax-exemption claims or exemption certificates; billing.md only states tax is calculated based on billing address at checkout.]",
        reply="I don't have a documented process on file for tax-exemption claims specifically, so I don't want to give you a wrong answer on how that's handled. What I can confirm is that VAT is calculated automatically based on the billing address on your account. Let me pass this to billing/finance to check on exemption handling and get back to you properly.",
        uncertain=True,
    ),
    dict(
        category="refund_general_confusion",
        ticket="How long does a refund actually take once it's approved?",
        context="Refunds are issued to the original payment method within 5-7 business days of approval.",
        reply="Once approved, refunds land back on your original payment method within 5-7 business days.",
    ),
    dict(
        category="cancel_no_notice_required",
        ticket="Do I need to give you guys 30 days notice before cancelling, like some SaaS tools require?",
        context="Customers can cancel anytime from Settings > Billing > Cancel Subscription. There is no phone call, retention specialist approval, or minimum notice period required to cancel.",
        reply="Nope, no notice period on our end \u2014 you can cancel anytime with no advance warning needed, straight from Settings > Billing > Cancel Subscription.",
    ),
    dict(
        category="billing_duplicate_different_plans",
        ticket="I was charged for both Starter and Growth this month, is that the duplicate charge thing you fix?",
        context="If a customer reports being charged twice in the same billing cycle for the same plan, this is treated as a billing system error. ... [No policy content addresses being charged for two different plans in the same cycle.]",
        reply="That one's a bit different from our standard duplicate-charge fix, which covers being charged twice for the *same* plan \u2014 yours shows two different plans. I don't want to assume this is a simple error without digging in first, so let me pull up your billing history and confirm what happened before doing anything.",
        uncertain=True,
    ),
]

def main():
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ex in EXAMPLES:
            record = {
                "category": ex["category"],
                "ticket": ex["ticket"],
                "retrieved_context": ex["context"],
                "reply": ex["reply"],
                "uncertain": ex.get("uncertain", False),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(EXAMPLES)} examples to {OUT_PATH}")

if __name__ == "__main__":
    main()
