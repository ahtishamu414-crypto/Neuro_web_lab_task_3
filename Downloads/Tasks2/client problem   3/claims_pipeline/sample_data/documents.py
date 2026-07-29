"""
A small mock claim bundle: one auto policy, one claim form, two repair
invoices. Deliberately seeded with the exact failure mode SecureClaim's
last two automations missed: an invoice line item inflated well above
both the other invoice's quote for the same repair and the policy's
per-incident cap -- with nothing else in the bundle obviously wrong.
"""

RAW_DOCUMENTS = {
    "doc_policy_001": {
        "file_name": "policy_AX-88213.pdf",
        "doc_type": "policy",
        "text": (
            "SecureClaim Insurance Group -- Auto Policy AX-88213\n"
            "Policyholder: J. Marsh\n"
            "Coverage: Collision, Comprehensive\n"
            "Page 3, Section 4.2: Collision repair claims are covered up to "
            "$6,000 per incident, subject to a $500 deductible.\n"
            "Page 3, Section 4.5: Rental car reimbursement excluded unless "
            "rider AX-RC-100 is attached to this policy. No rider AX-RC-100 "
            "is on file for this policy.\n"
        ),
    },
    "doc_claimform_001": {
        "file_name": "claim_form_CF-55210.pdf",
        "doc_type": "claim_form",
        "text": (
            "Claim Form CF-55210\n"
            "Policy: AX-88213\n"
            "Incident date: 2026-03-14\n"
            "Description: Rear-end collision, damage to rear bumper and "
            "trunk panel.\n"
            "Claimed repair amount: $5,850\n"
            "Repair shop: Lakeside Auto Body\n"
            "Rental car requested: yes, 10 days\n"
        ),
    },
    "doc_invoice_lakeside": {
        "file_name": "invoice_lakeside_body.pdf",
        "doc_type": "invoice",
        "text": (
            "Lakeside Auto Body -- Invoice #LB-3391\n"
            "Claim ref: CF-55210\n"
            "Line 1: Rear bumper replacement -- $1,450\n"
            "Line 2: Trunk panel repair and repaint -- $1,300\n"
            "Line 3: Labor (14 hrs @ $95/hr) -- $1,330\n"
            "Line 4: Parts and materials -- $1,770\n"
            "Total: $5,850\n"
        ),
    },
    "doc_invoice_independent": {
        "file_name": "independent_estimate_QuickAppraise.pdf",
        "doc_type": "damage_report",
        "text": (
            "QuickAppraise Independent Damage Assessment\n"
            "Ref: CF-55210 / vehicle inspection 2026-03-16\n"
            "Estimated fair repair cost for described damage "
            "(rear bumper + trunk panel, same vehicle/model): $2,950\n"
            "Note: line-item breakdown roughly consistent with bumper and "
            "panel work; labor and parts estimate is less than half of the "
            "Lakeside Auto Body invoice total for comparable scope.\n"
        ),
    },
}
