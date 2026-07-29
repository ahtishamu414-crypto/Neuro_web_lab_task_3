"""
Canned model outputs used by MockLLM. Each key is a marker string that
must appear in the prompt for that stage; the value is the JSON dict
the corresponding stage module expects back. Wire these to real model
calls by setting CLAIMS_PIPELINE_LIVE=1 instead.
"""

FIXTURES = {
    "doc_policy_001": {
        "fields": [
            {"field_name": "collision_coverage_limit", "value": "6000",
             "confidence": "high", "page": 3, "location": "Section 4.2"},
            {"field_name": "deductible", "value": "500",
             "confidence": "high", "page": 3, "location": "Section 4.2"},
            {"field_name": "rental_reimbursement_eligible", "value": "false",
             "confidence": "high", "page": 3, "location": "Section 4.5"},
        ],
        "notes": [],
    },
    "doc_claimform_001": {
        "fields": [
            {"field_name": "claimed_repair_amount", "value": "5850",
             "confidence": "high", "page": 1, "location": "body"},
            {"field_name": "incident_date", "value": "2026-03-14",
             "confidence": "high", "page": 1, "location": "body"},
            {"field_name": "rental_requested_days", "value": "10",
             "confidence": "high", "page": 1, "location": "body"},
        ],
        "notes": [],
    },
    "doc_invoice_lakeside": {
        "fields": [
            {"field_name": "invoice_total", "value": "5850",
             "confidence": "high", "page": 1, "location": "Total line"},
            {"field_name": "line_bumper", "value": "1450",
             "confidence": "high", "page": 1, "location": "Line 1"},
            {"field_name": "line_trunk_panel", "value": "1300",
             "confidence": "high", "page": 1, "location": "Line 2"},
            {"field_name": "line_labor", "value": "1330",
             "confidence": "medium", "page": 1, "location": "Line 3"},
            {"field_name": "line_parts", "value": "1770",
             "confidence": "medium", "page": 1, "location": "Line 4"},
        ],
        "notes": [],
    },
    "doc_invoice_independent": {
        "fields": [
            {"field_name": "independent_estimate_total", "value": "2950",
             "confidence": "high", "page": 1, "location": "body"},
        ],
        "notes": ["Estimate explicitly flags the Lakeside invoice as high "
                   "for comparable scope."],
    },

    "CROSS_DOCUMENT_VERIFICATION": {
        "discrepancies": [
            {
                "description": (
                    "Claimed/invoiced repair amount ($5,850) is roughly "
                    "double the independent damage assessment for the same "
                    "described damage ($2,950), with no explanation on file "
                    "for the gap."
                ),
                "severity": "major",
                "source_a": {"document_id": "doc_invoice_lakeside", "page": 1, "location": "Total line"},
                "source_b": {"document_id": "doc_invoice_independent", "page": 1, "location": "body"},
            },
            {
                "description": (
                    "Rental car was requested on the claim form (10 days) "
                    "but the policy has no rental reimbursement rider on file."
                ),
                "severity": "minor",
                "source_a": {"document_id": "doc_claimform_001", "page": 1, "location": "body"},
                "source_b": {"document_id": "doc_policy_001", "page": 3, "location": "Section 4.5"},
            },
        ],
    },

    "COVERAGE_DETERMINATION": {
        "findings": [
            {
                "claim_item": "Collision repair (bumper + trunk panel)",
                "covered": True,
                "policy_document_id": "doc_policy_001",
                "policy_page": 3,
                "policy_location": "Section 4.2",
                "rationale": (
                    "Collision damage is covered up to $6,000/incident with "
                    "a $500 deductible; the claimed $5,850 falls within the "
                    "cap, so coverage of the repair category applies. Note: "
                    "this determination covers eligibility, not the "
                    "disputed dollar amount -- see fraud/discrepancy findings."
                ),
                "confidence": "high",
            },
            {
                "claim_item": "Rental car reimbursement (10 days)",
                "covered": False,
                "policy_document_id": "doc_policy_001",
                "policy_page": 3,
                "policy_location": "Section 4.5",
                "rationale": (
                    "Rental reimbursement requires rider AX-RC-100, which is "
                    "not on file for this policy."
                ),
                "confidence": "high",
            },
        ],
    },

    "FRAUD_SCORING": {
        "score": 68,
        "signals": [
            {
                "name": "Invoice vs. independent estimate gap",
                "weight": 45,
                "evidence": "Invoiced total is ~98% higher than the independent "
                             "assessment for the same described damage.",
                "source_document_id": "doc_invoice_independent",
                "page": 1,
                "location": "body",
            },
            {
                "name": "Round, high labor/parts split",
                "weight": 13,
                "evidence": "Labor and parts line items are unusually high "
                             "relative to the two structural repair lines for "
                             "this scope of damage.",
                "source_document_id": "doc_invoice_lakeside",
                "page": 1,
                "location": "Line 3-4",
            },
            {
                "name": "Rental request despite no rider",
                "weight": 10,
                "evidence": "Rental reimbursement was requested despite no "
                             "rider being on file; may be an oversight rather "
                             "than fraud, kept as a minor signal.",
                "source_document_id": "doc_claimform_001",
                "page": 1,
                "location": "body",
            },
        ],
    },
}
