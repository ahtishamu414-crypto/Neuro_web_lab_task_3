# SecureClaim multi-document claims adjudication & fraud screening — demo

A working, runnable demo of the pipeline described in `technical_writeup.docx`.
Runs fully offline against canned model responses by default, so it can be
reviewed and tested without API access; swap in a real Anthropic API key to
run live (see below).

## Run it

```bash
pip install pydantic anthropic --break-system-packages
python demo.py                # offline demo, prints the full audit/decision trail
python -m pytest test_invariants.py -v   # proves the anti-fabrication guarantees
```

To run against the live Claude API instead of the mock:

```bash
export CLAIMS_PIPELINE_LIVE=1
export ANTHROPIC_API_KEY=sk-ant-...
python demo.py
```

## Files

| File | Stage | Client requirement it addresses |
|---|---|---|
| `models.py` | — | Citation-first data model: discrepancies and coverage approvals cannot be constructed without source citations (see `test_invariants.py`) |
| `ingestion.py` | 1 | Per-document classification and OCR-quality tagging |
| `extraction.py` | 2 | One isolated agent call per document — fixes "monolithic confusion" |
| `confidence_gate.py` | 3 | Blocks low-confidence financial fields from auto-approval — fixes "uncontained cascading errors" |
| `verification.py` | 4 | Cross-document discrepancy detection over structured fields — fixes "verification blind spot" |
| `coverage.py` | 4 | Coverage determination that refuses to guess on ambiguous/missing terms — fixes "silent fabrication" |
| `fraud_scoring.py` | 5 | Evidence-cited, weighted fraud score |
| `routing.py` | 6 | Deterministic (non-model) auto-approve / manual-review / fraud-investigation routing |
| `audit.py` | — | Renders the full regulator-facing decision trail from the same objects used to decide |
| `pipeline.py` | — | Orchestrates all six stages |
| `sample_data/` | — | A mock 4-document claim bundle with a seeded inflated invoice, plus mock model fixtures |

## What the sample run demonstrates

The sample bundle (`sample_data/documents.py`) is a real auto claim: a policy,
a claim form, a repair invoice, and an independent damage assessment. The
invoice is seeded to be ~2x the independent estimate for the same described
damage — the exact pattern that got past SecureClaim's last two automation
attempts. In this demo run, the pipeline:

1. Extracts each document independently (no figure mixing between the two
   dollar-amount-bearing documents).
2. Flags the invoice/estimate gap as a MAJOR discrepancy, citing both source
   documents.
3. Determines coverage eligibility for the repair category (True, cited to
   policy §4.2) and separately for the rental request (False, cited to §4.5)
   — never conflating "is this category covered" with "is this dollar amount
   legitimate".
4. Produces a fraud score of 68/100 driven mainly by the invoice/estimate gap.
5. Routes to `fraud_investigation` (score ≥ 60), not auto-approve — with a
   full, cited rationale in the printed audit trail.

See `sample_decision_trail.txt` for a full run's output.
