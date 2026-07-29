"""
Run the pipeline end-to-end on the sample claim bundle and print the
full audit trail. Runs fully offline against MockLLM by default.

To run against the live Anthropic API instead:
  export CLAIMS_PIPELINE_LIVE=1
  export ANTHROPIC_API_KEY=...
  python demo.py
(the extraction/verification/coverage/fraud prompts are unchanged --
only the client backing them switches)
"""

from pipeline import adjudicate_claim
from audit import render_audit_trail
from sample_data.documents import RAW_DOCUMENTS

if __name__ == "__main__":
    bundle = adjudicate_claim("CF-55210", RAW_DOCUMENTS)
    print(render_audit_trail(bundle))
