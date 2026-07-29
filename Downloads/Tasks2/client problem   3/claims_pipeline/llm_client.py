"""
Thin wrapper around the Anthropic API so every agent module calls one
interface. Two tiers are exposed on purpose:

  - FAST_MODEL   -> cheap/quick model, used for OCR classification and
                    straightforward per-document field extraction.
  - REASONING_MODEL -> stronger model, used only where it earns its cost:
                    cross-document verification, coverage judgement,
                    and fraud narrative synthesis.

This tiering is the main lever for the "cost must scale linearly with
claim volume and document count" constraint: the O(n) work (reading each
document) runs on the cheap tier, and the O(1)-per-claim reasoning work
(the parts that actually need judgement) runs on the stronger tier. A
20-document claim does NOT mean 20x reasoning-tier calls -- it means 20
fast-tier calls plus a small, roughly constant number of reasoning-tier
calls over the already-structured extractions.

Set CLAIMS_PIPELINE_LIVE=1 to make real API calls (requires network
access to api.anthropic.com and an API key configured in the
environment the API client reads from). Otherwise MockLLM returns
canned, deterministic responses so the pipeline can be demoed and
unit-tested without API access or nondeterminism.
"""

from __future__ import annotations
import json
import os
from typing import Any


FAST_MODEL = "claude-haiku-4-5-20251001"
REASONING_MODEL = "claude-sonnet-5"


class LLMClient:
    """Interface every agent module codes against."""

    def complete_json(self, model: str, system: str, user: str) -> dict[str, Any]:
        raise NotImplementedError


class AnthropicLLMClient(LLMClient):
    def __init__(self):
        import anthropic  # imported lazily so mock-mode has no hard dependency
        self._client = anthropic.Anthropic()

    def complete_json(self, model: str, system: str, user: str) -> dict[str, Any]:
        resp = self._client.messages.create(
            model=model,
            max_tokens=2000,
            system=system + "\n\nRespond with ONLY valid JSON, no prose, no markdown fences.",
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return json.loads(text)


class MockLLM(LLMClient):
    """
    Deterministic canned responses keyed by a marker in the prompt, so the
    demo pipeline is runnable and reviewable without live API access.
    Swap this for AnthropicLLMClient() to run against the real API.
    """

    def __init__(self, fixtures: dict[str, dict]):
        self.fixtures = fixtures

    def complete_json(self, model: str, system: str, user: str) -> dict[str, Any]:
        # Pick the LONGEST matching marker, not the first -- a per-document
        # marker like "doc_policy_001" can legitimately appear inside a
        # later-stage prompt (e.g. verification includes a field summary
        # that names every document), so length disambiguates the more
        # specific match rather than accidentally matching stage 2's
        # fixture for a stage 4/5 prompt.
        # Prefer a marker the prompt actually STARTS WITH (every stage in
        # this demo puts its marker on the first line) over a merely
        # substring-contained one -- a per-document id like
        # "doc_invoice_independent" can be longer than a stage marker like
        # "COVERAGE_DETERMINATION" and would otherwise win on length alone
        # even though it's just a cited document_id inside a later prompt.
        prefix_matches = [(m, r) for m, r in self.fixtures.items() if user.startswith(m)]
        if prefix_matches:
            _, response = max(prefix_matches, key=lambda pair: len(pair[0]))
            return response
        candidates = [(marker, resp) for marker, resp in self.fixtures.items() if marker in user]
        if candidates:
            _, response = max(candidates, key=lambda pair: len(pair[0]))
            return response
        raise KeyError(
            f"MockLLM has no fixture matching this prompt. Add one, or "
            f"switch to AnthropicLLMClient for live calls.\nPrompt head: {user[:200]}"
        )


def get_client() -> LLMClient:
    if os.environ.get("CLAIMS_PIPELINE_LIVE") == "1":
        return AnthropicLLMClient()
    from sample_data.fixtures import FIXTURES
    return MockLLM(FIXTURES)
