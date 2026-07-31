"""
Prompt templates for the two independent evaluation axes:

1. TONE_JUDGE_PROMPT      -- scores how well a reply matches ByteHive's brand voice,
                              *independent of whether the policy content is correct*.
2. CLAIM_EXTRACTION_PROMPT -- pulls out discrete factual/policy claims from a reply.
3. GROUNDING_JUDGE_PROMPT  -- for each extracted claim, checks whether it is
                              supported by the retrieved policy context, unsupported
                              (hallucinated), or contradicted by it.

Keeping these on two separate axes is the whole point: a reply that nails tone but
invents a refund rule must score high on (1) and low on (2), not get averaged into
a single misleading number. See eval/evaluate.py for how these are combined into a
report without conflating the two.
"""

TONE_JUDGE_PROMPT = """You are grading a customer support reply for adherence to a brand voice guide.
Score ONLY tone/style -- do not consider whether policy facts stated are correct.

Brand voice guide:
{tone_guide}

Reply to grade:
\"\"\"{reply}\"\"\"

Score each dimension 1 (not at all) to 5 (fully matches):
- directness: does it lead with the answer/outcome?
- friendliness: contractions, natural language, no "Dear Valued Customer"?
- specificity: concrete numbers/dates rather than vague words like "shortly"?
- no_boilerplate: absence of corporate phrases ("we sincerely apologize for any
  inconvenience", "please do not hesitate", "your satisfaction is our top priority")?
- conciseness: short paragraphs, no walls of text?

Respond ONLY with JSON in this exact schema, no other text:
{{"directness": <1-5>, "friendliness": <1-5>, "specificity": <1-5>, "no_boilerplate": <1-5>, "conciseness": <1-5>, "rationale": "<one sentence>"}}
"""

CLAIM_EXTRACTION_PROMPT = """Extract every discrete factual/policy claim made in this customer support reply
(refund amounts, time windows, billing rules, cancellation rules, dates, etc).
Ignore pure pleasantries/sign-offs that assert no policy fact.

Reply:
\"\"\"{reply}\"\"\"

Respond ONLY with a JSON list of strings, one per claim, no other text. If there are
no factual/policy claims, respond with [].
"""

GROUNDING_JUDGE_PROMPT = """You are fact-checking ONE claim made in a customer support reply against the
policy context that was available to the agent when writing it.

Policy context available:
\"\"\"{context}\"\"\"

Claim to check:
\"\"\"{claim}\"\"\"

Label the claim as exactly one of:
- "supported": the policy context explicitly backs this claim.
- "unsupported": the policy context does not address this claim at all (this is a
  hallucination risk -- the claim might be true but nothing in the context confirms it).
- "contradicted": the policy context directly conflicts with this claim.

Respond ONLY with JSON: {{"label": "supported" | "unsupported" | "contradicted", "reason": "<one sentence>"}}
"""
