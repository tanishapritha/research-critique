SUMMARY_PROMPT = """
Summarize the following paper in ~150 words. Include:
- Problem
- Method
- Findings
- Limitations

TITLE: {title}
ABSTRACT:
{abstract}
"""

SYNTH_PROMPT = """
You are a research synthesis assistant.

Combine these individual summaries into a cohesive field overview.
Include:
- Consensus
- Conflicting ideas
- Techniques used
- Gaps / challenges
- Future directions

SUMMARIES:
{summaries}
"""

CRIT_PROMPT = """
You are a critical research reviewer.

Critique the summary below. Describe:
- Weaknesses
- Missing perspectives
- Bias
- What could make it better

SUMMARY:
{s}
"""
