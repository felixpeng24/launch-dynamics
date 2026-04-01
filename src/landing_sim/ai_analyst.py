"""OpenAI-powered analysis assistant for interpreting Monte Carlo results."""

import os
import json
import numpy as np
from dotenv import load_dotenv

from landing_sim.loads import LoadStatistics

load_dotenv()

SYSTEM_PROMPT = """You are a senior structural dynamics engineer reviewing Monte Carlo landing \
load analysis results for a reusable launch vehicle (Starship-class). You specialize in loads \
and dynamics analysis for launch vehicles.

When analyzing results:
- Reference specific numbers from the data provided
- Identify which dispersion parameters most influence peak loads
- Flag any load margins that may be concerning
- Use proper engineering terminology (axial force, shear, overturning moment, etc.)
- Provide actionable insights for vehicle design
- Be concise but thorough

Format responses in clear paragraphs with key findings highlighted."""


def is_available() -> bool:
    """Check if OpenAI API key is configured."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def serialize_results(stats: dict[str, LoadStatistics],
                      n_cases: int,
                      settled_pct: float,
                      config_summary: dict | None = None) -> str:
    """Serialize Monte Carlo results into a context string for the LLM."""
    lines = [
        f"## Monte Carlo Landing Load Analysis — {n_cases} Cases",
        f"Settled successfully: {settled_pct:.1f}%",
        "",
        "### Peak Load Statistics",
        f"{'Quantity':<30} {'Mean':>12} {'Std':>12} {'P95':>12} {'P99':>12} {'Max':>12}",
        "-" * 90,
    ]

    for key, s in stats.items():
        scale = 1e-6 if "force" in key.lower() or "moment" in key.lower() else 1.0
        unit = "MN" if scale == 1e-6 else s.unit
        lines.append(
            f"{s.name:<30} {s.mean*scale:>12.3f} {s.std*scale:>12.3f} "
            f"{s.p95*scale:>12.3f} {s.p99*scale:>12.3f} {s.max_val*scale:>12.3f} {unit}"
        )

    if config_summary:
        lines.extend(["", "### Configuration"])
        for k, v in config_summary.items():
            lines.append(f"- {k}: {v}")

    return "\n".join(lines)


def query(user_question: str, results_context: str,
          model: str = "gpt-4o") -> str:
    """Send a query to the OpenAI API with Monte Carlo results as context.

    Args:
        user_question: The user's question about the results.
        results_context: Serialized results string from serialize_results().
        model: OpenAI model to use.

    Returns:
        The LLM's analysis response.

    Raises:
        RuntimeError: If API key is not set.
    """
    if not is_available():
        raise RuntimeError(
            "OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
        )

    from openai import OpenAI
    client = OpenAI()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Here are the Monte Carlo simulation results:\n\n"
            f"{results_context}\n\n"
            f"Question: {user_question}"
        )},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=1500,
    )

    return response.choices[0].message.content


def get_default_questions() -> list[str]:
    """Return a list of suggested analysis questions."""
    return [
        "Summarize the key findings from this Monte Carlo analysis.",
        "Which dispersion parameters most influence the peak axial force?",
        "Are any load margins concerning for structural design?",
        "Describe the worst-case landing scenario from these results.",
        "What design recommendations would you make based on these loads?",
        "How do the lateral loads compare to the axial loads in terms of design criticality?",
    ]
