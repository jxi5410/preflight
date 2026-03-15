"""Seed Input Generator.

Generates contextually appropriate inputs for personas to type into
input-first products (search engines, AI tools, URL analyzers, etc.).

Two modes:
- LLM-based: For full runs, uses smart tier to generate persona-specific inputs.
- Heuristic: For quick check, uses simple defaults based on input_type.
"""

from __future__ import annotations

import logging

from preflight.core.llm import LLMClient
from preflight.core.schemas import (
    AgentPersona,
    ProductIntentModel,
    SeedInput,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic seed inputs (no LLM needed, for quick check)
# ---------------------------------------------------------------------------

HEURISTIC_INPUTS: dict[str, str] = {
    "search": "test",
    "url": "example.com",
    "prompt": "Hello, can you help me?",
    "code": 'print("hello")',
    "data": "sample data",
    "free_text": "This is a test input",
}


def get_heuristic_seed_input(input_type: str) -> SeedInput:
    """Return a single heuristic seed input for quick check (no LLM call)."""
    text = HEURISTIC_INPUTS.get(input_type, HEURISTIC_INPUTS["free_text"])
    return SeedInput(
        input_text=text,
        purpose=f"Quick check: basic {input_type or 'text'} input",
        expected_outcome="Product should respond with relevant results",
    )


# ---------------------------------------------------------------------------
# LLM-based seed input generation (for full runs)
# ---------------------------------------------------------------------------

SEED_INPUT_SYSTEM_PROMPT = """You are a seed input generator for Preflight, an AI QA system.

Generate realistic inputs that a specific user persona would type into a product's
main input field. The inputs must be contextually appropriate for the product type,
input field type, and persona's expertise level and goals.

Rules:
- First input should be something simple and common (what most people would try first)
- Second input should test a realistic use case for this persona
- Third input (if applicable) should be an edge case or stress test
- Inputs must be realistic — not test data like "asdf" or "test123"
- Match the input type: if it's a URL field, give URLs; if search, give queries; etc.

Respond ONLY with valid JSON."""

SEED_INPUT_PROMPT_TEMPLATE = """Generate seed inputs for this persona to try on this product.

Product: {product_name} ({product_type})
Main input field placeholder: "{input_placeholder}"
Input type: {input_type}

Persona: {persona_name} — {persona_role}
Goals: {persona_goals}
Expertise: {expertise_level}

What would this person type into the input field? Generate 2-3 realistic inputs
this persona would try, ordered from most likely to least likely.

Respond with JSON:
{{
  "seed_inputs": [
    {{
      "input_text": "the text to type",
      "purpose": "why this persona would type this",
      "expected_outcome": "what a working product should show",
      "is_edge_case": false
    }}
  ]
}}"""


class SeedInputGenerator:
    """Generates contextually appropriate seed inputs for personas."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate_for_persona(
        self,
        intent: ProductIntentModel,
        persona: AgentPersona,
    ) -> list[SeedInput]:
        """Generate seed inputs for a persona using LLM (smart tier).

        Returns up to 3 seed inputs tailored to the persona and product.
        """
        if not intent.input_first:
            return []

        prompt = SEED_INPUT_PROMPT_TEMPLATE.format(
            product_name=intent.product_name,
            product_type=intent.product_type,
            input_placeholder=intent.input_placeholder,
            input_type=intent.input_type,
            persona_name=persona.name,
            persona_role=persona.role,
            persona_goals=", ".join(persona.goals),
            expertise_level=persona.expertise_level,
        )

        try:
            data = self.llm.complete_json(
                prompt, system=SEED_INPUT_SYSTEM_PROMPT, tier="smart",
            )
            raw_inputs = data.get("seed_inputs", [])
            seed_inputs = []
            for raw in raw_inputs[:3]:  # Max 3 per persona
                seed_inputs.append(SeedInput(
                    input_text=raw.get("input_text", ""),
                    purpose=raw.get("purpose", ""),
                    expected_outcome=raw.get("expected_outcome", ""),
                    is_edge_case=raw.get("is_edge_case", False),
                ))
            logger.info(
                "Generated %d seed inputs for persona %s",
                len(seed_inputs), persona.name,
            )
            return seed_inputs
        except Exception as e:
            logger.error("Seed input generation failed for %s: %s", persona.name, e)
            # Fallback to heuristic
            return [get_heuristic_seed_input(intent.input_type)]
