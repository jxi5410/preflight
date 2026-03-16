"""Dynamic Persona Generator.

Generates a tailored team of user agents based on the Product Intent Model.
Personas are created dynamically from product context, not from a static list.
"""

from __future__ import annotations

import logging

from preflight.core.llm import LLMClient
from preflight.core.schemas import (
    AgentPersona,
    CognitiveBehavior,
    EmotionalState,
    Platform,
    ProductIntentModel,
    RunConfig,
)

logger = logging.getLogger(__name__)

PERSONA_SYSTEM_PROMPT = """You are the persona generation engine for Preflight, an AI QA system.

Given a Product Intent Model, generate a team of realistic user agents who would
evaluate this product from different perspectives. Each persona must be specific
to THIS product — not generic testers.

Rules:
- Generate 4-8 personas depending on product complexity
- Each persona must have distinct goals and behavioral styles
- Include at least one novice/first-time user
- Include at least one skeptical or demanding user
- If institutional_relevance is moderate or high, include institutional reviewers
- Assign device preferences realistically (mobile users for consumer products, etc.)
- Make patience_level and expertise_level realistic for the persona type

Respond with a JSON array of persona objects."""

PERSONA_PROMPT_TEMPLATE = """Generate a team of user agent personas for evaluating this product.

## Product Intent Model
- Product: {product_name} ({product_type})
- Target Audience: {target_audience}
- Primary Jobs: {primary_jobs}
- Critical Journeys: {critical_journeys}
- Trust-Sensitive Actions: {trust_sensitive_actions}
- Institutional Relevance: {institutional_relevance}

## Additional Hints
{persona_hints}

## Focus Flows
{focus_flows}

For each persona, provide:
- name: A realistic name/label (e.g. "Alex Chen, First-time User")
- role: Their role relative to the product
- persona_type: Category slug (e.g. "first_time_user", "power_user", "risk_compliance_reviewer")
- goals: What they're trying to accomplish (list of 2-4 goals)
- expectations: What they expect from the product (list of 2-4)
- patience_level: "low" | "moderate" | "high"
- expertise_level: "novice" | "intermediate" | "expert"
- behavioral_style: Brief description of how they interact
- device_preference: "web" | "mobile_web" | "mobile_app"
- cognitive_behavior: An object with:
  - attention_span: "scanner" | "skimmer" | "reader"
  - patience_threshold: integer (1-10), how many confusing steps before abandoning
  - exploration_style: "linear" | "curious" | "goal-driven"
  - error_tolerance: "low" | "medium" | "high"
  - jargon_comfort: "none" | "some" | "fluent"
  - comparison_anchors: list of 2-3 product names this persona would compare against

Cognitive behavior guidelines:
- A first-time non-technical user: scanner attention, linear exploration, low error tolerance, no jargon comfort
- A developer evaluating a tool: skimmer attention, curious exploration, medium error tolerance, fluent jargon
- A busy executive: scanner attention, goal-driven exploration, low patience, some jargon comfort
- Include 2-3 comparison anchors — products this persona would naturally compare against based on the product category

Respond with a JSON array of persona objects."""


def compute_emotional_baseline(persona: AgentPersona) -> EmotionalState:
    """Infer starting emotional state from persona attributes.

    Different persona types start with different emotional baselines:
    - First-time non-technical users: low confidence, neutral trust, high engagement
    - Power users: high confidence, moderate trust, moderate engagement
    - Skeptical buyers: high confidence, low trust, low engagement
    """
    p_type = persona.persona_type.lower()
    expertise = persona.expertise_level.lower()
    patience = persona.patience_level.lower()

    # Default baseline
    confidence = 0.7
    frustration = 0.0
    trust = 0.5
    engagement = 0.7
    delight = 0.0

    # Adjust by persona type
    if "first_time" in p_type or "new" in p_type:
        confidence = 0.4
        trust = 0.5
        engagement = 0.8
    elif "power" in p_type or "advanced" in p_type:
        confidence = 0.9
        trust = 0.6
        engagement = 0.5
    elif "skeptic" in p_type or "buyer" in p_type or "evaluator" in p_type:
        confidence = 0.8
        trust = 0.3
        engagement = 0.4
    elif "compliance" in p_type or "risk" in p_type or "institutional" in p_type:
        confidence = 0.8
        trust = 0.3
        engagement = 0.5

    # Adjust by expertise
    if expertise == "novice":
        confidence = min(confidence, 0.4)
    elif expertise == "expert":
        confidence = max(confidence, 0.8)

    # Adjust by patience
    if patience == "low":
        engagement = min(engagement, 0.5)
    elif patience == "high":
        engagement = max(engagement, 0.7)

    return EmotionalState(
        confidence=confidence,
        frustration=frustration,
        trust=trust,
        engagement=engagement,
        delight=delight,
    )


class PersonaGenerator:
    """Generates tailored agent personas from product context."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def generate_personas(
        self,
        intent: ProductIntentModel,
        config: RunConfig,
        memory_context: str = "",
    ) -> list[AgentPersona]:
        """Generate agent team from intent model."""
        prompt = PERSONA_PROMPT_TEMPLATE.format(
            product_name=intent.product_name,
            product_type=intent.product_type,
            target_audience=", ".join(intent.target_audience),
            primary_jobs=", ".join(intent.primary_jobs),
            critical_journeys=", ".join(intent.critical_journeys),
            trust_sensitive_actions=", ".join(intent.trust_sensitive_actions),
            institutional_relevance=intent.institutional_relevance.value,
            persona_hints=", ".join(config.persona_hints) if config.persona_hints else "(none)",
            focus_flows=", ".join(config.focus_flows) if config.focus_flows else "(auto)",
        )

        if memory_context:
            prompt += f"\n\n## Prior Evaluation History\n{memory_context}"

        logger.info("Generating agent personas for %s", intent.product_name)

        try:
            data = self.llm.complete_json(prompt, system=PERSONA_SYSTEM_PROMPT, tier="fast")
            if not isinstance(data, list):
                data = data.get("personas", data.get("agents", []))

            personas = []
            for item in data:
                # Map device_preference string to enum
                dp = item.get("device_preference", "web")
                dp_map = {
                    "web": Platform.web,
                    "mobile_web": Platform.mobile_web,
                    "mobile_app": Platform.mobile_app,
                }
                item["device_preference"] = dp_map.get(dp, Platform.web)

                # Parse cognitive_behavior if provided by LLM
                cb_raw = item.pop("cognitive_behavior", None)
                if isinstance(cb_raw, dict):
                    item["cognitive_behavior"] = CognitiveBehavior(**cb_raw)

                personas.append(AgentPersona(**item))

            # Guarantee at least one mobile_web persona
            has_mobile = any(
                p.device_preference == Platform.mobile_web for p in personas
            )
            if not has_mobile and personas:
                # Promote the last persona to mobile_web
                personas[-1].device_preference = Platform.mobile_web
                logger.info("Promoted persona '%s' to mobile_web to ensure mobile coverage", personas[-1].name)

            # Apply emotional baselines
            for p in personas:
                p.emotional_state = compute_emotional_baseline(p)

            logger.info("Generated %d personas", len(personas))
            return personas

        except Exception as e:
            logger.error("Persona generation failed: %s", e)
            # Fallback: minimal set
            return [
                AgentPersona(
                    name="Default First-Time User",
                    role="New user evaluating the product",
                    persona_type="first_time_user",
                    goals=["Complete basic onboarding", "Understand what the product does"],
                    patience_level="moderate",
                    expertise_level="novice",
                    behavioral_style="Cautious, reads carefully",
                ),
                AgentPersona(
                    name="Default Skeptical User",
                    role="User evaluating whether to trust/adopt",
                    persona_type="skeptical_buyer",
                    goals=["Assess product quality", "Look for red flags"],
                    patience_level="low",
                    expertise_level="intermediate",
                    behavioral_style="Quick to judge, notices inconsistencies",
                ),
            ]
