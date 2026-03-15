"""LLM abstraction layer for HumanQA.

Supports Anthropic (Claude), OpenAI, and Google Gemini.
Includes tiered model configuration for cost optimization.

All LLM calls go through this module so prompts and model selection
are explicit and inspectable.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Tiered Model Configuration
# ---------------------------------------------------------------------------

@dataclass
class TierModels:
    """Model assignments for fast (cheap/quick) and smart (capable) tiers."""
    fast: str  # For dedup, classification, journey planning, summaries
    smart: str  # For evaluation, vision, complex analysis


# Presets: each maps a tier name to (provider, fast_model, smart_model)
TIER_PRESETS: dict[str, tuple[str, TierModels]] = {
    "balanced": (
        "gemini",
        TierModels(
            fast="gemini-2.0-flash",
            smart="gemini-2.0-flash",
        ),
    ),
    "budget": (
        "gemini",
        TierModels(
            fast="gemini-2.5-flash",
            smart="gemini-3-flash",
        ),
    ),
    "premium": (
        "anthropic",
        TierModels(
            fast="claude-sonnet-4-20250514",
            smart="claude-sonnet-4-20250514",
        ),
    ),
    "openai": (
        "openai",
        TierModels(
            fast="gpt-4.1",
            smart="gpt-5.4",
        ),
    ),
}

DEFAULT_TIER = "balanced"


def get_tier_config(tier: str) -> tuple[str, TierModels]:
    """Get (provider, tier_models) for a named tier preset."""
    if tier not in TIER_PRESETS:
        raise ValueError(
            f"Unknown tier '{tier}'. Available: {', '.join(TIER_PRESETS.keys())}"
        )
    return TIER_PRESETS[tier]


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Unified LLM interface with tiered model support."""

    def __init__(
        self,
        provider: str = "gemini",
        model: str | None = None,
        tier: str | None = None,
    ):
        self.provider = provider
        self._tier_models: TierModels | None = None
        self._client: Any = None
        self._gemini_deferred = False

        # If tier is specified, override provider and model
        if tier:
            self.provider, self._tier_models = get_tier_config(tier)
            # model param still wins if explicitly set
            if model:
                self._tier_models = TierModels(fast=model, smart=model)

        # Set up provider client
        if self.provider == "anthropic":
            import anthropic
            self.model = model or (self._tier_models.smart if self._tier_models else "claude-sonnet-4-20250514")
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            )
        elif self.provider == "openai":
            import openai
            self.model = model or (self._tier_models.smart if self._tier_models else "gpt-4o")
            self._client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
            )
        elif self.provider == "gemini":
            from google import genai
            self.model = model or (self._tier_models.smart if self._tier_models else "gemini-2.0-flash")
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
            if api_key:
                self._client = genai.Client(api_key=api_key)
            else:
                # Defer initialization — will fail at first actual call
                self._client = None
                self._gemini_deferred = True
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _ensure_client(self) -> None:
        """Lazily initialize provider client if deferred."""
        if self._client is None and getattr(self, "_gemini_deferred", False):
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY or GOOGLE_API_KEY environment variable required. "
                    "Get a free key at https://ai.google.dev/gemini-api/docs/api-key"
                )
            self._client = genai.Client(api_key=api_key)
            self._gemini_deferred = False

    def _resolve_model(self, tier_tag: str = "smart") -> str:
        """Resolve model name based on tier tag (fast or smart)."""
        tier_models = getattr(self, "_tier_models", None)
        if tier_models:
            return tier_models.fast if tier_tag == "fast" else tier_models.smart
        return self.model

    # ------------------------------------------------------------------
    # Text completions
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
        tier: str = "smart",
    ) -> str:
        """Get a text completion. Use tier='fast' for cheap tasks, 'smart' for complex ones."""
        model = self._resolve_model(tier)

        if self.provider == "anthropic":
            msg = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "You are HumanQA, an AI QA evaluation system.",
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text

        elif self.provider == "openai":
            messages: list[dict[str, Any]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""

        elif self.provider == "gemini":
            self._ensure_client()
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=system or "You are HumanQA, an AI QA evaluation system.",
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            response = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return response.text or ""

        raise ValueError(f"Unsupported provider: {self.provider}")

    # ------------------------------------------------------------------
    # Vision completions
    # ------------------------------------------------------------------

    def complete_with_vision(
        self,
        prompt: str,
        images: list[tuple[bytes, str]],  # (image_bytes, media_type)
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
        tier: str = "smart",
    ) -> str:
        """Get a text completion that includes image inputs."""
        model = self._resolve_model(tier)

        if self.provider == "anthropic":
            content: list[dict[str, Any]] = []
            for image_bytes, media_type in images:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                    },
                })
            content.append({"type": "text", "text": prompt})
            msg = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "You are HumanQA, an AI QA evaluation system.",
                messages=[{"role": "user", "content": content}],
            )
            return msg.content[0].text

        elif self.provider == "openai":
            content_parts: list[dict[str, Any]] = []
            for image_bytes, media_type in images:
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64}"},
                })
            content_parts.append({"type": "text", "text": prompt})
            messages: list[dict[str, Any]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": content_parts})
            resp = self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""

        elif self.provider == "gemini":
            self._ensure_client()
            from google.genai import types
            # Build multimodal content for Gemini
            parts: list[types.Part] = []
            for image_bytes, media_type in images:
                parts.append(types.Part.from_bytes(data=image_bytes, mime_type=media_type))
            parts.append(types.Part.from_text(text=prompt))

            config = types.GenerateContentConfig(
                system_instruction=system or "You are HumanQA, an AI QA evaluation system.",
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            response = self._client.models.generate_content(
                model=model,
                contents=parts,
                config=config,
            )
            return response.text or ""

        raise ValueError(f"Unsupported provider: {self.provider}")

    # ------------------------------------------------------------------
    # JSON completions
    # ------------------------------------------------------------------

    def complete_json(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tier: str = "smart",
    ) -> Any:
        """Get a JSON-structured completion. Extracts JSON from response."""
        full_system = (system or "You are HumanQA, an AI QA evaluation system.") + (
            "\n\nRespond ONLY with valid JSON. No markdown fences, no preamble, no explanation."
        )
        raw = self.complete(
            prompt, system=full_system, max_tokens=max_tokens,
            temperature=temperature, tier=tier,
        )
        return self._extract_json(raw)

    def complete_json_with_vision(
        self,
        prompt: str,
        images: list[tuple[bytes, str]],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tier: str = "smart",
    ) -> Any:
        """Get a JSON-structured completion that includes image inputs."""
        full_system = (system or "You are HumanQA, an AI QA evaluation system.") + (
            "\n\nRespond ONLY with valid JSON. No markdown fences, no preamble, no explanation."
        )
        raw = self.complete_with_vision(
            prompt, images, system=full_system,
            max_tokens=max_tokens, temperature=temperature, tier=tier,
        )
        return self._extract_json(raw)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(raw: str) -> Any:
        """Extract JSON from an LLM response, stripping markdown fences if present."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
