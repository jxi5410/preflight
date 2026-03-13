"""Webhook / Slack Summary — Post run results to a webhook endpoint.

Supports Slack-compatible webhook payloads.
"""

from __future__ import annotations

import json
import logging

import httpx

from humanqa.core.schemas import RunResult, Severity

logger = logging.getLogger(__name__)

SEVERITY_EMOJI = {
    "critical": "\U0001f534",  # red circle
    "high": "\U0001f7e0",     # orange circle
    "medium": "\U0001f7e1",   # yellow circle
    "low": "\U0001f535",      # blue circle
    "info": "\u26aa",         # white circle
}


def build_summary_text(result: RunResult) -> str:
    """Build a plain-text summary suitable for Slack or other webhooks."""
    intent = result.intent_model
    issues = result.issues

    severity_counts: dict[str, int] = {}
    for issue in issues:
        sev = issue.severity.value
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    lines: list[str] = []
    lines.append(f"*HumanQA Report: {intent.product_name or result.config.target_url}*")
    lines.append("")

    # Severity counts
    sev_parts: list[str] = []
    for sev in ["critical", "high", "medium", "low", "info"]:
        count = severity_counts.get(sev, 0)
        if count:
            emoji = SEVERITY_EMOJI.get(sev, "")
            sev_parts.append(f"{emoji} {count} {sev.title()}")
    if sev_parts:
        lines.append("  ".join(sev_parts))
    else:
        lines.append("No issues found.")

    # Top issue
    if issues:
        top = issues[0]
        lines.append(f'Top issue: "{top.title}"')

    # Scores
    scores = result.scores
    score_parts: list[str] = []
    if scores.get("trust_score") is not None:
        score_parts.append(f"Trust: {scores['trust_score']:.0%}")
    if scores.get("institutional_readiness_label"):
        score_parts.append(f"Inst. Readiness: {scores['institutional_readiness_label']}")
    if score_parts:
        lines.append(" | ".join(score_parts))

    return "\n".join(lines)


def build_slack_payload(result: RunResult, report_url: str | None = None) -> dict:
    """Build a Slack-compatible webhook payload."""
    text = build_summary_text(result)
    if report_url:
        text += f"\n<{report_url}|Full Report>"
    return {"text": text}


async def send_webhook(
    webhook_url: str,
    result: RunResult,
    report_url: str | None = None,
) -> bool:
    """Send a summary to a webhook URL. Returns True on success."""
    payload = build_slack_payload(result, report_url)

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code < 300:
                logger.info("Webhook sent successfully to %s", webhook_url)
                return True
            else:
                logger.warning(
                    "Webhook failed: %s %s", response.status_code, response.text[:200],
                )
                return False
    except Exception as e:
        logger.error("Webhook send failed: %s", e)
        return False
