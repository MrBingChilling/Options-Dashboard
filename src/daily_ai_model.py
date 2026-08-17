from __future__ import annotations

import json
import time
from difflib import SequenceMatcher
from typing import Any, Iterable, Sequence

import pandas as pd
import requests

from src.daily_ai_analysis import AnalysisPacket, build_analysis_packet
from src.daily_ai_summary import (
    DailySummary,
    GENERATOR_VERSION,
    SummaryBullet,
)
from src.skew_collector import AUTO_SYMBOLS


DEFAULT_MODEL = "gpt-5.6"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
MIN_BULLETS = 4
MAX_BULLETS = 8
MAX_SUMMARY_WORDS = 520


SYSTEM_PROMPT = """You are the senior volatility strategist writing a daily options dashboard summary.

This is a fresh research assignment every day. Independently examine all supplied data, decide which relationships are genuinely material, and write the best analysis for this session. Do not fit the facts into recurring categories, a fixed outline, yesterday's thesis, or reusable headline/body templates. Prior reports are continuity evidence only: reassess them, challenge them when today's data disagrees, and never copy their wording or structure.

Analytical requirements:
- Rank insights by portfolio relevance, magnitude, breadth, persistence and historical extremity.
- Distinguish a one-session move from a 1W or 1M regime using all three comparison horizons and the trailing 60-day percentiles.
- Cross-check spot moves, ATM IV, call IV, put IV and 25-delta skew across both 1W and 1M tenors.
- Use basket breadth and compare equal-weight with 10% trimmed results so outlier-driven moves are not presented as broad shifts.
- Surface important contradictions, dispersion and ticker-level exceptions; do not force every named basket into the report.
- When there is no material new change, say that directly and support it with specific evidence.
- Use only the supplied numerical data. Never claim news, catalysts, option trades, buying, selling, demand, flows, positioning or investor intent. IV and skew describe relative option pricing, not observed order flow.
- Every factual number in the prose must come from the packet. Use vol points for IV/skew changes and percentages for spot returns.

Writing requirements:
- Return 4 to 8 concise insights with original, descriptive titles followed by a decisive bottom line.
- Aim for roughly 350 to 520 words total. Favor interpretation over a recital of numbers, but cite enough exact evidence to make each conclusion auditable.
- Vary the organization naturally according to what matters today. Do not use a mandatory opening, category list or standard closing formula.
- Write in clear professional English for an options-literate investor. No generic market commentary, filler, trading recommendation or invented explanation.
"""


SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "bullets": {
            "type": "array",
            "minItems": MIN_BULLETS,
            "maxItems": MAX_BULLETS,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1},
                    "body": {"type": "string", "minLength": 1},
                },
                "required": ["title", "body"],
                "additionalProperties": False,
            },
        },
        "bottom_line": {"type": "string", "minLength": 1},
    },
    "required": ["bullets", "bottom_line"],
    "additionalProperties": False,
}


class SummaryGenerationError(RuntimeError):
    """Raised when fresh model analysis cannot be produced and validated."""


def _response_output_text(payload: dict[str, Any]) -> str:
    if payload.get("status") == "incomplete":
        details = payload.get("incomplete_details") or "unknown reason"
        raise SummaryGenerationError(f"OpenAI response was incomplete: {details}")
    if payload.get("error"):
        raise SummaryGenerationError(f"OpenAI response error: {payload['error']}")
    for output in payload.get("output") or []:
        if not isinstance(output, dict):
            continue
        for content in output.get("content") or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "refusal":
                raise SummaryGenerationError(
                    f"OpenAI refused the summary request: {content.get('refusal', '')}"
                )
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
    raise SummaryGenerationError("OpenAI response did not contain structured output text.")


def _validated_model_summary(
    payload: Any,
    prior_summaries: Sequence[DailySummary] = (),
) -> tuple[tuple[SummaryBullet, ...], str]:
    if not isinstance(payload, dict):
        raise SummaryGenerationError("The model summary was not a JSON object.")
    raw_bullets = payload.get("bullets")
    bottom_line = str(payload.get("bottom_line") or "").strip()
    if not isinstance(raw_bullets, list) or not MIN_BULLETS <= len(raw_bullets) <= MAX_BULLETS:
        raise SummaryGenerationError(
            f"The model returned an invalid bullet count; expected {MIN_BULLETS}-{MAX_BULLETS}."
        )
    bullets: list[SummaryBullet] = []
    titles: set[str] = set()
    for item in raw_bullets:
        if not isinstance(item, dict):
            raise SummaryGenerationError("A model summary bullet was not an object.")
        title = str(item.get("title") or "").strip()
        body = str(item.get("body") or "").strip()
        if not title or not body:
            raise SummaryGenerationError("A model summary bullet had an empty title or body.")
        normalized = title.casefold()
        if normalized in titles:
            raise SummaryGenerationError("The model summary repeated a bullet title.")
        titles.add(normalized)
        bullets.append(SummaryBullet(title, body))
    if not bottom_line:
        raise SummaryGenerationError("The model summary had an empty bottom line.")
    word_count = len(
        " ".join(
            [*(f"{bullet.title} {bullet.body}" for bullet in bullets), bottom_line]
        ).split()
    )
    if word_count > MAX_SUMMARY_WORDS:
        raise SummaryGenerationError(
            f"The model summary exceeded {MAX_SUMMARY_WORDS} words ({word_count})."
        )
    prior_titles = {
        bullet.title.strip().casefold()
        for summary in prior_summaries
        for bullet in summary.bullets
        if bullet.title.strip()
    }
    if titles.intersection(prior_titles):
        raise SummaryGenerationError(
            "The model reused a prior headline instead of writing a fresh analysis."
        )
    current_text = " ".join(
        [*(f"{bullet.title} {bullet.body}" for bullet in bullets), bottom_line]
    ).casefold()
    for summary in prior_summaries:
        prior_text = " ".join(
            [
                *(f"{bullet.title} {bullet.body}" for bullet in summary.bullets),
                summary.bottom_line,
            ]
        ).casefold()
        if prior_text and SequenceMatcher(None, current_text, prior_text).ratio() >= 0.90:
            raise SummaryGenerationError(
                "The model output was too similar to a prior report to count as fresh analysis."
            )
    return tuple(bullets), bottom_line


def _request_model_analysis(
    packet: AnalysisPacket,
    *,
    api_key: str,
    model: str,
    prior_summaries: Sequence[DailySummary] = (),
    timeout: float = 120.0,
    max_attempts: int = 3,
) -> tuple[tuple[SummaryBullet, ...], str]:
    if not api_key.strip():
        raise SummaryGenerationError(
            "OPENAI_API_KEY is required; no deterministic prose fallback is permitted."
        )
    request_body = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Analyze this complete daily evidence packet and return only the "
                    "requested structured summary.\n\n"
                    + json.dumps(packet.payload, separators=(",", ":"), allow_nan=False)
                ),
            },
        ],
        "reasoning": {"effort": "high"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "daily_options_market_summary",
                "strict": True,
                "schema": SUMMARY_SCHEMA,
            }
        },
        "max_output_tokens": 8000,
        "store": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    retryable_statuses = {408, 409, 429, 500, 502, 503, 504}
    last_error = "unknown error"
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            response = requests.post(
                OPENAI_RESPONSES_URL,
                headers=headers,
                json=request_body,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt >= max_attempts:
                break
            time.sleep(min(2 ** (attempt - 1), 4))
            continue
        if response.status_code == 200:
            try:
                response_payload = response.json()
                output_text = _response_output_text(response_payload)
                return _validated_model_summary(
                    json.loads(output_text), prior_summaries
                )
            except (ValueError, requests.JSONDecodeError, SummaryGenerationError) as exc:
                last_error = f"Invalid model output: {exc}"
                if attempt >= max_attempts:
                    break
                request_body["input"][1]["content"] += (
                    "\n\nThe previous attempt failed validation. Produce a genuinely fresh, "
                    "complete summary that follows every constraint."
                )
                time.sleep(min(2 ** (attempt - 1), 4))
                continue
        last_error = f"HTTP {response.status_code}: {response.text[:500]}"
        if response.status_code not in retryable_statuses or attempt >= max_attempts:
            break
        retry_after = response.headers.get("Retry-After") if hasattr(response, "headers") else None
        try:
            delay = min(float(retry_after), 15.0) if retry_after else min(2 ** attempt, 8)
        except (TypeError, ValueError):
            delay = min(2 ** attempt, 8)
        time.sleep(delay)
    raise SummaryGenerationError(
        f"Fresh model analysis failed after {max(1, max_attempts)} attempt(s): {last_error}"
    )


def generate_daily_summary(
    history: pd.DataFrame,
    symbols: Iterable[str] = AUTO_SYMBOLS,
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    prior_summaries: Sequence[DailySummary] = (),
    analysis_packet: AnalysisPacket | None = None,
) -> DailySummary:
    packet = analysis_packet or build_analysis_packet(history, symbols, prior_summaries)
    bullets, bottom_line = _request_model_analysis(
        packet,
        api_key=api_key,
        model=model,
        prior_summaries=prior_summaries,
    )
    return DailySummary(
        snapshot_date=packet.snapshot_date,
        comparison_date=packet.comparison_date,
        symbol_count=packet.symbol_count,
        expected_symbol_count=packet.expected_symbol_count,
        bullets=bullets,
        bottom_line=bottom_line,
        input_signature=packet.input_signature,
        week_comparison_date=packet.week_comparison_date,
        month_comparison_date=packet.month_comparison_date,
        generator_version=f"{GENERATOR_VERSION}:{model}",
    )
