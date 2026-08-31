from __future__ import annotations

import re
from dataclasses import dataclass
from starter import config


SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9$]+", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
REQUEST_PREFIX_RE = re.compile(
    r"\b(?:look(?:ing)?|search(?:ing)?|shop(?:ping)?)\s+for\b",
    re.IGNORECASE,
)
VAGUE_GOAL_RE = re.compile(
    r"\b(?:good|best|ideal|suitable|appropriate|recommended?|something)\b",
    re.IGNORECASE,
)
FOR_RE = re.compile(r"\bfor\s+", re.IGNORECASE)
ACTIONABLE_SCENARIO_RE = re.compile(
    r"\b(?:campus|cold|commut(?:e|er|ing)|gym|hiking|office|outdoor|rain|rainy|"
    r"running|school|snow|travel|uni|university|wet|winter|work)\b",
    re.IGNORECASE,
)
MAX_SCENARIO_CONTENT_TERMS = 10


@dataclass(frozen=True)
class ScenarioHypothesis:
    """Temporary scenario language for feature-vector recall only.

    The deterministic state supplies category and confirmed-feature queries.
    This value holds only inferred functional language, such as "waterproof
    traction", and is deliberately never treated as an active preference.
    """

    scenario_query: str
    basis: str
    confidence: float


def query_expansion_mode() -> str:
    if not _env_bool("TECHJAM_QUERY_EXPANSION_ENABLED", False):
        return "off"
    mode = config.getenv("TECHJAM_QUERY_EXPANSION_MODE", "shadow").strip().lower()
    return mode if mode in {"shadow", "recall"} else "shadow"


def query_expansion_enabled() -> bool:
    return query_expansion_mode() != "off"


def query_expansion_min_confidence() -> float:
    return _env_float("TECHJAM_QUERY_EXPANSION_MIN_CONFIDENCE", 0.60, 0.0, 1.0)


def query_expansion_max_hypotheses() -> int:
    return _env_int("TECHJAM_QUERY_EXPANSION_MAX_HYPOTHESES", 3, 1, 5)


def looks_like_scenario_query(message: str) -> bool:
    """Detect actionable, under-specified goals without guessing from location."""

    normalized = SPACE_RE.sub(" ", str(message)).strip()
    without_request_prefix = REQUEST_PREFIX_RE.sub("", normalized)
    content_terms = TOKEN_RE.findall(without_request_prefix)
    if len(content_terms) > MAX_SCENARIO_CONTENT_TERMS:
        return False
    if not FOR_RE.search(without_request_prefix):
        return False
    # A place name alone does not establish weather, season, or a functional
    # need. Only open-ended requests with an actionable scenario may use the
    # optional semantic-recall branch.
    return bool(
        VAGUE_GOAL_RE.search(without_request_prefix)
        and ACTIONABLE_SCENARIO_RE.search(without_request_prefix)
    )


def validate_scenario_hypotheses(
    hypotheses: list[ScenarioHypothesis], latest_message: str
) -> tuple[ScenarioHypothesis, ...]:
    """Validate scenario language before it enters the candidate-recall route."""

    if not query_expansion_enabled():
        return ()
    minimum = query_expansion_min_confidence()
    maximum = query_expansion_max_hypotheses()
    message_tokens = TOKEN_RE.findall(str(latest_message).lower())
    message_token_set = set(message_tokens)
    accepted: list[ScenarioHypothesis] = []
    seen_queries: set[str] = set()
    for item in hypotheses:
        raw_query = SPACE_RE.sub(" ", str(item.scenario_query)).strip(
            " \t\r\n,.;:"
        )[:240]
        basis = SPACE_RE.sub(" ", str(item.basis)).strip(" \t\r\n,.;:")[:120]
        confidence = min(1.0, max(0.0, float(item.confidence)))
        if not raw_query or not basis or confidence < minimum:
            continue
        if not _is_token_span(basis, message_tokens):
            continue
        # A generated number can silently change a hard budget, size, or model
        # identifier. Every numeric token in an expansion must already be explicit.
        query_numbers = {
            token.replace(",", "") for token in NUMBER_RE.findall(raw_query.lower())
        }
        message_numbers = {
            token.replace(",", "")
            for token in NUMBER_RE.findall(str(latest_message).lower())
        }
        if not query_numbers.issubset(message_numbers):
            continue
        # Keep this route semantically separate from the original/category
        # route. The category, audience, place, identifiers, and stated
        # constraints are already represented by deterministic routes. Leaving
        # only generated functional vocabulary avoids one mixed embedding in
        # which a location term can dominate candidate recall.
        scenario_terms = [
            token
            for token in TOKEN_RE.findall(raw_query.lower())
            if token not in message_token_set
        ]
        scenario_query = " ".join(_dedupe_terms(scenario_terms))[:240]
        if not scenario_query:
            continue
        normalized_query = scenario_query.lower()
        if normalized_query in seen_queries:
            continue
        seen_queries.add(normalized_query)
        accepted.append(
            ScenarioHypothesis(
                scenario_query=scenario_query,
                basis=basis,
                confidence=confidence,
            )
        )
        if len(accepted) >= maximum:
            break
    return tuple(accepted)


def _dedupe_terms(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _is_token_span(value: str, message_tokens: list[str]) -> bool:
    value_tokens = TOKEN_RE.findall(str(value).lower())
    if not value_tokens or len(value_tokens) > len(message_tokens):
        return False
    width = len(value_tokens)
    return any(
        message_tokens[index : index + width] == value_tokens
        for index in range(len(message_tokens) - width + 1)
    )


def _env_bool(name: str, default: bool) -> bool:
    value = config.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(config.getenv(name, str(default)))
    except ValueError:
        value = default
    return min(maximum, max(minimum, value))


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(config.getenv(name, str(default)))
    except ValueError:
        value = default
    return min(maximum, max(minimum, value))
