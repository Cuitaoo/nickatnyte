"""Optional LLM reranking of top retrieval candidates.

Retrieval boosts are coarse: many candidates share identical boost sets, so
their relative order inside the top 10 is fusion noise. When OpenAI is
enabled, one extra structured call per turn reorders the top candidates
against the accumulated preferences. Every failure path leaves the original
order untouched, and offline behavior is byte-identical to running without a
reranker.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from starter.retrieval import RankedCandidate
from starter.state import ShoppingState


DEFAULT_LIMIT = 20
SYSTEM_PROMPT = (
    "Rank the numbered products for this shopper, best match first. "
    "Judge only against the shopper's stated preferences and latest message. "
    "Return every product index exactly once."
)


class InvalidRerank(RuntimeError):
    """The model did not return a valid permutation of the candidate list."""


class RerankOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: list[int]


@dataclass(frozen=True)
class RerankResult:
    ordering: tuple[str, ...]
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _snippet(value: object, limit: int = 200) -> str:
    return str(value or "")[:limit]


class CandidateReranker:
    def __init__(self, model: object) -> None:
        self._structured_model = model.with_structured_output(
            RerankOutput, include_raw=True
        )

    @classmethod
    def from_environment(cls) -> "CandidateReranker | None":
        enabled = os.getenv("OPENAI_ENABLED", "true").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return None
        # Off by default: reranking doubles the per-turn API calls, so it must
        # be a deliberate opt-in (see the submission-posture section of the
        # README and docs/evaluations/fable-model-comparison.md).
        rerank_enabled = os.getenv("OPENAI_RERANK_ENABLED", "false").strip().lower()
        if rerank_enabled not in {"1", "true", "yes", "on"}:
            return None
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        from langchain_openai import ChatOpenAI

        from starter.llm_agent import _bounded_float, _bounded_int

        model_name = os.getenv("OPENAI_MODEL", "gpt-5.6-luna").strip()
        timeout = _bounded_float(
            os.getenv("OPENAI_TIMEOUT_SECONDS", "20"), minimum=1.0, maximum=60.0
        )
        max_retries = _bounded_int(
            os.getenv("OPENAI_MAX_RETRIES", "1"), minimum=0, maximum=3
        )
        model = ChatOpenAI(
            api_key=api_key,
            model=model_name or "gpt-5.6-luna",
            timeout=timeout,
            max_retries=max_retries,
            use_responses_api=True,
        )
        return cls(model)

    def rerank(
        self,
        state: ShoppingState,
        latest_message: str,
        candidates: Sequence[RankedCandidate],
        metadata: Mapping[str, dict],
        limit: int = DEFAULT_LIMIT,
    ) -> RerankResult:
        head = list(candidates[:limit])
        if len(head) < 2:
            return RerankResult(
                ordering=tuple(item.product_id for item in candidates)
            )
        payload = {
            "preferences": state.to_prompt_dict(),
            "latest_message": latest_message,
            "products": [
                {
                    "index": index,
                    "title": _snippet(metadata[item.product_id].get("title")),
                    "price": metadata[item.product_id].get("price"),
                    "details_snippet": _snippet(
                        metadata[item.product_id].get("details")
                    ),
                }
                for index, item in enumerate(head)
            ],
        }
        from langchain_core.messages import HumanMessage, SystemMessage

        result = self._structured_model.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload, sort_keys=True)),
            ]
        )
        if not isinstance(result, dict):
            raise InvalidRerank("structured rerank result has an unexpected shape")
        parsed = result.get("parsed")
        prompt_tokens, completion_tokens = _raw_usage(result.get("raw"))
        if not isinstance(parsed, RerankOutput):
            raise InvalidRerank("rerank output failed to parse")
        order = parsed.order
        if sorted(order) != list(range(len(head))):
            raise InvalidRerank("rerank order is not a permutation of the candidates")
        ordering = tuple(head[index].product_id for index in order) + tuple(
            item.product_id for item in candidates[limit:]
        )
        return RerankResult(
            ordering=ordering,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def _raw_usage(raw: object) -> tuple[int, int]:
    from starter.llm_agent import _usage_from_message

    try:
        return _usage_from_message(raw)  # type: ignore[arg-type]
    except Exception:
        return 0, 0
