from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from app.planning.coverage import EvidenceField


class CandidateStatus(str, Enum):
    """Lifecycle state of a URL candidate."""

    QUEUED = "queued"
    CRAWLED = "crawled"
    SKIPPED = "skipped"
    FAILED = "failed"


class CandidateURL(BaseModel):
    """
    A URL that may provide additional evidence.

    The planner scores candidates using both page-level signals and the
    current evidence gaps.
    """

    model_config = ConfigDict(extra="forbid")

    url: HttpUrl

    anchor_text: str | None = None
    surrounding_text: str | None = None
    source_page_id: str | None = None

    depth: int = 0

    # Cheap structural signals.
    url_relevance: float = 0.0
    anchor_relevance: float = 0.0
    semantic_relevance: float = 0.0
    source_quality: float = 0.5
    novelty: float = 1.0

    # Estimated probability that this URL improves each missing field.
    expected_field_gain: dict[EvidenceField, float] = Field(
        default_factory=dict
    )

    estimated_cost: float = 1.0

    priority: float = 0.0

    status: CandidateStatus = CandidateStatus.QUEUED

    failure_count: int = 0

    reason: str | None = None

    @field_validator(
        "url_relevance",
        "anchor_relevance",
        "semantic_relevance",
        "source_quality",
        "novelty",
    )
    @classmethod
    def validate_signal(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("ranking signals must be between 0.0 and 1.0")
        return value

    @field_validator("expected_field_gain")
    @classmethod
    def validate_expected_gain(
        cls,
        value: dict[EvidenceField, float],
    ) -> dict[EvidenceField, float]:
        for field, gain in value.items():
            if not 0.0 <= gain <= 1.0:
                raise ValueError(
                    f"expected gain for {field} must be between 0.0 and 1.0"
                )
        return value

    @field_validator("estimated_cost")
    @classmethod
    def validate_cost(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("estimated_cost must be greater than zero")
        return value

    @field_validator("failure_count")
    @classmethod
    def validate_failure_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("failure_count cannot be negative")
        return value


def calculate_expected_information_gain(
    candidate: CandidateURL,
    field_missingness: dict[EvidenceField, float],
) -> float:
    """
    Estimate how much useful evidence a candidate is expected to add.

    Missingness is 1.0 for a completely unknown field and 0.0 for a fully
    covered field.
    """

    gain = 0.0

    for field, expected_gain in candidate.expected_field_gain.items():
        missingness = field_missingness.get(field, 0.0)
        gain += missingness * expected_gain

    return gain


def calculate_priority(
    candidate: CandidateURL,
    field_missingness: dict[EvidenceField, float],
) -> float:
    """
    Score a candidate by expected verified information gained per cost.

    This is intentionally deterministic and cheap. An LLM is not required
    for the first version of URL planning.
    """

    expected_gain = calculate_expected_information_gain(
        candidate,
        field_missingness,
    )

    relevance = (
        0.35 * candidate.url_relevance
        + 0.25 * candidate.anchor_relevance
        + 0.20 * candidate.semantic_relevance
        + 0.20 * candidate.source_quality
    )

    raw_priority = (
        expected_gain
        * relevance
        * candidate.novelty
        / candidate.estimated_cost
    )

    failure_penalty = 1.0 / (1.0 + 0.5 * candidate.failure_count)

    return raw_priority * failure_penalty