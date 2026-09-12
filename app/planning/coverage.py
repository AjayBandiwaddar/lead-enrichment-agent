from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvidenceField(str, Enum):
    """Required intelligence fields produced by the system."""

    OVERVIEW = "overview"
    ICP = "icp"
    CONTACTS = "contacts"
    LEADERSHIP = "leadership"


class CoverageState(str, Enum):
    """Semantic state of evidence for one required field."""

    UNKNOWN = "unknown"
    WEAK = "weak"
    PARTIAL = "partial"
    SUPPORTED = "supported"
    STRONG = "strong"
    CONFLICTED = "conflicted"


class FieldCoverage(BaseModel):
    """
    Tracks how well a single required field is supported by current evidence.

    coverage_score is intentionally separate from state:
    - score is useful for algorithms
    - state is useful for reasoning/debugging
    """

    model_config = ConfigDict(extra="forbid")

    field: EvidenceField
    state: CoverageState = CoverageState.UNKNOWN
    coverage_score: float = 0.0

    supporting_page_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)

    @field_validator("coverage_score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("coverage_score must be between 0.0 and 1.0")
        return value

    def is_sufficient(self, threshold: float) -> bool:
        return self.coverage_score >= threshold

    def add_page(self, page_id: str) -> None:
        if page_id not in self.supporting_page_ids:
            self.supporting_page_ids.append(page_id)

    def add_claim(self, claim_id: str) -> None:
        if claim_id not in self.claim_ids:
            self.claim_ids.append(claim_id)