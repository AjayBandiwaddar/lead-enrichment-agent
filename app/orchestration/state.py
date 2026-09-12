from __future__ import annotations

from app.planning.ranking import CandidateStatus, CandidateURL

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.evidence.models import PageEvidence
from app.orchestration.budgets import Budget, BudgetUsage
from app.planning.coverage import EvidenceField, FieldCoverage
from app.planning.ranking import CandidateURL


class PipelineStatus(str, Enum):
    """Overall state of one domain-processing run."""

    INITIALIZING = "initializing"
    DISCOVERING = "discovering"
    CRAWLING = "crawling"
    EXTRACTING = "extracting"
    VERIFYING = "verifying"
    SEARCHING = "searching"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class TerminationReason(str, Enum):
    """Why the adaptive crawl stopped."""

    NONE = "none"
    COVERAGE_REACHED = "coverage_reached"
    SATURATED = "saturated"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FRONTIER_EXHAUSTED = "frontier_exhausted"
    REPEATED_FAILURES = "repeated_failures"
    FATAL_ERROR = "fatal_error"


class CrawlMetrics(BaseModel):
    """Lightweight metrics useful for the final report and Loom demo."""

    model_config = ConfigDict(extra="forbid")

    pages_discovered: int = 0
    pages_crawled: int = 0
    pages_failed: int = 0
    pages_skipped: int = 0

    evidence_items_added: int = 0

    crawl_iterations: int = 0

    average_information_gain: float = 0.0
    last_information_gain: float = 0.0

    external_searches: int = 0


class EvidenceState(BaseModel):
    """
    Current knowledge state for one company.

    This is the state that the adaptive planner reads and updates.
    """

    model_config = ConfigDict(extra="forbid")

    domain: str

    status: PipelineStatus = PipelineStatus.INITIALIZING

    # -------------------------
    # Evidence
    # -------------------------

    pages: dict[str, PageEvidence] = Field(default_factory=dict)

    # -------------------------
    # Research frontier
    # -------------------------

    candidates: dict[str, CandidateURL] = Field(default_factory=dict)

    visited_urls: set[str] = Field(default_factory=set)
    failed_urls: set[str] = Field(default_factory=set)

    # -------------------------
    # Field coverage
    # -------------------------

    coverage: dict[EvidenceField, FieldCoverage] = Field(
        default_factory=lambda: {
            field: FieldCoverage(field=field)
            for field in EvidenceField
        }
    )

    # -------------------------
    # Execution
    # -------------------------

    budget: Budget = Field(default_factory=Budget)
    usage: BudgetUsage = Field(default_factory=BudgetUsage)

    metrics: CrawlMetrics = Field(default_factory=CrawlMetrics)

    # -------------------------
    # Termination
    # -------------------------

    termination_reason: TerminationReason = TerminationReason.NONE

    # -------------------------
    # Errors
    # -------------------------

    errors: list[str] = Field(default_factory=list)

    def add_page(self, page: PageEvidence) -> None:
        self.pages[page.page_id] = page

        self.visited_urls.add(str(page.url))

        self.metrics.pages_crawled += 1

        if page.fetch_status.value != "success":
            self.metrics.pages_failed += 1
            self.failed_urls.add(str(page.url))

    def add_candidate(self, candidate: CandidateURL) -> None:
        url = str(candidate.url)

        # Don't re-add URLs that have already been successfully crawled.
        if url in self.visited_urls:
            return

        existing = self.candidates.get(url)

        if existing is None:
            self.candidates[url] = candidate
            self.metrics.pages_discovered += 1
            return

        # Keep the stronger candidate if the URL was discovered multiple times.
        if candidate.priority > existing.priority:
            self.candidates[url] = candidate

    def mark_candidate_failed(self, url: str) -> None:
        candidate = self.candidates.get(url)

        if candidate is None:
            return

        candidate.failure_count += 1

        if candidate.failure_count >= self.budget.max_retries_per_resource:
            candidate.status = candidate.status.FAILED
            self.failed_urls.add(url)

    def field_missingness(self) -> dict[EvidenceField, float]:
        """
        Convert current field coverage into missingness used by the planner.

        1.0 = completely missing
        0.0 = fully covered
        """
        return {
            field: 1.0 - field_coverage.coverage_score
            for field, field_coverage in self.coverage.items()
        }

    def all_required_fields_covered(self, threshold: float = 0.75) -> bool:
        return all(
            field_coverage.coverage_score >= threshold
            for field_coverage in self.coverage.values()
        )

    def viable_candidates(self) -> list[CandidateURL]:
        return [
            candidate
            for candidate in self.candidates.values()
            if candidate.status == CandidateStatus.QUEUED
            and str(candidate.url) not in self.visited_urls
            and str(candidate.url) not in self.failed_urls
        ]

    def add_error(self, message: str) -> None:
        self.errors.append(message)

        # Keep the most recent errors useful without allowing runaway growth.
        if len(self.errors) > 50:
            self.errors = self.errors[-50:]