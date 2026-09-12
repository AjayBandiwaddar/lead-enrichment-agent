from __future__ import annotations

from dataclasses import dataclass

from app.orchestration.state import EvidenceState, TerminationReason


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: TerminationReason
    message: str


def evaluate_stop(
    state: EvidenceState,
    *,
    low_gain_streak: int = 0,
    min_information_gain: float = 0.03,
    saturation_streak: int = 2,
) -> StopDecision:
    """
    Decide whether the research loop should terminate.

    Ordering matters:
    1. Required evidence coverage reached.
    2. Budget exhausted.
    3. Frontier exhausted.
    4. Repeated low information gain.
    """

    if state.all_required_fields_covered():
        return StopDecision(
            should_stop=True,
            reason=TerminationReason.COVERAGE_REACHED,
            message="Required evidence coverage reached.",
        )

    budget = state.budget
    usage = state.usage

    if usage.pages_crawled >= budget.max_pages:
        return StopDecision(
            should_stop=True,
            reason=TerminationReason.BUDGET_EXHAUSTED,
            message="Maximum page budget reached.",
        )

    if usage.runtime_seconds >= budget.max_runtime_seconds:
        return StopDecision(
            should_stop=True,
            reason=TerminationReason.BUDGET_EXHAUSTED,
            message="Maximum runtime budget reached.",
        )

    if not state.viable_candidates():
        return StopDecision(
            should_stop=True,
            reason=TerminationReason.FRONTIER_EXHAUSTED,
            message="No viable research candidates remain.",
        )

    if (
        state.metrics.crawl_iterations >= saturation_streak
        and low_gain_streak >= saturation_streak
        and state.metrics.last_information_gain < min_information_gain
    ):
        return StopDecision(
            should_stop=True,
            reason=TerminationReason.SATURATED,
            message=(
                "Recent crawls produced insufficient additional evidence."
            ),
        )

    return StopDecision(
        should_stop=False,
        reason=TerminationReason.NONE,
        message="Research should continue.",
    )