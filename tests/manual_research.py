from __future__ import annotations

import asyncio
import sys

from app.orchestration.pipeline import ResearchPipeline


async def research(domain: str) -> None:
    print("=" * 70)
    print(f"ADAPTIVE RESEARCH: {domain}")
    print("=" * 70)

    pipeline = ResearchPipeline()

    result = await pipeline.run(domain)
    state = result.state

    print()
    print("=" * 70)
    print("RESEARCH RESULT")
    print("=" * 70)

    print(f"domain:              {state.domain}")
    print(f"status:              {state.status.value}")
    print(
        f"termination:         "
        f"{state.termination_reason.value}"
    )

    print()
    print("Metrics:")
    print(
        f"  pages discovered:  "
        f"{state.metrics.pages_discovered}"
    )
    print(
        f"  pages crawled:     "
        f"{state.metrics.pages_crawled}"
    )
    print(
        f"  pages failed:      "
        f"{state.metrics.pages_failed}"
    )
    print(
        f"  iterations:        "
        f"{state.metrics.crawl_iterations}"
    )
    print(
        f"  avg information:   "
        f"{state.metrics.average_information_gain:.3f}"
    )

    print()
    print("Coverage:")

    for field, coverage in state.coverage.items():
        print(
            f"  {field.value:<12} "
            f"{coverage.state.value:<10} "
            f"{coverage.coverage_score:.2f}"
        )

    print()
    print("Research decisions:")

    for decision in result.decisions:
        print(
            f"  [{decision.iteration}] "
            f"priority={decision.priority:.3f} "
            f"gain={decision.information_gain:.3f}"
        )

        print(
            f"      {decision.url}"
        )

        print(
            f"      {decision.reason}"
        )

    print()
    print("Pages crawled:")

    for page in state.pages.values():
        print(
            f"  {page.source_kind.value:<14} "
            f"{page.url}"
        )

    print()
    print(
        f"Remaining candidates: "
        f"{len(state.viable_candidates())}"
    )

    if state.errors:
        print()
        print("Errors:")

        for error in state.errors:
            print(f"  - {error}")


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: "
            "python -m tests.manual_research <domain>"
        )
        raise SystemExit(1)

    asyncio.run(
        research(sys.argv[1])
    )


if __name__ == "__main__":
    main()