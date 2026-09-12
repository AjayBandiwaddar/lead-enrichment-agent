from __future__ import annotations

import asyncio

from dotenv import load_dotenv

load_dotenv()

from app.extraction import (
    CompanyExtraction,
    StructuredExtractor,
    build_extraction_context,
    verify_extraction,
)
from app.orchestration.pipeline import ResearchPipeline


def _context_text(context) -> str:
    if not context.blocks:
        return "[NO EVIDENCE BLOCKS SELECTED]"

    return context.as_prompt_text()


def _print_context_summary(contexts: dict) -> None:
    print("\n" + "=" * 70)
    print("FIELD CONTEXT SUMMARY")
    print("=" * 70)

    for field in (
        "overview",
        "icp",
        "contacts",
        "leadership",
    ):
        context = contexts[field]

        print(
            f"{field:12} "
            f"blocks={len(context.blocks):2d} "
            f"chars={context.character_count:4d}"
        )


def _merge_extractions(
    overview_icp,
    contacts,
    leadership,
) -> CompanyExtraction:
    return CompanyExtraction(
        overview=overview_icp.overview,
        icp_summary=overview_icp.icp_summary,
        target_segments=overview_icp.target_segments,
        buyer_roles=overview_icp.buyer_roles,
        use_cases=overview_icp.use_cases,
        contacts=contacts.contacts,
        leadership=leadership.leadership,
    )


def _print_usage(label: str, result) -> None:
    print(
        f"{label}: "
        f"provider={result.provider} "
        f"model={result.model} "
        f"input={result.usage.input_tokens} "
        f"output={result.usage.output_tokens}"
    )


async def main() -> None:
    pipeline = ResearchPipeline()

    result = await pipeline.run(
        "postman.com"
    )

    state = result.state
    pages = list(
        state.pages.values()
    )

    print("=" * 70)
    print("VERIFIED EXTRACTION V4")
    print("=" * 70)

    print(
        f"crawl status: {state.status.value}"
    )
    print(
        f"termination:  {state.termination_reason.value}"
    )
    print(
        f"pages:       {len(pages)}"
    )
    print(
        f"discovered:  "
        f"{state.metrics.pages_discovered}"
    )

    if not pages:
        raise RuntimeError(
            "No evidence pages returned"
        )

    contexts = build_extraction_context(
        pages,
        max_chars_per_field=4_500,
    )

    _print_context_summary(
        contexts
    )

    print(
        "\n"
        + "=" * 70
    )
    print("LEADERSHIP CONTEXT")
    print("=" * 70)

    print(
        _context_text(
            contexts["leadership"]
        )
    )

    extractor = StructuredExtractor()

    print(
        "\n"
        + "=" * 70
    )
    print("LLM EXTRACTION")
    print("=" * 70)

    overview = (
        extractor.extract_overview_icp(
            _context_text(
                contexts["overview"]
            )
            + "\n\n"
            + _context_text(
                contexts["icp"]
            )
        )
    )

    _print_usage(
        "overview/icp",
        overview,
    )

    contacts = (
        extractor.extract_contacts(
            _context_text(
                contexts["contacts"]
            )
        )
    )

    _print_usage(
        "contacts",
        contacts,
    )

    leadership = (
        extractor.extract_leadership(
            _context_text(
                contexts["leadership"]
            )
        )
    )

    _print_usage(
        "leadership",
        leadership,
    )

    raw = _merge_extractions(
        overview.value,
        contacts.value,
        leadership.value,
    )

    print(
        "\n"
        + "=" * 70
    )
    print("RAW EXTRACTION")
    print("=" * 70)

    print(
        raw.model_dump_json(
            indent=2
        )
    )

    verification = (
        verify_extraction(
            raw,
            pages,
        )
    )

    print(
        "\n"
        + "=" * 70
    )
    print("VERIFICATION")
    print("=" * 70)

    print(
        "overall confidence:",
        f"{verification.overall_confidence:.3f}",
    )

    for claim in verification.claims:
        print(
            f"\n[{claim.status}] "
            f"{claim.field} "
            f"confidence={claim.confidence:.3f}"
        )

        print(
            f"  {claim.value}"
        )

        if claim.evidence_ids:
            print(
                "  evidence: "
                + ", ".join(
                    claim.evidence_ids
                )
            )

        for reason in claim.reasons:
            print(
                f"  reason: {reason}"
            )

    print(
        "\n"
        + "=" * 70
    )
    print("FINAL VERIFIED EXTRACTION")
    print("=" * 70)

    print(
        verification.verified.model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    asyncio.run(main())