from __future__ import annotations

import asyncio

from dotenv import load_dotenv

load_dotenv()

from app.extraction import StructuredExtractor, build_extraction_context
from app.orchestration.pipeline import ResearchPipeline


def _combine_contexts(*contexts) -> str:
    sections = []

    for context in contexts:
        if context.blocks:
            sections.append(
                f"===== {context.field.upper()} EVIDENCE =====\n"
                f"{context.as_prompt_text()}"
            )

    return "\n\n".join(sections)


async def main() -> None:
    domain = "postman.com"

    print("=" * 72)
    print("END-TO-END EXTRACTION: CRAWLER -> EVIDENCE -> GEMINI")
    print("=" * 72)

    pipeline = ResearchPipeline()
    result = await pipeline.run(domain)
    state = result.state

    print(f"crawl status:      {state.status.value}")
    print(f"termination:       {state.termination_reason.value}")
    print(f"pages crawled:     {len(state.pages)}")
    print(f"pages discovered:  {state.metrics.pages_discovered}")
    print()

    if not state.pages:
        raise RuntimeError("Crawler returned no evidence pages")

    pages = list(state.pages.values())
    contexts = build_extraction_context(
        pages,
        max_chars_per_field=8000,
    )

    for field in ("overview", "icp", "contacts", "leadership"):
        context = contexts[field]
        print(
            f"{field:12} "
            f"pages={len(context.blocks):2d} "
            f"chars={context.character_count:5d}"
        )

    extractor = StructuredExtractor()

    overview_icp_text = _combine_contexts(
        contexts["overview"],
        contexts["icp"],
    )

    contacts_leadership_text = _combine_contexts(
        contexts["contacts"],
        contexts["leadership"],
    )

    print()
    print("Calling Gemini for overview + ICP...")
    overview_icp = extractor.extract_overview_icp(
        overview_icp_text
    )

    print("Calling Gemini for contacts + leadership...")
    contacts_leadership = extractor.extract_contacts_leadership(
        contacts_leadership_text
    )

    print()
    print("=" * 72)
    print("EXTRACTION RESULT")
    print("=" * 72)

    print(f"provider #1: {overview_icp.provider}")
    print(f"model #1:    {overview_icp.model}")
    print(
        f"tokens #1:   "
        f"{overview_icp.usage.input_tokens} in / "
        f"{overview_icp.usage.output_tokens} out"
    )
    print()

    print(f"provider #2: {contacts_leadership.provider}")
    print(f"model #2:    {contacts_leadership.model}")
    print(
        f"tokens #2:   "
        f"{contacts_leadership.usage.input_tokens} in / "
        f"{contacts_leadership.usage.output_tokens} out"
    )
    print()

    print("OVERVIEW + ICP")
    print(overview_icp.value.model_dump_json(indent=2))
    print()

    print("CONTACTS + LEADERSHIP")
    print(contacts_leadership.value.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
