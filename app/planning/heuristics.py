from __future__ import annotations

from urllib.parse import urlparse

from app.evidence.models import PageEvidence
from app.planning.coverage import CoverageState, EvidenceField, FieldCoverage


def estimate_page_field_coverage(
    page: PageEvidence,
) -> dict[EvidenceField, float]:
    """
    Estimate evidence coverage contributed by a page.

    IMPORTANT:
    These are evidence-acquisition heuristics, not factual claims.
    The LLM/verification layer is responsible for final claims.
    """

    url = page.url.lower()
    parsed = urlparse(url)
    path = parsed.path.lower()

    text = (
        f"{page.title} "
        f"{page.meta_description} "
        f"{page.clean_text}"
    ).lower()

    scores = {
        EvidenceField.OVERVIEW: 0.0,
        EvidenceField.ICP: 0.0,
        EvidenceField.CONTACTS: 0.0,
        EvidenceField.LEADERSHIP: 0.0,
    }

    # ---------------------------------------------------------
    # OVERVIEW
    # ---------------------------------------------------------

    if page.title:
        scores[EvidenceField.OVERVIEW] += 0.20

    if page.meta_description:
        scores[EvidenceField.OVERVIEW] += 0.20

    if len(page.clean_text) >= 250:
        scores[EvidenceField.OVERVIEW] += 0.25

    if any(
        token in path
        for token in (
            "/about",
            "/company",
            "/overview",
        )
    ):
        scores[EvidenceField.OVERVIEW] += 0.30

    if "we are" in text or "our company" in text:
        scores[EvidenceField.OVERVIEW] += 0.15

    # ---------------------------------------------------------
    # ICP / TARGET AUDIENCE
    # ---------------------------------------------------------

    icp_path_tokens = (
        "/pricing",
        "/customers",
        "/solutions",
        "/industries",
        "/use-cases",
        "/platform",
        "/product",
    )

    if any(token in path for token in icp_path_tokens):
        scores[EvidenceField.ICP] += 0.50

    icp_text_tokens = (
        "for developers",
        "for teams",
        "for enterprises",
        "for businesses",
        "customers",
        "used by",
        "built for",
        "designed for",
        "use cases",
    )

    matches = sum(token in text for token in icp_text_tokens)

    scores[EvidenceField.ICP] += min(
        0.40,
        matches * 0.08,
    )

    # Product/service pages are also useful ICP evidence.
    if len(page.clean_text) >= 500:
        scores[EvidenceField.ICP] += 0.10

    # ---------------------------------------------------------
    # CONTACTS
    # ---------------------------------------------------------

    if page.emails:
        scores[EvidenceField.CONTACTS] += 0.75

    contact_path_tokens = (
        "/contact",
        "/sales",
        "/support",
        "/help",
    )

    if any(token in path for token in contact_path_tokens):
        scores[EvidenceField.CONTACTS] += 0.40

    contact_text_tokens = (
        "contact us",
        "contact sales",
        "get in touch",
        "support",
        "email us",
    )

    if any(token in text for token in contact_text_tokens):
        scores[EvidenceField.CONTACTS] += 0.15

    # ---------------------------------------------------------
    # LEADERSHIP
    # ---------------------------------------------------------

    leadership_path_tokens = (
        "/team",
        "/leadership",
        "/company",
        "/about",
        "/people",
    )

    if any(token in path for token in leadership_path_tokens):
        scores[EvidenceField.LEADERSHIP] += 0.45

    if page.linkedin_urls:
        scores[EvidenceField.LEADERSHIP] += 0.20

    if any(
        entity.entity_type.lower() == "person"
        for entity in page.structured_entities
    ):
        scores[EvidenceField.LEADERSHIP] += 0.35

    leadership_text_tokens = (
        "founder",
        "co-founder",
        "chief executive",
        "ceo",
        "leadership",
        "our team",
    )

    matches = sum(
        token in text
        for token in leadership_text_tokens
    )

    scores[EvidenceField.LEADERSHIP] += min(
        0.30,
        matches * 0.06,
    )

    return {
        field: min(1.0, score)
        for field, score in scores.items()
    }


def update_coverage_from_page(
    coverage: dict[EvidenceField, FieldCoverage],
    page: PageEvidence,
) -> float:
    """
    Merge the page's evidence contribution into global coverage.

    Returns total positive information gain.
    """

    page_scores = estimate_page_field_coverage(page)
    total_gain = 0.0

    for field, page_score in page_scores.items():
        field_coverage = coverage[field]

        old_score = field_coverage.coverage_score

        # Evidence coverage is monotonic for this planning phase.
        new_score = max(old_score, page_score)

        field_coverage.coverage_score = new_score
        field_coverage.add_page(page.page_id)

        if new_score >= 0.85:
            field_coverage.state = CoverageState.STRONG
        elif new_score >= 0.65:
            field_coverage.state = CoverageState.SUPPORTED
        elif new_score >= 0.35:
            field_coverage.state = CoverageState.PARTIAL
        elif new_score > 0:
            field_coverage.state = CoverageState.WEAK
        else:
            field_coverage.state = CoverageState.UNKNOWN

        total_gain += max(0.0, new_score - old_score)

    return total_gain