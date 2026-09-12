from __future__ import annotations

from urllib.parse import urlparse

from app.evidence.models import LinkCandidate
from app.planning.coverage import EvidenceField
from app.planning.ranking import CandidateURL


# High-signal page semantics for the fields we care about.
FIELD_KEYWORDS: dict[EvidenceField, tuple[str, ...]] = {
    EvidenceField.OVERVIEW: (
        "about",
        "company",
        "product",
        "platform",
        "what-we-do",
    ),
    EvidenceField.ICP: (
        "customers",
        "solutions",
        "use-cases",
        "industries",
        "developers",
        "teams",
        "enterprise",
        "pricing",
    ),
    EvidenceField.CONTACTS: (
        "contact",
        "support",
        "sales",
        "help",
    ),
    EvidenceField.LEADERSHIP: (
        "team",
        "leadership",
        "founders",
        "founder",
        "people",
        "about",
        "company",
    ),
}


def build_candidate_urls(
    links: list[LinkCandidate],
    base_domain: str,
    source_page_id: str | None = None,
) -> list[CandidateURL]:
    """
    Convert extracted links into crawl candidates.

    Only same-domain HTTP(S) links are returned.
    """

    candidates: list[CandidateURL] = []

    normalized_domain = base_domain.lower().removeprefix("www.")

    for link in links:
        url = str(link.url)
        parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"}:
            continue

        host = parsed.netloc.lower().removeprefix("www.")

        if host != normalized_domain:
            continue

        text = " ".join(
            part
            for part in (
                parsed.path,
                link.anchor_text or "",
                link.surrounding_text or "",
            )
            if part
        ).lower()

        expected_gain = _estimate_field_gain(text)

        candidates.append(
            CandidateURL(
                url=url,
                anchor_text=link.anchor_text,
                surrounding_text=link.surrounding_text,
                source_page_id=source_page_id,
                url_relevance=_url_relevance(parsed.path),
                anchor_relevance=_anchor_relevance(link.anchor_text),
                semantic_relevance=max(expected_gain.values(), default=0.0),
                source_quality=0.8,
                novelty=1.0,
                expected_field_gain=expected_gain,
                estimated_cost=_estimate_cost(parsed.path),
                reason=_build_reason(expected_gain),
            )
        )

    return _deduplicate_candidates(candidates)


def _estimate_field_gain(
    text: str,
) -> dict[EvidenceField, float]:
    scores: dict[EvidenceField, float] = {}

    for field, keywords in FIELD_KEYWORDS.items():
        matches = sum(keyword in text for keyword in keywords)

        if matches == 0:
            continue

        # Two or more independent keyword signals should score higher.
        scores[field] = min(1.0, 0.45 + (0.2 * matches))

    return scores


def _url_relevance(path: str) -> float:
    normalized = path.lower().strip("/")

    if not normalized:
        return 0.35

    high_signal = {
        "about": 0.95,
        "team": 0.95,
        "leadership": 0.95,
        "company": 0.90,
        "contact": 0.90,
        "pricing": 0.75,
    }

    first_segment = normalized.split("/")[0]

    return high_signal.get(first_segment, 0.40)


def _anchor_relevance(anchor_text: str | None) -> float:
    if not anchor_text:
        return 0.20

    text = anchor_text.lower()

    strong_terms = (
        "team",
        "leadership",
        "founders",
        "about",
        "company",
        "contact",
        "pricing",
        "customers",
    )

    matches = sum(term in text for term in strong_terms)

    return min(1.0, 0.25 + matches * 0.2)


def _estimate_cost(path: str) -> float:
    """
    Rough crawl-cost heuristic.

    Documentation/blog sections are often larger than company/contact pages,
    so give them a slightly higher expected cost.
    """
    normalized = path.lower()

    if any(
        marker in normalized
        for marker in ("/docs", "/documentation", "/blog")
    ):
        return 1.5

    return 1.0


def _build_reason(
    expected_gain: dict[EvidenceField, float],
) -> str | None:
    if not expected_gain:
        return None

    strongest_field, score = max(
        expected_gain.items(),
        key=lambda item: item[1],
    )

    return (
        f"Likely useful for {strongest_field.value} "
        f"(estimated gain={score:.2f})"
    )


def _deduplicate_candidates(
    candidates: list[CandidateURL],
) -> list[CandidateURL]:
    unique: dict[str, CandidateURL] = {}

    for candidate in candidates:
        url = str(candidate.url)

        existing = unique.get(url)

        if existing is None or candidate.priority > existing.priority:
            unique[url] = candidate

    return list(unique.values())