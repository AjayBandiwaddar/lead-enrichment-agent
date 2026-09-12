from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class RecoveryCandidateScore:
    score: float
    reason: str


_LEADERSHIP_PATH_TERMS = (
    "/team",
    "/leadership",
    "/founder",
    "/founders",
    "/executive",
    "/executives",
    "/management",
    "/people",
    "/our-team",
    "/about",
    "/company",
)

_STRONG_ROLE_TERMS = (
    "ceo",
    "cto",
    "cfo",
    "coo",
    "chief executive",
    "chief technology",
    "chief product",
    "chief operating",
    "president",
    "co-founder",
    "founder",
    "vice president",
    "head of",
)

_BAD_EDITORIAL_TERMS = (
    "/press",
    "/press-media",
    "/news",
    "/blog",
    "/media",
)


def has_current_leadership_evidence(pages) -> bool:
    """
    Decide whether the collected evidence contains at least one
    reasonably strong current leadership source.

    Editorial pages such as press/news/blog are deliberately excluded.
    """

    qualifying_pages = 0

    for page in pages:
        url = str(
            getattr(page, "url", "")
        )

        path = urlparse(url).path.lower()

        if any(
            term in path
            for term in _BAD_EDITORIAL_TERMS
        ):
            continue

        text = str(
            getattr(
                page,
                "clean_text",
                "",
            )
            or ""
        ).lower()

        entities = (
            getattr(
                page,
                "structured_entities",
                [],
            )
            or []
        )

        linkedin_urls = (
            getattr(
                page,
                "linkedin_urls",
                [],
            )
            or []
        )

        person_entities = sum(
            1
            for entity in entities
            if "person"
            in str(
                getattr(
                    entity,
                    "type",
                    "",
                )
            ).lower()
        )

        role_hits = sum(
            1
            for term in _STRONG_ROLE_TERMS
            if term in text
        )

        direct_leadership_path = any(
            term in path
            for term in _LEADERSHIP_PATH_TERMS
        )

        # Strongest case:
        # structured Person data + explicit role or leadership path.
        if (
            person_entities >= 1
            and (
                role_hits >= 1
                or direct_leadership_path
            )
        ):
            qualifying_pages += 1
            continue

        # Dedicated about/team/leadership page with multiple explicit
        # leadership role signals.
        if (
            direct_leadership_path
            and role_hits >= 2
        ):
            qualifying_pages += 1
            continue

        # Multiple LinkedIn profiles plus multiple role signals.
        if (
            len(linkedin_urls) >= 2
            and role_hits >= 2
        ):
            qualifying_pages += 1

    return qualifying_pages > 0


def score_leadership_candidate(
    candidate,
) -> RecoveryCandidateScore:
    url = str(
        getattr(candidate, "url", "")
    )

    anchor = str(
        getattr(
            candidate,
            "anchor_text",
            "",
        )
        or ""
    ).lower()

    context = str(
        getattr(
            candidate,
            "context",
            "",
        )
        or ""
    ).lower()

    path = urlparse(url).path.lower()

    text = (
        f"{path} "
        f"{anchor} "
        f"{context}"
    )

    if not url:
        return RecoveryCandidateScore(
            score=0.0,
            reason="missing URL",
        )

    if any(
        term in path
        for term in _BAD_EDITORIAL_TERMS
    ):
        score = -3.0
    else:
        score = 0.0

    reasons: list[str] = []

    for term in _LEADERSHIP_PATH_TERMS:
        if term in path:
            score += 5.0
            reasons.append(
                f"leadership path: {term}"
            )
            break

    for token in (
        "leadership",
        "team",
        "founder",
        "founders",
        "executive",
        "people",
        "management",
    ):
        if token in text:
            score += 1.25
            reasons.append(
                f"leadership signal: {token}"
            )

    for token in _STRONG_ROLE_TERMS:
        if token in text:
            score += 1.5
            reasons.append(
                f"role signal: {token}"
            )

    depth = int(
        getattr(
            candidate,
            "depth",
            0,
        )
        or 0
    )

    score -= 0.15 * depth

    return RecoveryCandidateScore(
        score=score,
        reason=(
            "; ".join(reasons)
            or "generic leadership recovery candidate"
        ),
    )