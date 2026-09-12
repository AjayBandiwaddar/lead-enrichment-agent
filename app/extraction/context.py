from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class EvidenceBlock:
    evidence_id: str
    source_url: str
    title: str
    text: str
    relevance: float


@dataclass(frozen=True, slots=True)
class FieldContext:
    field: str
    blocks: tuple[EvidenceBlock, ...]
    character_count: int

    def as_prompt_text(self) -> str:
        return "\n\n---\n\n".join(
            f"[{block.evidence_id}] {block.title}\n"
            f"URL: {block.source_url}\n"
            f"{block.text}"
            for block in self.blocks
        )


_FIELD_KEYWORDS = {
    "overview": (
        "about",
        "company",
        "who-we-are",
        "story",
        "press",
        "homepage",
    ),
    "icp": (
        "customers",
        "customer",
        "solutions",
        "industries",
        "use-cases",
        "pricing",
        "products",
        "platform",
    ),
    "contacts": (
        "contact",
        "contact-us",
        "contact-sales",
        "support",
        "sales",
        "press",
        "careers",
    ),
    "leadership": (
        "team",
        "leadership",
        "founder",
        "founders",
        "executive",
        "executives",
        "about",
        "company",
        "management",
        "people",
        "our-team",
        "chief",
        "ceo",
        "cto",
        "cfo",
        "coo",
        "president",
    ),
}


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


_ROLE_TERMS = (
    "ceo",
    "cto",
    "cfo",
    "coo",
    "chief ",
    "president",
    "founder",
    "co-founder",
    "vice president",
    "vp ",
    "head of",
    "general manager",
    "chief executive",
    "chief technology",
    "chief product",
    "chief operating",
)


def _locale_penalty(path: str) -> float:
    parts = [
        part
        for part in path.split("/")
        if part
    ]

    if not parts:
        return 0.0

    first = parts[0].lower()

    locale_markers = {
        "jp",
        "ja",
        "de",
        "fr",
        "es",
        "pt",
        "it",
        "ko",
        "zh",
        "cn",
        "in",
        "br",
    }

    return (
        2.0
        if first in locale_markers
        or len(first) == 2
        else 0.0
    )


def _leadership_score(page) -> float:
    url = str(page.url)
    parsed = urlparse(url)
    path = parsed.path.lower()

    title = str(
        getattr(page, "title", "") or ""
    ).lower()

    text = str(
        getattr(page, "clean_text", "") or ""
    )

    haystack = (
        f"{path} {title} "
        f"{text[:6000].lower()}"
    )

    score = 0.0

    if any(
        term in path
        for term in _LEADERSHIP_PATH_TERMS
    ):
        score += 7.0

    score += sum(
        1.0
        for keyword in _FIELD_KEYWORDS["leadership"]
        if keyword in haystack
    )

    linkedin_count = len(
        getattr(
            page,
            "linkedin_urls",
            [],
        )
        or []
    )

    score += min(
        linkedin_count * 2.0,
        6.0,
    )

    entities = (
        getattr(
            page,
            "structured_entities",
            [],
        )
        or []
    )

    person_count = sum(
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

    score += min(
        person_count * 4.0,
        12.0,
    )

    score += min(
        sum(
            1
            for term in _ROLE_TERMS
            if term in haystack
        )
        * 1.5,
        12.0,
    )

    score -= _locale_penalty(path)

    return score


def _field_score(page, field: str) -> float:
    url = str(page.url)
    parsed = urlparse(url)
    path = parsed.path.lower()

    title = str(
        getattr(page, "title", "") or ""
    ).lower()

    text = str(
        getattr(page, "clean_text", "") or ""
    )

    haystack = (
        f"{path} {title} "
        f"{text[:3000].lower()}"
    )

    if field == "leadership":
        return _leadership_score(page)

    score = sum(
        1.0
        for keyword in _FIELD_KEYWORDS[field]
        if keyword in haystack
    )

    if field == "contacts":
        score += min(
            len(
                getattr(
                    page,
                    "emails",
                    [],
                )
                or []
            )
            * 3.0,
            6.0,
        )

    score -= _locale_penalty(path)

    return score


def _extract_leadership_windows(
    text: str,
    *,
    max_chars: int,
) -> str:
    normalized = text.strip()

    if len(normalized) <= max_chars:
        return normalized

    lowered = normalized.lower()

    signal_terms = (
        "ceo",
        "cto",
        "cfo",
        "coo",
        "chief ",
        "president",
        "founder",
        "co-founder",
        "vp ",
        "vice president",
        "head of",
        "executive",
        "leadership",
    )

    windows: list[tuple[int, int]] = []

    for term in signal_terms:
        start = 0

        while True:
            position = lowered.find(
                term,
                start,
            )

            if position < 0:
                break

            windows.append(
                (
                    max(0, position - 650),
                    min(
                        len(normalized),
                        position + 1000,
                    ),
                )
            )

            start = position + len(term)

            if len(windows) >= 30:
                break

        if len(windows) >= 30:
            break

    if not windows:
        return normalized[:max_chars]

    windows.sort()

    merged: list[list[int]] = []

    for start, end in windows:
        if (
            not merged
            or start > merged[-1][1]
        ):
            merged.append(
                [start, end]
            )
        else:
            merged[-1][1] = max(
                merged[-1][1],
                end,
            )

    pieces: list[str] = []
    used = 0

    for start, end in merged:
        remaining = max_chars - used

        if remaining <= 0:
            break

        piece = normalized[
            start : min(
                end,
                start + remaining,
            )
        ].strip()

        if not piece:
            continue

        pieces.append(piece)
        used += len(piece) + 2

    return "\n...\n".join(
        pieces
    )[:max_chars]


def build_field_context(
    pages,
    field: str,
    *,
    max_chars: int = 4_500,
    max_blocks: int = 4,
) -> FieldContext:
    if field not in _FIELD_KEYWORDS:
        raise ValueError(
            f"Unsupported extraction field: {field}"
        )

    ranked = []

    for index, page in enumerate(pages):
        text = str(
            getattr(
                page,
                "clean_text",
                "",
            )
            or ""
        ).strip()

        if not text:
            continue

        ranked.append(
            (
                _field_score(
                    page,
                    field,
                ),
                str(page.url),
                index,
                page,
                text,
            )
        )

    ranked.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    selected: list[EvidenceBlock] = []
    selected_urls: set[str] = set()
    used = 0

    for score, url, index, page, text in ranked:
        if (
            len(selected) >= max_blocks
            or used >= max_chars
            or url in selected_urls
        ):
            break

        remaining = max_chars - used

        if field == "leadership":
            excerpt = _extract_leadership_windows(
                text,
                max_chars=min(
                    2400,
                    remaining,
                ),
            )
        else:
            excerpt = text[
                : min(
                    2600,
                    remaining,
                )
            ]

        if not excerpt:
            continue

        selected.append(
            EvidenceBlock(
                evidence_id=f"page-{index + 1}",
                source_url=url,
                title=str(
                    getattr(
                        page,
                        "title",
                        "",
                    )
                    or "Untitled"
                ),
                text=excerpt,
                relevance=score,
            )
        )

        selected_urls.add(url)
        used += len(excerpt)

    return FieldContext(
        field=field,
        blocks=tuple(selected),
        character_count=used,
    )


def build_extraction_context(
    pages,
    *,
    max_chars_per_field: int = 4_500,
) -> dict[str, FieldContext]:
    return {
        field: build_field_context(
            pages,
            field,
            max_chars=max_chars_per_field,
        )
        for field in (
            "overview",
            "icp",
            "contacts",
            "leadership",
        )
    }