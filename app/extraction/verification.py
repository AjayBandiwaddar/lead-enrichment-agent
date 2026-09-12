from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urlparse

from .schemas import (
    CompanyExtraction,
    EvidenceRef,
)


VerificationStatus = str


@dataclass(frozen=True, slots=True)
class VerifiedClaim:
    field: str
    value: str
    status: VerificationStatus
    confidence: float
    reasons: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VerificationReport:
    verified: CompanyExtraction
    claims: tuple[VerifiedClaim, ...]
    overall_confidence: float


_EMAIL_RE = re.compile(
    r"^[^\s@]+@[^\s@]+\.[^\s@]+$"
)

_WORD_RE = re.compile(
    r"[a-z0-9]+(?:[-'][a-z0-9]+)*"
)

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "for",
    "in",
    "on",
    "at",
    "as",
    "is",
    "are",
    "be",
    "with",
    "by",
    "from",
    "that",
    "this",
    "it",
    "their",
    "its",
    "into",
    "across",
    "one",
    "our",
    "who",
    "which",
    "than",
}


def _norm(text: str) -> str:
    return " ".join(
        str(text).lower().split()
    )


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _WORD_RE.findall(
            _norm(text)
        )
        if token not in _STOPWORDS
        and len(token) > 2
    }


def _excerpt_support_score(
    excerpt: str,
    page_text: str,
) -> float:
    """
    Score whether the source supports a model-produced excerpt.

    Exact/near-exact spans score highest. Paraphrases can still score
    well when enough distinctive tokens overlap with the source.
    """

    excerpt_n = _norm(excerpt)
    source_n = _norm(page_text)

    if not excerpt_n or not source_n:
        return 0.0

    if excerpt_n in source_n:
        return 1.0

    excerpt_tokens = _tokens(
        excerpt_n
    )

    source_tokens = _tokens(
        source_n
    )

    if not excerpt_tokens:
        return 0.0

    overlap = (
        len(
            excerpt_tokens
            & source_tokens
        )
        / len(excerpt_tokens)
    )

    fragment_score = 0.0
    words = excerpt_n.split()

    if len(words) >= 6:
        for size in (8, 6, 5):
            if len(words) < size:
                continue

            hits = 0

            for i in range(
                len(words) - size + 1
            ):
                fragment = " ".join(
                    words[
                        i : i + size
                    ]
                )

                if fragment in source_n:
                    hits += 1

            if hits:
                fragment_score = min(
                    1.0,
                    0.55
                    + 0.10 * hits,
                )
                break

    return max(
        overlap * 0.85,
        fragment_score,
    )


def _source_quality(url: str) -> float:
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if not host:
        return 0.0

    path = parsed.path.lower()
    score = 0.80

    if any(
        token in path
        for token in (
            "/contact",
            "/about",
            "/company",
            "/team",
            "/leadership",
            "/pricing",
            "/solutions",
            "/customers",
            "/careers",
        )
    ):
        score += 0.15

    if any(
        token in path
        for token in (
            "/blog",
            "/press",
            "/news",
        )
    ):
        score -= 0.10

    return max(
        0.0,
        min(score, 1.0),
    )


def _verify_evidence(
    ref: EvidenceRef,
    pages_by_url: dict[str, object],
    pages_by_id: dict[str, object],
) -> tuple[
    bool,
    float,
    list[str],
]:
    url = str(ref.source_url)

    page = (
        pages_by_id.get(
            ref.evidence_id
        )
        or pages_by_url.get(url)
    )

    if page is None:
        return (
            False,
            0.0,
            ["source page not found"],
        )

    text = str(
        getattr(
            page,
            "clean_text",
            "",
        )
        or ""
    )

    support = _excerpt_support_score(
        ref.excerpt,
        text,
    )

    if support >= 0.92:
        reason = (
            "evidence excerpt "
            "exactly/near-exactly verified"
        )
    elif support >= 0.62:
        reason = (
            "evidence excerpt strongly "
            "supported by source text"
        )
    elif support >= 0.45:
        reason = (
            "evidence excerpt partially "
            "supported by source text"
        )
    else:
        reason = (
            "evidence excerpt insufficiently "
            "supported by source text"
        )

    return (
        support >= 0.62,
        support,
        [reason],
    )


def _verify_claim(
    field: str,
    value: str,
    refs: list[EvidenceRef],
    pages_by_url: dict[str, object],
    pages_by_id: dict[str, object],
) -> VerifiedClaim:
    valid_count = 0
    support_scores: list[float] = []
    ids: list[str] = []
    reasons: list[str] = []

    for ref in refs:
        (
            valid,
            score,
            ref_reasons,
        ) = _verify_evidence(
            ref,
            pages_by_url,
            pages_by_id,
        )

        reasons.extend(
            ref_reasons
        )

        support_scores.append(score)

        if valid:
            valid_count += 1
            ids.append(
                ref.evidence_id
            )

    if (
        not support_scores
        or valid_count == 0
    ):
        return VerifiedClaim(
            field=field,
            value=value,
            status="UNVERIFIED",
            confidence=0.0,
            reasons=tuple(reasons),
            evidence_ids=tuple(),
        )

    best_support = max(
        support_scores
    )

    corroboration = min(
        valid_count / 2.0,
        1.0,
    )

    source_quality = sum(
        _source_quality(
            str(ref.source_url)
        )
        for ref in refs
        if ref.evidence_id in ids
    ) / max(
        len(ids),
        1,
    )

    confidence = min(
        1.0,
        0.50 * best_support
        + 0.25 * source_quality
        + 0.25 * corroboration,
    )

    status = (
        "VERIFIED"
        if best_support >= 0.80
        else "SUPPORTED"
    )

    return VerifiedClaim(
        field=field,
        value=value,
        status=status,
        confidence=round(
            confidence,
            3,
        ),
        reasons=tuple(reasons),
        evidence_ids=tuple(ids),
    )


def verify_extraction(
    extraction: CompanyExtraction,
    pages,
) -> VerificationReport:
    pages_by_id = {
        f"page-{i + 1}": page
        for i, page in enumerate(pages)
    }

    pages_by_url = {
        str(page.url): page
        for page in pages
    }

    claims: list[VerifiedClaim] = []

    verified = extraction.model_copy(
        deep=True
    )

    if extraction.overview:
        claim = _verify_claim(
            "overview",
            extraction.overview.value,
            extraction.overview.evidence,
            pages_by_url,
            pages_by_id,
        )

        claims.append(claim)

        if claim.status not in {
            "VERIFIED",
            "SUPPORTED",
        }:
            verified.overview = None

    if extraction.icp_summary:
        claim = _verify_claim(
            "icp_summary",
            extraction.icp_summary.value,
            extraction.icp_summary.evidence,
            pages_by_url,
            pages_by_id,
        )

        claims.append(claim)

        if claim.status not in {
            "VERIFIED",
            "SUPPORTED",
        }:
            verified.icp_summary = None

    verified_contacts = []

    for contact in extraction.contacts:
        refs_ok = all(
            _verify_evidence(
                ref,
                pages_by_url,
                pages_by_id,
            )[0]
            for ref in contact.evidence
        )

        email_ok = bool(
            _EMAIL_RE.match(
                contact.email
            )
        )

        status = (
            "VERIFIED"
            if refs_ok and email_ok
            else "UNVERIFIED"
        )

        confidence = (
            0.9
            if status == "VERIFIED"
            else 0.0
        )

        claims.append(
            VerifiedClaim(
                "contact",
                contact.email,
                status,
                confidence,
                (),
                tuple(
                    ref.evidence_id
                    for ref in contact.evidence
                    if refs_ok
                ),
            )
        )

        if status == "VERIFIED":
            verified_contacts.append(
                contact
            )

    verified.contacts = (
        verified_contacts
    )

    verified_leadership = []

    for person in extraction.leadership:
        claim_value = (
            f"{person.name} — "
            f"{person.role}"
        )

        claim = _verify_claim(
            "leadership",
            claim_value,
            person.evidence,
            pages_by_url,
            pages_by_id,
        )

        linkedin_ok = (
            person.linkedin_url is None
            or str(
                person.linkedin_url
            )
            .lower()
            .startswith(
                "https://www.linkedin.com/"
            )
        )

        if not linkedin_ok:
            claim = VerifiedClaim(
                field=claim.field,
                value=claim.value,
                status="UNVERIFIED",
                confidence=0.0,
                reasons=(
                    claim.reasons
                    + ("invalid LinkedIn URL",)
                ),
                evidence_ids=claim.evidence_ids,
            )

        claims.append(claim)

        if claim.status in {
            "VERIFIED",
            "SUPPORTED",
        }:
            verified_leadership.append(
                person
            )

    verified.leadership = (
        verified_leadership
    )

    verified_confidences = [
        claim.confidence
        for claim in claims
        if claim.status
        in {
            "VERIFIED",
            "SUPPORTED",
        }
    ]

    overall = (
        round(
            sum(verified_confidences)
            / len(verified_confidences),
            3,
        )
        if verified_confidences
        else 0.0
    )

    return VerificationReport(
        verified=verified,
        claims=tuple(claims),
        overall_confidence=overall,
    )