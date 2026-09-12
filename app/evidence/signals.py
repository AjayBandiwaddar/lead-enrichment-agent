from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from app.evidence.models import LinkCandidate


EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

LINKEDIN_PATTERN = re.compile(
    r"https?://(?:www\.)?linkedin\.com/(?:in|company)/[A-Za-z0-9%._~:/?#\[\]@!$&'()*+,;=-]+",
    re.IGNORECASE,
)

SOCIAL_HOSTS = {
    "linkedin.com",
    "www.linkedin.com",
    "twitter.com",
    "www.twitter.com",
    "x.com",
    "www.x.com",
    "github.com",
    "www.github.com",
    "youtube.com",
    "www.youtube.com",
    "facebook.com",
    "www.facebook.com",
    "instagram.com",
    "www.instagram.com",
}


def extract_emails(text: str) -> list[str]:
    """Extract and normalize email addresses."""
    return sorted(
        {
            match.group(0).lower()
            for match in EMAIL_PATTERN.finditer(text)
        }
    )


def extract_linkedin_urls(html: str, base_url: str) -> list[str]:
    """Extract LinkedIn URLs from href attributes and visible text."""
    soup = BeautifulSoup(html, "lxml")
    found: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        absolute = urljoin(base_url, anchor["href"])

        if _is_linkedin_url(absolute):
            found.add(_normalize_url(absolute))

    for match in LINKEDIN_PATTERN.finditer(html):
        found.add(_normalize_url(match.group(0)))

    return sorted(found)


def extract_social_urls(html: str, base_url: str) -> list[str]:
    """Extract common social/profile URLs."""
    soup = BeautifulSoup(html, "lxml")
    found: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        absolute = urljoin(base_url, anchor["href"])
        parsed = urlparse(absolute)

        if parsed.netloc.lower() in SOCIAL_HOSTS:
            found.add(_normalize_url(absolute))

    return sorted(found)


def extract_link_candidates(
    html: str,
    base_url: str,
) -> list[LinkCandidate]:
    """
    Extract links together with anchor/context signals used by the planner.
    """
    soup = BeautifulSoup(html, "lxml")
    candidates: list[LinkCandidate] = []

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href")

        if not href:
            continue

        absolute_url = urljoin(base_url, href)

        if not absolute_url.startswith(("http://", "https://")):
            continue

        anchor_text = anchor.get_text(" ", strip=True) or None

        surrounding_text = _extract_surrounding_text(anchor)

        rel = anchor.get("rel") or []

        if isinstance(rel, str):
            rel = [rel]

        candidates.append(
            LinkCandidate(
                url=absolute_url,
                anchor_text=anchor_text,
                surrounding_text=surrounding_text,
                rel=rel,
            )
        )

    return _deduplicate_links(candidates)


def _extract_surrounding_text(anchor) -> str | None:
    """
    Capture a small amount of local context without trying to understand
    the entire page.
    """
    parent = anchor.parent

    if parent is None:
        return None

    text = parent.get_text(" ", strip=True)

    if not text:
        return None

    return " ".join(text.split())[:500]


def _is_linkedin_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    return host in {"linkedin.com", "www.linkedin.com"} and (
        parsed.path.startswith("/in/")
        or parsed.path.startswith("/company/")
    )


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)

    return parsed._replace(
        fragment="",
    ).geturl()


def _deduplicate_links(
    links: list[LinkCandidate],
) -> list[LinkCandidate]:
    seen: set[str] = set()
    unique: list[LinkCandidate] = []

    for link in links:
        url = str(link.url)

        if url in seen:
            continue

        seen.add(url)
        unique.append(link)

    return unique