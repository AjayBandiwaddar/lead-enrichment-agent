from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin

from app.discovery.robots import RobotsPolicy


@dataclass(frozen=True)
class DiscoveredSource:
    url: str
    kind: str
    priority: float
    reason: str


def normalize_domain(domain: str) -> str:
    """
    Normalize a user-provided domain into an HTTPS origin.
    """
    value = domain.strip()

    if not value:
        raise ValueError("Domain cannot be empty.")

    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"

    return value.rstrip("/")


def discover_sources(
    domain: str,
    *,
    robots: RobotsPolicy | None = None,
) -> list[DiscoveredSource]:
    """
    Discover cheap, high-signal sources.

    This function only produces candidates. It does not perform network I/O.
    """

    base_url = normalize_domain(domain)

    candidates = [
        DiscoveredSource(
            url=base_url,
            kind="homepage",
            priority=1.0,
            reason="Primary company entry point",
        ),
        DiscoveredSource(
            url=urljoin(f"{base_url}/", "llms.txt"),
            kind="llms_txt",
            priority=0.95,
            reason="Potential machine-readable AI-oriented site map",
        ),
        DiscoveredSource(
            url=urljoin(f"{base_url}/", "llms-full.txt"),
            kind="llms_txt",
            priority=0.90,
            reason="Potential full machine-readable site representation",
        ),
        DiscoveredSource(
            url=urljoin(f"{base_url}/", "sitemap.xml"),
            kind="sitemap",
            priority=0.85,
            reason="Potential URL discovery source",
        ),
    ]

    if robots is not None:
        for sitemap_url in robots.sitemaps:
            candidates.append(
                DiscoveredSource(
                    url=sitemap_url,
                    kind="sitemap",
                    priority=0.90,
                    reason="Sitemap declared by robots.txt",
                )
            )

    return _deduplicate_sources(candidates)


def _deduplicate_sources(
    sources: list[DiscoveredSource],
) -> list[DiscoveredSource]:
    by_url: dict[str, DiscoveredSource] = {}

    for source in sources:
        existing = by_url.get(source.url)

        if existing is None or source.priority > existing.priority:
            by_url[source.url] = source

    return sorted(
        by_url.values(),
        key=lambda source: source.priority,
        reverse=True,
    )