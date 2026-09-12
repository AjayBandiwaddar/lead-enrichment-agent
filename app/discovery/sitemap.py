from __future__ import annotations

import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import httpx


SITEMAP_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9"
}


async def fetch_sitemap_urls(
    sitemap_url: str,
    *,
    max_urls: int = 500,
) -> list[str]:
    """
    Fetch a sitemap or sitemap index and return bounded URL candidates.

    Handles:
    - <urlset>
    - <sitemapindex>
    """
    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
            headers={
                "User-Agent": "LeadEnrichmentAgent/1.0",
            },
        ) as client:
            response = await client.get(sitemap_url)

        if not response.is_success:
            return []

        content_type = response.headers.get("content-type", "").lower()

        # Sitemap XML is generally XML, but some servers return text/plain.
        if (
            "xml" not in content_type
            and not response.text.lstrip().startswith("<")
        ):
            return []

        return _parse_sitemap(
            response.text,
            max_urls=max_urls,
        )

    except (
        httpx.RequestError,
        ET.ParseError,
    ):
        return []


def _parse_sitemap(
    xml_text: str,
    *,
    max_urls: int,
) -> list[str]:
    root = ET.fromstring(xml_text)

    # Sitemap index → recursively fetch later at orchestration level.
    if _local_name(root.tag) == "sitemapindex":
        return [
            loc.text.strip()
            for loc in root.findall(
                ".//sm:sitemap/sm:loc",
                SITEMAP_NS,
            )
            if loc.text
        ][:max_urls]

    if _local_name(root.tag) == "urlset":
        return [
            loc.text.strip()
            for loc in root.findall(
                ".//sm:url/sm:loc",
                SITEMAP_NS,
            )
            if loc.text
        ][:max_urls]

    return []


def same_domain_urls(
    urls: list[str],
    domain: str,
) -> list[str]:
    """Keep only HTTP(S) URLs belonging to the target domain."""
    normalized_domain = domain.lower().removeprefix("www.")

    result: list[str] = []

    for url in urls:
        parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"}:
            continue

        host = parsed.netloc.lower().removeprefix("www.")

        if host == normalized_domain:
            result.append(url)

    return _deduplicate(result)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result