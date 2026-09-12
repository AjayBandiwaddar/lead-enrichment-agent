from __future__ import annotations

import asyncio
import sys
from urllib.parse import urlparse

from app.crawling.browser import BrowserFetcher
from app.discovery.sources import normalize_domain
from app.evidence.cleaner import clean_html
from app.evidence.models import FetchStatus, PageEvidence
from app.evidence.signals import (
    extract_emails,
    extract_linkedin_urls,
    extract_social_urls,
)
from app.evidence.structured_data import extract_structured_entities
from app.evidence.store import EvidenceStore


async def crawl_homepage(domain: str) -> None:
    base_url = normalize_domain(domain)

    print("=" * 70)
    print(f"CRAWLING: {base_url}")
    print("=" * 70)

    fetcher = BrowserFetcher()

    try:
        await fetcher.start()

        print("[1/5] Fetching homepage...")
        result = await fetcher.fetch(base_url)

        print(f"      method:       {result.method.value}")
        print(f"      success:      {result.success}")
        print(f"      status:       {result.status_code}")
        print(f"      content-type: {result.content_type}")
        print(f"      elapsed:      {result.elapsed_seconds:.2f}s")

        if not result.success or not result.html:
            print(f"\nFETCH FAILED: {result.error_type}")
            print(f"              {result.error_message}")
            return

        print(f"      HTML bytes:   {len(result.html.encode('utf-8')):,}")

        print("\n[2/5] Extracting structured data...")
        json_ld, structured_entities = extract_structured_entities(
            result.html
        )

        print(f"      JSON-LD objects: {len(json_ld)}")
        print(f"      entities:        {len(structured_entities)}")

        for entity in structured_entities[:10]:
            print(f"        - {entity.entity_type}")

        print("\n[3/5] Extracting deterministic signals...")
        emails = extract_emails(result.html)
        linkedin_urls = extract_linkedin_urls(
            result.html,
            base_url,
        )
        social_urls = extract_social_urls(
            result.html,
            base_url,
        )

        print(f"      emails:        {len(emails)}")
        print(f"      LinkedIn URLs: {len(linkedin_urls)}")
        print(f"      social URLs:   {len(social_urls)}")

        for email in emails:
            print(f"        email:    {email}")

        for url in linkedin_urls:
            print(f"        LinkedIn: {url}")

        print("\n[4/5] Cleaning and structuring page...")
        clean_text, text_blocks, headings, links = clean_html(
            result.html,
            base_url,
        )

        print(f"      headings:      {len(headings)}")
        print(f"      text blocks:   {len(text_blocks)}")
        print(f"      links:         {len(links)}")
        print(f"      clean chars:   {len(clean_text):,}")

        print("\n[5/5] Building PageEvidence...")

        parsed = urlparse(base_url)

        page = PageEvidence(
            page_id=f"homepage:{parsed.netloc}",
            url=result.url,
            canonical_url=None,
            source_kind="homepage",
            fetch_status=(
                FetchStatus.SUCCESS
                if result.success
                else FetchStatus.INVALID_CONTENT
            ),
            http_status=result.status_code,
            content_type=result.content_type,
            title=_extract_title(result.html),
            meta_description=_extract_meta_description(result.html),
            headings=headings,
            clean_text=clean_text,
            text_blocks=text_blocks,
            emails=emails,
            linkedin_urls=linkedin_urls,
            social_urls=social_urls,
            json_ld=json_ld,
            structured_entities=structured_entities,
            outgoing_links=links,
            content_hash=None,
        )

        page.content_hash = page.compute_content_hash()

        store = EvidenceStore()
        added = store.add(page)

        print("\n" + "=" * 70)
        print("EVIDENCE REPORT")
        print("=" * 70)

        print(f"page_id:          {page.page_id}")
        print(f"url:              {page.url}")
        print(f"title:            {page.title}")
        print(f"headings:         {len(page.headings)}")
        print(f"text blocks:      {len(page.text_blocks)}")
        print(f"emails:           {len(page.emails)}")
        print(f"LinkedIn URLs:    {len(page.linkedin_urls)}")
        print(f"social URLs:      {len(page.social_urls)}")
        print(f"JSON-LD objects:  {len(page.json_ld)}")
        print(f"structured:       {len(page.structured_entities)}")
        print(f"outgoing links:   {len(page.outgoing_links)}")
        print(f"clean chars:      {len(page.clean_text):,}")
        print(f"content hash:     {page.content_hash}")
        print(f"stored:           {added}")
        print(f"store size:       {len(store)}")

        print("\nTop headings:")
        for heading in page.headings[:15]:
            print(f"  - {heading}")

        print("\nFirst 5 text blocks:")
        for block in page.text_blocks[:5]:
            path = " > ".join(block.heading_path)
            print(f"  [{path or 'root'}]")
            print(f"  {block.text[:300]}")
            print()

        print("\nFirst 15 discovered links:")
        for link in page.outgoing_links[:15]:
            print(
                f"  {link.anchor_text!r:35} → {link.url}"
            )

    finally:
        await fetcher.close()


def _extract_title(html: str) -> str | None:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    title = soup.find("title")

    if title is None:
        return None

    text = title.get_text(" ", strip=True)
    return text or None


def _extract_meta_description(html: str) -> str | None:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    tag = soup.find(
        "meta",
        attrs={"name": "description"},
    )

    if tag is None:
        return None

    content = tag.get("content")

    if not content:
        return None

    return str(content).strip() or None


def main() -> None:
    domain = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "postman.com"
    )

    asyncio.run(crawl_homepage(domain))


if __name__ == "__main__":
    main()