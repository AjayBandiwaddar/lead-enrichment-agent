from __future__ import annotations

from collections.abc import Iterable

from app.evidence.models import PageEvidence


class EvidenceStore:
    """
    Lightweight in-memory evidence store for one pipeline run.

    Deduplicates by:
      1. canonical URL
      2. normalized URL
      3. content hash

    No database or vector store is necessary at this stage.
    """

    def __init__(self) -> None:
        self._pages: dict[str, PageEvidence] = {}
        self._url_index: dict[str, str] = {}
        self._content_hash_index: dict[str, str] = {}

    def add(self, page: PageEvidence) -> bool:
        """
        Add a page to the store.

        Returns True when new evidence was stored.
        Returns False when the page is a duplicate.
        """
        urls = {
            self._normalize_url(str(page.url)),
        }

        if page.canonical_url:
            urls.add(
                self._normalize_url(str(page.canonical_url))
            )

        # URL-level duplicate.
        for url in urls:
            if url in self._url_index:
                return False

        # Content-level duplicate.
        if page.content_hash:
            existing_page_id = self._content_hash_index.get(
                page.content_hash
            )

            if existing_page_id is not None:
                return False

        self._pages[page.page_id] = page

        for url in urls:
            self._url_index[url] = page.page_id

        if page.content_hash:
            self._content_hash_index[page.content_hash] = page.page_id

        return True

    def get(
        self,
        page_id: str,
    ) -> PageEvidence | None:
        return self._pages.get(page_id)

    def get_by_url(
        self,
        url: str,
    ) -> PageEvidence | None:
        page_id = self._url_index.get(
            self._normalize_url(url)
        )

        if page_id is None:
            return None

        return self._pages.get(page_id)

    def get_by_content_hash(
        self,
        content_hash: str,
    ) -> PageEvidence | None:
        page_id = self._content_hash_index.get(content_hash)

        if page_id is None:
            return None

        return self._pages.get(page_id)

    def all(self) -> list[PageEvidence]:
        return list(self._pages.values())

    def values(self) -> Iterable[PageEvidence]:
        return self._pages.values()

    def has_url(self, url: str) -> bool:
        return (
            self._normalize_url(url)
            in self._url_index
        )

    def __len__(self) -> int:
        return len(self._pages)

    def clear(self) -> None:
        self._pages.clear()
        self._url_index.clear()
        self._content_hash_index.clear()

    @staticmethod
    def _normalize_url(url: str) -> str:
        """
        Lightweight URL normalization.

        We intentionally do not aggressively rewrite query parameters yet.
        Some sites use them legitimately.
        """
        return url.rstrip("/")