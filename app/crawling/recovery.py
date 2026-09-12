from __future__ import annotations

import time
from datetime import datetime, timezone

import httpx
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from app.crawling.fetch import FetchMethod, FetchResult


class RecoveryFetcher:
    """Bounded HTTP fallback for pages that fail in the browser."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 12.0,
        max_attempts: int = 2,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts

    async def fetch_http(self, url: str) -> FetchResult:
        url = str(url)
        started = time.monotonic()
        retrieved_at = datetime.now(timezone.utc)

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_attempts),
                wait=wait_exponential(multiplier=0.35, min=0.35, max=1.5),
                reraise=True,
            ):
                with attempt:
                    async with httpx.AsyncClient(
                        timeout=self.timeout_seconds,
                        follow_redirects=True,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 (compatible; LeadEnrichmentAgent/1.0)"
                            ),
                            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                        },
                    ) as client:
                        response = await client.get(url)
                        content_type = response.headers.get("content-type", "")
                        html = response.text if "html" in content_type.lower() else ""
                        success = response.is_success and bool(html)

                        return FetchResult(
                            url=str(response.url),
                            method=FetchMethod.HTTP,
                            status_code=response.status_code,
                            content_type=content_type,
                            html=html,
                            success=success,
                            error_type=None if success else "http_error",
                            error_message=(
                                None
                                if success
                                else f"HTTP {response.status_code} or non-HTML response"
                            ),
                            elapsed_seconds=time.monotonic() - started,
                            retrieved_at=retrieved_at,
                        )

        except Exception as exc:
            return FetchResult(
                url=url,
                method=FetchMethod.HTTP,
                status_code=None,
                content_type="",
                html="",
                success=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
                elapsed_seconds=time.monotonic() - started,
                retrieved_at=retrieved_at,
            )

        return FetchResult(
            url=url,
            method=FetchMethod.HTTP,
            status_code=None,
            content_type="",
            html="",
            success=False,
            error_type="unknown",
            error_message="HTTP fallback returned no result",
            elapsed_seconds=time.monotonic() - started,
            retrieved_at=retrieved_at,
        )
