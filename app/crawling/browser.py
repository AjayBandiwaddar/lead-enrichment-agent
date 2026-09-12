from __future__ import annotations

import time
from datetime import datetime, timezone

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from app.crawling.fetch import FetchMethod, FetchResult


class BrowserFetcher:
    """Small, bounded Playwright fetcher.

    Important invariants:
    - every fetch owns and closes its Page;
    - FetchResult.url is always a plain str;
    - browser/context lifecycle is owned here, never by callers;
    - fetch failures become FetchResult objects instead of escaping.
    """

    def __init__(
        self,
        *,
        timeout_ms: int = 20_000,
        post_load_wait_ms: int = 1_250,
    ) -> None:
        self.timeout_ms = timeout_ms
        self.post_load_wait_ms = post_load_wait_ms
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    async def start(self) -> None:
        if self._context is not None:
            return

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        self._context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/140.0 Safari/537.36"
            ),
            ignore_https_errors=False,
        )

    async def _restart(self) -> None:
        await self.close()
        await self.start()

    async def fetch(self, url: str) -> FetchResult:
        url = str(url)
        started = time.monotonic()
        retrieved_at = datetime.now(timezone.utc)

        try:
            if self._context is None:
                await self.start()

            assert self._context is not None
            page: Page = await self._context.new_page()
            page.set_default_timeout(self.timeout_ms)

            try:
                response = await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=self.timeout_ms,
                )

                # Do not wait for networkidle: modern sites can maintain long-lived
                # connections indefinitely. A short bounded settle window gives JS
                # enough time to render without making fetches unbounded.
                if self.post_load_wait_ms > 0:
                    await page.wait_for_timeout(self.post_load_wait_ms)

                html = await page.content()
                final_url = str(page.url or url)
                status_code = response.status if response is not None else None
                content_type = ""
                if response is not None:
                    content_type = response.headers.get("content-type", "")

                elapsed = time.monotonic() - started
                success = bool(html) and (
                    status_code is None or 200 <= status_code < 400
                )

                return FetchResult(
                    url=final_url,
                    method=FetchMethod.PLAYWRIGHT,
                    status_code=status_code,
                    content_type=content_type,
                    html=html,
                    success=success,
                    error_type=None if success else "http_error",
                    error_message=None if success else f"HTTP status {status_code}",
                    elapsed_seconds=elapsed,
                    retrieved_at=retrieved_at,
                )

            finally:
                try:
                    await page.close()
                except Exception:
                    # Page cleanup must never mask the fetch result.
                    pass

        except (PlaywrightTimeoutError, TimeoutError) as exc:
            return FetchResult(
                url=url,
                method=FetchMethod.PLAYWRIGHT,
                status_code=None,
                content_type="",
                html="",
                success=False,
                error_type="timeout",
                error_message=str(exc),
                elapsed_seconds=time.monotonic() - started,
                retrieved_at=retrieved_at,
            )
        except Exception as exc:
            message = str(exc)
            error_name = type(exc).__name__

            # Playwright's TargetClosedError is not exported from the public
            # async_api surface in every Playwright release. Detect the closed
            # target by its stable error name/message instead of importing a
            # version-specific internal exception class.
            target_closed = (
                error_name == "TargetClosedError"
                or "target page, context or browser has been closed" in message.lower()
            )

            if target_closed:
                try:
                    await self._restart()
                except Exception:
                    pass

            return FetchResult(
                url=url,
                method=FetchMethod.PLAYWRIGHT,
                status_code=None,
                content_type="",
                html="",
                success=False,
                error_type="target_closed" if target_closed else error_name,
                error_message=message,
                elapsed_seconds=time.monotonic() - started,
                retrieved_at=retrieved_at,
            )

    async def close(self) -> None:
        context = self._context
        browser = self._browser
        playwright = self._playwright

        self._context = None
        self._browser = None
        self._playwright = None

        if context is not None:
            try:
                await context.close()
            except Exception:
                pass

        if browser is not None:
            try:
                await browser.close()
            except Exception:
                pass

        if playwright is not None:
            try:
                await playwright.stop()
            except Exception:
                pass
