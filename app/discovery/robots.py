from __future__ import annotations

from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from app.crawling.fetch import FetchMethod, FetchResult


DEFAULT_USER_AGENT = "LeadEnrichmentAgent/1.0"


class RobotsPolicy:
    """Parsed robots.txt policy for a target domain."""

    def __init__(
        self,
        base_url: str,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.user_agent = user_agent

        parsed = urlparse(self.base_url)
        self.robots_url = urljoin(
            f"{parsed.scheme}://{parsed.netloc}/",
            "robots.txt",
        )

        self._parser = RobotFileParser()
        self._parser.set_url(self.robots_url)

        self.loaded = False
        self.sitemaps: list[str] = []

    async def load(self) -> FetchResult:
        """
        Fetch and parse robots.txt.

        robots.txt is advisory policy. Failure to fetch it does not
        automatically mean that every URL is allowed.
        """
        try:
            timeout = httpx.Timeout(10.0)

            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            ) as client:
                response = await client.get(self.robots_url)

            if response.status_code == 404:
                # No robots.txt is a normal situation.
                self.loaded = True
                return FetchResult(
                    url=self.robots_url,
                    method=FetchMethod.HTTP,
                    status_code=404,
                    content_type=response.headers.get("content-type"),
                    success=True,
                    html="",
                )

            if not response.is_success:
                return FetchResult(
                    url=self.robots_url,
                    method=FetchMethod.HTTP,
                    status_code=response.status_code,
                    content_type=response.headers.get("content-type"),
                    error_type="robots_http_error",
                    error_message=(
                        f"robots.txt returned HTTP {response.status_code}"
                    ),
                )

            text = response.text

            self._parser.parse(text.splitlines())
            self.sitemaps = self._parser.site_maps() or []
            self.loaded = True

            return FetchResult(
                url=self.robots_url,
                method=FetchMethod.HTTP,
                status_code=response.status_code,
                content_type=response.headers.get("content-type"),
                html=text,
                success=True,
            )

        except httpx.TimeoutException as exc:
            return FetchResult(
                url=self.robots_url,
                method=FetchMethod.HTTP,
                error_type="timeout",
                error_message=str(exc),
            )

        except httpx.RequestError as exc:
            return FetchResult(
                url=self.robots_url,
                method=FetchMethod.HTTP,
                error_type="robots_request_error",
                error_message=str(exc),
            )

    def can_fetch(self, url: str) -> bool:
        """
        Check whether our user agent is allowed to fetch a URL.

        If robots could not be loaded, we conservatively return True here;
        the caller can separately record that policy information was
        unavailable.
        """
        if not self.loaded:
            return True

        return self._parser.can_fetch(self.user_agent, url)