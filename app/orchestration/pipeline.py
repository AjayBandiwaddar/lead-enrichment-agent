from __future__ import annotations

import inspect
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import (
    parse_qsl,
    urlencode,
    urldefrag,
    urlparse,
    urlsplit,
    urlunsplit,
)

from bs4 import BeautifulSoup

from app.crawling.browser import BrowserFetcher
from app.crawling.fetch import FetchResult
from app.crawling.recovery import RecoveryFetcher
from app.discovery.links import build_candidate_urls
from app.discovery.robots import RobotsPolicy
from app.evidence.cleaner import clean_html
from app.evidence.models import FetchStatus, PageEvidence, SourceKind
from app.evidence.signals import (
    extract_emails,
    extract_linkedin_urls,
    extract_social_urls,
)
from app.evidence.store import EvidenceStore
from app.evidence.structured_data import (
    extract_json_ld,
    normalize_structured_entities,
)
from app.orchestration.state import (
    EvidenceState,
    PipelineStatus,
    TerminationReason,
)
from app.planning.heuristics import update_coverage_from_page
from app.planning.ranking import CandidateStatus
from app.planning.stopping import evaluate_stop


@dataclass
class ResearchDecision:
    iteration: int
    url: str
    priority: float
    reason: str
    information_gain: float = 0.0
    success: bool = True


@dataclass
class ResearchResult:
    state: EvidenceState
    decisions: list[ResearchDecision] = field(default_factory=list)


class ResearchPipeline:
    """Budget-aware adaptive evidence acquisition before LLM extraction."""

    # Common BCP-47-ish locale prefixes. These are deprioritized, not banned,
    # because a localized page can occasionally be the only useful source.
    _LOCALE_CODES = {
        "ar", "au", "bg", "br", "ca", "cs", "da", "de", "el", "en",
        "es", "fi", "fr", "he", "hi", "hk", "id", "in", "it", "ja",
        "jp", "ko", "kr", "ms", "mx", "nl", "no", "nz", "pl", "pt",
        "ro", "ru", "sg", "sk", "sv", "th", "tr", "tw", "uk", "us",
        "vi", "zh",
    }

    _FIELD_SIGNALS = {
        "contacts": (
            2.6,
            (
                "contact",
                "contact-us",
                "contact-sales",
                "sales",
                "support",
                "press",
                "media",
                "email",
                "talk-to-sales",
            ),
        ),
        "icp": (
            2.5,
            (
                "customers",
                "customer-stories",
                "case-studies",
                "industries",
                "solutions",
                "use-cases",
                "usecases",
                "pricing",
                "platform",
                "developers",
                "teams",
                "enterprise",
            ),
        ),
        "leadership": (
            1.7,
            (
                "team",
                "leadership",
                "founder",
                "founders",
                "executive",
                "people",
                "management",
            ),
        ),
        "overview": (
            1.0,
            (
                "about",
                "company",
                "who-we-are",
                "our-story",
                "mission",
            ),
        ),
    }

    _TRACKING_QUERY_PREFIXES = (
        "utm_",
    )

    _TRACKING_QUERY_KEYS = {
        "gclid",
        "fbclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "_ga",
        "_gl",
    }

    _LEADERSHIP_RECOVERY_PATHS = (
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
        "/company/about",
        "/company",
    )

    _LEADERSHIP_ROLE_TERMS = (
        "ceo",
        "cto",
        "cfo",
        "coo",
        "chief executive",
        "chief technology",
        "chief product",
        "chief operating",
        "president",
        "founder",
        "co-founder",
        "vice president",
        "vp ",
        "head of",
        "general manager",
    )

    _LEADERSHIP_BAD_PATHS = (
        "/press",
        "/press-media",
        "/news",
        "/blog",
        "/media",
    )

    def __init__(
        self,
        *,
        browser: BrowserFetcher | None = None,
        recovery: RecoveryFetcher | None = None,
    ) -> None:
        self.browser = browser or BrowserFetcher()
        self.recovery = recovery or RecoveryFetcher()
        self.store = EvidenceStore()
        self.robots: RobotsPolicy | None = None
        self._robots_loaded = False
        self._robots_failures = 0
        self._scheduled_urls: set[str] = set()
        self._source_cluster_low_gain: dict[str, int] = {}
        self._seen_content_hashes: set[str] = set()
        self._leadership_recovery_attempts = 0
        self._max_leadership_recovery_attempts = 2

    async def run(self, domain: str) -> ResearchResult:
        normalized_domain = self._normalize_domain(domain)
        state = EvidenceState(domain=normalized_domain)
        decisions: list[ResearchDecision] = []
        low_gain_streak = 0
        started_at = time.monotonic()
        browser_started = False

        self._scheduled_urls.clear()
        self._source_cluster_low_gain.clear()
        self._seen_content_hashes.clear()
        self._leadership_recovery_attempts = 0
        self.robots = None
        self._robots_loaded = False
        self._robots_failures = 0

        # crawl_iterations is an attempt counter. The hard page budget is
        # measured by distinct successful evidence pages, not attempts.
        max_attempts = max(
            state.budget.max_pages * (state.budget.max_retries_per_resource + 1),
            state.budget.max_pages + 3,
        )

        try:
            state.status = PipelineStatus.DISCOVERING

            await self.browser.start()
            browser_started = True

            self.robots = RobotsPolicy(
                base_url=f"https://{normalized_domain}"
            )
            await self._load_robots()

            homepage_url = f"https://{normalized_domain}/"
            homepage = await self._crawl(
                homepage_url,
                source_kind=SourceKind.HOMEPAGE,
                state=state,
            )

            if homepage is None:
                state.status = PipelineStatus.FAILED
                state.termination_reason = TerminationReason.FATAL_ERROR
                state.add_error(f"Homepage crawl failed: {homepage_url}")
                return ResearchResult(state=state, decisions=decisions)

            homepage_added = self._record_page(state=state, page=homepage)
            if not homepage_added:
                state.status = PipelineStatus.FAILED
                state.termination_reason = TerminationReason.FATAL_ERROR
                state.add_error("Homepage was rejected as duplicate evidence")
                return ResearchResult(state=state, decisions=decisions)

            homepage_for_coverage = self._coverage_safe_page(homepage)
            homepage_gain = update_coverage_from_page(
                state.coverage,
                homepage_for_coverage,
            )
            state.metrics.last_information_gain = homepage_gain
            state.metrics.average_information_gain = homepage_gain

            await self._discover_candidates(state=state, page=homepage)
            self._rerank_candidates(state)
            state.status = PipelineStatus.CRAWLING

            while True:
                state.usage.runtime_seconds = time.monotonic() - started_at

                # Exact distinct-successful-page budget. A browser+HTTP fallback
                # is still only one page when the resulting evidence is accepted.
                if state.usage.pages_crawled >= state.budget.max_pages:
                    state.termination_reason = TerminationReason.BUDGET_EXHAUSTED
                    break

                if state.metrics.crawl_iterations >= max_attempts:
                    state.termination_reason = TerminationReason.BUDGET_EXHAUSTED
                    break

                stop = evaluate_stop(
                    state,
                    low_gain_streak=low_gain_streak,
                )

                if stop.should_stop:
                    if stop.reason == TerminationReason.SATURATED:
                        # Saturation is a guard against low-yield repetition,
                        # not permission to stop while a strong path to an
                        # uncovered field still exists.
                        if (
                            not self._coverage_is_complete(state)
                            and self._has_high_value_candidates(state)
                        ):
                            low_gain_streak = 0
                            continue

                    # A generic coverage stop can still be premature for
                    # extraction. In particular, editorial/press evidence
                    # must not satisfy the leadership requirement.
                    if stop.reason == TerminationReason.COVERAGE_REACHED:
                        recovered = await self._run_leadership_recovery(
                            state=state,
                            decisions=decisions,
                        )
                        if recovered:
                            low_gain_streak = 0
                            self._rerank_candidates(state)
                            continue

                    state.termination_reason = stop.reason
                    break

                candidate = self._select_next_candidate(state)
                if candidate is None:
                    state.termination_reason = TerminationReason.FRONTIER_EXHAUSTED
                    break

                candidate_url = self._canonicalize_url(candidate.url)
                candidate.status = CandidateStatus.CRAWLED
                state.metrics.crawl_iterations += 1
                iteration = state.metrics.crawl_iterations

                page = await self._crawl(
                    candidate_url,
                    source_kind=SourceKind.INTERNAL_PAGE,
                    state=state,
                )

                if page is None:
                    self._handle_candidate_failure(
                        state=state,
                        candidate=candidate,
                    )
                    low_gain_streak += 1
                    decisions.append(
                        ResearchDecision(
                            iteration=iteration,
                            url=candidate_url,
                            priority=float(candidate.priority),
                            reason=f"failed: {candidate.reason}",
                            information_gain=0.0,
                            success=False,
                        )
                    )
                    continue

                added = self._record_page(state=state, page=page)

                if not added:
                    # Canonical/content duplicate: successful fetch, but no new
                    # evidence. Do not burn the page budget or rediscover links.
                    candidate.status = CandidateStatus.SKIPPED
                    self._register_cluster_gain(candidate, 0.0)
                    low_gain_streak += 1
                    decisions.append(
                        ResearchDecision(
                            iteration=iteration,
                            url=candidate_url,
                            priority=float(candidate.priority),
                            reason="duplicate evidence; skipped",
                            information_gain=0.0,
                            success=True,
                        )
                    )
                    continue

                page_for_coverage = self._coverage_safe_page(page)
                gain = update_coverage_from_page(
                    state.coverage,
                    page_for_coverage,
                )

                state.metrics.last_information_gain = gain
                state.metrics.average_information_gain = self._running_average(
                    state.metrics.average_information_gain,
                    gain,
                    iteration,
                )
                low_gain_streak = low_gain_streak + 1 if gain < 0.03 else 0
                self._register_cluster_gain(candidate, gain)

                candidate.status = CandidateStatus.CRAWLED
                decisions.append(
                    ResearchDecision(
                        iteration=iteration,
                        url=self._canonicalize_url(page.url),
                        priority=float(candidate.priority),
                        reason=candidate.reason,
                        information_gain=gain,
                        success=True,
                    )
                )

                await self._discover_candidates(state=state, page=page)
                self._rerank_candidates(state)

            self._finalize_status(state)
            return ResearchResult(state=state, decisions=decisions)

        except Exception as exc:
            state.status = PipelineStatus.FAILED
            state.termination_reason = TerminationReason.FATAL_ERROR
            state.add_error(
                f"Pipeline error: {type(exc).__name__}: {exc}"
            )
            return ResearchResult(state=state, decisions=decisions)

        finally:
            state.usage.runtime_seconds = time.monotonic() - started_at
            if browser_started:
                try:
                    await self.browser.close()
                except Exception as exc:
                    if len(state.errors) < 8:
                        state.add_error(
                            "Browser cleanup failed: "
                            f"{type(exc).__name__}: {exc}"
                        )

    async def _crawl(
        self,
        url: str,
        *,
        source_kind: SourceKind,
        state: EvidenceState,
    ) -> PageEvidence | None:
        """Fetch one URL; isolate failures to this resource."""
        url = self._canonicalize_url(url)

        if not url or self._url_seen(state, url):
            return None

        fetch_result: FetchResult | None = None

        try:
            fetch_result = await self.browser.fetch(url)
        except Exception as exc:
            self._record_resource_error(state, url, exc)

        if not self._valid_fetch_result(fetch_result):
            try:
                fetch_result = await self.recovery.fetch_http(url)
            except Exception as exc:
                self._record_resource_error(state, url, exc)
                fetch_result = None

        if not self._valid_fetch_result(fetch_result):
            state.metrics.pages_failed += 1
            return None

        assert fetch_result is not None
        effective_source = (
            SourceKind.HTTP_FALLBACK
            if self._method_is_http(fetch_result)
            else source_kind
        )

        page = self._build_page_evidence(
            fetch_result,
            source_kind=effective_source,
        )
        if page is None:
            state.metrics.pages_failed += 1
            return None

        return page

    def _build_page_evidence(
        self,
        result: FetchResult,
        *,
        source_kind: SourceKind,
    ) -> PageEvidence | None:
        """Convert a successful FetchResult into normalized PageEvidence."""
        try:
            result_url = self._as_url_string(result.url)
            html = str(result.html or "")
            soup = BeautifulSoup(html, "lxml")

            title = (
                soup.title.get_text(" ", strip=True)
                if soup.title
                else ""
            )
            meta = soup.find(
                "meta",
                attrs={"name": "description"},
            )
            meta_description = (
                str(meta.get("content", "")).strip()
                if meta
                else ""
            )

            canonical_url = self._extract_canonical_url(
                soup,
                result_url,
            )

            json_ld = extract_json_ld(html)
            entities = normalize_structured_entities(json_ld)
            emails = extract_emails(html)
            linkedin_urls = extract_linkedin_urls(html, result_url)
            social_urls = extract_social_urls(html, result_url)
            clean_text, text_blocks, headings, links = clean_html(
                html,
                result_url,
            )

            retrieved_at = getattr(result, "retrieved_at", None)
            if retrieved_at is None:
                retrieved_at = datetime.now(timezone.utc)

            page = PageEvidence(
                page_id=self._page_id(source_kind, result_url),
                url=result_url,
                canonical_url=canonical_url,
                source_kind=source_kind,
                fetch_status=(
                    FetchStatus.SUCCESS
                    if result.success
                    else FetchStatus.HTTP_ERROR
                ),
                http_status=result.status_code,
                content_type=str(result.content_type or ""),
                retrieved_at=retrieved_at,
                title=title,
                meta_description=meta_description,
                headings=headings,
                clean_text=clean_text,
                text_blocks=text_blocks,
                emails=emails,
                linkedin_urls=linkedin_urls,
                social_urls=social_urls,
                json_ld=json_ld,
                structured_entities=entities,
                outgoing_links=links,
            )
            page.compute_content_hash()
            return page
        except Exception as exc:
            self._record_local_evidence_error(result, exc)
            return None

    def _record_page(
        self,
        *,
        state: EvidenceState,
        page: PageEvidence,
    ) -> bool:
        """Persist genuinely new evidence and charge one page-budget unit."""
        fetched_url = self._canonicalize_url(page.url)
        canonical_url = self._canonicalize_url(page.canonical_url)

        if self._url_seen(state, fetched_url) or self._url_seen(
            state,
            canonical_url,
        ):
            return False

        content_hash = str(getattr(page, "content_hash", "") or "")
        if content_hash and content_hash in self._seen_content_hashes:
            return False

        self.store.add(page)
        state.add_page(page)
        state.visited_urls.add(fetched_url)
        state.visited_urls.add(canonical_url)
        self._scheduled_urls.add(fetched_url)
        self._scheduled_urls.add(canonical_url)

        if content_hash:
            self._seen_content_hashes.add(content_hash)

        state.metrics.pages_crawled += 1
        state.usage.pages_crawled += 1
        return True

    async def _discover_candidates(
        self,
        *,
        state: EvidenceState,
        page: PageEvidence,
    ) -> None:
        """Convert page links into deduplicated, same-domain candidates."""
        normalized_links = []

        for link in page.outgoing_links:
            try:
                if hasattr(link, "model_copy"):
                    normalized_links.append(
                        link.model_copy(
                            update={"url": self._as_url_string(link.url)}
                        )
                    )
                else:
                    normalized_links.append(link)
            except Exception:
                continue

        try:
            candidates = build_candidate_urls(
                links=normalized_links,
                base_domain=str(state.domain),
                source_page_id=str(page.page_id),
            )
        except Exception as exc:
            state.add_error(
                "Candidate discovery failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return

        for candidate in candidates:
            candidate_url = self._canonicalize_url(candidate.url)
            if not candidate_url:
                continue

            if candidate_url in self._scheduled_urls:
                continue
            if self._url_seen(state, candidate_url):
                continue

            try:
                allowed = await self._robots_allowed(candidate_url)
            except Exception:
                allowed = True

            if not allowed:
                candidate.status = CandidateStatus.SKIPPED
                continue

            try:
                candidate.url = candidate_url
            except Exception:
                pass

            state.add_candidate(candidate)
            self._scheduled_urls.add(candidate_url)

    async def _load_robots(self) -> None:
        if self.robots is None or self._robots_loaded:
            return

        loader = getattr(self.robots, "load", None)
        if loader is None:
            self._robots_loaded = True
            return

        try:
            result = loader()
            if inspect.isawaitable(result):
                await result
            self._robots_loaded = True
        except Exception as exc:
            self._robots_failures += 1
            self._robots_loaded = True
            print(
                "      [robots-warning] "
                f"load failed: {type(exc).__name__}: {exc}"
            )

    async def _robots_allowed(self, url: str) -> bool:
        if self.robots is None:
            return True

        try:
            result = self.robots.can_fetch(self._as_url_string(url))
            if inspect.isawaitable(result):
                result = await result
            return bool(result)
        except Exception as exc:
            self._robots_failures += 1
            if self._robots_failures <= 2:
                print(
                    "      [robots-warning] "
                    f"can_fetch failed: {type(exc).__name__}: {exc}"
                )
            return True

    def _rerank_candidates(self, state: EvidenceState) -> None:
        """Rank frontier by current field gaps, expected gain, and diversity."""
        missingness = state.field_missingness()
        candidates = self._frontier_candidates(state)
        non_locale_exists = any(
            not self._is_locale_url(candidate.url)
            for candidate in candidates
        )

        for candidate in candidates:
            url = self._canonicalize_url(candidate.url).lower()
            anchor = str(getattr(candidate, "anchor_text", "") or "").lower()
            context = str(getattr(candidate, "context", "") or "").lower()
            text = f"{url} {anchor} {context}"

            score = 0.0
            matched_fields: list[str] = []

            for field, (weight, tokens) in self._FIELD_SIGNALS.items():
                missing = max(float(missingness.get(field, 0.0)), 0.0)
                if missing <= 0.05:
                    continue
                if any(token in text for token in tokens):
                    score += weight * missing
                    matched_fields.append(field)

            expected_gain = self._float_attr(candidate, "expected_field_gain")
            relevance = self._float_attr(candidate, "relevance")
            quality = self._float_attr(candidate, "source_quality")

            # Expected gain is useful, but field gaps dominate once the agent
            # knows what is still missing.
            score += 0.70 * expected_gain
            score += 0.20 * relevance
            score += 0.10 * quality

            novelty = max(self._float_attr(candidate, "novelty", 1.0), 0.0)
            cost = max(self._float_attr(candidate, "estimated_cost", 1.0), 0.01)
            depth = self._int_attr(candidate, "depth")
            failures = self._int_attr(candidate, "failure_count")

            score *= novelty
            score /= cost
            score -= depth * 0.025
            score -= failures * 1000.0

            cluster_keys = self._candidate_clusters(candidate)
            zero_gain_penalty = sum(
                self._source_cluster_low_gain.get(cluster, 0)
                for cluster in cluster_keys
            )
            score -= 0.80 * zero_gain_penalty

            # Do not spend scarce pages on generic editorial/PR material when
            # structured ICP/contact fields remain weak.
            if any(
                token in url
                for token in ("/blog/", "/news/", "/press-media/")
            ):
                if (
                    missingness.get("contacts", 0.0) > 0.4
                    or missingness.get("icp", 0.0) > 0.4
                ):
                    score -= 0.8

            # Locale pages are legitimate fallback sources, but a default-
            # language path should win whenever one exists in the frontier.
            if self._is_locale_url(candidate.url) and non_locale_exists:
                score -= 1.5

            # Slight diversity reward when the candidate opens a source cluster
            # not yet explored successfully.
            if cluster_keys and all(
                self._source_cluster_low_gain.get(cluster, 0) == 0
                for cluster in cluster_keys
            ):
                score += 0.12

            candidate.priority = float(score)

    def _select_next_candidate(self, state: EvidenceState):
        self._rerank_candidates(state)
        viable = self._frontier_candidates(state)
        if not viable:
            return None

        return max(
            viable,
            key=lambda candidate: (
                float(candidate.priority),
                self._float_attr(candidate, "expected_field_gain"),
                -self._int_attr(candidate, "failure_count"),
                -self._int_attr(candidate, "depth"),
                self._canonicalize_url(candidate.url),
            ),
        )

    def _handle_candidate_failure(self, *, state: EvidenceState, candidate) -> None:
        candidate.failure_count += 1
        url = self._canonicalize_url(candidate.url)

        if candidate.failure_count >= state.budget.max_retries_per_resource:
            candidate.status = CandidateStatus.FAILED
            state.failed_urls.add(url)
            self._scheduled_urls.add(url)
            if len(state.errors) < 8:
                state.add_error(f"Candidate exhausted retries: {url}")
            return

        candidate.status = CandidateStatus.QUEUED
        candidate.priority = -1000.0 - candidate.failure_count

    def _register_cluster_gain(self, candidate, gain: float) -> None:
        clusters = self._candidate_clusters(candidate)
        if not clusters:
            return

        if gain < 0.03:
            for cluster in clusters:
                self._source_cluster_low_gain[cluster] = (
                    self._source_cluster_low_gain.get(cluster, 0) + 1
                )
        else:
            # Evidence from a productive page reopens the cluster modestly.
            for cluster in clusters:
                self._source_cluster_low_gain[cluster] = max(
                    0,
                    self._source_cluster_low_gain.get(cluster, 0) - 1,
                )

    def _candidate_clusters(self, candidate) -> list[str]:
        url = self._canonicalize_url(candidate.url).lower()
        anchor = str(getattr(candidate, "anchor_text", "") or "").lower()
        context = str(getattr(candidate, "context", "") or "").lower()
        text = f"{url} {anchor} {context}"

        path_parts = [part for part in urlparse(url).path.split("/") if part]
        if path_parts and path_parts[0] in self._LOCALE_CODES:
            path_parts = path_parts[1:]
        first_path = path_parts[0] if path_parts else "root"

        clusters = []
        for field, (_, tokens) in self._FIELD_SIGNALS.items():
            if any(token in text for token in tokens):
                clusters.append(f"{field}:{first_path}")
        return clusters

    def _frontier_candidates(self, state: EvidenceState) -> list:
        return [
            candidate
            for candidate in state.candidates.values()
            if candidate.status == CandidateStatus.QUEUED
            and not self._url_seen(state, candidate.url)
            and self._canonicalize_url(candidate.url) not in self._scheduled_failed_urls(state)
        ]

    def _scheduled_failed_urls(self, state: EvidenceState) -> set[str]:
        return {self._canonicalize_url(url) for url in state.failed_urls}

    def _has_high_value_candidates(
        self,
        state: EvidenceState,
        threshold: float = 0.45,
    ) -> bool:
        self._rerank_candidates(state)
        return any(
            float(candidate.priority) >= threshold
            for candidate in self._frontier_candidates(state)
        )

    @staticmethod
    def _coverage_safe_page(page: PageEvidence) -> PageEvidence:
        return page.model_copy(
            update={
                "url": str(page.url),
                "canonical_url": str(page.canonical_url),
            }
        )

    def _url_seen(self, state: EvidenceState, url) -> bool:
        canonical = self._canonicalize_url(url)
        if not canonical:
            return True
        return canonical in {
            self._canonicalize_url(value)
            for value in state.visited_urls
        }

    def _needs_current_leadership_evidence(
        self,
        state: EvidenceState,
    ) -> bool:
        """
        Return True when the collected evidence does not contain a strong
        first-party/current leadership source.

        Editorial pages (press/news/blog/media) do not count as current
        leadership evidence on their own.
        """
        pages = list(state.pages.values())

        for page in pages:
            url = self._canonicalize_url(
                getattr(page, "url", "")
            )
            path = urlparse(url).path.lower()

            if any(
                bad in path
                for bad in self._LEADERSHIP_BAD_PATHS
            ):
                continue

            text = str(
                getattr(
                    page,
                    "clean_text",
                    "",
                )
                or ""
            ).lower()

            entities = (
                getattr(
                    page,
                    "structured_entities",
                    [],
                )
                or []
            )

            linkedin_urls = (
                getattr(
                    page,
                    "linkedin_urls",
                    [],
                )
                or []
            )

            person_entities = sum(
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

            role_hits = sum(
                1
                for term in self._LEADERSHIP_ROLE_TERMS
                if term in text
            )

            direct_path = any(
                term in path
                for term in self._LEADERSHIP_RECOVERY_PATHS
            )

            if (
                person_entities >= 1
                and role_hits >= 1
            ):
                return False

            if (
                direct_path
                and role_hits >= 2
            ):
                return False

            if (
                len(linkedin_urls) >= 2
                and role_hits >= 2
            ):
                return False

        return True

    def _score_leadership_recovery_candidate(
        self,
        candidate,
    ) -> tuple[float, str]:
        url = self._canonicalize_url(
            candidate.url
        )

        path = urlparse(url).path.lower()
        anchor = str(
            getattr(
                candidate,
                "anchor_text",
                "",
            )
            or ""
        ).lower()
        context = str(
            getattr(
                candidate,
                "context",
                "",
            )
            or ""
        ).lower()

        text = (
            f"{path} "
            f"{anchor} "
            f"{context}"
        )

        score = 0.0
        reasons: list[str] = []

        if any(
            bad in path
            for bad in self._LEADERSHIP_BAD_PATHS
        ):
            score -= 6.0
            reasons.append(
                "editorial path penalty"
            )

        path_hits = [
            term
            for term in self._LEADERSHIP_RECOVERY_PATHS
            if term in path
        ]

        if path_hits:
            score += 8.0
            reasons.append(
                "direct leadership/about path: "
                + path_hits[0]
            )

        role_hits = [
            term
            for term in self._LEADERSHIP_ROLE_TERMS
            if term in text
        ]

        if role_hits:
            score += min(
                5.0,
                len(role_hits) * 1.5,
            )
            reasons.append(
                "role signals: "
                + ", ".join(role_hits[:3])
            )

        for signal in (
            "team",
            "leadership",
            "founder",
            "founders",
            "executive",
            "people",
            "management",
        ):
            if signal in text:
                score += 1.0
                reasons.append(
                    f"leadership signal: {signal}"
                )

        expected_gain = self._float_attr(
            candidate,
            "expected_field_gain",
        )

        relevance = self._float_attr(
            candidate,
            "relevance",
        )

        score += 1.2 * expected_gain
        score += 0.4 * relevance

        depth = self._int_attr(
            candidate,
            "depth",
        )

        score -= 0.10 * depth

        return (
            float(score),
            "; ".join(reasons)
            or "leadership recovery candidate",
        )

    def _leadership_recovery_frontier(
        self,
        state: EvidenceState,
    ) -> list:
        scored = []

        for candidate in self._frontier_candidates(state):
            score, reason = (
                self._score_leadership_recovery_candidate(
                    candidate
                )
            )

            if score <= 0:
                continue

            scored.append(
                (
                    score,
                    candidate,
                    reason,
                )
            )

        scored.sort(
            key=lambda item: (
                -item[0],
                self._canonicalize_url(
                    item[1].url
                ),
            )
        )

        return scored

    async def _run_leadership_recovery(
        self,
        *,
        state: EvidenceState,
        decisions: list[ResearchDecision],
    ) -> bool:
        """
        Spend at most two additional successful-page opportunities on
        leadership evidence, and only when current leadership evidence is weak.

        This runs only when normal stopping says coverage is reached.
        """
        if (
            self._leadership_recovery_attempts
            >= self._max_leadership_recovery_attempts
        ):
            return False

        if not self._needs_current_leadership_evidence(
            state
        ):
            return False

        # Preserve the exact page budget.
        if (
            state.usage.pages_crawled
            >= state.budget.max_pages
        ):
            return False

        frontier = self._leadership_recovery_frontier(
            state
        )

        if not frontier:
            return False

        used_any = False

        while (
            frontier
            and self._leadership_recovery_attempts
            < self._max_leadership_recovery_attempts
            and state.usage.pages_crawled
            < state.budget.max_pages
        ):
            score, candidate, reason = frontier.pop(0)

            candidate_url = self._canonicalize_url(
                candidate.url
            )

            if self._url_seen(
                state,
                candidate_url,
            ):
                candidate.status = (
                    CandidateStatus.SKIPPED
                )
                continue

            self._leadership_recovery_attempts += 1
            state.metrics.crawl_iterations += 1

            iteration = (
                state.metrics.crawl_iterations
            )

            candidate.status = (
                CandidateStatus.CRAWLED
            )

            page = await self._crawl(
                candidate_url,
                source_kind=SourceKind.INTERNAL_PAGE,
                state=state,
            )

            if page is None:
                self._handle_candidate_failure(
                    state=state,
                    candidate=candidate,
                )

                decisions.append(
                    ResearchDecision(
                        iteration=iteration,
                        url=candidate_url,
                        priority=float(score),
                        reason=(
                            "leadership recovery failed: "
                            + reason
                        ),
                        information_gain=0.0,
                        success=False,
                    )
                )

                continue

            added = self._record_page(
                state=state,
                page=page,
            )

            if not added:
                candidate.status = (
                    CandidateStatus.SKIPPED
                )

                decisions.append(
                    ResearchDecision(
                        iteration=iteration,
                        url=candidate_url,
                        priority=float(score),
                        reason=(
                            "leadership recovery duplicate: "
                            + reason
                        ),
                        information_gain=0.0,
                        success=True,
                    )
                )

                continue

            gain = update_coverage_from_page(
                state.coverage,
                self._coverage_safe_page(page),
            )

            self._register_cluster_gain(
                candidate,
                gain,
            )

            candidate.status = (
                CandidateStatus.CRAWLED
            )

            decisions.append(
                ResearchDecision(
                    iteration=iteration,
                    url=self._canonicalize_url(
                        page.url
                    ),
                    priority=float(score),
                    reason=(
                        "leadership recovery: "
                        + reason
                    ),
                    information_gain=gain,
                    success=True,
                )
            )

            used_any = True

            # The new page may expose an even better first-party team/about
            # page, so discover its links before deciding whether another
            # recovery attempt is warranted.
            await self._discover_candidates(
                state=state,
                page=page,
            )

            if not self._needs_current_leadership_evidence(
                state
            ):
                break

            frontier = self._leadership_recovery_frontier(
                state
            )

        return used_any

    @staticmethod
    def _valid_fetch_result(result: FetchResult | None) -> bool:
        if result is None:
            return False
        try:
            return bool(
                result.success
                and bool(result.html)
                and bool(str(result.url))
            )
        except Exception:
            return False

    @staticmethod
    def _method_is_http(result: FetchResult) -> bool:
        method = getattr(result, "method", None)
        method_value = getattr(method, "value", method)
        return str(method_value).lower() == "http"

    @staticmethod
    def _record_resource_error(
        state: EvidenceState,
        url: str,
        exc: Exception,
    ) -> None:
        if len(state.errors) >= 8:
            return
        state.add_error(
            f"Resource failed {str(url)}: "
            f"{type(exc).__name__}: {exc}"
        )

    @staticmethod
    def _record_local_evidence_error(
        result: FetchResult,
        exc: Exception,
    ) -> None:
        print(
            "      [evidence-build-error] "
            f"{type(exc).__name__}: {exc}"
        )

    def _coverage_is_complete(self, state: EvidenceState) -> bool:
        checker = getattr(state, "all_required_fields_covered", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                pass

        coverage = getattr(state, "coverage", None)
        if coverage is None:
            return False

        items = []
        values = getattr(coverage, "values", None)
        if callable(values):
            items = list(values())
        elif isinstance(coverage, dict):
            items = list(coverage.values())

        if not items:
            return False

        for item in items:
            checker = getattr(item, "is_sufficient", None)
            if callable(checker):
                if not checker():
                    return False
            elif float(getattr(item, "score", 0.0)) < 0.8:
                return False
        return True

    @classmethod
    def _extract_canonical_url(
        cls,
        soup: BeautifulSoup,
        source_url: str,
    ) -> str:
        source = cls._canonicalize_url(source_url)
        try:
            tag = soup.find(
                "link",
                attrs={
                    "rel": lambda value: cls._rel_contains_canonical(value),
                },
            )
            href = str(tag.get("href", "")).strip() if tag else ""
            if not href:
                return source

            from urllib.parse import urljoin

            candidate = cls._canonicalize_url(urljoin(source_url, href))
            if not candidate:
                return source

            source_host = (urlparse(source).hostname or "").lower()
            candidate_host = (urlparse(candidate).hostname or "").lower()
            if candidate_host and candidate_host != source_host:
                return source

            return candidate
        except Exception:
            return source

    @staticmethod
    def _rel_contains_canonical(value) -> bool:
        if isinstance(value, (list, tuple)):
            return any(str(item).lower() == "canonical" for item in value)
        return "canonical" in str(value).lower().split()

    @classmethod
    def _canonicalize_url(cls, value) -> str:
        raw = str(value).strip()
        if not raw:
            return ""

        try:
            raw, _ = urldefrag(raw)
            parsed = urlsplit(raw)
            scheme = parsed.scheme.lower()
            host = (parsed.hostname or "").lower()

            if not host:
                return raw.rstrip("/")

            port = parsed.port
            netloc = host
            if port is not None and not (
                (scheme == "https" and port == 443)
                or (scheme == "http" and port == 80)
            ):
                netloc = f"{host}:{port}"

            path = parsed.path or "/"
            path = re.sub(r"/{2,}", "/", path)
            if path != "/":
                path = path.rstrip("/")

            query_items = []
            for key, val in parse_qsl(
                parsed.query,
                keep_blank_values=True,
            ):
                lower_key = key.lower()
                if lower_key in cls._TRACKING_QUERY_KEYS:
                    continue
                if any(lower_key.startswith(prefix) for prefix in cls._TRACKING_QUERY_PREFIXES):
                    continue
                query_items.append((key, val))

            query = urlencode(query_items, doseq=True)

            return urlunsplit(
                (scheme, netloc, path, query, "")
            )
        except Exception:
            return raw.rstrip("/")

    @classmethod
    def _is_locale_url(cls, value) -> bool:
        try:
            path = urlparse(cls._canonicalize_url(value)).path
            parts = [part for part in path.split("/") if part]
            if not parts:
                return False
            first = parts[0].lower()
            return first in cls._LOCALE_CODES
        except Exception:
            return False

    @staticmethod
    def _running_average(previous: float, value: float, count: int) -> float:
        if count <= 1:
            return value
        return previous + ((value - previous) / count)

    @staticmethod
    def _float_attr(obj, name: str, default: float = 0.0) -> float:
        try:
            return float(getattr(obj, name, default) or default)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _int_attr(obj, name: str, default: int = 0) -> int:
        try:
            return int(getattr(obj, name, default) or default)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _finalize_status(state: EvidenceState) -> None:
        reason = state.termination_reason.name
        if reason == "FATAL_ERROR":
            state.status = PipelineStatus.FAILED
        elif reason == "COVERAGE_REACHED":
            state.status = PipelineStatus.COMPLETED
        else:
            state.status = PipelineStatus.PARTIAL

    @staticmethod
    def _normalize_domain(domain: str) -> str:
        value = str(domain).strip()
        if not value.startswith(("http://", "https://")):
            value = f"https://{value}"

        parsed = urlparse(value)
        host = str(parsed.netloc).lower()
        host = host.rsplit("@", 1)[-1]
        host = host.split(":", 1)[0]

        if host.startswith("www."):
            host = host[4:]

        return host

    @staticmethod
    def _as_url_string(value) -> str:
        return str(value)

    @staticmethod
    def _page_id(source_kind: SourceKind, url: str) -> str:
        return f"{source_kind.value}:{str(url)}"
