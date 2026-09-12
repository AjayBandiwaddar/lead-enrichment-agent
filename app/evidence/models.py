from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SourceKind(str, Enum):
    """Where a piece of evidence came from."""

    HOMEPAGE = "homepage"
    INTERNAL_PAGE = "internal_page"
    SITEMAP = "sitemap"
    LLMS_TXT = "llms_txt"
    MARKDOWN = "markdown"
    HTTP_FALLBACK = "http_fallback"
    EXTERNAL_SEARCH = "external_search"


class FetchStatus(str, Enum):
    """Outcome of attempting to retrieve a source."""

    SUCCESS = "success"
    HTTP_ERROR = "http_error"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    INVALID_CONTENT = "invalid_content"
    JS_RENDER_FAILURE = "js_render_failure"


# ---------------------------------------------------------------------------
# Navigation evidence
# ---------------------------------------------------------------------------


class LinkCandidate(BaseModel):
    """
    A link discovered on a page.

    This is intentionally richer than just a URL because the planner later
    uses anchor text and surrounding context when estimating the value of
    crawling the link.
    """

    model_config = ConfigDict(extra="forbid")

    url: HttpUrl
    anchor_text: str | None = None
    surrounding_text: str | None = None
    rel: list[str] = Field(default_factory=list)

    @field_validator("anchor_text", "surrounding_text")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = " ".join(value.split())
        return normalized or None


# ---------------------------------------------------------------------------
# Text / content evidence
# ---------------------------------------------------------------------------


class TextBlock(BaseModel):
    """
    A semantically meaningful piece of cleaned page text.

    Evidence spans in later Claim objects can point back to block_id instead
    of storing or searching the entire page again.
    """

    model_config = ConfigDict(extra="forbid")

    block_id: str
    heading_path: list[str] = Field(default_factory=list)
    text: str
    char_count: int
    content_hash: str

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = " ".join(value.split())

        if not normalized:
            raise ValueError("TextBlock.text cannot be empty.")

        return normalized

    @staticmethod
    def make_content_hash(text: str) -> str:
        """Create a stable SHA-256 hash for deduplication."""
        normalized = " ".join(text.split()).encode("utf-8")
        return sha256(normalized).hexdigest()


# ---------------------------------------------------------------------------
# Machine-readable structured data
# ---------------------------------------------------------------------------


class StructuredEntity(BaseModel):
    """
    Normalized representation of useful machine-readable entities.

    Examples:
      Organization
      Person
      ContactPoint

    `properties` deliberately remains generic because Schema.org markup can
    vary substantially across websites.
    """

    model_config = ConfigDict(extra="allow")

    entity_type: str
    properties: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Page evidence
# ---------------------------------------------------------------------------


class PageEvidence(BaseModel):
    """
    Canonical evidence record produced by the crawling/extraction layer.

    This object contains normalized evidence only. It should not contain LLM
    claims, confidence scores, or verification results.
    """

    model_config = ConfigDict(extra="forbid")

    # -------------------------
    # Identity
    # -------------------------

    page_id: str

    url: HttpUrl
    canonical_url: HttpUrl | None = None

    source_kind: SourceKind

    # -------------------------
    # Fetch metadata
    # -------------------------

    fetch_status: FetchStatus
    http_status: int | None = None
    content_type: str | None = None

    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # -------------------------
    # Page metadata
    # -------------------------

    title: str | None = None
    meta_description: str | None = None
    headings: list[str] = Field(default_factory=list)

    # -------------------------
    # Clean textual evidence
    # -------------------------

    clean_text: str = ""
    text_blocks: list[TextBlock] = Field(default_factory=list)

    # -------------------------
    # Deterministic signals
    # -------------------------

    emails: list[str] = Field(default_factory=list)
    linkedin_urls: list[HttpUrl] = Field(default_factory=list)
    social_urls: list[HttpUrl] = Field(default_factory=list)

    # -------------------------
    # Structured / machine data
    # -------------------------

    json_ld: list[dict[str, Any]] = Field(default_factory=list)
    structured_entities: list[StructuredEntity] = Field(default_factory=list)

    # -------------------------
    # Navigation
    # -------------------------

    outgoing_links: list[LinkCandidate] = Field(default_factory=list)

    # -------------------------
    # Deduplication
    # -------------------------

    content_hash: str | None = None

    # -------------------------
    # Validation
    # -------------------------

    @field_validator("title", "meta_description")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = " ".join(value.split())
        return normalized or None

    @field_validator("headings")
    @classmethod
    def normalize_headings(cls, values: list[str]) -> list[str]:
        return [
            normalized
            for value in values
            if (normalized := " ".join(value.split()))
        ]

    @field_validator("clean_text")
    @classmethod
    def normalize_clean_text(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("emails")
    @classmethod
    def normalize_emails(cls, values: list[str]) -> list[str]:
        normalized = {
            value.strip().lower()
            for value in values
            if value and value.strip()
        }

        return sorted(normalized)

    def has_content(self) -> bool:
        """Whether this page contains usable textual or structured evidence."""
        return bool(
            self.clean_text
            or self.text_blocks
            or self.json_ld
            or self.structured_entities
            or self.emails
            or self.linkedin_urls
        )

    def compute_content_hash(self) -> str | None:
        """
        Compute a stable hash from normalized page text.

        This is intentionally separate from validation so callers can decide
        when they want to pay the hashing cost.
        """
        if not self.clean_text:
            return None

        return sha256(
            self.clean_text.encode("utf-8")
        ).hexdigest()