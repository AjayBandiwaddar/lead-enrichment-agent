from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class FetchMethod(str, Enum):
    PLAYWRIGHT = "playwright"
    HTTP = "http"


class FetchResult(BaseModel):
    """
    Raw result of attempting to retrieve a URL.

    This intentionally contains no parsed evidence. Parsing belongs to
    app.evidence.
    """

    model_config = ConfigDict(extra="forbid")

    url: str
    method: FetchMethod

    status_code: int | None = None
    content_type: str | None = None

    html: str = ""

    success: bool = False

    error_type: str | None = None
    error_message: str | None = None

    elapsed_seconds: float = 0.0

    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def has_html(self) -> bool:
        return bool(self.html.strip())

    @property
    def is_http_error(self) -> bool:
        return (
            self.status_code is not None
            and self.status_code >= 400
        )