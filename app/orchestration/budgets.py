from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Budget(BaseModel):
    """
    Hard limits for one domain-processing run.

    These are safety boundaries, not targets.
    """

    model_config = ConfigDict(extra="forbid")

    max_pages: int = 8
    max_runtime_seconds: float = 60.0

    max_llm_input_tokens: int = 12_000
    max_llm_output_tokens: int = 3_000

    max_search_queries: int = 3
    max_retries_per_resource: int = 2

    @field_validator(
        "max_pages",
        "max_llm_input_tokens",
        "max_llm_output_tokens",
        "max_search_queries",
        "max_retries_per_resource",
    )
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("budget limits must be greater than zero")
        return value

    @field_validator("max_runtime_seconds")
    @classmethod
    def validate_runtime(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("max_runtime_seconds must be greater than zero")
        return value


class BudgetUsage(BaseModel):
    """Runtime resource consumption."""

    model_config = ConfigDict(extra="forbid")

    pages_crawled: int = 0
    runtime_seconds: float = 0.0

    llm_input_tokens: int = 0
    llm_output_tokens: int = 0

    search_queries: int = 0
    retries: int = 0

    @property
    def total_tokens(self) -> int:
        return self.llm_input_tokens + self.llm_output_tokens

    def can_crawl_page(self, budget: Budget) -> bool:
        return self.pages_crawled < budget.max_pages

    def can_search(self, budget: Budget) -> bool:
        return self.search_queries < budget.max_search_queries

    def has_llm_budget(self, budget: Budget) -> bool:
        return (
            self.llm_input_tokens < budget.max_llm_input_tokens
            and self.llm_output_tokens < budget.max_llm_output_tokens
        )

    def has_runtime_budget(self, budget: Budget) -> bool:
        return self.runtime_seconds < budget.max_runtime_seconds

    def record_page(self) -> None:
        self.pages_crawled += 1

    def record_search(self) -> None:
        self.search_queries += 1

    def record_retry(self) -> None:
        self.retries += 1

    def record_runtime(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("runtime increment cannot be negative")
        self.runtime_seconds += seconds

    def record_llm_usage(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts cannot be negative")

        self.llm_input_tokens += input_tokens
        self.llm_output_tokens += output_tokens


class BudgetDecision(BaseModel):
    """
    Result of asking whether an operation is currently allowed.
    """

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    reason: str | None = None


class BudgetExceeded(RuntimeError):
    """Raised only when a caller explicitly requests a forbidden operation."""

    pass