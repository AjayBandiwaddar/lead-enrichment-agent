from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, TypeVar

from pydantic import BaseModel

from .prompts import (
    SYSTEM_PROMPT,
    contacts_prompt,
    leadership_prompt,
    overview_icp_prompt,
)
from .schemas import (
    ContactsExtraction,
    ContactsLeadershipExtraction,
    LeadershipExtraction,
    OverviewICPExtraction,
)


SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class LLMResult:
    value: BaseModel
    usage: LLMUsage
    provider: str
    model: str


class StructuredExtractor:
    """Gemini-first structured extractor with Groq fallback."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        gemini_client: Any | None = None,
        groq_client: Any | None = None,
        gemini_model: str | None = None,
        groq_model: str | None = None,
        max_retries: int = 1,
    ) -> None:
        self.provider = (
            provider
            or os.getenv(
                "EXTRACTION_PROVIDER",
                "gemini",
            )
        ).strip().lower()

        self.gemini_model = (
            gemini_model
            or os.getenv(
                "GEMINI_MODEL",
                "gemini-3.8-flash",
            )
        )

        self.groq_model = (
            groq_model
            or os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-120b",
            )
        )

        self.max_retries = max(
            0,
            int(max_retries),
        )

        self.gemini_client = gemini_client
        self.groq_client = groq_client

    def extract_overview_icp(
        self,
        context_text: str,
    ) -> LLMResult:
        return self._extract(
            schema=OverviewICPExtraction,
            user_prompt=overview_icp_prompt(
                context_text
            ),
        )

    def extract_contacts(
        self,
        context_text: str,
    ) -> LLMResult:
        return self._extract(
            schema=ContactsExtraction,
            user_prompt=contacts_prompt(
                context_text
            ),
        )

    def extract_leadership(
        self,
        context_text: str,
    ) -> LLMResult:
        return self._extract(
            schema=LeadershipExtraction,
            user_prompt=leadership_prompt(
                context_text
            ),
        )

    def extract_contacts_leadership(
        self,
        context_text: str,
    ) -> LLMResult:
        """Backward-compatible combined call."""
        return self._extract(
            schema=ContactsLeadershipExtraction,
            user_prompt=contacts_prompt(context_text)
            + "\n\n"
            + leadership_prompt(context_text),
        )

    def _extract(
        self,
        *,
        schema: type[SchemaT],
        user_prompt: str,
    ) -> LLMResult:
        providers = self._provider_order()
        errors: list[str] = []

        for provider in providers:
            for attempt in range(
                self.max_retries + 1
            ):
                try:
                    if provider == "gemini":
                        return self._call_gemini(
                            schema,
                            user_prompt,
                        )

                    if provider == "groq":
                        return self._call_groq(
                            schema,
                            user_prompt,
                        )

                    raise ValueError(
                        "Unsupported extraction provider: "
                        f"{provider}"
                    )

                except Exception as exc:
                    errors.append(
                        f"{provider} attempt "
                        f"{attempt + 1}: "
                        f"{type(exc).__name__}: {exc}"
                    )

        raise RuntimeError(
            "All extraction providers failed: "
            + " | ".join(errors[-6:])
        )

    def _provider_order(self) -> list[str]:
        if self.provider == "gemini":
            return ["gemini", "groq"]

        if self.provider == "groq":
            return ["groq", "gemini"]

        if self.provider == "auto":
            return ["gemini", "groq"]

        raise ValueError(
            "EXTRACTION_PROVIDER must be "
            "'gemini', 'groq', or 'auto'"
        )

    def _get_gemini_client(self):
        if self.gemini_client is not None:
            return self.gemini_client

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set"
            )

        from openai import OpenAI

        self.gemini_client = OpenAI(
            api_key=api_key,
            base_url=(
                "https://generativelanguage.googleapis.com/"
                "v1beta/openai/"
            ),
        )

        return self.gemini_client

    def _get_groq_client(self):
        if self.groq_client is not None:
            return self.groq_client

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set"
            )

        from openai import OpenAI

        self.groq_client = OpenAI(
            api_key=api_key,
            base_url=(
                "https://api.groq.com/openai/v1"
            ),
        )

        return self.groq_client

    def _call_gemini(
        self,
        schema: type[SchemaT],
        user_prompt: str,
    ) -> LLMResult:
        client = self._get_gemini_client()

        response = (
            client.beta.chat.completions.parse(
                model=self.gemini_model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                response_format=schema,
            )
        )

        message = response.choices[0].message

        refusal = getattr(
            message,
            "refusal",
            None,
        )

        if refusal:
            raise RuntimeError(
                f"Gemini refused extraction: {refusal}"
            )

        parsed = getattr(
            message,
            "parsed",
            None,
        )

        if parsed is None:
            raise RuntimeError(
                "Gemini returned no parsed structured output"
            )

        usage = getattr(
            response,
            "usage",
            None,
        )

        return LLMResult(
            value=schema.model_validate(
                parsed
            ),
            usage=LLMUsage(
                input_tokens=int(
                    getattr(
                        usage,
                        "prompt_tokens",
                        0,
                    )
                    or 0
                ),
                output_tokens=int(
                    getattr(
                        usage,
                        "completion_tokens",
                        0,
                    )
                    or 0
                ),
            ),
            provider="gemini",
            model=self.gemini_model,
        )

    def _call_groq(
        self,
        schema: type[SchemaT],
        user_prompt: str,
    ) -> LLMResult:
        client = self._get_groq_client()

        response = (
            client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "strict": True,
                        "schema": _strict_json_schema(
                            schema
                        ),
                    },
                },
            )
        )

        message = response.choices[0].message

        refusal = getattr(
            message,
            "refusal",
            None,
        )

        if refusal:
            raise RuntimeError(
                f"Groq refused extraction: {refusal}"
            )

        content = getattr(
            message,
            "content",
            None,
        )

        if not content:
            raise RuntimeError(
                "Groq returned empty structured output"
            )

        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Groq returned invalid JSON"
            ) from exc

        usage = getattr(
            response,
            "usage",
            None,
        )

        return LLMResult(
            value=schema.model_validate(
                raw
            ),
            usage=LLMUsage(
                input_tokens=int(
                    getattr(
                        usage,
                        "prompt_tokens",
                        0,
                    )
                    or 0
                ),
                output_tokens=int(
                    getattr(
                        usage,
                        "completion_tokens",
                        0,
                    )
                    or 0
                ),
            ),
            provider="groq",
            model=self.groq_model,
        )


def _strict_json_schema(
    schema: type[BaseModel],
) -> dict[str, Any]:
    document = schema.model_json_schema()

    _normalize_schema_node(
        document
    )

    return document


def _normalize_schema_node(
    node: Any,
) -> None:
    if not isinstance(
        node,
        dict,
    ):
        return

    if node.get("type") == "object":
        properties = node.get(
            "properties",
            {},
        )

        node["required"] = list(
            properties.keys()
        )

        node["additionalProperties"] = False

        for child in properties.values():
            _normalize_schema_node(
                child
            )

    if "$defs" in node:
        for child in node["$defs"].values():
            _normalize_schema_node(
                child
            )

    if "items" in node:
        _normalize_schema_node(
            node["items"]
        )

    for key in (
        "anyOf",
        "oneOf",
        "allOf",
    ):
        for child in (
            node.get(key, [])
            or []
        ):
            _normalize_schema_node(
                child
            )