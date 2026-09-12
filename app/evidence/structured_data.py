from __future__ import annotations

import json
from typing import Any

from bs4 import BeautifulSoup

from app.evidence.models import StructuredEntity


def extract_json_ld(html: str) -> list[dict[str, Any]]:
    """
    Extract valid JSON-LD objects from a page.

    Handles:
    - regular JSON objects
    - arrays of JSON-LD objects
    - @graph containers
    - malformed individual script blocks without crashing
    """
    soup = BeautifulSoup(html, "lxml")
    results: list[dict[str, Any]] = []

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text()

        if not raw or not raw.strip():
            continue

        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # One malformed JSON-LD block must not break the page.
            continue

        results.extend(_flatten_json_ld(parsed))

    return results


def _flatten_json_ld(value: Any) -> list[dict[str, Any]]:
    """Normalize common JSON-LD container shapes into a flat list."""
    if isinstance(value, dict):
        graph = value.get("@graph")

        if isinstance(graph, list):
            base = {
                key: item
                for key, item in value.items()
                if key != "@graph"
            }

            results: list[dict[str, Any]] = []

            for item in graph:
                if isinstance(item, dict):
                    if base:
                        merged = {**base, **item}
                        results.append(merged)
                    else:
                        results.append(item)

            return results

        return [value]

    if isinstance(value, list):
        return [
            item
            for element in value
            for item in _flatten_json_ld(element)
        ]

    return []


def normalize_structured_entities(
    json_ld: list[dict[str, Any]],
) -> list[StructuredEntity]:
    """
    Convert raw JSON-LD into lightweight normalized entities.

    We intentionally do not attempt full Schema.org interpretation here.
    The raw JSON-LD remains available for auditing.
    """
    entities: list[StructuredEntity] = []

    for item in json_ld:
        entity_type = item.get("@type")

        if isinstance(entity_type, list):
            entity_type = entity_type[0] if entity_type else None

        if not isinstance(entity_type, str):
            continue

        entities.append(
            StructuredEntity(
                entity_type=entity_type,
                properties=item,
            )
        )

    return entities


def extract_structured_entities(
    html: str,
) -> tuple[list[dict[str, Any]], list[StructuredEntity]]:
    """Convenience function used by the evidence pipeline."""
    json_ld = extract_json_ld(html)
    entities = normalize_structured_entities(json_ld)

    return json_ld, entities