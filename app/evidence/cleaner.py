from __future__ import annotations

import hashlib
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from app.evidence.models import LinkCandidate, TextBlock


HARD_REMOVE_TAGS = {
    "script",
    "style",
    "svg",
    "noscript",
    "template",
    "canvas",
    "iframe",
    "object",
    "embed",
}

BOILERPLATE_TAGS = {
    "nav",
    "footer",
    "aside",
}

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

BOILERPLATE_TEXT_PATTERNS = [
    r"^skip to (main )?content$",
    r"^cookie settings?$",
    r"^privacy settings?$",
    r"^accept cookies?$",
]

# Generic containers used only as a fallback for modern div/span-heavy pages.
FALLBACK_CONTAINER_TAGS = {
    "main",
    "article",
    "section",
    "div",
}

MIN_FALLBACK_CHARS = 20
MAX_FALLBACK_CHARS = 1000


def clean_html(
    html: str,
    base_url: str,
) -> tuple[str, list[TextBlock], list[str], list[LinkCandidate]]:
    """
    Convert raw HTML into compact structured evidence.

    Important:
    - Links are extracted BEFORE DOM cleanup.
    - DOM cleanup is defensive and must never crash the pipeline.
    - Text extraction is conservative first, then uses a bounded fallback
      for div/span-heavy modern websites.
    """

    soup = BeautifulSoup(html or "", "lxml")

    # Extract links before deleting navigation/footer/etc.
    links = _extract_links(soup, base_url)

    # Now clean the DOM used for textual evidence.
    _remove_unwanted_nodes(soup)

    headings = _extract_headings(soup)
    text_blocks = _extract_text_blocks(soup)

    clean_text = "\n\n".join(block.text for block in text_blocks)

    return clean_text, text_blocks, headings, links


def _remove_unwanted_nodes(soup: BeautifulSoup) -> None:
    """Remove nodes unlikely to contain useful textual evidence."""

    for node in list(soup.find_all(True)):
        try:
            if not isinstance(node, Tag):
                continue

            name = (node.name or "").lower()

            if name in HARD_REMOVE_TAGS:
                node.decompose()
                continue

            attrs = getattr(node, "attrs", None) or {}

            classes = attrs.get("class", [])
            if isinstance(classes, str):
                classes = classes.split()
            elif not isinstance(classes, (list, tuple, set)):
                classes = []

            class_text = " ".join(str(x).lower() for x in classes)
            node_id = str(attrs.get("id", "") or "").lower()

            if "hidden" in attrs:
                node.decompose()
                continue

            aria_hidden = str(attrs.get("aria-hidden", "")).lower()
            if aria_hidden == "true":
                node.decompose()
                continue

            style = str(attrs.get("style", "") or "").lower()
            if _is_hidden_style(style):
                node.decompose()
                continue

            if name in BOILERPLATE_TAGS:
                node.decompose()
                continue

            marker = f"{node_id} {class_text}"

            if _looks_like_boilerplate_container(marker):
                node.decompose()

        except Exception:
            # A malformed DOM node cannot kill the domain.
            continue


def _is_hidden_style(style: str) -> bool:
    normalized = re.sub(r"\s+", "", style or "")

    return (
        "display:none" in normalized
        or "visibility:hidden" in normalized
        or "opacity:0" in normalized
    )


def _looks_like_boilerplate_container(marker: str) -> bool:
    patterns = (
        "cookie-banner",
        "cookie-consent",
        "cookie-modal",
        "consent-banner",
        "privacy-banner",
        "newsletter-popup",
        "newsletter-modal",
        "modal-backdrop",
        "modal-overlay",
        "mobile-menu",
        "mobile-nav",
    )

    return any(pattern in marker for pattern in patterns)


def _extract_headings(soup: BeautifulSoup) -> list[str]:
    headings: list[str] = []

    for node in soup.find_all(HEADING_TAGS):
        try:
            text = _normalize_text(node.get_text(" ", strip=True))

            if text and not _is_boilerplate_text(text):
                headings.append(text)

        except Exception:
            continue

    return _dedupe_preserve_order(headings)


def _extract_text_blocks(soup: BeautifulSoup) -> list[TextBlock]:
    """
    Two-pass extraction:

    Pass 1:
        p/li elements — highest precision.

    Pass 2:
        bounded fallback over semantic containers whose meaningful text lives
        directly in div/span nodes. This is important for modern JS-heavy sites.
    """

    blocks: list[TextBlock] = []
    seen_hashes: set[str] = set()

    heading_stack: list[str] = []

    # ---------------------------------------------------------
    # Pass 1: explicit semantic text elements
    # ---------------------------------------------------------

    for node in soup.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]
    ):
        try:
            if not isinstance(node, Tag):
                continue

            name = (node.name or "").lower()
            text = _normalize_text(node.get_text(" ", strip=True))

            if not text or _is_boilerplate_text(text):
                continue

            if name in HEADING_TAGS:
                level = int(name[1])

                heading_stack = heading_stack[: level - 1]
                heading_stack.append(text)
                continue

            if len(text) < 3:
                continue

            _append_block(
                blocks=blocks,
                seen_hashes=seen_hashes,
                text=text,
                heading_path=heading_stack,
            )

        except Exception:
            continue

    # ---------------------------------------------------------
    # Pass 2: div/span-heavy fallback
    # ---------------------------------------------------------

    for node in soup.find_all(FALLBACK_CONTAINER_TAGS):
        try:
            if not isinstance(node, Tag):
                continue

            # Skip containers that simply wrap other block-level containers.
            direct_block_children = node.find_all(
                ["p", "li", "article", "section"],
                recursive=False,
            )

            if direct_block_children:
                continue

            direct_text = _extract_direct_text(node)

            if not direct_text:
                continue

            if _is_boilerplate_text(direct_text):
                continue

            if not (
                MIN_FALLBACK_CHARS
                <= len(direct_text)
                <= MAX_FALLBACK_CHARS
            ):
                continue

            # Avoid giant structural wrappers.
            descendant_divs = node.find_all("div")
            if len(descendant_divs) > 12:
                continue

            heading_path = _infer_heading_path(node)

            _append_block(
                blocks=blocks,
                seen_hashes=seen_hashes,
                text=direct_text,
                heading_path=heading_path,
            )

        except Exception:
            continue

    return blocks


def _append_block(
    *,
    blocks: list[TextBlock],
    seen_hashes: set[str],
    text: str,
    heading_path: list[str],
) -> None:
    content_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    if content_hash in seen_hashes:
        return

    seen_hashes.add(content_hash)

    blocks.append(
        TextBlock(
            block_id=f"block-{len(blocks) + 1}",
            heading_path=list(heading_path),
            text=text,
            char_count=len(text),
            content_hash=content_hash,
        )
    )


def _extract_direct_text(node: Tag) -> str:
    """
    Extract text directly contained by the node.

    This deliberately avoids collecting the complete descendant subtree,
    which would recreate giant duplicate blocks.
    """

    pieces: list[str] = []

    try:
        for child in node.children:
            # NavigableString-like objects have no .name.
            if not getattr(child, "name", None):
                text = str(child).strip()

                if text:
                    pieces.append(text)

        return _normalize_text(" ".join(pieces))

    except Exception:
        return ""


def _infer_heading_path(node: Tag) -> list[str]:
    """
    Approximate heading context by looking at preceding headings in document
    order. This is intentionally bounded and cheap.
    """

    try:
        previous = node.find_all_previous(HEADING_TAGS, limit=6)

        if not previous:
            return []

        headings = [
            _normalize_text(item.get_text(" ", strip=True))
            for item in reversed(previous)
        ]

        headings = [
            heading
            for heading in headings
            if heading and not _is_boilerplate_text(heading)
        ]

        return headings[-3:]

    except Exception:
        return []


def _extract_links(
    soup: BeautifulSoup,
    base_url: str,
) -> list[LinkCandidate]:
    """
    Extract links from the original DOM before boilerplate removal.

    Navigation links are intentionally preserved because they are valuable
    discovery candidates such as /about, /team, /contact, /pricing, etc.
    """

    links: list[LinkCandidate] = []
    seen_urls: set[str] = set()

    for node in soup.find_all("a"):
        try:
            if not isinstance(node, Tag):
                continue

            attrs = getattr(node, "attrs", None) or {}

            href = attrs.get("href")
            if not isinstance(href, str):
                continue

            href = href.strip()

            if not href:
                continue

            if (
                href.startswith("#")
                or href.startswith("javascript:")
                or href.startswith("mailto:")
                or href.startswith("tel:")
            ):
                continue

            absolute_url = urljoin(base_url, href).split("#", 1)[0]

            if not absolute_url or absolute_url in seen_urls:
                continue

            seen_urls.add(absolute_url)

            anchor_text = _normalize_text(
                node.get_text(" ", strip=True)
            )

            surrounding_text = _extract_surrounding_context(node) or ""

            # Pydantic contract: rel must be list[str].
            rel_value = attrs.get("rel")

            if isinstance(rel_value, str):
                rel = rel_value.split()
            elif isinstance(rel_value, (list, tuple, set)):
                rel = [str(item) for item in rel_value]
            else:
                rel = []

            links.append(
                LinkCandidate(
                    url=absolute_url,
                    anchor_text=anchor_text,
                    surrounding_text=surrounding_text,
                    rel=rel,
                )
            )

        except Exception as exc:
            print(
                f"      [link-skip] "
                f"{type(exc).__name__}: {exc}"
            )
            continue

    return links


def _extract_surrounding_context(
    node: Tag,
    max_chars: int = 500,
) -> str | None:
    try:
        parent = node.parent

        if not isinstance(parent, Tag):
            return None

        text = _normalize_text(
            parent.get_text(" ", strip=True)
        )

        if not text:
            return None

        return text[:max_chars]

    except Exception:
        return None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _is_boilerplate_text(text: str) -> bool:
    normalized = _normalize_text(text).lower()

    if not normalized:
        return True

    return any(
        re.search(pattern, normalized)
        for pattern in BOILERPLATE_TEXT_PATTERNS
    )


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)

    return result