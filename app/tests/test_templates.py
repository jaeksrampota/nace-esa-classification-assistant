"""The page templates: what a browser needs from them that a server-side test can check.

Both checks guard real defects found in the browser on 23 Sept 2026: the htmx script carried
an integrity hash that was not htmx 1.9.12's, so every browser blocked it and the page only
ever worked as a plain form post; and htmx, once loaded, does not swap 4xx/5xx answers, which
would have left the page silent on a 503 "Číselníky nejsou k dispozici".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[1] / "ui"
SUGGEST = UI / "templates" / "suggest.html"
PROTOTYPE = UI / "prototype" / "suggest.html"

#: The SRI cdnjs publishes for htmx 1.9.12 htmx.min.js (api.cdnjs.com, checked 2026-09-23;
#: equal to the SHA-512 of the served file). Change it only together with the version.
HTMX_SRC = "https://cdnjs.cloudflare.com/ajax/libs/htmx/1.9.12/htmx.min.js"
HTMX_SRI = "sha512-JvpjarJlOl4sW26MnEb3IdSAcGdeTeOaAlu2gUZtfFrRgnChdzELOZKl0mN6ZvI0X+xiX5UMvxjK2Rx2z/fliw=="


def _script_tag(html: str, src: str) -> str:
    match = re.search(r"<script[^>]*" + re.escape(src) + r"[^>]*>", html, re.DOTALL)
    assert match, f"no <script> loads {src}"
    return match.group(0)


def test_htmx_is_loaded_with_its_published_integrity_hash() -> None:
    """A wrong hash makes the browser block the script silently - the 22 Sept page did that."""
    tag = _script_tag(SUGGEST.read_text(encoding="utf-8"), HTMX_SRC)
    assert f'integrity="{HTMX_SRI}"' in tag
    assert 'crossorigin="anonymous"' in tag


def test_error_answers_are_swapped_in() -> None:
    """htmx 1.9 skips 4xx/5xx by default; the 503 page must still reach the user."""
    html = SUGGEST.read_text(encoding="utf-8")
    assert "htmx:beforeSwap" in html
    assert "shouldSwap = true" in html


@pytest.mark.parametrize("phrase", ["navrhovaný kód", "podle pravidel", "jistota"])
def test_each_state_label_appears_only_where_it_is_rendered(phrase: str) -> None:
    """The page tests tell the states apart by these words (a proposal, a rule's proposal, a
    model's confidence). A CSS comment or a legend repeating one - comments are sent with the
    page - would make every state look like every other."""
    assert SUGGEST.read_text(encoding="utf-8").count(phrase) == 1


def _css_rule(html: str, selector: str) -> str:
    match = re.search(r"^\s*" + re.escape(selector) + r" \{([^}]*)\}", html, re.MULTILINE)
    assert match, f"no CSS rule for {selector}"
    return match.group(1)


@pytest.mark.parametrize("selector", [".code", ".ctsid", ".issuer .meta"])
def test_codes_and_identifiers_are_set_in_a_face_with_a_plain_zero(selector: str) -> None:
    """MO read a CTS ID's dotted 0 as an 8 (12 Oct 2026): Windows' monospace fonts mark the zero
    (Cascadia Mono a dot, Consolas a slash). The values copied into CTS - the code, its CTS ID,
    the ISIN/LEI/IČO line - are set in the sans face, whose zero is plain, with tabular figures."""
    rule = _css_rule(SUGGEST.read_text(encoding="utf-8"), selector)
    assert "var(--sans)" in rule
    assert "var(--mono)" not in rule
    assert "tabular-nums" in rule


def test_prototype_carries_the_template_stylesheet() -> None:
    """``ui/prototype/suggest.html`` is the design reference: its CSS is a verbatim copy of the
    template's (CLAUDE.md: change both together), so a look fixed on one page is fixed on both."""
    style = re.compile(r"<style>.*?</style>", re.DOTALL)
    template_css = style.search(SUGGEST.read_text(encoding="utf-8"))
    prototype_css = style.search(PROTOTYPE.read_text(encoding="utf-8"))
    assert template_css and prototype_css
    assert prototype_css.group(0) == template_css.group(0)
