"""The model searches the web for the issuer: who it is, what it does, who controls it.

Jakub, 30 Sept 2026: the tool is to look at the web on every lookup, with the model ("use
llm for each"). No search API is procured (roadmap Q13), but the OpenAI key production already
uses can search: the Responses API's ``web_search`` tool, which ``gpt-5.6-luna`` supports
(model page, checked 30 Sept 2026). One call per issuer, cached like the classifications, so
the page, the download and a report's re-run read the same text, and a repeated issuer costs
nothing.

The answer is evidence, not a decision. The model returns the issuer's official name and two
to four Czech sentences - the activity, the kind of institution, the seat, the group and, when
the sources say so, who owns or controls it, which the ESA control digit turns on (Q7). The
sentences join the description after whatever was typed or found on Wikipedia, labelled as the
model's, and the pages it cites join the evidence list. The Czech registers are excluded from
the search (``blocked_domains``) and a cited page on them is dropped, as the scraping ban
requires.

Fail-soft like every source: no model, no time, a failed call or an unreadable answer is a
note on the page at most, and the lookup goes on with what it has.

Cost (pricing page, 30 Sept 2026): 10 USD per 1,000 searches plus the search content as input
tokens at the model's rate - about 0.01 USD a lookup, some ten times the two classification
calls. The ledger records the call as ``WEB``, charged to the signed-in user like the others.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

from core.classify.cache import ClassificationCache, NullCache
from core.classify.errors import LlmError, LlmNotConfiguredError
from core.classify.prompts import Prompt
from core.classify.provider import LlmProvider, NullLlmProvider
from core.sources.names import fold
from core.sources.web import (
    BLOCKED_HOSTS,
    NO_SEARCH_PROVIDER_NOTE,
    EvidenceSource,
    IssuerEvidence,
    is_blocked,
)

if TYPE_CHECKING:
    from config.settings import Settings
    from core.classify.llm import LlmClassifier

LOGGER = logging.getLogger(__name__)

#: Bump when the prompt's wording or the answer's shape changes: it is in the cache key.
WEB_PROMPT_VERSION: Final[str] = "web/1"
#: Longest description kept; the prompt asks for four sentences at most.
MAX_DESCRIPTION_CHARS: Final[int] = 1200
#: Cited pages kept for the evidence list.
MAX_SOURCES: Final[int] = 5
#: How much of a typed description goes into the search prompt, as a hint.
TYPED_HINT_CHARS: Final[int] = 300

_SYSTEM: Final[str] = (
    "Jsi rešeršista oddělení, které zakládá zahraniční emitenty cenných papírů do systému CTS "
    "a přiřazuje jim kódy NACE (činnost) a ESA 2010 (sektor). Najdi na webu, kdo je emitent a "
    "čím se zabývá. Hledej podle identifikátorů (ISIN, LEI) i podle názvu; přednost mají "
    "oficiální stránky emitenta, výroční zprávy, prospekty, regulátoři a Wikipedie.\n\n"
    "Odpověz česky, přesně ve dvou řádcích:\n"
    "NÁZEV: oficiální název emitenta (právnické osoby, která cenný papír vydala)\n"
    "POPIS: 2-4 věty - hlavní činnost emitenta; zda jde o banku, pojišťovnu, investiční fond, "
    "finanční nebo účelovou společnost skupiny, veřejnou instituci, nebo nefinanční podnik; "
    "země sídla; skupina, do které patří; a kdo emitenta vlastní nebo ovládá (stát, zahraniční "
    "mateřská společnost, rozptýlení akcionáři), pokud to zdroje uvádějí.\n\n"
    "Nevymýšlej si nic, co zdroje neuvádějí. Když emitenta spolehlivě neurčíš, odpověz jediným "
    "slovem: NENALEZENO"
)

_NOT_FOUND: Final[str] = "nenalezeno"
_NAME_RE: Final[re.Pattern[str]] = re.compile(
    r"^[\s*_]*N[ÁA]ZEV[\s*_]*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
)
_DESCRIPTION_RE: Final[re.Pattern[str]] = re.compile(
    r"^[\s*_]*POPIS[\s*_]*:\s*(.+)", re.IGNORECASE | re.MULTILINE | re.DOTALL
)
#: " ([example.com](https://...))" - the citation chips the model puts inline.
_CITATION_CHIP_RE: Final[re.Pattern[str]] = re.compile(r"\s*\(\[[^\]]*\]\(https?://[^)]*\)\)")
#: "[text](https://...)" -> "text".
_LINK_RE: Final[re.Pattern[str]] = re.compile(r"\[([^\]]+)\]\(https?://[^)]*\)")


@dataclass(frozen=True, slots=True)
class WebFinding:
    """What the model found on the web about one issuer; both texts ``None`` = not found."""

    issuer_name: str | None
    description: str | None
    sources: tuple[tuple[str, str], ...]
    model: str
    searched_at: datetime

    def to_json(self) -> str:
        return json.dumps(
            {
                "issuer_name": self.issuer_name,
                "description": self.description,
                "sources": [list(source) for source in self.sources],
                "model": self.model,
                "searched_at": self.searched_at.isoformat(),
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, text: str) -> WebFinding | None:
        """The cached finding, or ``None`` when the row cannot be read (it is then redone)."""
        try:
            data = json.loads(text)
            return cls(
                issuer_name=data.get("issuer_name"),
                description=data.get("description"),
                sources=tuple((str(url), str(title)) for url, title in data.get("sources", [])),
                model=str(data["model"]),
                searched_at=datetime.fromisoformat(data["searched_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("ignoring an unreadable web-search cache row: %s", exc)
            return None


def web_prompt(
    *,
    name: str | None,
    isin: str | None,
    lei: str | None,
    country: str | None,
    instrument: str | None,
    typed: str | None,
) -> Prompt:
    """The search request: only public identifiers and names, and MO's own words as a hint."""
    hint = " ".join((typed or "").split())[:TYPED_HINT_CHARS]
    lines = [
        f"{label}: {value}"
        for label, value in (
            ("ISIN", isin),
            ("LEI", lei),
            ("Název", name),
            ("Země sídla podle GLEIF", country),
            ("Nástroj podle OpenFIGI", instrument),
            ("Popis zadaný uživatelem", hint),
        )
        if value
    ]
    return Prompt(
        kind="WEB",
        system=_SYSTEM,
        user="\n".join(lines),
        schema={},
        version=WEB_PROMPT_VERSION,
        web_search=True,
        blocked_domains=tuple(sorted(BLOCKED_HOSTS)),
    )


def parse_answer(content: str) -> tuple[str | None, str | None]:
    """``(name, description)`` from the model's "NÁZEV: / POPIS:" answer.

    ``(None, None)`` for "NENALEZENO". An answer that ignored the format is taken whole as
    the description; inline citation links are reduced to their text.
    """
    text = _LINK_RE.sub(r"\1", _CITATION_CHIP_RE.sub("", content)).strip()
    if not text or fold(text).startswith(_NOT_FOUND):
        return None, None
    named = _NAME_RE.search(text)
    described = _DESCRIPTION_RE.search(text)
    name = named.group(1).strip(" *_\"'„“") if named else None
    body = described.group(1) if described else _NAME_RE.sub("", text)
    description = " ".join(body.split()).strip(" *_")
    if name and fold(name) in {_NOT_FOUND, ""}:
        name = None
    if not description or fold(description).startswith(_NOT_FOUND):
        description = None
    elif len(description) > MAX_DESCRIPTION_CHARS:
        description = description[:MAX_DESCRIPTION_CHARS].rstrip() + "…"
    return name, description


def web_cache_key(
    *, name: str | None, isin: str | None, lei: str | None, typed: str | None, model: str
) -> str:
    """Everything the search's answer depends on, as for :func:`core.classify.cache.cache_key`."""
    material = json.dumps(
        [
            "WEB",
            fold(name or ""),
            isin or "",
            lei or "",
            " ".join((typed or "").split()).casefold(),
            model,
            WEB_PROMPT_VERSION,
        ],
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class LlmWebSearch:
    """One web search per issuer through the (budgeted) model provider, cached.

    Args:
        provider: The classifier's provider, so the budget and the ledger cover this call.
        cache: The classifier's cache; the finding is stored as text under the kind ``WEB``.
        call_seconds: The longest one model call can take
            (:func:`~core.classify.provider.worst_case_call_seconds`).
        clock: Monotonic seconds; injected by the tests.
    """

    def __init__(
        self,
        provider: LlmProvider,
        *,
        cache: ClassificationCache | None = None,
        call_seconds: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._provider = provider
        self._cache = NullCache() if cache is None else cache
        self._call_seconds = max(0.0, call_seconds)
        self._clock = clock

    def find(
        self,
        *,
        name: str | None,
        isin: str | None,
        lei: str | None,
        country: str | None = None,
        instrument: str | None = None,
        typed: str | None = None,
        deadline: float | None = None,
    ) -> tuple[WebFinding | None, str | None]:
        """The finding (from the cache when there is one), or ``None`` and why, if anything.

        With a ``deadline`` the search starts only when a classification call still fits after
        it in the worst case: the codes matter more than one more paragraph of evidence.
        """
        key = web_cache_key(name=name, isin=isin, lei=lei, typed=typed, model=self._provider.model)
        cached = self._cache.get_text(key)
        finding = WebFinding.from_json(cached) if cached else None
        if finding is not None:
            return finding, None
        if deadline is not None and self._call_seconds > 0:
            left = deadline - self._clock()
            if left < 2 * self._call_seconds:
                return None, (
                    f"na hledání na webu nezbyl čas (zbývá {max(left, 0.0):.0f} s, hledání "
                    f"a klasifikace mohou trvat až {2 * self._call_seconds:.0f} s)"
                )
        prompt = web_prompt(
            name=name, isin=isin, lei=lei, country=country, instrument=instrument, typed=typed
        )
        try:
            response = self._provider.complete(prompt)
        except LlmNotConfiguredError:
            return None, None
        except LlmError as exc:
            LOGGER.warning("web search for %s failed: %s", isin or lei or name, exc)
            return None, f"hledání na webu (model) selhalo: {exc}"
        found_name, description = parse_answer(response.content)
        sources = tuple((url, title) for url, title in response.citations if not is_blocked(url))[
            :MAX_SOURCES
        ]
        finding = WebFinding(
            issuer_name=found_name,
            description=description,
            sources=sources if description else (),
            model=response.model,
            searched_at=datetime.now(UTC),
        )
        self._cache.put_text(
            key, finding.to_json(), kind="WEB", issuer_name=name, model=response.model
        )
        return finding, None

    def enrich(
        self,
        evidence: IssuerEvidence,
        *,
        name: str | None,
        isin: str | None,
        lei: str | None,
        country: str | None = None,
        instrument: str | None = None,
        typed: str | None = None,
        deadline: float | None = None,
    ) -> IssuerEvidence:
        """``evidence`` with the model's finding after its description, its pages and a note.

        The gatherer's "vyhledávání na webu není zapojené" is dropped whatever the outcome:
        the web is searched here, and a failure says so in its own note (live, 30 Sept 2026,
        the two notes stood side by side).
        """
        if not (name or isin or lei):
            return evidence
        evidence = replace(
            evidence, notes=tuple(n for n in evidence.notes if n != NO_SEARCH_PROVIDER_NOTE)
        )
        finding, note = self.find(
            name=name,
            isin=isin,
            lei=lei,
            country=country,
            instrument=instrument,
            typed=typed,
            deadline=deadline,
        )
        if finding is None:
            return replace(evidence, notes=(*evidence.notes, note)) if note else evidence
        if finding.description is None:
            return replace(
                evidence,
                notes=(*evidence.notes, f"model {finding.model} emitenta na webu nenašel"),
                found_name=finding.issuer_name,
            )
        label = f"Podle webu (vyhledal model {finding.model})"
        description = (
            f"{evidence.description}\n\n{label}: {finding.description}"
            if evidence.description
            else f"{label}: {finding.description}"
        )
        pages = tuple(
            EvidenceSource(
                url=url,
                title=f"Web (citoval model) – {title or url}",
                snippet="",
                fetched=False,
                retrieved_at=finding.searched_at,
            )
            for url, title in finding.sources
        )
        return replace(
            evidence,
            description=description,
            sources=(*evidence.sources, *pages),
            notes=(
                *evidence.notes,
                f"popis doplněn z webu: vyhledal model {finding.model} "
                f"(zdroje: {len(finding.sources)}) – ověřte, že popisuje právě tohoto emitenta",
            ),
            found_name=finding.issuer_name,
            issuer_name=evidence.issuer_name or finding.issuer_name,
        )


def build_web_search(settings: Settings, classifier: LlmClassifier) -> LlmWebSearch | None:
    """The web search on the classifier's provider and cache; ``None`` when it is off or no
    model is configured (a lookup without a model makes no call it could not answer)."""
    if not (settings.llm_web_search and settings.web_enabled):
        return None
    if classifier.provider.model == NullLlmProvider.model:
        return None
    return LlmWebSearch(
        classifier.provider, cache=classifier.cache, call_seconds=classifier.call_seconds
    )


__all__ = [
    "LlmWebSearch",
    "WEB_PROMPT_VERSION",
    "WebFinding",
    "build_web_search",
    "parse_answer",
    "web_cache_key",
    "web_prompt",
]
