"""A Czech subject by its name: ARES's name search, judged by a rule that never guesses.

Why this is here (5 Oct 2026, Jakub: "can we fix this?"): a Czech name GLEIF cannot match -
a subject without a LEI, or one whose Czech legal form GLEIF's matcher does not know
("Komerční banka" for "Komerční banka, a.s.") - ended in "zadejte IČO". Now ARES is asked,
and the subject is taken **only when exactly one active subject has the same normalised name,
judged over the complete set of hits**. Anything else lists the namesakes for MO to retype as
an IČO; nothing is ever picked. A wrong subject would carry wrong codes from RES straight
into the proposal, which is worse than a lookup MO finishes by hand.

The rule and the normalisation are ported from the RES a OR tool (``jaeksrampota/res-or-
lookup``, ``src/core/namematch.py`` and ``lookup._search_name``, Jakub's own code), where an
audit of 893 golden names gave 0 false accepts (``docs/MATCHING.md`` there, 2026-09-23):

- **normalise**: no diacritics, lower case, every non-alphanumeric run one space, then the
  trailing legal form (``a.s.``, ``s.r.o.``, ``spol. s r.o.``, the written-out forms...) and a
  status tag ("v likvidaci") stripped, repeatedly. "ALZA.CZ A.S." = "Alza.cz a.s." = "Alza cz";
  "Kofola a.s." and "Kofola ČeskoSlovensko a.s." stay apart.
- **the legal form does not tell subjects apart**: ARES wants every word, so a name typed with
  its legal form is searched without it too, and that search decides - "HARMONIE PLUS,
  s.r.o." alone never fetches the namesake "HARMONIE PLUS".
- **complete before deciding**: ARES lists hits in ascending IČO order, so a second holder of
  the name can sit on any page; the rest is paged (200 a page, up to the 1 000 ARES allows).
  A set that cannot be completed accepts nothing.
- **punctuation**: ARES keeps ``-``, ``&`` and quotes inside a word ("Frýdek-Místek"), so a
  name carrying them is also searched with them written the other ways, and the union decides.
- **warnings that stop an accept**: the one holder carries another legal form than the one
  typed ("EG.D, a.s." is the former name of EG.D Holding; EG.D, s.r.o. is another company),
  or the typed name is "v likvidaci" and the holder's is not.

Left out on purpose, as candidates-only steps there (word dropout, former names from the
commercial register) or as a data file this tool does not load (ČSÚ's open data of recently
founded and dissolved subjects). A dissolved namesake is therefore invisible here, as it is
to ARES's search.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover
    from core.sources.ares import AresHit, AresSource, NameSearch

#: The first page of a search; enough for the usual single exact hit.
FIRST_PAGE: Final[int] = 10
#: The page used to complete a hit set.
PAGE: Final[int] = 200
#: ARES lists no more than this many hits for one name.
MAX_HITS: Final[int] = 1000
#: Searches one name may cost before the set counts as incomplete (and nothing is accepted).
MAX_CALLS: Final[int] = 12
#: Namesakes named in a note.
NAMED: Final[int] = 5

# Legal-form suffixes as they appear at the end of a Czech business name once diacritics and
# punctuation are gone: "s.r.o." and "s. r. o." both arrive as "s r o". Glued spellings ("sro",
# "as", "ks") are absent on purpose: they collide with real words (the town Aš).
_LEGAL_FORM_SUFFIXES: Final[tuple[str, ...]] = (
    "akciova spolecnost",
    "spolecnost s rucenim omezenym",
    "verejna obchodni spolecnost",
    "komanditni spolecnost",
    "statni podnik",
    "zapsany spolek",
    "zapsany ustav",
    "obecne prospesna spolecnost",
    "prispevkova organizace",
    "bytove druzstvo",
    "vyrobni druzstvo",
    "spotrebni druzstvo",
    "zemedelske druzstvo",
    "druzstvo",
    "spol s r o",
    "s r o",
    "a s",
    "v o s",
    "k s",
    "s p",
    "z s",
    "z u",
    "o p s",
)
_SUFFIX_ALTERNATIVES: Final[str] = "|".join(
    suffix.replace(" ", r"\s+") for suffix in _LEGAL_FORM_SUFFIXES
)
_SUFFIX_PATTERN: Final[re.Pattern[str]] = re.compile(rf"(?:\s+(?:{_SUFFIX_ALTERNATIVES}))+$")
#: Every stripped suffix by the legal form it names: "a. s." and "akciová společnost" are one.
_SUFFIX_FAMILIES: Final[dict[str, str]] = {
    "akciova spolecnost": "as",
    "a s": "as",
    "spolecnost s rucenim omezenym": "sro",
    "spol s r o": "sro",
    "s r o": "sro",
    "verejna obchodni spolecnost": "vos",
    "v o s": "vos",
    "komanditni spolecnost": "ks",
    "k s": "ks",
    "statni podnik": "sp",
    "s p": "sp",
    "zapsany spolek": "zs",
    "z s": "zs",
    "zapsany ustav": "zu",
    "z u": "zu",
    "obecne prospesna spolecnost": "ops",
    "o p s": "ops",
    "prispevkova organizace": "po",
    "bytove druzstvo": "druzstvo",
    "vyrobni druzstvo": "druzstvo",
    "spotrebni druzstvo": "druzstvo",
    "zemedelske druzstvo": "druzstvo",
    "druzstvo": "druzstvo",
}
# A state, not a name: "Sberbank CZ, a.s. v likvidaci" is still Sberbank CZ.
_STATUS_TAG_KEY: Final[re.Pattern[str]] = re.compile(
    r"\s+v\s+(?:likvidaci|konkursu|upadku|insolvenci)$"
)
# The designation the law puts into a venture capital entity's name, wherever it stands.
_DESIGNATION_KEY: Final[re.Pattern[str]] = re.compile(
    r"(?:^|\s)osoba\s+rizikoveho\s+kapitalu(?=\s|$)"
)
# "Hrubý a spol., s.r.o.": "a spol." (and partners) is part of the name, only "s.r.o." after the
# comma is its legal form.
_A_SPOL_SRO: Final[re.Pattern[str]] = re.compile(
    r"(\sa\s+spol\.?)\s*,\s*s\.?\s*r\.?\s*o\.?\s*$", re.IGNORECASE
)
# The look-behind lets a match start only where a run of separators starts (else O(n²)).
_STATUS_TAG_DISPLAY: Final[re.Pattern[str]] = re.compile(
    r"(?<![\s,\-–])[\s,\-–]*[\"„]?\s*v\s+(?:likvidaci|konkursu|úpadku|insolvenci)"
    r"\s*[\"“]?\s*$",
    re.IGNORECASE,
)
_NON_ALNUM: Final[re.Pattern[str]] = re.compile(r"[^0-9a-z]+")
# Legal forms as register names spell them, stripped with case and diacritics kept (the name
# searched without its form must stay readable; normalize() is the comparison key).
_LEGAL_FORM_DISPLAY: Final[re.Pattern[str]] = re.compile(
    r"(?<![,\s])[,\s]+(?:a\.\s?s\.|akciová společnost|spol\.\s?s\s?r\.\s?o\.|s\.\s?r\.\s?o\.|"
    r"společnost s ručením omezeným|v\.\s?o\.\s?s\.|veřejná obchodní společnost|"
    r"k\.\s?s\.|komanditní společnost|s\.\s?p\.|státní podnik|z\.\s?s\.|zapsaný spolek|"
    r"z\.\s?ú\.|zapsaný ústav|o\.\s?p\.\s?s\.|obecně prospěšná společnost|"
    r"příspěvková organizace|(?:(?:bytové|výrobní|spotřební|zemědělské)\s+)?družstvo)\s*$",
    re.IGNORECASE,
)
_SE_SUFFIX: Final[re.Pattern[str]] = re.compile(r"[,\s]+SE\s*$")  # Societas Europaea, capitals
#: How many stacked suffixes one name may shed; a bound, since each pass scans the name.
MAX_SUFFIX_PASSES: Final[int] = 6
#: Quotation marks a name may be typed with; the register rarely has them.
_QUOTES: Final[re.Pattern[str]] = re.compile("[\"'„“”‚‘’«»]")
_SPACED_JOINER: Final[re.Pattern[str]] = re.compile(r"\s*([-&])\s*")


# -- the comparison key ----------------------------------------------------------------


def strip_diacritics(text: str) -> str:
    """``"Škoda"`` -> ``"Skoda"``; everything else is left alone."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _key_text(text: str) -> str:
    """Lower case, no diacritics, every non-alphanumeric run one space."""
    return _NON_ALNUM.sub(" ", strip_diacritics(text).lower()).strip()


def _drop_designation(key: str) -> str:
    """A key text without "osoba rizikového kapitálu", wherever it stood."""
    return " ".join(_DESIGNATION_KEY.sub(" ", key).split())


def normalize(name: str | None) -> str:
    """The comparison key of a Czech business name; ``""`` when nothing is left."""
    if not name:
        return ""
    text = _SE_SUFFIX.sub("", " ".join(str(name).split()))
    bare = _STATUS_TAG_DISPLAY.sub("", text).strip(" ,")
    if _A_SPOL_SRO.search(bare):
        text = _A_SPOL_SRO.sub(r"\1", bare)
    text = _drop_designation(_key_text(text))
    previous = None
    for _ in range(MAX_SUFFIX_PASSES):  # "Alza a.s. s.r.o." -> "alza"
        if not text or text == previous:
            break
        previous = text
        text = _STATUS_TAG_KEY.sub("", text).strip()
        text = _SUFFIX_PATTERN.sub("", text).strip()
    return text


def strip_legal_form(name: str) -> str:
    """``"Kofola a.s."`` -> ``"Kofola"``, case and diacritics kept; the name when nothing goes.

    A trailing status tag goes too, and every spelling :func:`normalize` strips ("a.s" without
    its last dot, "akciova spolecnost" without diacritics), so the bare name can be searched.
    """
    original = " ".join(str(name or "").split())
    text = original
    previous = None
    for _ in range(MAX_SUFFIX_PASSES):
        if not text or text == previous:
            break
        previous = text
        text = _STATUS_TAG_DISPLAY.sub("", text).strip(" ,")
        text = _SE_SUFFIX.sub("", _LEGAL_FORM_DISPLAY.sub("", text)).strip(" ,")
    return _strip_key_suffix(text) or original


def _strip_key_suffix(text: str) -> str:
    """``text`` cut where :func:`normalize` stops keeping words; ``text`` when it keeps all."""
    words = _key_text(text).split()
    kept = normalize(text).split()
    if not kept or len(kept) >= len(words) or words[: len(kept)] != kept:
        return text
    spans = _key_spans(text)
    if len(spans) != len(words):  # a character normalize() splits: leave it
        return text
    return text[: spans[len(kept)][0]].rstrip(" ,-–") or text


def _key_spans(text: str) -> list[tuple[int, int]]:
    """``[start, end)`` in ``text`` of every word :func:`normalize` sees, in order."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    end = 0
    for index, char in enumerate(text):
        folded = strip_diacritics(char).lower()
        if not folded:  # a combining mark belongs to the letter before it
            continue
        if _NON_ALNUM.sub("", folded) != folded:
            if start is not None:
                spans.append((start, end))
                start = None
            continue
        if start is None:
            start = index
        end = index + 1
    if start is not None:
        spans.append((start, end))
    return spans


def legal_form_family(name: str | None) -> str:
    """Which legal form a name ends in (``"as"``, ``"sro"``, ``"se"``...); ``""`` for none."""
    text = " ".join(str(name or "").split())
    text = _STATUS_TAG_DISPLAY.sub("", text).strip(" ,")
    if _SE_SUFFIX.search(text):
        return "se"
    key = _STATUS_TAG_KEY.sub("", _drop_designation(_key_text(text))).strip()
    for suffix in _LEGAL_FORM_SUFFIXES:
        if key == suffix or key.endswith(" " + suffix):
            return _SUFFIX_FAMILIES[suffix]
    return ""


def has_status_tag(text: str | None) -> bool:
    """Whether ``text`` ends in a status tag: "v likvidaci", "v konkursu", ..."""
    return bool(_STATUS_TAG_DISPLAY.search(" ".join(str(text or "").split())))


def punctuation_variants(name: str) -> list[str]:
    """The name with its ``-``, ``&`` and quotes written the other ways ARES tokenises.

    Quotes removed; spaces around ``-`` and ``&`` closed ("Frýdek - Místek" -> "Frýdek-Místek");
    ``-`` and ``&`` read as a space. Only variants that differ from the name, once each.
    """
    variants: list[str] = []
    unquoted = " ".join(_QUOTES.sub(" ", name).split())
    for variant in (
        unquoted,
        _SPACED_JOINER.sub(r"\1", unquoted),
        " ".join(re.sub(r"[-&]", " ", unquoted).split()),
    ):
        if variant and variant.lower() != name.lower() and variant not in variants:
            variants.append(variant)
    return variants


# -- the decision ----------------------------------------------------------------------

ACCEPTED: Final[str] = "accepted"
AMBIGUOUS: Final[str] = "ambiguous"
LEGAL_FORM_DIFFERS: Final[str] = "legal_form_differs"
STATUS_TAG_DIFFERS: Final[str] = "status_tag_differs"
INCOMPLETE: Final[str] = "incomplete"
TOO_MANY: Final[str] = "too_many"
INACTIVE: Final[str] = "inactive"
NOT_EXACT: Final[str] = "not_exact"
NOTHING: Final[str] = "nothing"


@dataclass(frozen=True, slots=True)
class CzechNameMatch:
    """What ARES's name search found for one name, and whether a subject was taken.

    Attributes:
        query: The name as typed (whitespace collapsed).
        outcome: :data:`ACCEPTED`, or why not (:data:`AMBIGUOUS`, :data:`LEGAL_FORM_DIFFERS`,
            :data:`STATUS_TAG_DIFFERS`, :data:`INCOMPLETE`, :data:`TOO_MANY`,
            :data:`INACTIVE`, :data:`NOT_EXACT`, :data:`NOTHING`).
        hit: The subject taken; only with :data:`ACCEPTED`.
        holders: The subjects whose name is the query's (the one taken, or the namesakes).
        others: Hits with another name, when no subject has the query's.
        total: Every hit ARES counted for the name.
        calls: Searches made.
        without_ico: Hits without an IČO of their own (branches), never candidates.
    """

    query: str
    outcome: str
    hit: AresHit | None = None
    holders: tuple[AresHit, ...] = ()
    others: tuple[AresHit, ...] = ()
    total: int = 0
    calls: int = 0
    without_ico: int = 0

    @property
    def ico(self) -> str | None:
        return self.hit.ico if self.hit is not None else None

    def note(self) -> str:
        """What the lookup says about it, in Czech, for the page's notes."""
        name = self.query
        if self.hit is not None:
            return (
                f"emitent dohledán v ARES podle názvu, ne podle IČO: {self.hit.label()} – jediný "
                "aktivní český subjekt s tímto názvem; ověřte, že jde o správný subjekt"
            )
        listed = "; ".join(item.label() for item in self.holders[:NAMED])
        if self.outcome == AMBIGUOUS:
            return (
                f"název „{name}“ nese v ARES {_active_subjects(len(self.holders))}, emitenta "
                f"nelze určit – zadejte IČO toho správného: {listed}"
            )
        if self.outcome == LEGAL_FORM_DIFFERS:
            return (
                f"jediný aktivní český subjekt s názvem „{name}“ má jinou právní formu, než je "
                f"zadaná: {listed} – nepřevzat; je-li to on, zadejte jeho IČO"
            )
        if self.outcome == STATUS_TAG_DIFFERS:
            return (
                f"zadaný název je „v likvidaci“ (nebo v jiném stavu), jediný aktivní subjekt "
                f"s tímto názvem ne: {listed} – nepřevzat; je-li to on, zadejte jeho IČO"
            )
        if self.outcome == INCOMPLETE:
            found = f" (mezi nalezenými: {listed})" if listed else ""
            return (
                f"v ARES nelze ověřit, že název „{name}“ nese jediný subjekt – výsledků je příliš "
                f"mnoho{found}; zadejte IČO"
            )
        if self.outcome == TOO_MANY:
            return (
                f"název „{name}“ odpovídá v ARES {_thousands(self.total)} subjektům (ARES jich "
                "vrací nejvýše 1 000) – upřesněte ho, nebo zadejte IČO"
            )
        if self.outcome == INACTIVE:
            return (
                f"název „{name}“ nese v ARES jen zaniklý subjekt: {listed} – kódy z RES "
                "nepřevzaty; zadejte IČO"
            )
        if self.outcome == NOT_EXACT:
            if self.others and self.total <= NAMED:
                similar = "; ".join(item.label() for item in self.others)
                return (
                    f"v ARES není subjekt s přesně tímto názvem; podobné: {similar} – je-li to "
                    "jeden z nich, zadejte jeho IČO"
                )
            return (
                f"v ARES není subjekt s přesně tímto názvem ({_thousands(self.total)} "
                "podobných subjektů) – zadejte celý oficiální název, nebo IČO"
            )
        if self.without_ico:
            return (
                f"ARES zná název „{name}“ jen u záznamu bez vlastního IČO (např. odštěpný "
                "závod) – zadejte IČO"
            )
        return f"ani ARES (české subjekty) nezná subjekt s názvem „{name}“"


def find_czech_subject(source: AresSource, name: str) -> CzechNameMatch:
    """Search ARES for ``name`` and take the subject only when the rule allows it.

    Raises the source's :class:`~core.sources.base.SourceError` when ARES cannot be asked: an
    outage must never read as "no such subject".
    """
    query = " ".join(name.split())
    wanted = normalize(query)
    if not wanted:
        return CzechNameMatch(query=query, outcome=NOTHING)
    searcher = _Searcher(source)
    first = searcher.first(query)
    if first.too_many:
        return CzechNameMatch(query=query, outcome=TOO_MANY, total=first.total, calls=1)

    decisive_query, decisive, total = query, list(first.hits), first.total
    complete = True
    stripped = strip_legal_form(query)
    has_form = stripped.lower() != query.lower()
    if has_form:
        # ARES wants every word, so the typed form never fetches a namesake with another legal
        # form or none - which the rule counts as a holder. The bare name's hits decide.
        second = searcher.first(stripped)
        if second.too_many:
            complete = False
        elif any(hit.ico for hit in second.hits):
            decisive_query = stripped
            decisive = _merge(list(second.hits), decisive)
            total = max(second.total, total)
    if complete:
        decisive, complete = searcher.complete(decisive_query, decisive, total)
    if complete:
        for variant in punctuation_variants(stripped if has_form else query):
            found = searcher.first(variant)
            if found.too_many:
                complete = False
                break
            if not any(hit.ico for hit in found.hits):
                continue
            hits, complete = searcher.complete(variant, list(found.hits), found.total)
            decisive = _merge(decisive, hits)
            total = max(total, found.total)
            if not complete:
                break

    usable = [hit for hit in decisive if hit.ico]
    exact = [hit for hit in usable if normalize(hit.name) == wanted]
    active = [hit for hit in exact if hit.active]
    common = {
        "query": query,
        "total": max(total, len(usable)),
        "calls": searcher.calls,
        "without_ico": len(decisive) - len(usable),
    }
    if len(active) == 1:
        holder = active[0]
        if not complete:
            return CzechNameMatch(outcome=INCOMPLETE, holders=(holder,), **common)
        form = legal_form_family(query)
        if form and legal_form_family(holder.name) != form:
            return CzechNameMatch(outcome=LEGAL_FORM_DIFFERS, holders=(holder,), **common)
        if has_status_tag(query) and not has_status_tag(holder.name):
            return CzechNameMatch(outcome=STATUS_TAG_DIFFERS, holders=(holder,), **common)
        return CzechNameMatch(outcome=ACCEPTED, hit=holder, holders=(holder,), **common)
    if active:
        return CzechNameMatch(outcome=AMBIGUOUS, holders=tuple(active), **common)
    if not complete:
        return CzechNameMatch(outcome=INCOMPLETE, **common)
    if exact:
        return CzechNameMatch(outcome=INACTIVE, holders=tuple(exact), **common)
    if usable:
        return CzechNameMatch(outcome=NOT_EXACT, others=tuple(usable[:NAMED]), **common)
    return CzechNameMatch(outcome=NOTHING, **common)


class _Searcher:
    """The searches of one name, counted against :data:`MAX_CALLS`."""

    def __init__(self, source: AresSource) -> None:
        self._source = source
        self.calls = 0

    def first(self, query: str) -> NameSearch:
        """The first page of ``query``; past the budget, an empty page that is "too many"."""
        return self._page(query, FIRST_PAGE, 0)

    def complete(self, query: str, hits: list[AresHit], total: int) -> tuple[list[AresHit], bool]:
        """Every hit of ``query``; ``False`` when the set could not be completed."""
        if total <= len(hits):
            return hits, True
        if total > MAX_HITS:
            return hits, False
        start = 0
        while start < total:
            page = self._page(query, PAGE, start)
            if page.too_many:
                return hits, False
            hits = _merge(hits, list(page.hits))
            start += PAGE
        return hits, True

    def _page(self, query: str, limit: int, start: int) -> NameSearch:
        if self.calls >= MAX_CALLS:
            from core.sources.ares import NameSearch

            return NameSearch(query=query, total=MAX_HITS + 1, hits=(), too_many=True)
        self.calls += 1
        return self._source.search_by_name(query, limit=limit, start=start)


def _merge(first: list[AresHit], more: list[AresHit]) -> list[AresHit]:
    """Hits of several searches, the first occurrence of an IČO kept, order kept."""
    seen: set[object] = set()
    merged: list[AresHit] = []
    for hit in [*first, *more]:
        key: object = hit.ico or ("bez IČO", hit.name, hit.legal_form, hit.town)
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)
    return merged


def _thousands(number: int) -> str:
    """``2825`` -> ``2 825`` with a no-break space, as Czech writes it."""
    return f"{number:,}".replace(",", " ")


def _active_subjects(count: int) -> str:
    """``2 aktivní subjekty``, ``5 aktivních subjektů``: the Czech plural after a number."""
    if 2 <= count <= 4:
        return f"{count} aktivní subjekty"
    return f"{_thousands(count)} aktivních subjektů"


__all__ = [
    "ACCEPTED",
    "AMBIGUOUS",
    "INACTIVE",
    "INCOMPLETE",
    "LEGAL_FORM_DIFFERS",
    "NOTHING",
    "NOT_EXACT",
    "STATUS_TAG_DIFFERS",
    "TOO_MANY",
    "CzechNameMatch",
    "find_czech_subject",
    "has_status_tag",
    "legal_form_family",
    "normalize",
    "punctuation_variants",
    "strip_legal_form",
]
