"""Issuer-name comparison shared by the registers that match on a name (GLEIF, Wikidata)."""

from __future__ import annotations

import re
import unicodedata
from typing import Final

#: Legal-form suffixes stripped for the looser comparison ("Kommuninvest i Sverige AB"
#: -> "Kommuninvest i Sverige"). Deliberately a list of *legal forms*, not of words.
LEGAL_FORM_RE: Final[re.Pattern[str]] = re.compile(
    r"""(?:[\s,]+(?:AG|SE|SA|S\.A\.|SpA|S\.p\.A\.|N\.?V\.?|B\.?V\.?|AB|ASA|AS|A/S|Oyj|plc|
    Ltd\.?|Limited|LLC|Inc\.?|Incorporated|Corp\.?|Corporation|GmbH|KGaA|S\.à\s?r\.l\.|Sàrl|
    S\.A\.S\.|SAS|Aktiengesellschaft|Aktiebolag|Aktieselskab|Realkreditaktieselskab|
    Societ[àa]'?\s+per\s+azioni|Naamloze\s+Vennootschap|Besloten\s+Vennootschap|
    Gesellschaft\s+mit\s+beschr[äa]nkter\s+Haftung|Public\s+Limited\s+Company|
    Soci[ée]t[ée]\s+anonyme))+\s*$""",
    re.IGNORECASE | re.VERBOSE,
)
_NON_ALNUM_RE: Final[re.Pattern[str]] = re.compile(r"[^0-9a-z]+")

#: Long legal forms -> their abbreviation, as folded text: "OMV AG" is "OMV Aktiengesellschaft".
_FORM_ALIASES: Final[dict[str, str]] = {
    "aktiengesellschaft": "ag",
    "aktiebolag": "ab",
    "aktieselskab": "as",
    "societaperazioni": "spa",
    "naamlozevennootschap": "nv",
    "beslotenvennootschap": "bv",
    "gesellschaftmitbeschrankterhaftung": "gmbh",
    "publiclimitedcompany": "plc",
    "limited": "ltd",
    "incorporated": "inc",
    "corporation": "corp",
    "societeanonyme": "sa",
}


def fold(name: str) -> str:
    """``"Assicurazioni Generali S.p.A."`` -> ``"assicurazionigeneralispa"``: the comparison key."""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _NON_ALNUM_RE.sub("", ascii_only.lower())


def strip_legal_form(name: str) -> str:
    """``"adidas AG"`` -> ``"adidas"``; a name with no legal-form suffix is returned as is."""
    return LEGAL_FORM_RE.sub("", name).strip()


def legal_form_key(name: str) -> str | None:
    """``"OMV AG"`` and ``"OMV Aktiengesellschaft"`` -> ``"omv|ag"``; ``None`` with no legal form."""
    found = LEGAL_FORM_RE.search(name)
    if found is None:
        return None
    form = fold(found.group(0))
    return f"{fold(name[: found.start()])}|{_FORM_ALIASES.get(form, form)}"


#: Words too common in issuer names to tell two issuers apart ("Erste Bank" vs "Deutsche Bank").
_GENERIC: Final[frozenset[str]] = frozenset(
    [
        "the",
        "of",
        "and",
        "und",
        "de",
        "la",
        "le",
        "bank",
        "banka",
        "group",
        "gruppe",
        "holding",
        "holdings",
        "international",
        "finance",
        "financial",
        "capital",
        "company",
        "fund",
        "funds",
        "trust",
    ]
)


def _words(name: str) -> list[str]:
    return [w for w in (fold(part) for part in strip_legal_form(name).split()) if w]


def names_agree(typed: str, *known: str | None) -> bool:
    """Could ``typed`` be one of the ``known`` names? Lenient: only a clear mismatch is False.

    Agrees on a shared distinctive word, on the typed name inside a known one ("Heineken" in
    "Heineken N.V."), or on initials ("BMW" = Bayerische Motoren Werke). A typed name of
    generic words only cannot be judged and agrees.
    """
    typed_words = {w for w in _words(typed) if w not in _GENERIC}
    if not typed_words:
        return True
    compact = fold(strip_legal_form(typed))
    for name in filter(None, known):
        words = _words(name)
        if typed_words & set(words) or compact in fold(name):
            return True
        if len(words) > 1 and compact == "".join(w[0] for w in words):
            return True
    return False


__all__ = ["LEGAL_FORM_RE", "fold", "legal_form_key", "names_agree", "strip_legal_form"]
