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
    Societ[àa]\s+per\s+azioni|Naamloze\s+Vennootschap|Besloten\s+Vennootschap|
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


__all__ = ["LEGAL_FORM_RE", "fold", "legal_form_key", "strip_legal_form"]
