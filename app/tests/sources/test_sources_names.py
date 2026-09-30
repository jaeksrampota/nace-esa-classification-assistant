"""Issuer-name comparison: folding, stripping and equal legal forms."""

from __future__ import annotations

import pytest

from core.sources.names import fold, legal_form_key, strip_legal_form


@pytest.mark.parametrize(
    ("short", "long"),
    [
        ("OMV AG", "OMV AKTIENGESELLSCHAFT"),
        ("Kommuninvest i Sverige AB", "Kommuninvest i Sverige Aktiebolag"),
        ("Enel S.p.A.", "Enel Società per azioni"),
        ("Shell plc", "Shell Public Limited Company"),
        ("Heineken N.V.", "Heineken Naamloze Vennootschap"),
        ("Foo GmbH", "Foo Gesellschaft mit beschränkter Haftung"),
    ],
)
def test_a_short_and_a_long_legal_form_are_one(short: str, long: str) -> None:
    assert legal_form_key(short) == legal_form_key(long)


def test_different_legal_forms_differ() -> None:
    assert legal_form_key("OMV AG") != legal_form_key("OMV - S.P.A.")
    assert legal_form_key("OMV AG") != legal_form_key("ÖMV AB")


def test_no_legal_form_has_no_key() -> None:
    assert legal_form_key("OMV") is None


def test_strip_and_fold() -> None:
    assert strip_legal_form("adidas AG") == "adidas"
    assert fold("ÖMV AB") == "omvab"


@pytest.mark.parametrize(
    ("typed", "known", "agree"),
    [
        ("Deutsche Bank AG", "DEUTSCHE BANK AKTIENGESELLSCHAFT", True),
        ("adidas", "adidas AG", True),
        ("BMW", "Bayerische Motoren Werke Aktiengesellschaft", True),
        ("Heineken", "Heineken N.V.", True),
        ("Siemens", "DEUTSCHE BANK AKTIENGESELLSCHAFT", False),
        ("Erste Bank", "DEUTSCHE BANK AKTIENGESELLSCHAFT", False),
        ("Bank AG", "DEUTSCHE BANK AKTIENGESELLSCHAFT", True),
    ],
)
def test_names_agree(typed: str, known: str, agree: bool) -> None:
    from core.sources.names import names_agree

    assert names_agree(typed, known) is agree


@pytest.mark.parametrize(
    ("typed", "known", "shared"),
    [
        ("Kongsberg Gruppen ASA", "Kongsberg Gruppen", 2),
        ("Kongsberg Gruppen ASA", "Kongsberg", 1),  # the town: ranked below the company
        ("BMW Finance N.V.", "BMW Bank", 1),  # "finance" and "bank" tell nothing apart
        ("Nordkap Funding B.V.", "Nordkapp", 0),
    ],
)
def test_shared_words_rank_the_wikipedia_hits(typed: str, known: str, shared: int) -> None:
    from core.sources.names import shared_words

    assert shared_words(typed, known) == shared


def test_an_italian_legal_form_with_an_apostrophe() -> None:
    assert legal_form_key("Cassa Depositi e Prestiti S.p.A.") == legal_form_key(
        "CASSA DEPOSITI E PRESTITI SOCIETA' PER AZIONI"
    )
