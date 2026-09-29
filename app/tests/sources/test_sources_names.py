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
