"""A Czech subject by its name: the comparison key and the rule that takes one or none.

The search runs through the real ``AresSource`` over a mock transport (``conftest.py``), so the
paging, the legal-form search and the punctuation variants are exercised as ARES would answer
them; nothing touches the network. The names are real Czech companies, as ARES listed them on
5 Oct 2026, and the traps are the RES a OR tool's golden cases (``docs/MATCHING.md`` there).
"""

from __future__ import annotations

import pytest

from config.settings import Settings
from core.sources import czech_names
from core.sources.ares import AresSource
from core.sources.base import SourceUnavailableError
from core.sources.czech_names import (
    ACCEPTED,
    AMBIGUOUS,
    INACTIVE,
    INCOMPLETE,
    LEGAL_FORM_DIFFERS,
    NOT_EXACT,
    NOTHING,
    STATUS_TAG_DIFFERS,
    TOO_MANY,
    find_czech_subject,
    has_status_tag,
    legal_form_family,
    normalize,
    punctuation_variants,
    strip_legal_form,
)
from tests.sources.conftest import ares_client, search_hit

KB = search_hit("45317054", "Komerční banka, a.s.")
#: Trade-union branches whose names contain the bank's, listed before it in IČO order.
UNIONS = [
    search_hit(
        f"6573{index:04d}",
        f"Základní organizace odborového svazu Komerční banka, a.s. - pobočka {index}",
        legal_form="707",
    )
    for index in range(20)
]


def source(search: dict, *, searched: list | None = None) -> AresSource:
    settings = Settings(ares_min_interval_seconds=0.0, ares_max_attempts=1)  # type: ignore[call-arg]
    client = ares_client(search=search, searched=searched)
    return AresSource(settings, client=client, sleep=lambda _: None)


def find(name: str, search: dict, *, searched: list | None = None):
    return find_czech_subject(source(search, searched=searched), name)


class TestKey:
    @pytest.mark.parametrize(
        ("name", "key"),
        [
            ("ALZA.CZ A.S.", "alza cz"),
            ("Alza.cz a.s.", "alza cz"),
            ("Alza cz", "alza cz"),
            ("ČEZ, a. s.", "cez"),
            ("Komerční banka, a.s.", "komercni banka"),
            ("Artesa, spořitelní družstvo", "artesa sporitelni"),
            ("Kofola ČeskoSlovensko a.s.", "kofola ceskoslovensko"),
            ("Sberbank CZ, a.s. v likvidaci", "sberbank cz"),
            ("Hrubý a spol., s.r.o.", "hruby a spol"),
            ("T R I M E D I C A, spol. s r.o.", "t r i m e d i c a"),
            ("iEnergy capital s.r.o., osoba rizikového kapitálu", "ienergy capital"),
            ("Kofola akciova spolecnost", "kofola"),
            ("Allianz SE", "allianz"),
            ("středisko Smrčina Aš", "stredisko smrcina as"),
            ("", ""),
        ],
    )
    def test_normalize(self, name: str, key: str) -> None:
        assert normalize(name) == key

    def test_the_legal_form_does_not_tell_kofola_apart_but_the_words_do(self) -> None:
        assert normalize("Kofola a.s.") == normalize("Kofola s.r.o.") == "kofola"
        assert normalize("Kofola a.s.") != normalize("Kofola ČeskoSlovensko a.s.")

    @pytest.mark.parametrize(
        ("name", "bare"),
        [
            ("Kofola a.s.", "Kofola"),
            ("ČEZ, a. s.", "ČEZ"),
            ("HARMONIE PLUS, s.r.o", "HARMONIE PLUS"),
            ("ADRA, spol. s r.o. - v likvidaci", "ADRA"),
            ("Komerční banka", "Komerční banka"),
        ],
    )
    def test_strip_legal_form_keeps_the_readable_name(self, name: str, bare: str) -> None:
        assert strip_legal_form(name) == bare

    @pytest.mark.parametrize(
        ("name", "family"),
        [
            ("EG.D, a.s.", "as"),
            ("EG.D, akciová společnost", "as"),
            ("EG.D, s.r.o.", "sro"),
            ("Artesa, spořitelní družstvo", "druzstvo"),
            ("Allianz SE", "se"),
            ("Komerční banka", ""),
        ],
    )
    def test_legal_form_family(self, name: str, family: str) -> None:
        assert legal_form_family(name) == family

    def test_status_tag(self) -> None:
        assert has_status_tag("Sberbank CZ, a.s. v likvidaci")
        assert not has_status_tag("Sberbank CZ, a.s.")

    def test_punctuation_variants(self) -> None:
        assert punctuation_variants("J & T Banka") == ["J&T Banka", "J T Banka"]
        assert punctuation_variants("Komerční banka") == []


class TestTaken:
    def test_the_one_active_holder_among_namesakes_is_taken_after_paging(self) -> None:
        """28 hits, KB the 21st: the rest is paged before the name is judged unique."""
        searched: list[tuple[str, int, int]] = []
        match = find("Komerční banka", {"Komerční banka": [*UNIONS, KB]}, searched=searched)
        assert match.outcome == ACCEPTED and match.ico == "45317054"
        assert searched == [("Komerční banka", 10, 0), ("Komerční banka", 200, 0)]
        assert "emitent dohledán v ARES podle názvu" in match.note()
        assert "Komerční banka, a.s. (IČO 45317054, Praha)" in match.note()

    def test_a_typed_legal_form_is_searched_away_and_the_bare_name_decides(self) -> None:
        searched: list[tuple[str, int, int]] = []
        match = find(
            "Komerční banka, a.s.",
            {"Komerční banka, a.s.": [KB], "Komerční banka": [KB]},
            searched=searched,
        )
        assert match.outcome == ACCEPTED
        assert [name for name, _, _ in searched] == ["Komerční banka, a.s.", "Komerční banka"]

    def test_a_branch_of_a_foreign_bank_is_a_subject_with_its_own_ico(self) -> None:
        ing = search_hit("49279866", "ING Bank N.V.", legal_form="421")
        match = find("ING Bank N.V.", {"ING Bank N.V.": [ing]})
        assert match.outcome == ACCEPTED and match.ico == "49279866"

    def test_punctuation_typed_otherwise_still_finds_the_name(self) -> None:
        jt = search_hit("47115378", "J&T BANKA, a.s.")
        match = find("J & T Banka a.s.", {"J&T Banka": [jt]})
        assert match.outcome == ACCEPTED and match.ico == "47115378"


class TestNotTaken:
    def test_two_holders_once_the_legal_form_is_set_aside(self) -> None:
        """F01 of the RES a OR test report: the typed form must not narrow the hit set."""
        sro = search_hit("25119273", "HARMONIE PLUS, s.r.o.", legal_form="112")
        bare = search_hit("48383201", "HARMONIE PLUS", legal_form="706", town="Rokycany")
        match = find(
            "HARMONIE PLUS, s.r.o.",
            {"HARMONIE PLUS, s.r.o.": [sro], "HARMONIE PLUS": [sro, bare]},
        )
        assert match.outcome == AMBIGUOUS and match.ico is None
        note = match.note()
        assert "nese v ARES 2 aktivní subjekty" in note
        assert "(IČO 25119273, Praha)" in note and "(IČO 48383201, Rokycany)" in note

    def test_a_second_holder_on_a_later_page_is_found(self) -> None:
        late = search_hit("99999999", "KOMERČNÍ BANKA", legal_form="706", town="Brno")
        match = find("Komerční banka", {"Komerční banka": [KB, *UNIONS, late]})
        assert match.outcome == AMBIGUOUS
        assert "nese v ARES 2 aktivní subjekty" in match.note()

    def test_the_only_holder_of_another_legal_form_is_not_taken(self) -> None:
        """EG.D, a.s. is EG.D Holding's former name; EG.D, s.r.o. is another company."""
        egd = search_hit("21055050", "EG.D, s.r.o.", legal_form="112", town="Brno")
        match = find("EG.D, a.s.", {"EG.D": [egd]})
        assert match.outcome == LEGAL_FORM_DIFFERS and match.ico is None
        assert "má jinou právní formu, než je zadaná" in match.note()
        assert "EG.D, s.r.o. (IČO 21055050, Brno)" in match.note()

    def test_a_name_in_liquidation_is_not_the_active_namesake(self) -> None:
        holder = search_hit("12345679", "Sberbank CZ, a.s.")
        match = find(
            "Sberbank CZ, a.s. v likvidaci",
            {"Sberbank CZ, a.s. v likvidaci": [], "Sberbank CZ": [holder]},
        )
        assert match.outcome == STATUS_TAG_DIFFERS
        assert "v likvidaci" in match.note()

    def test_too_many_hits_asks_to_refine(self) -> None:
        match = find("Stavby", {"Stavby": 2825})
        assert match.outcome == TOO_MANY and match.total == 2825
        assert "odpovídá v ARES 2 825 subjektům" in match.note()

    def test_a_bare_name_with_too_many_hits_accepts_nothing(self) -> None:
        holder = search_hit("11111111", "Stavby Praha a.s.")
        match = find("Stavby Praha a.s.", {"Stavby Praha a.s.": [holder], "Stavby Praha": 1500})
        assert match.outcome == INCOMPLETE and match.ico is None
        assert "nelze ověřit" in match.note()

    def test_a_set_the_budget_cannot_complete_accepts_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(czech_names, "MAX_CALLS", 1)
        match = find("Komerční banka", {"Komerční banka": [*UNIONS, KB]})
        assert match.outcome == INCOMPLETE and match.ico is None

    def test_only_a_dissolved_holder(self) -> None:
        gone = search_hit("14893649", "Max banka a.s.", dissolved="2024-10-18")
        match = find("Max banka", {"Max banka": [gone]})
        assert match.outcome == INACTIVE
        assert "jen zaniklý subjekt" in match.note()

    def test_a_holder_no_register_reports_active_is_not_taken(self) -> None:
        gone = search_hit(
            "14893649", "Max banka a.s.", states={"stavZdrojeRes": "ZANIKLY", "stavZdrojeVr": ""}
        )
        assert find("Max banka", {"Max banka": [gone]}).outcome == INACTIVE

    def test_similar_names_are_listed_when_there_are_few(self) -> None:
        kooperativa = search_hit("47116617", "Kooperativa pojišťovna, a.s., Vienna Insurance Group")
        match = find("Kooperativa pojišťovna", {"Kooperativa pojišťovna": [kooperativa]})
        assert match.outcome == NOT_EXACT
        assert "podobné: Kooperativa pojišťovna, a.s., Vienna Insurance Group" in match.note()

    def test_many_similar_names_are_counted_not_listed(self) -> None:
        match = find("banka", {"banka": UNIONS})
        assert match.outcome == NOT_EXACT
        assert "(20 podobných subjektů)" in match.note()

    def test_a_hit_without_an_ico_is_never_a_candidate(self) -> None:
        branch = search_hit(None, "Tracasa s.r.o.", legal_form="112")
        match = find("Tracasa", {"Tracasa": [branch]})
        assert match.outcome == NOTHING and match.without_ico == 1
        assert "bez vlastního IČO" in match.note()

    def test_nothing_found(self) -> None:
        match = find("EGAP", {})
        assert match.outcome == NOTHING
        assert match.note() == "ani ARES (české subjekty) nezná subjekt s názvem „EGAP“"


def test_an_outage_is_raised_never_read_as_nothing() -> None:
    with pytest.raises(SourceUnavailableError):
        find("Komerční banka", {"Komerční banka": 503})
