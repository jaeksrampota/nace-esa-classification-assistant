"""ECB lists: parsing, the table, the fact line, and what the identity step does with it."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from config.settings import Settings
from core.classify.candidates import EsaCandidateFilter
from core.classify.hints import register_esa_families
from core.db import Database, DatabaseError
from core.sources.base import Provenance
from core.sources.ecb import EcbEntry, EcbRegister, parse_mfi, parse_sheet
from core.sources.gleif import LeiRecord
from core.sources.identity import IssuerIdentifier
from tests.classify.conftest import build_codebooks

DB_LEI = "7LTWFZYICNSX8D621K86"
VW_LEI = "529900NNUPAGGOMPXZ31"
ISIN = "DE0005140008"

MFI_TSV = (
    "RIAD_CODE\tLEI\tCOUNTRY_OF_REGISTRATION\tNAME\tBOX\tADDRESS\tPOSTAL\tCITY\tCATEGORY\t"
    "HEAD_COUNTRY_OF_REGISTRATION\tHEAD_NAME\tHEAD_RIAD_CODE\tHEAD_LEI\tREPORT\n"
    f"DE1\t{DB_LEI}\tDE\tDEUTSCHE BANK AG\t\t\t\tFrankfurt\tCredit Institution\t\t\t\t\tx\n"
    "LU1\t5299002SSK89DA5VPJ96\tLU\tAMUNDI FUNDS -- CASH EUR\t\t\t\tLux\tMoney Market Fund\t\t\t\t\tx\n"
    "PL1\t\tPL\tSome branch without LEI\t\t\t\tWarsaw\tCredit Institution\t\t\t\t\tx\n"
    "SE1\t213800HW2E5VN9FK1Y53\tSE\tNordea, filial i Sverige\t\t\t\tSthlm\tCredit Institution\t"
    "FI\tNordea Bank Abp\tFI1\t529900ODI3047E2LIV03\tx\n"
    "XX1\tABCDEFGHIJKLMNOPQRST\tXX\tOdd\t\t\t\t-\tSomething New\t\t\t\t\tx\n"
)

IF_ROWS = [
    ("List of investment funds", None, None),
    ("Country of Residence", "ID", "LEI", "Name", "Type of investment fund"),
    (
        "IE",
        "IE1",
        "549300QS4Q1IT6XCA514",
        "iShares Core MSCI World UCITS ETF",
        "Exchange traded fund (ETF)",
    ),
    ("LU", "LU1", None, "A fund without a LEI", "Investment fund other than an ETF"),
]


def database(tmp_path: Path) -> Database:
    return Database(f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}")


def register(tmp_path: Path) -> EcbRegister:
    reg = EcbRegister(database(tmp_path))
    reg.replace("MFI", parse_mfi(MFI_TSV, "2026-09-28"))
    reg.replace("IF", parse_sheet("IF", IF_ROWS, "2026-07"))
    return reg


class TestParsing:
    def test_mfi_rows_with_a_lei_and_a_known_category(self) -> None:
        entries = list(parse_mfi(MFI_TSV, "2026-09-28"))
        assert [e.code for e in entries] == [
            "ECB_MFI:CREDIT_INSTITUTION",
            "ECB_MFI:MONEY_MARKET_FUND",
            "ECB_MFI:CREDIT_INSTITUTION",
        ]
        assert entries[0].lei == DB_LEI and entries[0].country == "DE"

    def test_a_branch_carries_its_head_office(self) -> None:
        branch = list(parse_mfi(MFI_TSV, "2026-09-28"))[2]
        assert branch.head_lei == "529900ODI3047E2LIV03"
        assert "Pobočka, ústředí LEI 529900ODI3047E2LIV03" in branch.facts()[0]

    def test_a_sheet_finds_its_header_and_the_sub_type(self) -> None:
        (fund,) = parse_sheet("IF", IF_ROWS, "2026-07")
        assert fund.code == "ECB_IF"
        assert fund.subtype == "Exchange traded fund (ETF)"
        assert fund.as_of == "2026-07"


class TestRegister:
    def test_find_by_lei(self, tmp_path: Path) -> None:
        entry = register(tmp_path).find(DB_LEI.lower())
        assert entry is not None and entry.code == "ECB_MFI:CREDIT_INSTITUTION"

    def test_an_unknown_or_malformed_lei_is_none(self, tmp_path: Path) -> None:
        reg = register(tmp_path)
        assert reg.find(VW_LEI) is None
        assert reg.find("x' OR 1=1 --") is None

    def test_a_refresh_replaces_only_its_own_list(self, tmp_path: Path) -> None:
        reg = register(tmp_path)
        assert reg.replace("MFI", []) == 0
        assert reg.find(DB_LEI) is None
        assert reg.find("549300QS4Q1IT6XCA514") is not None

    def test_as_of_is_none_while_empty(self, tmp_path: Path) -> None:
        reg = EcbRegister(database(tmp_path))
        assert reg.as_of() is None
        reg.replace("IF", parse_sheet("IF", IF_ROWS, "2026-07"))
        assert reg.as_of() == "2026-07"


class TestFactsAndRules:
    def test_the_fact_line_carries_the_code_the_rule_matches(self) -> None:
        (fund,) = parse_sheet("IF", IF_ROWS, "2026-07")
        line = fund.facts()[0]
        assert "[ECB_IF]" in line and "ke dni 2026-07" in line
        assert list(register_esa_families(line)) == [
            "investicni fondy jine nez fondy penezniho trhu"
        ]

    def test_prose_never_fires_a_rule(self) -> None:
        assert register_esa_families("an ECB credit institution and an investment fund") == {}

    def test_the_ecb_fact_outranks_a_captive_keyword(self) -> None:
        """'funding vehicle' says captive; the ECB list says bank - the register wins."""
        entry = next(iter(parse_mfi(MFI_TSV, "2026-09-28")))
        text = "Funding vehicle of the group.\n\n" + entry.facts()[0]
        result = EsaCandidateFilter(build_codebooks()).shortlist(text)
        assert result.candidates[0].code.startswith("200221")
        assert result.candidates[0].reasons[0] == "register: ECB MFI list: credit institution"


class FakeGleif:
    def __init__(self, lei: str) -> None:
        self.record = LeiRecord(
            lei=lei,
            legal_name="X AG",
            other_names=(),
            jurisdiction="DE",
            legal_address_country="DE",
            headquarters_country="DE",
            category="GENERAL",
            sub_category=None,
            legal_form_id=None,
            legal_form_text=None,
            entity_status="ACTIVE",
            registration_status="ISSUED",
            provenance=Provenance(source="GLEIF", retrieved_at=datetime(2026, 9, 29, tzinfo=UTC)),
        )

    def find_by_isin(self, isin: str) -> LeiRecord:
        return self.record

    def close(self) -> None: ...


class BrokenRegister(EcbRegister):
    def find(self, lei: str) -> EcbEntry | None:
        raise DatabaseError("down")


def identify(lei: str, ecb: EcbRegister | None):
    ident = IssuerIdentifier(
        Settings(openfigi_enabled=False),
        gleif=FakeGleif(lei),
        ecb=ecb,  # type: ignore[arg-type]
    )
    return ident.identify(ISIN)


class TestIdentity:
    def test_a_listed_lei_adds_the_fact_and_the_citation(self, tmp_path: Path) -> None:
        identity = identify(DB_LEI, register(tmp_path))
        assert "[ECB_MFI:CREDIT_INSTITUTION]" in identity.fact_sheet()
        assert any(e.title.startswith("ECB – seznam") for e in identity.evidence)

    def test_an_unlisted_lei_is_stated_as_absent(self, tmp_path: Path) -> None:
        sheet = identify(VW_LEI, register(tmp_path)).fact_sheet()
        assert "LEI není v žádném seznamu finančních institucí ECB" in sheet
        assert register_esa_families(sheet) == {}

    def test_an_empty_table_states_nothing(self, tmp_path: Path) -> None:
        sheet = identify(VW_LEI, EcbRegister(database(tmp_path))).fact_sheet()
        assert "ECB" not in sheet

    def test_a_database_failure_is_a_note_not_an_absence(self, tmp_path: Path) -> None:
        identity = identify(DB_LEI, BrokenRegister(database(tmp_path)))
        assert "ECB seznamy: databázi se nepodařilo dotázat" in identity.notes
        assert "ECB" not in identity.fact_sheet()

    @pytest.mark.parametrize("ecb", [None])
    def test_without_a_register_nothing_changes(self, ecb: None) -> None:
        assert "ECB" not in identify(DB_LEI, ecb).fact_sheet()
