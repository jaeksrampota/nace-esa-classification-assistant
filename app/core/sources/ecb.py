"""ECB lists of financial institutions: a LEI -> which kind of financial institution it is.

The ESA *family* of a financial issuer (bank, money market fund, investment fund,
securitisation vehicle, insurer, pension fund) is otherwise read from words. The ECB
publishes, for every EU country, the statistical populations those families are made of,
most rows with a LEI (checked 2026-09-29):

* MFI - central banks, credit institutions, money market funds, other deposit-takers
  (S.121-S.123); daily, tab-separated UTF-16 CSV named ``mfi_csv_<yymmdd>.csv.gz``;
* IF  - investment funds (S.124); monthly xlsx in a yearly zip (68 MB, 78k rows);
* FVC - financial vehicle corporations engaged in securitisation (S.125); quarterly;
* IC  - insurance corporations (S.128); quarterly;
* PF  - pension funds (S.129); quarterly.

The files are far too big to read during a lookup (the fund list alone takes 12 s to
open), so ``python -m core.sources.ecb --refresh`` loads them into the ``ecb_institutions``
table of the central database and a lookup reads one row by LEI. Membership states the
entity family; it never states the control variant (roadmap Q7), and absence is a fact,
never a decision. Source: ECB, lists of financial institutions (reuse with attribution).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import logging
import re
import sys
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import httpx

from core.db import Database, DatabaseError

LOGGER = logging.getLogger(__name__)

ECB: Final[str] = "https://www.ecb.europa.eu"
#: The public page citing the lists, shown under "Podklady".
LISTS_PAGE: Final[str] = (
    f"{ECB}/stats/financial_corporations/list_of_financial_institutions/html/index.en.html"
)
MFI_PAGE: Final[str] = (
    f"{ECB}/stats/financial_corporations/list_of_financial_institutions/html/daily_list-MID.en.html"
)
MFI_FILE: Final[str] = f"{ECB}/stats/money/mfi/general/html/dla/mfi_MID/{{name}}"
#: Zips per list; ``{year}`` is tried for this year, then the last.
ZIPS: Final[dict[str, str]] = {
    "IF": f"{ECB}/stats/pdf/money/ecb.ifd_overview_{{year}}.en.zip",
    "FVC": f"{ECB}/stats/pdf/money/fvc/ecb.fvcd_overview_{{year}}.en.zip",
    "IC": f"{ECB}/stats/pdf/money/ic/ecb.icd_overview.en.zip",
    "PF": f"{ECB}/stats/pdf/money/pf/ecb.pfd_overview.en.zip",
}
#: The column carrying the sub-type, per list (matched case-insensitively).
TYPE_COLUMNS: Final[dict[str, str]] = {
    "IF": "type of investment fund",
    "FVC": "nature of securitisation",
    "IC": "type of insurance corporation",
    "PF": "type of pension fund",
}
#: MFI categories as the bracketed code of the fact sheet.
MFI_CODES: Final[dict[str, str]] = {
    "Central Bank": "CENTRAL_BANK",
    "Credit Institution": "CREDIT_INSTITUTION",
    "Money Market Fund": "MONEY_MARKET_FUND",
    "Other Institution": "OTHER_INSTITUTION",
}
#: What each code means, Czech with the English words the keyword table knows.
GLOSS_CS: Final[dict[str, str]] = {
    "ECB_MFI:CENTRAL_BANK": "centrální banka (central bank)",
    "ECB_MFI:CREDIT_INSTITUTION": "úvěrová instituce (credit institution)",
    "ECB_MFI:MONEY_MARKET_FUND": "fond peněžního trhu (money market fund)",
    "ECB_MFI:OTHER_INSTITUTION": "jiná instituce přijímající vklady (other deposit-taking corporation)",
    "ECB_IF": "investiční fond jiný než fond peněžního trhu (investment fund)",
    "ECB_FVC": "účelová jednotka pro sekuritizaci (securitisation vehicle, FVC)",
    "ECB_IC": "pojišťovna (insurance corporation)",
    "ECB_PF": "penzijní fond (pension fund)",
}
LIST_CS: Final[dict[str, str]] = {
    "MFI": "měnových finančních institucí (MFI)",
    "IF": "investičních fondů (IF)",
    "FVC": "sekuritizačních jednotek (FVC)",
    "IC": "pojišťoven (IC)",
    "PF": "penzijních fondů (PF)",
}
_LEI_RE: Final[re.Pattern[str]] = re.compile(r"[A-Z0-9]{20}")
_MFI_NAME_RE: Final[re.Pattern[str]] = re.compile(r"mfi_csv_(\d{6})\.csv\.gz")
_BATCH: Final[int] = 2000


@dataclass(frozen=True, slots=True)
class EcbEntry:
    """One row of an ECB list, keyed by LEI."""

    lei: str
    list_name: str
    code: str
    name: str | None
    country: str | None
    subtype: str | None
    head_lei: str | None
    as_of: str

    @property
    def url(self) -> str:
        return LISTS_PAGE

    def facts(self) -> tuple[str, ...]:
        """One Czech line for the fact sheet, carrying the bracketed code the rules match."""
        detail = [GLOSS_CS.get(self.code, self.code)]
        if self.subtype and self.subtype.lower() not in ("unspecified",):
            detail.append(self.subtype)
        line = (
            f"ECB, seznam {LIST_CS.get(self.list_name, self.list_name)} ke dni {self.as_of}: "
            f"{self.name or '(název neuveden)'} ({self.country or '?'}) - "
            + ", ".join(detail)
            + f" [{self.code}]."
        )
        if self.head_lei:
            line += f" Pobočka, ústředí LEI {self.head_lei}."
        return (line,)


def absence_fact(as_of: str) -> str:
    """The fact sheet line for a LEI on none of the lists."""
    # No family words at all: "penzijní fondy" fired a keyword and "finančních institucí"
    # scored every financial family - Adidas was offered pension funds (29 Sept 2026).
    return f"ECB: LEI se ke dni {as_of} nevyskytuje v žádném statistickém seznamu ECB [ECB_NONE]."


class EcbRegister:
    """Reads and refreshes the ``ecb_institutions`` table."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def find(self, lei: str) -> EcbEntry | None:
        """The entry for ``lei``; ``None`` when no list has it. Raises :class:`DatabaseError`."""
        lei = lei.strip().upper()
        if not _LEI_RE.fullmatch(lei):
            return None
        rows = self._database.query(
            "SELECT lei, list_name, code, name, country, subtype, head_lei, as_of "
            "FROM ecb_institutions WHERE lei = ? ORDER BY list_name LIMIT 1",
            (lei,),
        )
        return EcbEntry(*rows[0]) if rows else None

    def as_of(self) -> str | None:
        """The oldest list date loaded, or ``None`` while the table is empty."""
        rows = self._database.query("SELECT MIN(as_of) FROM ecb_institutions")
        return rows[0][0] if rows and rows[0][0] else None

    def replace(self, list_name: str, entries: Iterable[EcbEntry]) -> int:
        """Swap one list's rows for ``entries`` in one transaction; returns the count."""
        insert = self._database.sql(
            "INSERT INTO ecb_institutions "
            "(lei, list_name, code, name, country, subtype, head_lei, as_of) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (lei, list_name) DO NOTHING"
        )
        count = 0
        with self._database.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                self._database.sql("DELETE FROM ecb_institutions WHERE list_name = ?"),
                (list_name,),
            )
            batch: list[tuple[Any, ...]] = []
            for entry in entries:
                batch.append(
                    (
                        entry.lei,
                        entry.list_name,
                        entry.code,
                        entry.name,
                        entry.country,
                        entry.subtype,
                        entry.head_lei,
                        entry.as_of,
                    )
                )
                if len(batch) >= _BATCH:
                    cursor.executemany(insert, batch)
                    count += len(batch)
                    batch = []
            if batch:
                cursor.executemany(insert, batch)
                count += len(batch)
        return count


# -- parsing ------------------------------------------------------------------------------


def _clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _lei(value: object) -> str | None:
    text = (_clean(value) or "").upper()
    return text if _LEI_RE.fullmatch(text) else None


def parse_mfi(text: str, as_of: str) -> Iterator[EcbEntry]:
    """Rows of the MFI CSV (already decoded) that carry a LEI."""
    for row in csv.DictReader(io.StringIO(text), delimiter="\t"):
        lei = _lei(row.get("LEI"))
        category = _clean(row.get("CATEGORY")) or ""
        if lei is None or category not in MFI_CODES:
            continue
        yield EcbEntry(
            lei=lei,
            list_name="MFI",
            code=f"ECB_MFI:{MFI_CODES[category]}",
            name=_clean(row.get("NAME")),
            country=_clean(row.get("COUNTRY_OF_REGISTRATION")),
            subtype=None,
            head_lei=_lei(row.get("HEAD_LEI")),
            as_of=as_of,
        )


def parse_sheet(list_name: str, rows: Iterable[tuple[Any, ...]], as_of: str) -> Iterator[EcbEntry]:
    """Rows of an IF/FVC/IC/PF sheet that carry a LEI; the header is the row naming ``LEI``."""
    header: list[str] | None = None
    for row in rows:
        if header is None:
            if row and "LEI" in [str(cell).strip() for cell in row if cell is not None]:
                header = [str(cell or "").strip().lower() for cell in row]
            continue
        values = dict(zip(header, row, strict=False))
        lei = _lei(values.get("lei"))
        if lei is None:
            continue
        yield EcbEntry(
            lei=lei,
            list_name=list_name,
            code=f"ECB_{list_name}",
            name=_clean(values.get("name")),
            country=_clean(values.get("country of residence")),
            subtype=_clean(values.get(TYPE_COLUMNS[list_name])),
            head_lei=_lei(values.get("head lei")),
            as_of=as_of,
        )


# -- downloading --------------------------------------------------------------------------


def _client() -> httpx.Client:
    from config.settings import get_settings

    return httpx.Client(
        timeout=120.0,
        follow_redirects=True,
        headers={"User-Agent": get_settings().web_user_agent},
    )


def fetch_mfi(client: httpx.Client) -> tuple[str, str]:
    """The newest MFI CSV, decoded, and its date (``2026-09-28``)."""
    page = client.get(MFI_PAGE)
    page.raise_for_status()
    names = sorted(set(_MFI_NAME_RE.findall(page.text)))
    if not names:
        raise RuntimeError("the ECB MFI page lists no mfi_csv_<date>.csv.gz file")
    stamp = names[-1]
    response = client.get(MFI_FILE.format(name=f"mfi_csv_{stamp}.csv.gz"))
    response.raise_for_status()
    text = gzip.decompress(response.content).decode("utf-16")
    return text, f"20{stamp[:2]}-{stamp[2:4]}-{stamp[4:]}"


def fetch_zip_sheet(client: httpx.Client, list_name: str) -> tuple[Iterator[tuple[Any, ...]], str]:
    """The newest workbook in a list's zip, as rows, and its period (``2026-07``, ``2026-Q2``)."""
    import openpyxl

    year = datetime.now(UTC).year
    content: bytes | None = None
    for candidate in (year, year - 1):
        response = client.get(ZIPS[list_name].format(year=candidate))
        if response.status_code == 200:
            content = response.content
            break
    if content is None:
        raise RuntimeError(f"no ECB {list_name} download found for {year} or {year - 1}")
    archive = zipfile.ZipFile(io.BytesIO(content))
    books = sorted(
        name
        for name in archive.namelist()
        if name.lower().endswith(".xlsx") and not name.split("/")[-1].startswith("~$")
    )
    if not books:
        raise RuntimeError(f"the ECB {list_name} zip holds no xlsx")
    newest = books[-1]
    period = re.search(r"(\d{4}_(?:Q\d|\d{2}))", newest)
    workbook = openpyxl.load_workbook(io.BytesIO(archive.read(newest)), read_only=True)
    as_of = period.group(1).replace("_", "-") if period else newest
    return workbook.worksheets[0].iter_rows(values_only=True), as_of


def refresh(
    register: EcbRegister, lists: Iterable[str] = ("MFI", "IF", "FVC", "IC", "PF")
) -> dict[str, int]:
    """Download each list and replace its rows; one failing list leaves the others loaded."""
    loaded: dict[str, int] = {}
    with _client() as client:
        for list_name in lists:
            try:
                if list_name == "MFI":
                    text, as_of = fetch_mfi(client)
                    entries: Iterable[EcbEntry] = parse_mfi(text, as_of)
                else:
                    rows, as_of = fetch_zip_sheet(client, list_name)
                    entries = parse_sheet(list_name, rows, as_of)
                loaded[list_name] = register.replace(list_name, entries)
                LOGGER.info("ECB %s (%s): %d rows", list_name, as_of, loaded[list_name])
            except (httpx.HTTPError, RuntimeError, zipfile.BadZipFile, DatabaseError) as exc:
                LOGGER.error("ECB %s not refreshed: %s", list_name, exc)
    return loaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m core.sources.ecb", description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="download and load every list")
    parser.add_argument("--lei", help="show the entry for one LEI")
    parser.add_argument("--lists", default="MFI,IF,FVC,IC,PF", help="lists to refresh")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from core.db import get_database

    database = get_database()
    if database is None:
        print("DATABASE_URL is not set; the ECB lists live in the central database.")
        return 2
    register = EcbRegister(database)
    if args.refresh:
        loaded = refresh(register, [name.strip().upper() for name in args.lists.split(",")])
        print(", ".join(f"{name} {count}" for name, count in loaded.items()) or "nothing loaded")
        return 0 if loaded else 1
    if args.lei:
        entry = register.find(args.lei)
        print(entry.facts()[0] if entry else absence_fact(register.as_of() or "?"))
        return 0
    print(f"ECB lists loaded as of {register.as_of() or 'never'} in {database.describe()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["EcbEntry", "EcbRegister", "absence_fact", "parse_mfi", "parse_sheet", "refresh"]
