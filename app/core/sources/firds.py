"""ESMA FIRDS: an ISIN -> its issuer's LEI, for the ISINs GLEIF has no mapping for.

GLEIF's ISIN mapping misses many Eurobonds and funds (BMW Finance N.V., Shell International
Finance B.V., iShares Core MSCI World, Bavarian Sky - 13 of 108 test lookups, 29 Sept 2026).
FIRDS, the EU register of instruments admitted to trading, states the issuer's LEI for every
one of them. Verified live 29 Sept 2026:

    GET https://registers.esma.europa.eu/solr/esma_registers_firds/select
        ?q=isin:XS1948611840&wt=json&rows=1&fl=lei   ->  docs[0].lei = 5299006ZHG3IXU0PNJ56

Fail-soft like the other registers: ``None`` = FIRDS has no LEI for the ISIN;
:class:`~core.sources.base.SourceUnavailableError` = it could not be asked.
"""

from __future__ import annotations

import re
from typing import Final

import httpx

from config.settings import Settings
from core.sources.base import SourceUnavailableError

FIRDS_SELECT: Final[str] = "https://registers.esma.europa.eu/solr/esma_registers_firds/select"
_ISIN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]")
_LEI_RE: Final[re.Pattern[str]] = re.compile(r"[A-Z0-9]{20}")


def firds_lei(isin: str, settings: Settings, *, client: httpx.Client | None = None) -> str | None:
    """The issuer LEI FIRDS records for ``isin`` (already normalised), or ``None``."""
    if not _ISIN_RE.fullmatch(isin):
        return None
    try:
        http = client or httpx.Client(
            timeout=settings.gleif_timeout_seconds,
            headers={"User-Agent": settings.web_user_agent},
        )
        with http:
            response = http.get(
                FIRDS_SELECT, params={"q": f"isin:{isin}", "wt": "json", "rows": "1", "fl": "lei"}
            )
    except httpx.HTTPError as exc:
        raise SourceUnavailableError(f"FIRDS is unreachable: {exc}") from exc
    if response.status_code != 200:
        raise SourceUnavailableError(f"FIRDS returned HTTP {response.status_code}")
    try:
        docs = response.json()["response"]["docs"]
    except (ValueError, KeyError, TypeError) as exc:
        raise SourceUnavailableError(f"FIRDS returned an unexpected document: {exc}") from exc
    lei = str(docs[0].get("lei") or "").strip().upper() if docs else ""
    return lei if _LEI_RE.fullmatch(lei) else None


__all__ = ["firds_lei"]
