"""Identifier normalization and validation (ISIN).

Pure functions only: no I/O, no logging. The algorithm and the policy for messy
Excel-derived input are documented in :mod:`core.identifiers.isin`; this package re-exports
its public names.
"""

from core.identifiers.isin import (
    ISIN_LENGTH,
    ISIN_PATTERN,
    ISIN_REASONS,
    InvalidIsinError,
    IsinError,
    IsinReason,
    is_isin_format,
    is_valid_isin,
    isin_checksum_ok,
    isin_country_code,
    normalize_isin,
    try_normalize_isin,
)

__all__ = [
    "ICO_LENGTH",
    "ICO_REASONS",
    "ISIN_LENGTH",
    "ISIN_PATTERN",
    "ISIN_REASONS",
    "InvalidIsinError",
    "IsinError",
    "IsinReason",
    "is_isin_format",
    "is_valid_isin",
    "isin_checksum_ok",
    "isin_country_code",
    "normalize_isin",
    "try_normalize_isin",
]
