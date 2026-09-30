"""Tool 1 end to end: an ISIN, a name or a description in, two ranked suggestions out.

The pipeline the brief describes, in one place so the API, the UI and a CLI all get the same
behaviour:

    input -> identity (GLEIF, OpenFIGI) -> evidence (typed, Wikipedia, the model's web search)
          -> shortlist per codebook -> classifier -> suggestions

The identity step is what makes an ISIN a useful input on its own. GLEIF turns it into the
issuer's LEI record - legal name, country, legal form, entity category, parents - and
OpenFIGI into the instrument's type and market sector. The legal name becomes the web search
query and the name on the page; the facts go to the pre-filter and the model next to the
web description, so a bank is a bank because the register says so, not because the model
recognised the name. The web is looked at on every lookup (30 Sept 2026): Wikipedia by LEI
or name, then the model's own web search (:mod:`core.sources.llm_web`) when a model is on.

What the pipeline deliberately does **not** do is fail. Every step degrades: a malformed ISIN
becomes a note, a register that cannot be asked becomes a note, an issuer the web cannot
describe produces an abstention, an exhausted budget produces an abstention. A reviewer
always gets a row back, with the reason attached, because MO's fallback is to research the
issuer themselves - which they can only do if they can see that the tool did not.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from core.classify.candidates import DEFAULT_LIMIT, EsaCandidateFilter, NaceCandidateFilter
from core.classify.llm import LlmClassifier
from core.classify.models import ESA, NACE, CandidateSet, Classification
from core.classify.proposal import Proposal, propose
from core.codebooks.models import CodebookSet
from core.identifiers.isin import InvalidIsinError, normalize_isin
from core.sources.identity import NO_IDENTITY, IssuerIdentifier, IssuerIdentity
from core.sources.llm_web import LlmWebSearch
from core.sources.names import fold
from core.sources.web import EvidenceSource, IssuerEvidence, WebEvidenceGatherer

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SuggestionRequest:
    """What the user asked for. At least one field must be filled."""

    isin: str | None = None
    name: str | None = None
    description: str | None = None

    @property
    def is_empty(self) -> bool:
        return not any((self.isin or "").strip() for _ in (1,)) and not (
            (self.name or "").strip() or (self.description or "").strip()
        )

    def cleaned(self) -> tuple[SuggestionRequest, tuple[str, ...]]:
        """Trim the inputs and normalise the ISIN, collecting notes about what was wrong.

        A malformed ISIN is a note, not a rejection: the name or description may still be
        enough, and telling the user their ISIN looks wrong is more useful than refusing the
        whole request. It is also never sent to a register - only a valid ISIN is looked up.
        """
        notes: list[str] = []
        isin = (self.isin or "").strip() or None
        name = " ".join((self.name or "").split()) or None
        description = (self.description or "").strip() or None

        if isin is not None:
            try:
                isin = normalize_isin(isin)
            except InvalidIsinError as exc:
                notes.append(f"ISIN {isin!r} nevypadá jako platný ISIN ({exc.reason})")
                if not (name or description):
                    notes.append("bez názvu nebo popisu nelze pokračovat")
                isin = None
        return SuggestionRequest(isin=isin, name=name, description=description), tuple(notes)


@dataclass(frozen=True, slots=True)
class IssuerSuggestion:
    """Everything one lookup produced, including why a half of it may be empty."""

    request: SuggestionRequest
    evidence: IssuerEvidence
    nace: Classification
    esa: Classification
    nace_candidates: CandidateSet
    esa_candidates: CandidateSet
    codebook_version: str | None = None
    created_at: datetime | None = None
    notes: tuple[str, ...] = field(default=())
    identity: IssuerIdentity = NO_IDENTITY
    warnings: tuple[str, ...] = field(default=())

    @property
    def issuer_name(self) -> str | None:
        """The name to show and export ("Jméno emitenta"): the one found, not the one typed.

        See :func:`issuer_name_of`. What was typed is shown beside it when it differs
        (:attr:`typed_name_apart`).
        """
        return issuer_name_of(self.request, self.identity, self.evidence)

    @property
    def typed_name_apart(self) -> str | None:
        """The typed name, when the name shown is a different one; the page shows both."""
        typed = (self.request.name or "").strip()
        return typed if typed and fold(typed) != fold(self.issuer_name or "") else None

    @property
    def description(self) -> str | None:
        return self.evidence.description

    @property
    def classifier_text(self) -> str:
        """What the pre-filter and the model read: the description, then the register facts."""
        return "\n\n".join(
            part for part in (self.evidence.description, self.identity.fact_sheet()) if part
        )

    @property
    def evidence_sources(self) -> tuple[EvidenceSource, ...]:
        """Register pages first, then the web hits - the "Podklady" list."""
        return (*self.identity.evidence, *self.evidence.sources)

    @property
    def sources(self) -> tuple[str, ...]:
        """Registers that answered, then ``WEB``: the audit trail and the row's ``source``."""
        return (*self.identity.sources, "WEB")

    @property
    def source_label(self) -> str:
        """``"GLEIF+OPENFIGI+WEB"``, or plain ``"WEB"`` for a lookup without an ISIN."""
        return "+".join(self.sources)

    @property
    def answered(self) -> bool:
        """True when the model produced a suggestion for at least one codebook."""
        return bool(self.nace.suggestions or self.esa.suggestions)

    @property
    def nace_proposal(self) -> Proposal | None:
        """The proposed NACE code ("navrhovaný kód"): the model's pick, else a rule's."""
        return propose(self.nace, self.nace_candidates)

    @property
    def esa_proposal(self) -> Proposal | None:
        """The proposed ESA code ("navrhovaný kód"): the model's pick, else a rule's."""
        return propose(self.esa, self.esa_candidates)

    @property
    def all_notes(self) -> tuple[str, ...]:
        """Request notes, register notes, evidence notes and each abstention reason, deduplicated."""
        collected: list[str] = [
            *self.warnings,
            *self.notes,
            *self.identity.notes,
            *self.evidence.notes,
        ]
        for classification in (self.nace, self.esa):
            if classification.abstained and classification.abstain_reason:
                collected.append(f"{classification.kind}: {classification.abstain_reason}")
        seen: list[str] = []
        for note in collected:
            if note and note not in seen:
                seen.append(note)
        return tuple(seen)


class SuggestionService:
    """Runs the pipeline. One instance is shared by the API; it holds no per-request state."""

    def __init__(
        self,
        codebooks: CodebookSet,
        *,
        gatherer: WebEvidenceGatherer,
        classifier: LlmClassifier,
        identifier: IssuerIdentifier | None = None,
        limit: int = DEFAULT_LIMIT,
        deadline_seconds: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
        web_search: LlmWebSearch | None = None,
    ) -> None:
        self._codebooks = codebooks
        self._gatherer = gatherer
        self._classifier = classifier
        self._identifier = identifier
        self._web_search = web_search
        self._limit = limit
        self._deadline_seconds = max(0.0, deadline_seconds)
        self._clock = clock
        self._nace = NaceCandidateFilter(codebooks)
        self._esa = EsaCandidateFilter(codebooks)

    @property
    def codebooks(self) -> CodebookSet:
        return self._codebooks

    def suggest(self, request: SuggestionRequest) -> IssuerSuggestion:
        """Run one lookup. Never raises for ordinary failures.

        With ``deadline_seconds`` set (``LOOKUP_DEADLINE_SECONDS``), the model calls must be
        over that long after this started: whatever the registers took is subtracted.
        """
        started = self._clock()
        deadline = started + self._deadline_seconds if self._deadline_seconds else None
        cleaned, notes = request.cleaned()

        identity = (
            self._identifier.identify(cleaned.isin, name=cleaned.name)
            if self._identifier is not None
            else NO_IDENTITY
        )
        # The LEI lets the gatherer find the issuer's Wikipedia article by identifier; without
        # one, what the user typed, else the register's name, is searched on Wikipedia.
        evidence = self._gatherer.gather(
            name=cleaned.name or identity.legal_name,
            isin=cleaned.isin,
            description=cleaned.description,
            lei=identity.lei,
            deadline=deadline,
        )
        if self._web_search is not None:
            # "Use the LLM for each" (Jakub, 30 Sept 2026): the model searches the web too,
            # whatever was typed or found on Wikipedia, and its finding follows theirs.
            evidence = self._web_search.enrich(
                evidence,
                name=identity.lei_record.legal_name if identity.lei_record else cleaned.name,
                isin=cleaned.isin,
                lei=identity.lei,
                country=identity.country,
                instrument=identity.instrument.name if identity.instrument else None,
                typed=cleaned.description,
                deadline=deadline,
            )
        issuer_name = issuer_name_of(cleaned, identity, evidence)
        text = "\n\n".join(part for part in (evidence.description, identity.fact_sheet()) if part)

        nace_candidates = self._nace.shortlist(text, limit=self._limit)
        esa_candidates = self._esa.shortlist(text, limit=self._limit)
        nace, esa = self._classifier.classify_both(
            nace_candidates,
            esa_candidates,
            issuer_name=issuer_name,
            description=text,
            deadline=deadline,
        )

        return IssuerSuggestion(
            request=cleaned,
            evidence=evidence,
            nace=nace,
            esa=esa,
            nace_candidates=nace_candidates,
            esa_candidates=esa_candidates,
            codebook_version=self._codebooks.version.id,
            created_at=datetime.now(UTC),
            notes=notes,
            identity=identity,
            warnings=_input_warnings(cleaned, identity),
        )


def build_service(
    settings: object = None,
    *,
    codebooks: CodebookSet | None = None,
) -> SuggestionService:
    """The standard service: real codebooks, the configured registers, search provider and model."""
    from config.settings import Settings, get_settings
    from core.classify.llm import build_classifier
    from core.codebooks.loaders import load_and_check
    from core.sources.identity import build_identifier
    from core.sources.llm_web import build_web_search

    resolved: Settings = settings if isinstance(settings, Settings) else get_settings()
    if codebooks is None:
        codebooks, _ = load_and_check(resolved, strict=False)
    classifier = build_classifier(resolved, codebook_version=codebooks.version.id)
    return SuggestionService(
        codebooks,
        gatherer=WebEvidenceGatherer(resolved),
        classifier=classifier,
        identifier=build_identifier(resolved),
        deadline_seconds=resolved.lookup_deadline_seconds,
        web_search=build_web_search(resolved, classifier),
    )


def issuer_name_of(
    request: SuggestionRequest, identity: IssuerIdentity, evidence: IssuerEvidence
) -> str | None:
    """The issuer's name as found, not as typed (Jakub, 30 Sept 2026: "find the issuer's name").

    GLEIF's legal name first; then the official name the model's web search found; then what
    the user typed; then OpenFIGI's market name and a web page's title. Until 30 Sept 2026 the
    typed name came first, so "adidas" was exported where GLEIF says "adidas AG".
    """
    record = identity.lei_record
    return (
        (record.legal_name if record is not None else None)
        or evidence.found_name
        or request.name
        or identity.legal_name
        or evidence.issuer_name
    )


def _input_warnings(request: SuggestionRequest, identity: IssuerIdentity) -> tuple[str, ...]:
    """A typed name that does not fit the ISIN's issuer: one of the two inputs is wrong."""
    from core.sources.names import names_agree

    record = identity.lei_record
    resident = [
        f"Emitent je podle {where} rezident ČR – nástroj je určen pro zahraniční emitenty a nabízí "
        "jen nerezidentské kódy ESA. Kód pro rezidenta ověřte zvlášť."
        for where, country in (
            ("GLEIF", identity.country),
            ("ISIN", (request.isin or "")[:2] if not identity.country else None),
        )
        if country == "CZ"
    ][:1]
    if not (request.isin and request.name and identity.legal_name):
        return tuple(resident)
    known = (
        identity.legal_name,
        *(record.other_names if record else ()),
        identity.instrument.name if identity.instrument else None,
    )
    if names_agree(request.name, *known):
        return tuple(resident)
    return (
        *resident,
        f"Zadaný název „{request.name}“ neodpovídá emitentovi ISIN {request.isin} "
        f"(podle registru „{identity.legal_name}“). Zkontrolujte ISIN i název – výsledek "
        "vychází z ISIN.",
    )


__all__ = [
    "ESA",
    "NACE",
    "IssuerSuggestion",
    "SuggestionRequest",
    "SuggestionService",
    "build_service",
    "issuer_name_of",
]
