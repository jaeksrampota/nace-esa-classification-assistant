"""The model's web search for the issuer (core.sources.llm_web). No key, no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import Settings
from core.classify.cache import SqliteCache
from core.classify.errors import LlmUnavailableError
from core.classify.llm import LlmClassifier
from core.classify.prompts import Prompt
from core.classify.provider import LlmResponse, NullLlmProvider, StubLlmProvider
from core.sources.llm_web import LlmWebSearch, build_web_search, parse_answer, web_prompt
from core.sources.web import BLOCKED_HOSTS, NO_SEARCH_PROVIDER_NOTE, IssuerEvidence

ANSWER = (
    "NÁZEV: Kongsberg Gruppen ASA\n"
    "POPIS: Norská technologická skupina, dodává obranné a námořní systémy. "
    "Přibližně polovinu akcií vlastní norský stát."
)
CITED = (
    ("https://www.kongsberg.com/about", "About Kongsberg"),
    ("https://or.justice.cz/ias/ui/rejstrik", "Obchodní rejstřík"),
)


def stub(content: str = ANSWER, citations: tuple[tuple[str, str], ...] = CITED) -> StubLlmProvider:
    return StubLlmProvider({"WEB": content}, citations=citations)


def lookup(search: LlmWebSearch, **overrides: object):
    arguments: dict[str, object] = {"name": "Kongsberg Gruppen ASA", "isin": None, "lei": None}
    arguments.update(overrides)
    return search.find(**arguments)  # type: ignore[arg-type]


class TestAnswer:
    def test_the_name_and_the_description_are_read(self) -> None:
        name, description = parse_answer(ANSWER)
        assert name == "Kongsberg Gruppen ASA"
        assert description is not None and description.startswith("Norská technologická skupina")
        assert "norský stát" in description  # who controls it: what the ESA digit turns on

    @pytest.mark.parametrize("answer", ["NENALEZENO", "Nenalezeno.", "NÁZEV: X\nPOPIS: NENALEZENO"])
    def test_not_found_is_no_description(self, answer: str) -> None:
        assert parse_answer(answer)[1] is None

    def test_inline_citation_links_are_reduced_to_text(self) -> None:
        _, description = parse_answer(
            "NÁZEV: X\nPOPIS: Banka ([x.com](https://x.com/a)). Viz [web](https://x.com/b)."
        )
        assert description == "Banka. Viz web."

    def test_an_answer_ignoring_the_format_is_taken_whole(self) -> None:
        assert parse_answer("Emitent je banka se sídlem ve Vídni.") == (
            None,
            "Emitent je banka se sídlem ve Vídni.",
        )


class TestPrompt:
    def test_only_public_facts_and_the_typed_hint_are_sent(self) -> None:
        prompt = web_prompt(
            name="X AG",
            isin="DE0000000000",
            lei="5299000FUKEMR5ZJ0K48",
            country="DE",
            instrument="X AG 2030",
            typed="Úvěrová instituce. " * 40,
        )
        assert prompt.kind == "WEB" and prompt.web_search
        assert "ISIN: DE0000000000" in prompt.user
        assert "LEI: 5299000FUKEMR5ZJ0K48" in prompt.user
        assert len(prompt.user) < 600  # the typed text is only a hint
        assert set(prompt.blocked_domains) == set(BLOCKED_HOSTS)


class TestFind:
    def test_the_finding_drops_a_page_on_the_czech_registers(self) -> None:
        finding, note = lookup(LlmWebSearch(stub()))
        assert note is None and finding is not None
        assert finding.issuer_name == "Kongsberg Gruppen ASA"
        assert finding.sources == (("https://www.kongsberg.com/about", "About Kongsberg"),)

    def test_a_repeated_issuer_is_served_from_the_cache(self, tmp_path: Path) -> None:
        provider = stub()
        search = LlmWebSearch(provider, cache=SqliteCache(tmp_path / "c.sqlite3"))
        first, _ = lookup(search)
        second, _ = lookup(search)
        assert len(provider.calls) == 1
        assert second == first

    def test_no_search_starts_that_would_leave_no_time_to_classify(self) -> None:
        provider = stub()
        search = LlmWebSearch(provider, call_seconds=20.0, clock=lambda: 100.0)
        finding, note = lookup(search, deadline=130.0)  # 30 s left, 40 s needed
        assert finding is None and provider.calls == []
        assert note is not None and note.startswith("na hledání na webu nezbyl čas")

    def test_no_model_means_no_search_and_no_note(self) -> None:
        assert lookup(LlmWebSearch(NullLlmProvider())) == (None, None)

    def test_a_failed_call_is_a_note_not_an_error(self) -> None:
        class Down:
            name, model = "down", "m"

            def complete(self, prompt: Prompt) -> LlmResponse:
                raise LlmUnavailableError("model call timed out")

        finding, note = lookup(LlmWebSearch(Down()))  # type: ignore[arg-type]
        assert finding is None
        assert note == "hledání na webu (model) selhalo: model call timed out"


class TestEnrich:
    def test_the_finding_follows_the_description_labelled_as_the_models(self) -> None:
        evidence = IssuerEvidence(query="x", description="Popis z Wikipedie.")
        enriched = LlmWebSearch(stub()).enrich(
            evidence, name="Kongsberg Gruppen ASA", isin=None, lei=None
        )
        assert enriched.description is not None
        assert enriched.description.startswith(
            "Popis z Wikipedie.\n\nPodle webu (vyhledal model stub-model): Norská"
        )
        assert enriched.found_name == "Kongsberg Gruppen ASA"
        assert [source.url for source in enriched.sources] == ["https://www.kongsberg.com/about"]
        assert enriched.notes[-1].startswith("popis doplněn z webu: vyhledal model stub-model")

    def test_alone_it_becomes_the_description(self) -> None:
        enriched = LlmWebSearch(stub()).enrich(
            IssuerEvidence(query="x"), name="Kongsberg Gruppen ASA", isin=None, lei=None
        )
        assert enriched.has_description
        assert enriched.description.startswith("Podle webu (vyhledal model stub-model): ")

    def test_not_found_leaves_the_description_and_says_so(self) -> None:
        evidence = IssuerEvidence(query="x", description="Zadaný popis.")
        enriched = LlmWebSearch(stub("NENALEZENO")).enrich(evidence, name="X", isin=None, lei=None)
        assert enriched.description == "Zadaný popis."
        assert enriched.notes[-1] == "model stub-model emitenta na webu nenašel"

    @pytest.mark.parametrize("answer", [ANSWER, "NENALEZENO"])
    def test_the_no_search_provider_note_goes_the_web_is_searched(self, answer: str) -> None:
        # Live, 30 Sept 2026: "není zapojené" stood next to "popis doplněn z webu".
        evidence = IssuerEvidence(query="x", notes=("a", NO_SEARCH_PROVIDER_NOTE))
        enriched = LlmWebSearch(stub(answer)).enrich(evidence, name="X", isin=None, lei=None)
        assert NO_SEARCH_PROVIDER_NOTE not in enriched.notes
        assert enriched.notes[0] == "a"

    def test_nothing_to_search_for_asks_nothing(self) -> None:
        provider = stub()
        evidence = IssuerEvidence(query="")
        assert LlmWebSearch(provider).enrich(evidence, name=None, isin=None, lei=None) is evidence
        assert provider.calls == []


class TestBuild:
    def test_on_only_with_a_model_and_the_switches(self) -> None:
        with_model = LlmClassifier(StubLlmProvider({}))
        assert isinstance(build_web_search(Settings(llm_api_key=None), with_model), LlmWebSearch)
        assert (
            build_web_search(Settings(llm_api_key=None), LlmClassifier(NullLlmProvider())) is None
        )
        off = Settings(llm_api_key=None, llm_web_search=False)
        assert build_web_search(off, with_model) is None
        assert build_web_search(Settings(llm_api_key=None, web_enabled=False), with_model) is None
