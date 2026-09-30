"""Wikidata/Wikipedia: LEI -> item -> article summary, and how the gatherer uses it.

Payloads are trimmed live responses (see ``conftest.py``); nothing here touches the network.
"""

from __future__ import annotations

import httpx
import pytest

from config.settings import Settings
from core.sources.base import SourceResponseError, SourceUnavailableError
from core.sources.web import SearchHit, StaticSearchProvider, WebEvidenceGatherer
from core.sources.wikimedia import WikimediaSource, _clean
from tests.sources.conftest import (
    WIKIDATA_DB_LEI,
    WIKIDATA_EMPTY_SEARCH,
    WIKIPEDIA_DB_CS,
    WIKIPEDIA_DB_EN,
    WIKIPEDIA_DISAMBIGUATION,
    wikimedia_client,
)

LEI = WIKIDATA_DB_LEI


def wikipedia_searches(calls: list[httpx.Request]) -> list[httpx.Request]:
    """The Wikipedia searches among ``calls`` (Wikidata's API has the same path)."""
    return [
        call
        for call in calls
        if call.url.host.endswith(".wikipedia.org") and call.url.path == "/w/api.php"
    ]


def settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "wikimedia_enabled": True,
        "wikimedia_min_interval_seconds": 0.0,
        "wikimedia_max_attempts": 1,
        "wikipedia_languages": "cs,en",
        "web_enabled": True,
        "web_min_interval_seconds": 0.0,
        "web_search_url": None,
    }
    base.update(overrides)
    return Settings(**base)


def source(client: httpx.Client | None = None, **overrides: object) -> WikimediaSource:
    return WikimediaSource(
        settings(**overrides), client=client or wikimedia_client(), sleep=lambda _: None
    )


class TestItemByLei:
    def test_maps_the_live_payload(self) -> None:
        item = source().find_by_lei(LEI)
        assert item is not None
        assert item.qid == "Q66048"
        assert item.label == "Deutsche Bank"
        assert item.description_cs == "německá banka"
        assert item.description_en == "German global banking and financial services company"
        assert item.url == "https://www.wikidata.org/wiki/Q66048"
        assert [link.lang for link in item.sitelinks] == ["cs", "en"]
        assert item.provenance.source == "WEB"
        assert item.provenance.detail == "web:wikidata"

    def test_industries_put_preferred_first_and_drop_deprecated(self) -> None:
        item = source().find_by_lei(LEI)
        assert item is not None
        assert [industry.qid for industry in item.industries] == [
            "Q29585689",
            "Q837171",
            "Q29584334",
        ]
        assert item.industries[0].text == (
            "ostatní peněžní zprostředkování (other monetary intermediation)"
        )
        # Only an English label: shown as it is.
        assert item.industries[2].text.startswith("financial service activities")

    def test_facts_are_czech_with_the_english_in_parentheses(self) -> None:
        item = source().find_by_lei(LEI)
        assert item is not None
        facts = item.facts()
        assert facts[0] == (
            "Wikidata: německá banka (German global banking and financial services company)."
        )
        assert facts[1].startswith("Odvětví podle Wikidat: ostatní peněžní zprostředkování")

    def test_the_search_asks_for_the_lei_property(self) -> None:
        calls: list[httpx.Request] = []
        source(wikimedia_client(calls=calls)).find_by_lei(LEI)
        assert calls[0].url.params["srsearch"] == f"haswbstatement:P1278={LEI}"
        # The whole item (443 KB for Deutsche Bank) is never requested.
        assert all("claims" not in (call.url.params.get("props") or "") for call in calls)

    def test_the_sitelink_filter_follows_the_configured_languages(self) -> None:
        calls: list[httpx.Request] = []
        item = source(wikimedia_client(calls=calls), wikipedia_languages="en").find_by_lei(LEI)
        assert item is not None
        assert [link.lang for link in item.sitelinks] == ["en"]
        entity_call = next(
            call for call in calls if call.url.params.get("props", "").startswith("labels|desc")
        )
        assert entity_call.url.params["sitefilter"] == "enwiki"

    def test_no_item_with_the_lei_is_none(self) -> None:
        assert source(wikimedia_client(search=WIKIDATA_EMPTY_SEARCH)).find_by_lei(LEI) is None

    @pytest.mark.parametrize("lei", ["", "SHORT", "7ltwfzyicnsx8d621k8!", f"{LEI} OR x"])
    def test_a_malformed_lei_is_never_sent(self, lei: str) -> None:
        calls: list[httpx.Request] = []
        assert source(wikimedia_client(calls=calls)).find_by_lei(lei) is None
        assert calls == []

    def test_an_item_without_industries_asks_nothing_more(self) -> None:
        calls: list[httpx.Request] = []
        item = source(wikimedia_client(claims={"claims": {}}, calls=calls)).find_by_lei(LEI)
        assert item is not None
        assert item.industries == ()
        assert len(calls) == 3  # search, entity, claims - no industry labels


class TestArticleByName:
    """The name path: a Wikipedia search, and the first article whose title fits the name.

    Until 30 Sept 2026 only an exact, unique Wikidata label with no other entity's LEI counted;
    that check is gone, so these tests pin what replaced it - including the namesake it lets in.
    """

    EIB = {"European Investment Bank": ["European Investment Bank", "EIB Group"]}

    def test_the_article_the_search_finds_is_taken_with_its_item(self) -> None:
        wiki = source(
            wikimedia_client(search=WIKIDATA_EMPTY_SEARCH, wiki_search=self.EIB)
        ).describe("5493006YXS1U5GIHE750", name="European Investment Bank")
        assert wiki is not None and wiki.summary is not None
        assert wiki.item.matched_by == "name"
        assert wiki.item.qid == "Q66048"  # the canned article's wikibase_item
        assert wiki.summary.lang == "cs"

    def test_the_lei_is_tried_first_and_the_name_only_after_a_miss(self) -> None:
        calls: list[httpx.Request] = []
        wiki = source(wikimedia_client(calls=calls, wiki_search=self.EIB)).describe(
            LEI, name="European Investment Bank"
        )
        assert wiki is not None and wiki.item.matched_by == "lei"
        assert not wikipedia_searches(calls)

    def test_a_name_alone_is_enough(self) -> None:
        wiki = source(wikimedia_client(wiki_search=self.EIB)).describe(
            None, name="European Investment Bank"
        )
        assert wiki is not None and wiki.item.matched_by == "name"

    def test_the_search_leaves_the_legal_form_out(self) -> None:
        calls: list[httpx.Request] = []
        source(wikimedia_client(calls=calls)).describe(None, name="Kongsberg Gruppen ASA")
        searched = [call.url.params.get("srsearch") for call in wikipedia_searches(calls)]
        assert searched == ["Kongsberg Gruppen", "Kongsberg Gruppen"]  # cs, then en

    def test_a_title_sharing_no_word_with_the_name_is_passed_over(self) -> None:
        # "Nordkapp" is a cape in Norway, not the financing vehicle.
        calls: list[httpx.Request] = []
        client = wikimedia_client(
            calls=calls, wiki_search={"Nordkap Funding": ["Nordkapp", "Nordkap Funding"]}
        )
        assert source(client).describe_by_name("Nordkap Funding B.V.") is not None
        read = [call.url.path for call in calls if "/page/summary/" in call.url.path]
        assert read[0] == "/api/rest_v1/page/summary/Nordkap_Funding"
        assert not any("Nordkapp" in path for path in read)

    def test_the_title_sharing_most_words_is_tried_first(self) -> None:
        # Live, 30 Sept 2026: the Czech search put the town Kongsberg before the company.
        calls: list[httpx.Request] = []
        client = wikimedia_client(
            calls=calls, wiki_search={"Kongsberg Gruppen": ["Kongsberg", "Kongsberg Gruppen"]}
        )
        assert source(client).describe_by_name("Kongsberg Gruppen ASA") is not None
        read = [call.url.path for call in calls if "/page/summary/" in call.url.path]
        assert read[0] == "/api/rest_v1/page/summary/Kongsberg_Gruppen"

    def test_english_is_searched_first_and_the_czech_article_preferred(self) -> None:
        calls: list[httpx.Request] = []
        wiki = source(wikimedia_client(calls=calls, wiki_search=self.EIB)).describe_by_name(
            "European Investment Bank"
        )
        assert wikipedia_searches(calls)[0].url.host == "en.wikipedia.org"
        # Found in English; the item's Czech article is the text, as on the LEI path.
        assert wiki is not None and wiki.summary is not None and wiki.summary.lang == "cs"

    def test_a_group_article_is_taken_for_its_vehicle(self) -> None:
        """The accepted risk: BMW Finance N.V. may get BMW's article; the gatherer flags every
        name match for the reviewer and the model reads it next to the register facts."""
        wiki = source(wikimedia_client(wiki_search={"BMW Finance": ["BMW"]})).describe_by_name(
            "BMW Finance N.V."
        )
        assert wiki is not None and wiki.item.matched_by == "name"

    def test_czech_is_searched_when_english_finds_nothing(self) -> None:
        client = wikimedia_client(wiki_search=self.EIB, wiki_search_langs={"cs"})
        wiki = source(client).describe_by_name("European Investment Bank")
        assert wiki is not None and wiki.summary is not None
        assert wiki.summary.lang == "cs"

    def test_a_disambiguation_page_is_passed_over(self) -> None:
        client = wikimedia_client(
            wiki_search={"Mercury": ["Mercury"]},
            summaries={"cs": WIKIPEDIA_DISAMBIGUATION, "en": WIKIPEDIA_DISAMBIGUATION},
        )
        assert source(client).describe_by_name("Mercury") is None

    def test_a_family_name_article_is_passed_over(self) -> None:
        client = wikimedia_client(wiki_search={"Generali": ["Generali (příjmení)"]})
        assert source(client).describe_by_name("Generali") is None

    def test_an_article_without_a_wikidata_item_is_passed_over(self) -> None:
        bare = {key: value for key, value in WIKIPEDIA_DB_CS.items() if key != "wikibase_item"}
        client = wikimedia_client(wiki_search=self.EIB, summaries={"cs": bare, "en": bare})
        assert source(client).describe_by_name("European Investment Bank") is None

    def test_too_short_a_name_is_never_searched(self) -> None:
        calls: list[httpx.Request] = []
        assert source(wikimedia_client(calls=calls)).describe_by_name("AB") is None
        assert calls == []

    def test_the_switch_turns_the_name_path_off(self) -> None:
        calls: list[httpx.Request] = []
        client = wikimedia_client(calls=calls, search=WIKIDATA_EMPTY_SEARCH, wiki_search=self.EIB)
        assert (
            source(client, wikimedia_name_match=False).describe(
                LEI, name="European Investment Bank"
            )
            is None
        )
        assert not wikipedia_searches(calls)


class TestSummary:
    def test_czech_comes_first(self) -> None:
        wiki = source().describe(LEI)
        assert wiki is not None and wiki.summary is not None
        assert wiki.summary.lang == "cs"
        assert wiki.summary.url == "https://cs.wikipedia.org/wiki/Deutsche_Bank"
        assert wiki.summary.provenance.detail == "web:wikipedia:cs"
        assert wiki.summary.revised_at is not None

    def test_editorial_marks_are_dropped(self) -> None:
        wiki = source().describe(LEI)
        assert wiki is not None and wiki.summary is not None
        assert "[kdy?]" not in wiki.summary.extract
        assert "\n" in wiki.summary.extract  # paragraphs kept

    def test_english_when_there_is_no_czech_article(self) -> None:
        wiki = source(wikimedia_client(summaries={"en": WIKIPEDIA_DB_EN})).describe(LEI)
        assert wiki is not None and wiki.summary is not None
        assert wiki.summary.lang == "en"

    def test_a_disambiguation_page_is_not_a_description(self) -> None:
        client = wikimedia_client(summaries={"cs": WIKIPEDIA_DISAMBIGUATION, "en": WIKIPEDIA_DB_EN})
        wiki = source(client).describe(LEI)
        assert wiki is not None and wiki.summary is not None
        assert wiki.summary.lang == "en"

    def test_a_wikipedia_outage_leaves_the_item_and_a_note(self) -> None:
        wiki = source(wikimedia_client(summaries={"cs": 503, "en": 503})).describe(LEI)
        assert wiki is not None
        assert wiki.summary is None
        assert len(wiki.notes) == 2
        assert wiki.notes[0].startswith("Wikipedie (cs): článek se nepodařilo načíst")
        # The item alone still describes the issuer.
        assert wiki.paragraphs and wiki.paragraphs[0].startswith("Wikidata: německá banka")

    def test_no_article_at_all_is_noted(self) -> None:
        entity = {
            "entities": {
                "Q66048": {"id": "Q66048", "labels": {}, "descriptions": {}, "sitelinks": {}}
            }
        }
        wiki = source(wikimedia_client(entity=entity)).describe(LEI)
        assert wiki is not None
        assert wiki.summary is None
        assert "nemá článek na Wikipedii" in wiki.notes[0]

    def test_no_item_means_no_description(self) -> None:
        assert source(wikimedia_client(search=WIKIDATA_EMPTY_SEARCH)).describe(LEI) is None


class TestFailures:
    def test_a_wikidata_outage_is_unavailable_not_a_miss(self) -> None:
        with pytest.raises(SourceUnavailableError):
            source(wikimedia_client(wikidata_status=503)).find_by_lei(LEI)

    def test_an_api_error_inside_a_200_is_a_response_error(self) -> None:
        client = wikimedia_client(search={"error": {"code": "badvalue", "info": "x"}})
        with pytest.raises(SourceResponseError, match="badvalue"):
            source(client).find_by_lei(LEI)

    def test_429_is_retried(self) -> None:
        answers = iter([httpx.Response(429), httpx.Response(200, json=WIKIDATA_EMPTY_SEARCH)])
        client = httpx.Client(transport=httpx.MockTransport(lambda request: next(answers)))
        assert source(client, wikimedia_max_attempts=2).find_by_lei(LEI) is None

    def test_a_4xx_is_not_retried(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(403, text="Please respect our robot policy")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(SourceResponseError):
            source(client, wikimedia_max_attempts=3).find_by_lei(LEI)
        assert len(calls) == 1

    def test_no_request_is_started_that_could_outlive_the_deadline(self) -> None:
        calls: list[httpx.Request] = []
        wiki = WikimediaSource(
            settings(wikimedia_timeout_seconds=5.0),
            client=wikimedia_client(calls=calls),
            sleep=lambda _: None,
            monotonic=lambda: 100.0,
        )
        with pytest.raises(SourceUnavailableError, match="nezbyl čas"):
            wiki.find_by_lei(LEI, deadline=104.0)
        assert calls == []

    def test_requests_are_spaced_out(self) -> None:
        now = {"t": 0.0}
        slept: list[float] = []
        wiki = WikimediaSource(
            settings(wikimedia_min_interval_seconds=0.3),
            client=wikimedia_client(),
            sleep=slept.append,
            monotonic=lambda: now["t"],
        )
        wiki.find_by_lei(LEI)
        assert slept and all(wait == pytest.approx(0.3) for wait in slept)


def test_clean_keeps_text_and_drops_marks() -> None:
    assert _clean("A  bank.[citation needed]\n\n\nIt lends.") == "A bank.\nIt lends."


# -- the gatherer ------------------------------------------------------------------------


class RecordingProvider(StaticSearchProvider):
    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        super().__init__(hits or [])
        self.queries: list[str] = []

    def search(self, query: str, *, limit: int) -> tuple[SearchHit, ...]:
        self.queries.append(query)
        return super().search(query, limit=limit)


def gatherer(
    client: httpx.Client | None = None,
    provider: RecordingProvider | None = None,
    **overrides: object,
) -> WebEvidenceGatherer:
    config = settings(**overrides)
    return WebEvidenceGatherer(
        config,
        provider=provider or RecordingProvider(),
        wikimedia=WikimediaSource(
            config, client=client or wikimedia_client(), sleep=lambda _: None
        ),
        sleep=lambda _: None,
    )


class TestGatherer:
    def test_a_lei_without_a_typed_description_is_described_from_wikipedia(self) -> None:
        provider = RecordingProvider()
        evidence = gatherer(provider=provider).gather(name="DEUTSCHE BANK AG", lei=LEI)
        assert evidence.has_description
        assert evidence.description.startswith("Deutsche Bank AG je největší německá banka")
        assert "Odvětví podle Wikidat" in evidence.description
        assert evidence.provenance is not None
        assert evidence.provenance.source == "WEB"
        assert evidence.provenance.detail == "web:wikipedia:cs"
        assert provider.queries == []  # the paid search is not needed

    def test_both_pages_are_cited(self) -> None:
        evidence = gatherer().gather(name="DEUTSCHE BANK AG", lei=LEI)
        assert [source.url for source in evidence.sources] == [
            "https://www.wikidata.org/wiki/Q66048",
            "https://cs.wikipedia.org/wiki/Deutsche_Bank",
        ]
        assert all(source.fetched for source in evidence.sources)

    def test_the_reviewer_is_told_to_check_the_match(self) -> None:
        evidence = gatherer().gather(name="DEUTSCHE BANK AG", lei=LEI)
        assert any("ověřte, že popisuje právě tohoto emitenta" in n for n in evidence.notes)

    def test_the_typed_name_stays_the_name(self) -> None:
        assert gatherer().gather(name="Deutsche Bank AG", lei=LEI).issuer_name == (
            "Deutsche Bank AG"
        )
        assert gatherer().gather(lei=LEI).issuer_name == "Deutsche Bank"

    def test_a_typed_description_stays_first_and_wikipedia_follows_it(self) -> None:
        """The tool looks at the web every time (30 Sept 2026); what MO typed still leads."""
        calls: list[httpx.Request] = []
        evidence = gatherer(wikimedia_client(calls=calls)).gather(
            name="Deutsche Bank AG", lei=LEI, description="Univerzální banka."
        )
        assert evidence.description.startswith(
            "Univerzální banka.\n\nPodle Wikipedie (cs): Deutsche Bank AG je největší"
        )
        assert calls  # Wikimedia was asked
        assert evidence.provenance is not None and evidence.provenance.detail == "user-supplied"
        assert evidence.notes[0] == "popis zadal uživatel, doplněn o Wikipedii"
        assert [source.url for source in evidence.sources][-1].startswith("https://cs.wikipedia")

    def test_a_typed_description_stands_alone_when_wikipedia_has_nothing(self) -> None:
        evidence = gatherer(wikimedia_client(search=WIKIDATA_EMPTY_SEARCH)).gather(
            name="Nordkap Funding B.V.", lei=LEI, description="Kaptivní financování skupiny."
        )
        assert evidence.description == "Kaptivní financování skupiny."
        assert evidence.notes[0] == "popis zadal uživatel"

    def test_no_item_falls_through_to_the_search_provider(self) -> None:
        provider = RecordingProvider()
        evidence = gatherer(wikimedia_client(search=WIKIDATA_EMPTY_SEARCH), provider).gather(
            name="BMW Finance N.V.", lei=LEI
        )
        assert provider.queries == ["BMW Finance N.V."]
        assert not evidence.has_description
        assert "Wikidata nemá položku s tímto LEI a Wikipedie nenašla článek podle názvu" in (
            evidence.notes
        )

    def test_a_name_match_is_flagged_as_such(self) -> None:
        client = wikimedia_client(
            search=WIKIDATA_EMPTY_SEARCH,
            wiki_search={"European Investment Bank": ["European Investment Bank"]},
        )
        evidence = gatherer(client).gather(name="European Investment Bank", lei=LEI)
        assert evidence.has_description
        assert any("podle shody názvu, ne identifikátoru" in n for n in evidence.notes)
        assert not any("podle LEI" in n for n in evidence.notes)

    def test_the_name_alone_can_bring_a_description(self) -> None:
        provider = RecordingProvider()
        client = wikimedia_client(
            wiki_search={"European Investment Bank": ["European Investment Bank"]}
        )
        evidence = gatherer(client, provider).gather(name="European Investment Bank")
        assert evidence.has_description
        assert provider.queries == []

    def test_an_outage_is_a_note_and_the_search_still_runs(self) -> None:
        provider = RecordingProvider()
        evidence = gatherer(wikimedia_client(wikidata_status=503), provider).gather(
            name="Deutsche Bank AG", lei=LEI
        )
        assert provider.queries == ["Deutsche Bank AG"]
        assert any(
            note.startswith("Wikidata: zdroj se nepodařilo dotázat") for note in evidence.notes
        )

    @pytest.mark.parametrize("switch", ["wikimedia_enabled", "web_enabled"])
    def test_either_switch_turns_it_off(self, switch: str) -> None:
        calls: list[httpx.Request] = []
        gatherer(wikimedia_client(calls=calls), **{switch: False}).gather(
            name="Deutsche Bank AG", lei=LEI
        )
        assert calls == []

    def test_without_a_lei_or_a_name_wikimedia_is_never_asked(self) -> None:
        calls: list[httpx.Request] = []
        gatherer(wikimedia_client(calls=calls)).gather(isin="DE0005140008")
        assert calls == []

    def test_without_a_lei_only_the_name_is_asked(self) -> None:
        calls: list[httpx.Request] = []
        evidence = gatherer(wikimedia_client(calls=calls)).gather(name="Deutsche Bank AG")
        assert calls and calls == wikipedia_searches(calls)  # no LEI: only the name search
        assert "Wikipedie nenašla článek podle názvu emitenta" in evidence.notes

    def test_the_description_is_capped_like_any_web_description(self) -> None:
        evidence = gatherer(web_max_description_chars=200).gather(lei=LEI)
        assert evidence.description is not None
        assert len(evidence.description) <= 201
