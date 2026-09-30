import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.briefings import (
    backtranslate_message,
    build_branch_facts,
    build_borrower_facts,
    check_backtranslation_violations,
    generate_branch_briefing,
    generate_outreach_message,
    is_grounded,
    load_final_portfolio,
    run_backtranslation_audit,
    template_branch_briefing,
    template_outreach_message,
)


class FakeClient:
    """Stands in for groq.Groq: returns the next canned response text on
    each call to chat.completions.create(), in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        text = self._responses[self.calls]
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


@pytest.fixture(scope="module")
def portfolio():
    return load_final_portfolio()


@pytest.fixture(scope="module")
def sample_branch_id(portfolio):
    counts = portfolio[portfolio["tier"] == "high"]["branch_id"].value_counts()
    return counts.index[0]  # branch with the most urgent borrowers - a rich fixture


@pytest.fixture(scope="module")
def sample_borrower_row(portfolio):
    return portfolio[portfolio["tier"] == "high"].iloc[0]


def test_build_branch_facts_counts_match_source_data(portfolio, sample_branch_id):
    facts = build_branch_facts(portfolio, sample_branch_id)
    branch_df = portfolio[portfolio["branch_id"] == sample_branch_id]

    assert facts["urgent_count"] == (branch_df["tier"] == "high").sum()
    assert facts["watch_count"] == (branch_df["tier"] == "medium").sum()
    assert facts["exposed_outstanding_inr"] == pytest.approx(
        branch_df.loc[branch_df["tier"] != "not_exposed", "outstanding"].sum()
    )
    assert len(facts["top_villages"]) <= 3
    # sorted descending by urgent_count
    counts = [v["urgent_count"] for v in facts["top_villages"]]
    assert counts == sorted(counts, reverse=True)


def test_build_branch_facts_includes_derived_values_the_model_tends_to_cite(portfolio, sample_branch_id):
    facts = build_branch_facts(portfolio, sample_branch_id)

    assert facts["exposed_outstanding_lakh"] == pytest.approx(facts["exposed_outstanding_inr"] / 100_000, abs=0.005)
    assert facts["exposed_outstanding_cr"] == pytest.approx(facts["exposed_outstanding_inr"] / 10_000_000, abs=0.005)
    assert facts["top_villages_urgent_total"] == sum(v["urgent_count"] for v in facts["top_villages"])


def test_build_branch_facts_top_occupations_are_real_and_sorted(portfolio, sample_branch_id):
    facts = build_branch_facts(portfolio, sample_branch_id)
    branch_df = portfolio[(portfolio["branch_id"] == sample_branch_id) & (portfolio["tier"] == "high")]

    assert len(facts["top_occupations"]) <= 3
    counts = [o["borrower_count"] for o in facts["top_occupations"]]
    assert counts == sorted(counts, reverse=True)
    assert sum(counts) <= len(branch_df)
    for entry in facts["top_occupations"]:
        assert entry["occupation"]  # a real (label-mapped) occupation string, not blank


def test_build_borrower_facts_has_placeholder_not_a_real_name(sample_borrower_row):
    facts = build_borrower_facts(sample_borrower_row)
    assert facts["first_name_placeholder"] == "[Borrower Name]"
    assert facts["occupation"]
    assert facts["village_id"] == sample_borrower_row["village_id"]
    assert facts["missed_emis"] == int(sample_borrower_row["missed_emis"])
    assert isinstance(facts["reasons"], list)


def test_is_grounded_accepts_only_numbers_present_in_facts():
    facts = {"urgent_count": 42, "nested": {"km": 50.0}, "reasons": ["missed 2 EMIs"]}
    assert is_grounded("We found 42 urgent cases within 50 km, 2 EMIs missed.", facts)
    assert not is_grounded("We found 99 urgent cases.", facts)


def test_templates_are_always_number_grounded(portfolio, sample_branch_id, sample_borrower_row):
    branch_facts = build_branch_facts(portfolio, sample_branch_id)
    assert is_grounded(template_branch_briefing(branch_facts), branch_facts)

    borrower_facts = build_borrower_facts(sample_borrower_row)
    assert is_grounded(template_outreach_message(borrower_facts), borrower_facts)


def test_generate_branch_briefing_accepts_grounded_llm_output(portfolio, sample_branch_id):
    facts = build_branch_facts(portfolio, sample_branch_id)
    good_text = f"Call the {facts['urgent_count']} urgent borrowers first."
    client = FakeClient([good_text])

    text, source = generate_branch_briefing(facts, client=client)

    assert source == "llm"
    assert text == good_text
    assert client.calls == 1


def test_generate_branch_briefing_retries_then_falls_back(portfolio, sample_branch_id):
    facts = build_branch_facts(portfolio, sample_branch_id)
    bogus_text = "Call the 999999 urgent borrowers first."  # not grounded in facts
    client = FakeClient([bogus_text, bogus_text])

    text, source = generate_branch_briefing(facts, client=client)

    assert source == "template"
    assert text == template_branch_briefing(facts)
    assert client.calls == 2  # one try + one retry, both rejected


def test_generate_outreach_message_accepts_grounded_llm_json(sample_borrower_row):
    facts = build_borrower_facts(sample_borrower_row)
    payload = json.dumps({"en": "Hello, we hope you are safe.", "hi": "नमस्ते", "or": "ନମସ୍କାର"})
    client = FakeClient([payload])

    messages, source = generate_outreach_message(facts, client=client)

    assert source == "llm"
    assert messages == {"en": "Hello, we hope you are safe.", "hi": "नमस्ते", "or": "ନମସ୍କାର"}


def test_generate_outreach_message_falls_back_on_malformed_json(sample_borrower_row):
    facts = build_borrower_facts(sample_borrower_row)
    client = FakeClient(["not valid json", "still not valid json"])

    messages, source = generate_outreach_message(facts, client=client)

    assert source == "template"
    assert messages == template_outreach_message(facts)


def test_backtranslate_message_returns_llm_text():
    client = FakeClient(["This is the English translation."])

    result = backtranslate_message("नमस्ते", "hi", client=client)

    assert result == "This is the English translation."
    assert client.calls == 1


def test_check_backtranslation_violations_flags_forbidden_language():
    cache = {
        "B1": {"hi": "We hope you are safe and well.", "or": "We will restructure your loan."},
        "B2": {"hi": "Please call us anytime.", "or": "Have a good day."},
    }

    violations = check_backtranslation_violations(cache)

    assert violations["hi"] == []
    assert [bid for bid, _ in violations["or"]] == ["B1"]


def test_run_backtranslation_audit_is_resumable(tmp_path, monkeypatch):
    import src.briefings as briefings_module

    outreach_path = tmp_path / "outreach.json"
    backtranslations_path = tmp_path / "outreach_backtranslations.json"
    outreach_path.write_text(
        json.dumps({"B1": {"messages": {"en": "Hi", "hi": "नमस्ते", "or": "ନମସ୍କାର"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(briefings_module, "OUTREACH_PATH", outreach_path)
    monkeypatch.setattr(briefings_module, "BACKTRANSLATIONS_PATH", backtranslations_path)

    client = FakeClient(["Hindi translation", "Odia translation"])
    cache = run_backtranslation_audit(client=client)

    assert cache == {"B1": {"hi": "Hindi translation", "or": "Odia translation"}}
    assert client.calls == 2
    assert json.loads(backtranslations_path.read_text(encoding="utf-8")) == cache

    # Rerun: both languages already cached, so this must make zero new LLM calls.
    client2 = FakeClient([])
    cache2 = run_backtranslation_audit(client=client2)

    assert cache2 == cache
    assert client2.calls == 0


def test_generate_outreach_message_falls_back_on_hallucinated_number(sample_borrower_row):
    facts = build_borrower_facts(sample_borrower_row)
    payload = json.dumps(
        {"en": "We can waive 50% of your loan.", "hi": "50%", "or": "50%"}
    )  # invented number + a promise the prompt forbids, but validation only checks numbers
    client = FakeClient([payload, payload])

    messages, source = generate_outreach_message(facts, client=client)

    assert source == "template"
    assert messages == template_outreach_message(facts)
