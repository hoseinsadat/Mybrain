from mybrain.providers.llm import _append_markers
from mybrain.providers.util import parse_json_text
from mybrain.retrieval.planner import plan_query


class StubLLM:
    provider = "stub"
    model = "stub"

    def __init__(self, data):
        self.data = data

    def complete_json(self, system, user, max_tokens=4096):
        if isinstance(self.data, Exception):
            raise self.data
        return self.data


def test_parse_json_text_tolerates_fences_and_prose():
    assert parse_json_text('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_text('Sure! {"a": {"b": [1,2]}} done') == {"a": {"b": [1, 2]}}


def test_append_markers_keeps_trailing_whitespace():
    assert _append_markers("Claim.\n\n", [1, 2]) == "Claim.[1][2]\n\n"
    assert _append_markers("Claim.", []) == "Claim."


def test_plan_query_dedupes_and_limits():
    llm = StubLLM(
        {
            "language": "en",
            "intent": "relationship",
            "entities": [{"name": "Ali Rezaei", "type": "person", "aliases": ["علی رضایی"]}],
            "queries": [
                {"text": "who is CFO of Pars"},
                {"text": "WHO IS CFO OF PARS"},
                "مدیر مالی پارس",
                {"text": "q3"},
                {"text": "q4"},
                {"text": "q5"},
            ],
            "keywords": ["40m"],
            "time": {"from": "2023-01-01", "to": None},
        }
    )
    plan = plan_query(llm, "Who is CFO of Pars?")
    assert plan.queries[0] == "Who is CFO of Pars?"
    assert len(plan.queries) == 5  # original + 4
    assert plan.entities[0].aliases == ["علی رضایی"]
    assert plan.time_from == "2023-01-01" and plan.keywords == ["40m"]


def test_plan_query_falls_back_on_error():
    plan = plan_query(StubLLM(RuntimeError("boom")), "question?")
    assert plan.queries == ["question?"] and plan.error == "boom"
    assert plan_query(None, "q").queries == ["q"]
