import pytest

from finagent.agents.aggregator import CAPS, _agreement


def M(s):
    return {"sentiment": s}


def Q(s):
    return {"bias": s}


@pytest.mark.parametrize(
    "macro,quant,expected",
    [
        ("bullish", "bullish", "agree"),
        ("neutral", "neutral", "agree"),
        ("bearish", "bearish", "agree"),
        ("neutral", "bullish", "partial"),
        ("neutral", "bearish", "partial"),
        ("bullish", "neutral", "partial"),
        ("bearish", "neutral", "partial"),
        ("bullish", "bearish", "conflict"),
        ("bearish", "bullish", "conflict"),
    ],
)
def test_agreement_matrix(macro, quant, expected):
    assert _agreement(M(macro), Q(quant)) == expected


def test_missing_analyst_is_incomplete():
    assert _agreement(None, Q("bullish")) == "incomplete"
    assert _agreement(M("bullish"), None) == "incomplete"
    assert _agreement(None, None) == "incomplete"


def test_confidence_caps_are_ordered():
    assert CAPS["agree"] > CAPS["partial"] > CAPS["conflict"] > CAPS["incomplete"]