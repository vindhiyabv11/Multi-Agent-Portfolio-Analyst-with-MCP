import asyncio

import pytest

from finagent.config import OPENAI_API_KEY


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
def test_graph_end_to_end():
    from finagent.graph import run_analysis

    state = asyncio.run(run_analysis("TCS.NS"))
    final = state.get("final")
    assert final is not None
    assert final["verdict"] in {"healthy", "watch", "at_risk"}
    assert 0 <= final["overall_confidence"] <= 1