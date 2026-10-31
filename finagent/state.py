import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    ticker: str
    run_id: str
    hitl: bool                      # ask a human when analysts conflict?
    holding: Optional[Dict[str, Any]]
    macro: Optional[Dict[str, Any]]
    quant: Optional[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]]
    candles: Optional[List[Dict[str, Any]]]
    human_guidance: Optional[str]   # weigh_news | weigh_technicals | neither
    human_note: str
    final: Optional[Dict[str, Any]]
    # Reducers: parallel nodes can both append without overwriting each other
    errors: Annotated[List[str], operator.add]
    warnings: Annotated[List[str], operator.add]