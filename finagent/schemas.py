from typing import List, Literal

from pydantic import BaseModel, Field

Sentiment = Literal["bullish", "neutral", "bearish"]


class MacroView(BaseModel):
    sentiment: Sentiment = Field(description="Overall news sentiment")
    score: float = Field(description="-1.0 very bearish to 1.0 very bullish")
    key_themes: List[str] = Field(description="2-4 short themes")
    evidence: List[str] = Field(description="2-4 shortened headlines supporting the call")
    confidence: float = Field(description="0.0 to 1.0")


class QuantView(BaseModel):
    trend: Literal["uptrend", "sideways", "downtrend"]
    momentum: Literal["overbought", "neutral", "oversold"]
    risk_level: Literal["low", "medium", "high"]
    bias: Sentiment = Field(description="Net technical bias")
    signals: List[str] = Field(description="2-5 short signals citing the provided numbers")
    summary: str = Field(description="2-3 sentence technical summary")
    confidence: float = Field(description="0.0 to 1.0")


class FinalReport(BaseModel):
    verdict: Literal["healthy", "watch", "at_risk"]
    overall_confidence: float = Field(description="0.0 to 1.0")
    suggested_action: Literal["hold", "accumulate", "reduce", "review"]
    summary: str = Field(description="3-4 sentence unified asset-health summary")
    conflict_explanation: str = Field(
        description="If the macro and quant views disagree, explain why and which you weigh more. Otherwise say they agree."
    )
    portfolio_impact: str = Field(
        description="What this means for the user's position (or say it is not held)"
    )
    risks: List[str] = Field(description="2-4 key risks")


    