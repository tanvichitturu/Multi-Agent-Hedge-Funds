"""
Central data contracts. Every agent reads/writes these shapes -
this is what lets the Fund Manager reason over analyst output
programmatically instead of parsing prose.
"""
from pydantic import BaseModel, Field
from typing import Literal


class AnalystVerdict(BaseModel):
    """Structured output every asset-class analyst must produce."""
    asset: str = Field(description="Ticker or asset class this verdict covers")
    signal: Literal["bullish", "bearish", "neutral"] = Field(
        description="Overall directional stance"
    )
    confidence: float = Field(ge=0, le=1, description="0 (no conviction) to 1 (high conviction)")
    reasoning: str = Field(description="1-3 sentence justification grounded in the data provided")


class PortfolioDecision(BaseModel):
    """Structured output the Fund Manager produces after reading all analysts."""
    crypto_weight: float = Field(ge=0, le=1, description="Fraction of capital allocated to crypto")
    stock_weight: float = Field(ge=0, le=1, description="Fraction of capital allocated to stocks")
    forex_weight: float = Field(ge=0, le=1, description="Fraction of capital allocated to forex")
    cash_weight: float = Field(ge=0, le=1, default=0.0, description="Fraction held as uninvested cash")
    rationale: str = Field(description="Why this allocation, referencing analyst signals")


class RiskAssessment(BaseModel):
    """Output of the volatility check that gates whether the veto fires (Phase D)."""
    triggered: bool = Field(description="True if the Extreme Market Conference trigger fired")
    reason: str = Field(description="Which asset(s) and move(s) caused the trigger, or why not")
