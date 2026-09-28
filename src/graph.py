r"""
Graph structure as of Phase E:

        START
       /  |  \
  crypto stock forex   <- run in parallel, each writes its own verdict
       \  |  /
    fund_manager        <- waits for ALL three, decides allocation (real math)
        |
    risk_check           <- checks real volatility, writes RiskAssessment
        |
   (conditional edge)
     /        \
 risk_veto   (nothing - straight through)
     \        /
       logger            <- writes the full run to logs/decisions.jsonl
        |
       END
"""
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from src.schemas import AnalystVerdict, PortfolioDecision, RiskAssessment
from src.agents.crypto_analyst import crypto_analyst_node
from src.agents.stock_analyst import stock_analyst_node
from src.agents.forex_analyst import forex_analyst_node
from src.agents.fund_manager import fund_manager_node
from src.agents.risk_manager import risk_check_node, route_after_risk_check, risk_veto_node
from src.logging_utils import logger_node


class HedgeFundState(TypedDict):
    crypto_verdict: Optional[AnalystVerdict]
    stock_verdict: Optional[AnalystVerdict]
    forex_verdict: Optional[AnalystVerdict]
    portfolio_decision: Optional[PortfolioDecision]
    risk_assessment: Optional[RiskAssessment]


def build_graph():
    graph = StateGraph(HedgeFundState)

    graph.add_node("crypto_analyst", crypto_analyst_node)
    graph.add_node("stock_analyst", stock_analyst_node)
    graph.add_node("forex_analyst", forex_analyst_node)
    graph.add_node("fund_manager", fund_manager_node)
    graph.add_node("risk_check", risk_check_node)
    graph.add_node("risk_veto", risk_veto_node)
    graph.add_node("logger", logger_node)

    # Fan-out: all three analysts start from START and run concurrently -
    # LangGraph runs any nodes with no dependency on each other's output
    # in parallel automatically.
    graph.add_edge(START, "crypto_analyst")
    graph.add_edge(START, "stock_analyst")
    graph.add_edge(START, "forex_analyst")

    # Join: fund_manager has three incoming edges, so LangGraph will not
    # run it until all three analyst nodes have completed and written
    # their verdicts into state.
    graph.add_edge("crypto_analyst", "fund_manager")
    graph.add_edge("stock_analyst", "fund_manager")
    graph.add_edge("forex_analyst", "fund_manager")

    # fund_manager -> risk_check is a plain sequential edge: risk_check
    # NEEDS portfolio_decision to already exist in state before it runs.
    graph.add_edge("fund_manager", "risk_check")

    # Conditional edge: route_after_risk_check reads state["risk_assessment"]
    # (written by risk_check_node) and returns "veto" or "approve". This is
    # the exact same pattern as Module 1's check_size function - just with
    # a real financial condition instead of a toy number comparison.
    graph.add_conditional_edges(
        "risk_check",
        route_after_risk_check,
        {
            "veto": "risk_veto",
            "approve": "logger",
        },
    )

    # Both paths converge on logger before finishing - this guarantees
    # every run gets logged exactly once, regardless of which branch fired.
    graph.add_edge("risk_veto", "logger")
    graph.add_edge("logger", END)

    return graph.compile()