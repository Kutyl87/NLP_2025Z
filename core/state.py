from __future__ import annotations
from typing import TypedDict, Any, List


class GraphState(TypedDict, total=False):
    input: str

    # Analyst outputs
    analysis: str
    data_path: str
    viz_plan: dict[str, Any]

    # Visualizer outputs
    plots: List[str]
    plots_desc: List[str]

    # Report outputs
    report_path: str
    report_markdown: str

    # Critic / routing
    critic_decision: str
    critic_notes: str
    critic_llm_decision: str
    critic_llm_raw: str
    route_decision: str
    do_rerun: bool
    rerun_count: int
