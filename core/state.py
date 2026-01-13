from __future__ import annotations
from typing import TypedDict, Any, List


class GraphState(TypedDict, total=False):

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


class GraphParallelState(TypedDict, total=False):

    # Analyst outputs
    analysis: str
    data_path: str
    viz_plan: dict[str, Any]

    # Visualizer outputs
    plots: List[str]
    plots_desc: List[str]
    vis_report_path: str
    vis_report_markdown: str

    # Report outputs
    rep_report_path: str
    rep_report_markdown: str

    # Critic / routing
    vis_critic_decision: str
    vis_critic_notes: str
    vis_critic_llm_decision: str
    vis_critic_llm_raw: str
    vis_route_decision: str
    vis_do_rerun: bool
    vis_rerun_count: int

    # Critic / routing
    rep_critic_decision: str
    rep_critic_notes: str
    rep_critic_llm_decision: str
    rep_critic_llm_raw: str
    rep_route_decision: str
    rep_do_rerun: bool
    rep_rerun_count: int

    # Assembler outputs
    final_report_path: str
    final_report_markdown: str
