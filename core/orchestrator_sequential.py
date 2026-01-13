from __future__ import annotations
from langgraph.graph import StateGraph, END

from agents.analyst import AnalystAgent
from agents.visualizer import VisualizationAgent
from agents.critic import CriticAgent
from agents.report import ReportAgent
from core.orchestrator import Orchestrator
from .state import GraphState


class OrchestratorSequential(Orchestrator):
    MAX_RERUNS = 2

    def __init__(
            self,
            analyst: AnalystAgent,
            visualizer: VisualizationAgent,
            critic: CriticAgent,
            reporter: ReportAgent,
    ) -> None:
        super().__init__(analyst, visualizer, critic, reporter)

    def _node_visualizer(self, state: GraphState) -> GraphState:
        res = self.visualizer.run(
            data_path=state["data_path"],
            viz_plan=state["viz_plan"],
        )
        return {"plots": res["plots"], "plots_desc": res.get("plots_desc", [])}

    def _node_report_draft(self, state: GraphState) -> GraphState:
        previous_notes = state.get("critic_notes")
        previous_decision = state.get("critic_decision")
        print(f"--- [Orchestrator] Drafting Report (Has feedback: {bool(previous_notes)}) ---")
        res = self.reporter.run(
            title="Measurement Data Report",
            overview="Auto-generated report from multi-agent pipeline.",
            analysis=state.get("analysis", ""),
            plots=state.get("plots", []),
            critic_decision=previous_decision,
            critic_notes=previous_notes,
        )
        return {
            "report_path": res["report_path"],
            "report_markdown": res.get("report_markdown", ""),
        }

    def _node_critic(self, state: GraphState) -> GraphState:
        res = self.critic.run(
            report_path=state.get("report_path"),
            report_markdown=state.get("report_markdown"),
            analysis=state.get("analysis", ""),
            plots=state.get("plots", []),
        )
        decision_norm = self._normalize_decision(res)
        rerun_count = int(state.get("rerun_count", 0))
        notes = res.get("critic_llm_feedback") or res.get("critic_llm_raw") or "No details."
        needs_correction = decision_norm in ["RERUN", "REJECT", "AMBIGUOUS"]
        do_rerun = needs_correction and (rerun_count < self.MAX_RERUNS)
        print(f"--- [Orchestrator] Critic Decision: {decision_norm} (Raw: {res.get('critic_llm_decision')}) ---")
        return {
            **res,
            "route_decision": decision_norm,
            "do_rerun": do_rerun,
            "rerun_count": rerun_count + 1 if do_rerun else rerun_count,
            "critic_notes": notes,
            "critic_decision": decision_norm
        }

    def _node_report_final(self, state: GraphState) -> GraphState:
        decision = state.get("critic_decision")
        if not decision:
            decision = state.get("critic_llm_decision", "ACCEPT")
        notes = state.get("critic_notes")
        print(f"--- [Orchestrator] Finalizing Report with Status: {decision} ---")
        res = self.reporter.run(
            title="Measurement Data Report - FINAL",
            overview="Final verified version.",
            analysis=state.get("analysis", ""),
            plots=state.get("plots", []),
            critic_decision=decision,
            critic_notes=notes,
            out_name="report_final.md"
        )
        return {
            "report_path": res["report_path"],
            "report_markdown": res.get("report_markdown", ""),
        }

    def _route_after_critic(self, state: GraphState) -> str:
        if state.get("do_rerun"):
            return "report_draft"
        return "report_final"

    def run(self, data_path: str | None = None) -> GraphState:
        initial: GraphState = {"rerun_count": 0}
        initial["data_path"] = data_path or self.analyst.input_path
        return self._app.invoke(initial)

    def _build(self):
        wf = StateGraph(GraphState)
        wf.add_node("analyst", self._node_analyst)
        wf.add_node("visualizer", self._node_visualizer)
        wf.add_node("report_draft", self._node_report_draft)
        wf.add_node("critic", self._node_critic)
        wf.add_node("report_final", self._node_report_final)
        wf.set_entry_point("analyst")
        wf.add_edge("analyst", "visualizer")
        wf.add_edge("visualizer", "report_draft")
        wf.add_edge("report_draft", "critic")
        wf.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {
                "report_draft": "report_draft",
                "report_final": "report_final"
            },
        )
        wf.add_edge("report_final", END)
        return wf.compile()
