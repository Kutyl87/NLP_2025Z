from __future__ import annotations
from langgraph.graph import StateGraph, END

from agents.analyst import AnalystAgent
from agents.visualizer import VisualizationAgent, VisualizationParallelAgent
from agents.critic import CriticAgent, CriticRepAgent, CriticVisAgent
from agents.report import ReportAgent, ReportParallelAgent
from agents.assemble import AssemblerAgent
from .state import GraphState, GraphParallelState


class Orchestrator:
    MAX_RERUNS = 2

    def __init__(
            self,
            analyst: AnalystAgent | None = None,
            visualizer: VisualizationAgent | None = None,
            critic_vis: CriticAgent | None = None,
            critic_rep: CriticAgent | None = None,
            reporter: ReportAgent | None = None,
            assembler: ReportAgent | None = None,
            data_path: str = "data/input/winequality-red.csv",
    ) -> None:
        self.analyst = analyst or AnalystAgent(input_path=data_path)
        self.visualizer = visualizer or VisualizationParallelAgent()
        self.reporter = reporter or ReportParallelAgent()
        self.critic_vis = critic_vis or CriticVisAgent()
        self.critic_rep = critic_rep or CriticRepAgent()
        self.assembler = assembler or AssemblerAgent()
        self._app = self._build()

    def _node_analyst(self, state: GraphState | GraphParallelState) -> GraphState | GraphParallelState:
        return self.analyst.run(
            data_path=state.get("data_path", self.analyst.input_path),
        )

    def _node_visualizer(self, state: GraphState | GraphParallelState) -> GraphState | GraphParallelState:
        feedback = state.get("vis_critic_notes")
        print(f"--- [Orchestrator] Running Visualizer Branch (Feedback present: {bool(feedback)}) ---")

        res = self.visualizer.run(
            data_path=state["data_path"],
            viz_plan=state["viz_plan"],
            critic_notes=feedback,
            critic_decision=state.get("vis_critic_decision")
        )
        return {
            "plots": res["plots"],
            "plots_desc": res.get("plots_desc", []),
            "vis_report_path": res["report_path"],
            "vis_report_markdown": res["report_markdown"]
        }

    def _node_report_draft(self, state: GraphParallelState) -> GraphParallelState:
        feedback = state.get("rep_critic_notes")
        decision = state.get("rep_critic_decision")
        print(f"--- [Orchestrator] Running Reporter Branch (Feedback present: {bool(feedback)}) ---")

        res = self.reporter.run(
            title="Measurement Data Report",
            analysis=state.get("analysis", ""),
            plots=state.get("plots", []),
            critic_decision=decision,
            critic_notes=feedback,
        )
        return {
            "rep_report_path": res["report_path"],
            "rep_report_markdown": res["report_markdown"],
        }

    def _normalize_decision(self, res: dict, prefix: str) -> str:
        raw_decision = res.get(f"{prefix}_critic_llm_decision") or res.get(f"{prefix}_critic_decision", "")
        d = str(raw_decision).upper().strip()
        if "ACCEPT" in d: return "ACCEPT"
        if "RERUN" in d or "REGENERATE" in d: return "RERUN"
        if "REJECT" in d: return "REJECT"
        if "AMBIGUOUS" in d: return "AMBIGUOUS"
        return "AMBIGUOUS"

    def _node_critic_vis(self, state: GraphParallelState) -> GraphParallelState:
        res = self.critic_vis.run(
            report_path=state.get("vis_report_path"),
            report_markdown=state.get("vis_report_markdown"),
            analysis=state.get("analysis", ""),
            plots=state.get("plots", []),
        )
        decision_norm = self._normalize_decision(res, prefix="vis")
        rerun_count = int(state.get("vis_rerun_count", 0))
        notes = res.get("vis_critic_llm_feedback") or res.get("vis_critic_llm_raw") or "No details."

        needs_correction = decision_norm in ["RERUN", "REJECT", "AMBIGUOUS"]
        do_rerun = needs_correction and (rerun_count < self.MAX_RERUNS)

        print(f"--- [Orchestrator] VIS Critic: {decision_norm} (Attempt {rerun_count}) ---")

        return {
            **res,
            "vis_route_decision": decision_norm,
            "vis_do_rerun": do_rerun,
            "vis_rerun_count": rerun_count + 1 if do_rerun else rerun_count,
            "vis_critic_notes": notes,
            "vis_critic_decision": decision_norm
        }

    def _node_critic_rep(self, state: GraphParallelState) -> GraphParallelState:
        res = self.critic_rep.run(
            report_path=state.get("rep_report_path"),
            report_markdown=state.get("rep_report_markdown"),
            analysis=state.get("analysis", ""),
            plots=state.get("plots", []),
        )
        decision_norm = self._normalize_decision(res, prefix="rep")
        rerun_count = int(state.get("rep_rerun_count", 0))
        notes = res.get("rep_critic_llm_feedback") or res.get("rep_critic_llm_raw") or "No details."

        needs_correction = decision_norm in ["RERUN", "REJECT", "AMBIGUOUS"]
        do_rerun = needs_correction and (rerun_count < self.MAX_RERUNS)

        print(f"--- [Orchestrator] REP Critic: {decision_norm} (Attempt {rerun_count}) ---")

        return {
            **res,
            "rep_route_decision": decision_norm,
            "rep_do_rerun": do_rerun,
            "rep_rerun_count": rerun_count + 1 if do_rerun else rerun_count,
            "rep_critic_notes": notes,
            "rep_critic_decision": decision_norm
        }

    def _node_assembler(self, state: GraphParallelState) -> GraphParallelState:
        print(f"--- [Orchestrator] Assembling Final Report ---")
        res = self.assembler.run(
            title="Measurement Data Report - ASSEMBLED",
            overview="Auto-generated report from multi-agent pipeline.",
            analysis=state.get("analysis", ""),
            plots=state.get("plots", []),
            rep_report_markdown=state.get("rep_report_markdown"),
            vis_report_markdown=state.get("vis_report_markdown"),
            rep_report_path=state.get("rep_report_path"),
            vis_report_path=state.get("vis_report_path"),
            vis_critic_notes=state.get("vis_critic_notes"),
            rep_critic_notes=state.get("rep_critic_notes"),
            vis_critic_decision=state.get("vis_critic_decision"),
            rep_critic_decision=state.get("rep_critic_decision")
        )
        return {
            "final_report_path": res["final_report_path"],
            "final_report_markdown": res.get("final_report_markdown", ""),
            "report_path": res["final_report_path"] # For POC compatibility
        }

    def _route_vis(self, state: GraphParallelState) -> str:
        if state.get("vis_do_rerun"):
            return "visualizer"
        return "assembler"

    def _route_rep(self, state: GraphParallelState) -> str:
        if state.get("rep_do_rerun"):
            return "report_draft"
        return "assembler"

    def _build(self):
        wf = StateGraph(GraphParallelState)
        wf.set_entry_point("analyst")
        wf.add_node("analyst", self._node_analyst)
        wf.add_node("visualizer", self._node_visualizer)
        wf.add_node("report_draft", self._node_report_draft)
        wf.add_node("critic_vis", self._node_critic_vis)
        wf.add_node("critic_rep", self._node_critic_rep)
        wf.add_node("assembler", self._node_assembler)
        wf.add_edge("analyst", "visualizer")
        wf.add_edge("analyst", "report_draft")
        wf.add_edge("visualizer", "critic_vis")
        wf.add_edge("report_draft", "critic_rep")
        wf.add_conditional_edges(
            "critic_vis",
            self._route_vis,
            {
                "visualizer": "visualizer",
                "assembler": "assembler"
            },
        )
        wf.add_conditional_edges(
            "critic_rep",
            self._route_rep,
            {
                "report_draft": "report_draft",
                "assembler": "assembler"
            },
        )
        wf.add_edge("assembler", END)

        return wf.compile()

    @property
    def app(self):
        return self._app

    # def run(self, data_path: str | None = None) -> GraphState | GraphParallelState:
    #     initial: GraphState | GraphParallelState = {"rerun_count": 0}
    #     initial["data_path"] = data_path or self.analyst.input_path
    #     return self._app.invoke(initial)

    # def save_graph_png(self, path: str = "pipeline_graph.png") -> None:
    #     self._app.get_graph().draw_mermaid_png(output_file_path=path)

    # def print_ascii(self) -> None:
    #     self._app.get_graph().print_ascii()

    def run(self, data_path: str | None = None) -> GraphParallelState:
        initial: GraphParallelState = {
            "vis_rerun_count": 0,
            "rep_rerun_count": 0,
            "data_path": data_path or self.analyst.input_path
        }
        return self._app.invoke(initial)

    def save_graph_png(self, path: str = "pipeline_graph.png") -> None:
        try:
            self._app.get_graph().draw_mermaid_png(output_file_path=path)
        except Exception as e:
            print(f"Could not save graph image (requires graphviz/mermaid): {e}")

    def print_ascii(self) -> None:
        try:
            self._app.get_graph().print_ascii()
        except Exception as e:
            print(f"Could not print ascii graph: {e}")