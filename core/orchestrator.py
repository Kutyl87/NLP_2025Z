from __future__ import annotations
from langgraph.graph import StateGraph, END

from agents.analyst import AnalystAgent
from agents.visualizer import VisualizationAgent
from agents.critic import CriticAgent
from agents.report import ReportAgent
from .state import GraphState


class Orchestrator:
    MAX_RERUNS = 1

    def __init__(
        self,
        analyst: AnalystAgent | None = None,
        visualizer: VisualizationAgent | None = None,
        critic: CriticAgent | None = None,
        reporter: ReportAgent | None = None,
        data_path: str = "data/input/winequality-red.csv",
    ) -> None:
        self.analyst = analyst or AnalystAgent(input_path=data_path)
        self.visualizer = visualizer or VisualizationAgent()
        self.critic = critic or CriticAgent()
        self.reporter = reporter or ReportAgent()
        self._app = self._build()

    # Node handlers
    def _node_analyst(self, state: GraphState) -> GraphState:
        return self.analyst.run(
            input=state["input"],
            data_path=state.get("data_path", self.analyst.input_path),
        )

    def _node_visualizer(self, state: GraphState) -> GraphState:
        res = self.visualizer.run(
            data_path=state["data_path"],
            viz_plan=state["viz_plan"],
        )
        return {"plots": res["plots"], "plots_desc": res.get("plots_desc", [])}

    def _node_report_draft(self, state: GraphState) -> GraphState:
        res = self.reporter.run(
            title="Measurement Data Report",
            overview="Auto-generated report from multi-agent pipeline.",
            analysis=state.get("analysis", ""),
            plots=state.get("plots", []),
            critic_decision=None,
            critic_notes=None,
        )
        return {
            "report_path": res["report_path"],
            "report_markdown": res.get("report_markdown", ""),
        }

    def _normalize_decision(self, res: dict) -> str:
        if "critic_llm_decision" in res:
            return str(res["critic_llm_decision"]).upper()
        if "critic_decision" in res:
            m = str(res["critic_decision"]).lower()
            return {"accept": "ACCEPT", "revise": "RERUN", "fail": "REJECT"}.get(
                m, "AMBIGUOUS"
            )
        return "AMBIGUOUS"

    def _node_critic(self, state: GraphState) -> GraphState:
        res = self.critic.run(
            report_path=state.get("report_path"),
            report_markdown=state.get("report_markdown"),
            analysis=state.get("analysis", ""),
            plots=state.get("plots", []),
        )
        decision_norm = self._normalize_decision(res)
        rerun_count = int(state.get("rerun_count", 0))
        do_rerun = decision_norm == "RERUN" and rerun_count < self.MAX_RERUNS
        return {
            **res,
            "route_decision": decision_norm,
            "do_rerun": do_rerun,
            "rerun_count": rerun_count + 1 if do_rerun else rerun_count,
        }

    def _node_report_final(self, state: GraphState) -> GraphState:
        res = self.reporter.run(
            title="Measurement Data Report",
            overview="Auto-generated report from multi-agent pipeline.",
            analysis=state.get("analysis", ""),
            plots=state.get("plots", []),
            critic_decision=state.get("critic_llm_decision")
            or state.get("critic_decision"),
            critic_notes=state.get("critic_notes") or state.get("critic_llm_raw"),
        )
        return {
            "report_path": res["report_path"],
            "report_markdown": res.get("report_markdown", ""),
        }

    def _route_after_critic(self, state: GraphState) -> str:
        return "analyst" if state.get("do_rerun") else "report_final"

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
            {"analyst": "analyst", "report_final": "report_final"},
        )
        wf.add_edge("report_final", END)
        return wf.compile()

    @property
    def app(self):
        return self._app

    def run(self, input_text: str, data_path: str | None = None) -> GraphState:
        initial: GraphState = {"input": input_text, "rerun_count": 0}
        initial["data_path"] = data_path or self.analyst.input_path
        return self._app.invoke(initial)

    def save_graph_png(self, path: str = "pipeline_graph.png") -> None:
        self._app.get_graph().draw_mermaid_png(output_file_path=path)

    def print_ascii(self) -> None:
        self._app.get_graph().print_ascii()
