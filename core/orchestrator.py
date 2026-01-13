from __future__ import annotations
from langgraph.graph import StateGraph, END

from agents.analyst import AnalystAgent
from agents.visualizer import VisualizationAgent
from agents.critic import CriticAgent
from agents.report import ReportAgent
from .state import GraphState
from abc import ABC, abstractmethod


class Orchestrator(ABC):
    MAX_RERUNS = 2

    def __init__(
            self,
            analyst: AnalystAgent,
            visualizer: VisualizationAgent,
            critic: CriticAgent,
            reporter: ReportAgent,
    ) -> None:
        self.analyst = analyst
        self.visualizer = visualizer
        self.critic = critic
        self.reporter = reporter
        self._app = self._build()

    def _node_analyst(self, state: GraphState) -> GraphState:
        return self.analyst.run(
            data_path=state.get("data_path", self.analyst.input_path),
        )

    @abstractmethod
    def _node_report_draft(self, state: GraphState) -> GraphState:
        pass

    def _normalize_decision(self, res: dict, prefix: str="") -> str:
        raw_decision = res.get(f"{prefix}critic_llm_decision") or res.get(f"{prefix}critic_decision", "")
        d = str(raw_decision).upper().strip()
        if "ACCEPT" in d: return "ACCEPT"
        if "RERUN" in d or "REGENERATE" in d: return "RERUN"
        if "REJECT" in d: return "REJECT"
        if "AMBIGUOUS" in d: return "AMBIGUOUS"
        return "AMBIGUOUS"

    @abstractmethod
    def _node_visualizer(self, state: GraphState) -> GraphState:
        pass

    @abstractmethod
    def _build(self):
        pass

    @property
    def app(self):
        return self._app

    @abstractmethod
    def run(self, data_path: str | None = None) -> GraphState:
        pass

    def save_graph_png(self, path: str = "pipeline_graph.png") -> None:
        self._app.get_graph().draw_mermaid_png(output_file_path=path)

    def print_ascii(self) -> None:
        self._app.get_graph().print_ascii()
