from __future__ import annotations

import argparse

from agents.analyst import AnalystAgent
from agents.assemble import AssemblerAgent
from agents.critic import CriticAgent, CriticRepAgent, CriticVisAgent
from agents.visualizer import VisualizationAgent, VisualizationParallelAgent
from agents.report import ReportAgent, ReportParallelAgent
from core.orchestrator_sequential import OrchestratorSequential
from core.orchestrator_parallel import ParallelOrchestrator
from utils.utils import ensure_dirs
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uruchomienie Orkiestratora Systemu NLP")
    ensure_dirs()
    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['seq', 'par'],
        default='seq',
        help="Tryb działania: 'seq' (sekwencyjny) lub 'par' (równoległy). Domyślnie: seq"
    )
    args = parser.parse_args()
    if args.mode == 'seq':
        print("Hello")
        orch = OrchestratorSequential(
            analyst=AnalystAgent(),
            visualizer=VisualizationAgent(),
            critic=CriticAgent(),
            reporter=ReportAgent()
        )
    elif args.mode == 'par':
        orch = ParallelOrchestrator(
            analyst=AnalystAgent(),
            visualizer=VisualizationParallelAgent(),
            critic_vis=CriticVisAgent(),
            critic_rep=CriticRepAgent(),
            reporter=ReportParallelAgent(),
            assembler=AssemblerAgent()
        )

    print("\n=== ASCII diagram ===")
    orch.print_ascii()
    orch.save_graph_png("pipeline_graph.png")
    print("Diagram saved to: pipeline_graph.png")

    print("\n=== RUN ===")
    # result = orch.run()
    # result = orch.run(data_path="data/input/sales_data.csv")
    result = orch.run(data_path="data/input/winequality-red.csv")

    print("\n=== FINAL RESULT ===")
    for k, v in result.items():
        print(f"{k.upper()}:\n{v}\n")

    if "report_path" in result:
        print(f"Report saved to: {result['report_path']}")
