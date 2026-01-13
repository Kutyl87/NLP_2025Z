from __future__ import annotations
from core.orchestrator import Orchestrator as OrchSequential
from core.orchestrator_parallel import Orchestrator as OrchParallel
from utils.utils import ensure_dirs
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    ensure_dirs()

    # orch = OrchSequential()
    orch = OrchParallel()

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
