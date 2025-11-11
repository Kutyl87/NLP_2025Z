from __future__ import annotations
from core.orchestrator import Orchestrator
from utils.utils import ensure_dirs

if __name__ == "__main__":
    ensure_dirs()

    orch = Orchestrator()  # default agents

    print("\n=== ASCII diagram ===")
    orch.print_ascii()
    orch.save_graph_png("pipeline_graph.png")
    print("Zapisano diagram do: pipeline_graph.png")

    print("\n=== RUN ===")
    result = orch.run(
        input_text=(
            "The WUT Formula Student team built a new active front-wing system "
            "to improve downforce and cornering stability at low speeds."
        )
    )

    print("\n=== WYNIK KOŃCOWY ===")
    for k, v in result.items():
        print(f"{k.upper()}:\n{v}\n")

    if "report_path" in result:
        print(f"Report saved to: {result['report_path']}")