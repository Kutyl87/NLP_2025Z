from __future__ import annotations
from core.orchestrator import Orchestrator
from utils.utils import ensure_dirs

if __name__ == "__main__":
    ensure_dirs()

    orch = Orchestrator()

    print("\n=== ASCII diagram ===")
    orch.print_ascii()
    orch.save_graph_png("pipeline_graph.png")
    print("Diagram saved to: pipeline_graph.png")

    print("\n=== RUN ===")
    result = orch.run(
        input_text=(
            "Analyze the dataset containing physicochemical properties of red wine samples. "
            "Perform data cleaning, exploratory analysis, and identify key variables influencing wine quality. "
            "Prepare visualizations to show correlations between alcohol, acidity, and quality, "
            "and generate a summary report with conclusions."
        )
    )

    print("\n=== FINAL RESULT ===")
    for k, v in result.items():
        print(f"{k.upper()}:\n{v}\n")

    if "report_path" in result:
        print(f"Report saved to: {result['report_path']}")