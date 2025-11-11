import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Any, Dict
from .base import Agent


class VisualizationAgent(Agent):
    MIN_COLS_FOR_HEATMAP = 3

    def __init__(
        self,
        name: str = "Visualizer",
        out_dir: str = "data/output/plots",
    ) -> None:
        super().__init__(name)
        self.out_dir = out_dir

    def _ensure_out(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)

    def _plot_hist(self, df: pd.DataFrame, col: str) -> str:
        self._ensure_out()
        plt.figure()
        df[col].plot(kind="hist", bins=20)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("count")
        out = os.path.join(self.out_dir, f"hist_{col}.png")
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        return out

    def _plot_scatter(self, df: pd.DataFrame, a: str, b: str) -> str:
        self._ensure_out()
        plt.figure()
        plt.scatter(df[a], df[b], s=18)
        plt.title(f"{a} vs {b}")
        plt.xlabel(a)
        plt.ylabel(b)
        out = os.path.join(self.out_dir, f"scatter_{a}_vs_{b}.png")
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        return out

    def _plot_corr_heatmap(self, df: pd.DataFrame, num_cols: List[str]) -> str | None:
        if len(num_cols) < self.MIN_COLS_FOR_HEATMAP:
            return None
        corr = df[num_cols].corr()
        self._ensure_out()
        plt.figure()
        im = plt.imshow(corr, interpolation="nearest")
        plt.colorbar(im)
        plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha="right")
        plt.yticks(range(len(num_cols)), num_cols)
        plt.title("Correlation heatmap")
        plt.tight_layout()
        out = os.path.join(self.out_dir, "corr_heatmap.png")
        plt.savefig(out)
        plt.close()
        return out

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        data_path: str = kwargs["data_path"]
        viz_plan: Dict[str, Any] = kwargs["viz_plan"]

        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Visualizer expected cleaned file at '{data_path}'")

        df = pd.read_csv(data_path)
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()

        plots: List[str] = []

        for col in viz_plan.get("hists", []):
            if col in df.columns:
                plots.append(self._plot_hist(df, col))

        for pair in viz_plan.get("pairs", []):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            a, b = pair
            if a in df.columns and b in df.columns:
                plots.append(self._plot_scatter(df, a, b))

        if viz_plan.get("heatmap", False):
            path = self._plot_corr_heatmap(df, num_cols)
            if path:
                plots.append(path)

        return {"plots": plots}
