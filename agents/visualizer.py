import os

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from typing import List, Any, Dict, Optional
from .base import Agent
import json
from langchain_core.messages import HumanMessage

from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain_google_genai import ChatGoogleGenerativeAI


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
            raise FileNotFoundError(
                f"Visualizer expected cleaned file at '{data_path}'"
            )

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


class VisualizationParallelAgent(VisualizationAgent):
    def __init__(
        self,
        name: str = "Gemini-Visualizer",
        out_plt_dir: str = "data/output/plots",
        templates_dir: str = "templates",
        template_name: str = "report.md.j2",
        out_dir: str = "data/output",
        out_name: str = "report-vis.md",
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None
    ) -> None:
        super().__init__(name, out_dir=out_plt_dir)

        self.templates_dir = templates_dir
        self.template_name = template_name

        self.report_out_dir = out_dir
        self.out_name = out_name

        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.3,
        )

        self._env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=select_autoescape(enabled_extensions=(".html", ".xml")),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _curate_content_with_llm(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        analysis_text = raw_data.get("analysis", "No raw analysis provided.")
        available_plots = raw_data.get("plots", [])
        critic_feedback = raw_data.get("critic_notes")

        feedback_context = f"CRITIC FEEDBACK: {critic_feedback}" if critic_feedback else "NO PRIOR FEEDBACK."

        prompt = f"""
        You are an expert Senior Data Reporter. Your task is to generate a structured markdown report where visual evidence drives the narrative.

        INPUT DATA:
        1. Raw Analysis: {analysis_text}
        2. Available Plot Files: {available_plots}
        3. Context/Feedback: {feedback_context}

        INSTRUCTIONS:

        1. **Executive Overview**:
           - SKIP. Another agent is doing this.

        2. **Structured Analysis (The Core)**:
           - Organize the report into **logical sections**.
           - For each key insight, select **ONE** relevant plot.
           - **Structure**: The plot comes first, followed by the deep-dive analysis of that specific plot.
           - **Selection Limit**: Create MAXIMUM 3-5 sections (one plot per section). Choose only the most impactful plots.

        3. **Final Synthesis & Recommendations**:
           - SKIP. Another agent is doing this.

        4. **Handling Feedback**:
           - If 'Context' indicates CRITIC FEEDBACK, generate a `change_log`.
           - Explain how you fixed the issues (e.g., "Reviewer requested X, so I added section Y").
           - If no feedback, `change_log` is null.

        JSON STRUCTURE:
        {{
            "title": "Measurement Data Report",
            "overview": null,
            "sections": [
                {{
                    "heading": "Subheader for this insight...",
                    "plot_path": "path/to/selected_plot.png",
                    "content": "Deep dive analysis specifically describing this plot..."
                }}
            ],
            "conclusion": null,
            "change_log": "Explanation of fixes or null"
        }}
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)

        except Exception as e:
            print(f"[{self.name}] LLM Error (fallback): {e}")
            fallback_plots = raw_data.get("plots", [])[:3]
            sections = []
            for p in fallback_plots:
                sections.append({
                    "heading": "Analysis (Fallback)",
                    "plot_path": p,
                    "content": "Automated description unavailable due to LLM error."
                })

            return {
                "title": raw_data.get("title", "Report (Fallback)"),
                "overview": None,
                "sections": sections,
                "change_log": None
            }

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        plots = []
        if "plots" in kwargs and kwargs["plots"]:
            plots = kwargs["plots"]
        else:
            plot_res = super().run(**kwargs)
            plots = plot_res["plots"]

        print(f"[{self.name}] Generating structured analysis section...")
        curated_content = self._curate_content_with_llm({**kwargs, "plots": plots})

        c_notes = kwargs.get("critic_notes") or ""
        used_plots = [s.get("plot_path") for s in curated_content.get("sections", []) if s.get("plot_path")]

        payload = {
            "title": curated_content.get("title", "Analysis Report"),
            "generated_at": "",
            "overview": None,
            "sections": curated_content.get("sections", []),
            "conclusion": None,
            "change_log": curated_content.get("change_log"),
            "critic_notes": c_notes,
            "critic_decision": kwargs.get("critic_decision"),
            "plots": used_plots
        }

        template = self._env.get_template(self.template_name)
        md = template.render(**payload)

        os.makedirs(self.report_out_dir, exist_ok=True)
        out_path = os.path.join(self.report_out_dir, self.out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)

        return {
            "report_path": out_path,
            "report_markdown": md,
            "plots": plots
        }
