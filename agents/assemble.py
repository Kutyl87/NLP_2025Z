from typing import Any, Dict, Optional, Tuple
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import json
from datetime import datetime
from .base import Agent


class AssemblerAgent(Agent):

    def __init__(
        self,
        name: str = "Gemini-Assembler",
        templates_dir: str = "templates",
        template_name: str = "report.md.j2",
        out_dir: str = "data/output",
        out_name: str = "report.md",
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None
    ) -> None:
        super().__init__(name)

        self.templates_dir = templates_dir
        self.template_name = template_name
        self.out_dir = out_dir
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
        plots = raw_data.get("plots", [])
        rep_md = raw_data.get("rep_report_markdown") or ""
        vis_md = raw_data.get("vis_report_markdown") or ""

        prompt = f"""
        You are an expert Senior Data Reporter. Your task is to synthesize a final, stakeholder-ready markdown report by
        combining two draft markdown documents and any available analysis and plots.

        INPUTS:
        - Reporter draft markdown (REPORTER - Overview & Conclusion):
        {rep_md}

        - Visualizer draft markdown (VISUALIZER - Analysis Sections):
        {vis_md}

        - Raw analysis text:
        {analysis_text}

        - Available plot files:
        {plots}

        INSTRUCTIONS:

        1. **Executive Overview**:
           - Paste text from this section from REPORTER draft.

        2. **Structured Analysis (The Core)**:
            - Paste text from this section from VISUALIZER.

        3. **Final Synthesis & Recommendations**:
           - Write a strong **Conclusion** paragraph based on the REPORTER draft.

        4. **Handling Feedback**:
           - Ignore this for JSON generation. Set `change_log` to null in JSON. We will handle it programmatically.

        JSON STRUCTURE:
        {{
            "title": "A professional title",
            "overview": "The executive summary...",
            "sections": [
                {{
                    "heading": "Subheader for this insight (e.g. Alcohol vs Quality)",
                    "plot_path": "path/to/selected_plot.png",
                    "content": "Deep dive analysis specifically describing this plot..."
                }},
                {{
                    "heading": "Another insight...",
                    "plot_path": "path/to/another_plot.png",
                    "content": "Analysis..."
                }}
            ],
            "conclusion": "Final summary and recommendations text...",
            "change_log": null
        }}
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)

        except Exception as e:
            print(f"[{self.name}] LLM Error (fallback): {e}")
            sections = []
            for p in (plots or [])[:3]:
                sections.append({
                    "heading": "Analysis",
                    "plot_path": p,
                    "content": "See attached plot and reporter draft for details."
                })
            return {
                "title": "Assembled Report",
                "overview": (rep_md.splitlines()[0] if rep_md else "Combined overview."),
                "sections": sections,
                "conclusion": "See reporter and visualizer drafts for details.",
                "change_log": None
            }

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"[{self.name}] Generating structured report...")

        curated_content = self._curate_content_with_llm(kwargs)

        vis_notes = kwargs.get("vis_critic_notes")
        rep_notes = kwargs.get("rep_critic_notes")

        change_log_parts = []
        if vis_notes and len(str(vis_notes)) > 5:
             change_log_parts.append(f"#### Visualizer Branch Feedback\n{vis_notes}")

        if rep_notes and len(str(rep_notes)) > 5:
             change_log_parts.append(f"#### Reporter Branch Feedback\n{rep_notes}")

        manual_change_log = "\n\n".join(change_log_parts) if change_log_parts else None


        sections = curated_content.get("sections") or []
        used_plots = [s.get("plot_path") for s in sections if s.get("plot_path")]

        payload = {
            "title": curated_content.get("title", kwargs.get("title")),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "overview": curated_content.get("overview"),
            "sections": sections,
            "conclusion": curated_content.get("conclusion"),
            "change_log": manual_change_log,
            "critic_notes": manual_change_log,
            "critic_decision": "; ".join([d for d in (kwargs.get("vis_critic_decision"), kwargs.get("rep_critic_decision")) if d]),
            "plots": used_plots
        }

        template = self._env.get_template(self.template_name)
        md = template.render(**payload)
        out_path = os.path.join(self.out_dir, self.out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        return {"final_report_path": out_path, "final_report_markdown": md}
