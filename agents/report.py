from __future__ import annotations
import os
import json
from datetime import datetime
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from .base import Agent


class ReportAgent(Agent):
    def __init__(
            self,
            name: str = "Gemini-Reporter",
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
           - Write a compelling "Big Picture" summary for stakeholders.

        2. **Structured Analysis (The Core)**:
           - Instead of one big text block, organize the report into **logical sections**.
           - For each key insight, select **ONE** relevant plot.
           - **Structure**: The plot comes first, followed by the deep-dive analysis of that specific plot.
           - **Selection Limit**: Create MAXIMUM 3-5 sections (one plot per section). Choose only the most impactful plots.
           - **Connect dots**: Explain trends/anomalies visible in the specific image selected for that section.

        3. **Final Synthesis & Recommendations**:
           - Write a strong **Conclusion** paragraph.
           - Synthesize findings from the plots and raw analysis.
           - Provide concrete recommendations based on data.

        4. **Handling Feedback**:
           - If 'Context' indicates CRITIC FEEDBACK, generate a `change_log`.
           - Explain how you fixed the issues (e.g., "Reviewer requested X, so I added section Y").
           - If no feedback, `change_log` is null.

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
                "overview": "Overview failed.",
                "sections": sections,
                "change_log": None
            }

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"[{self.name}] Generating structured report...")
        curated_content = self._curate_content_with_llm(kwargs)
        c_notes = kwargs.get("critic_notes")
        if not c_notes: c_notes = ""
        used_plots = [s.get("plot_path") for s in curated_content.get("sections", []) if s.get("plot_path")]
        payload = {
            "title": curated_content.get("title", kwargs.get("title")),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "overview": curated_content.get("overview"),
            "sections": curated_content.get("sections", []),
            "conclusion": curated_content.get("conclusion"),
            "change_log": curated_content.get("change_log"),
            "critic_notes": c_notes,
            "critic_decision": kwargs.get("critic_decision"),
            "plots": used_plots
        }
        template = self._env.get_template(self.template_name)
        md = template.render(**payload)
        out_path = os.path.join(self.out_dir, self.out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        return {"report_path": out_path, "report_markdown": md}



class ReportParallelAgent(ReportAgent):
    def __init__(
            self,
            name: str = "Gemini-Reporter",
            templates_dir: str = "templates",
            template_name: str = "report.md.j2",
            out_dir: str = "data/output",
            out_name: str = "report-rep.md",
            model_name: str = "gemini-2.5-flash",
            api_key: Optional[str] = None
    ) -> None:
        super().__init__(name, templates_dir, template_name, out_dir, out_name, model_name, api_key)

    def _curate_content_with_llm(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        analysis_text = raw_data.get("analysis", "No raw analysis provided.")
        available_plots = raw_data.get("plots", [])
        critic_feedback = raw_data.get("critic_notes")

        feedback_context = f"CRITIC FEEDBACK: {critic_feedback}" if critic_feedback else "NO PRIOR FEEDBACK."

        prompt = f"""
        You are an expert Senior Data Reporter. Your task is to generate the Overview and Conclusion for a report.

        INPUT DATA:
        1. Raw Analysis: {analysis_text}
        2. Available Plot Files: {available_plots}
        3. Context/Feedback: {feedback_context}

        INSTRUCTIONS:

        1. **Executive Overview**:
           - Write a compelling "Big Picture" summary for stakeholders.

        2. **Structured Analysis (The Core)**:
           - SKIP. Another agent is doing this.

        3. **Final Synthesis & Recommendations**:
           - Write a strong **Conclusion** paragraph.
           - Synthesize findings from the plots and raw analysis.
           - Provide concrete recommendations based on data.

        4. **Handling Feedback**:
           - If 'Context' indicates CRITIC FEEDBACK, generate a `change_log`.

        JSON STRUCTURE:
        {{
            "title": "Measurement Data Report",
            "overview": "The executive summary...",
            "sections": null,
            "conclusion": "Final summary and recommendations text...",
            "change_log": "Explanation of fixes or null"
        }}
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)

        except Exception as e:
            print(f"[{self.name}] LLM Error (fallback): {e}")
            return {
                "title": raw_data.get("title", "Report (Fallback)"),
                "overview": "Overview failed.",
                "sections": None,
                "conclusion": "Conclusion failed.",
                "change_log": None
            }

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"[{self.name}] Generating structured report (Overview/Conclusion)...")
        curated_content = self._curate_content_with_llm(kwargs)

        c_notes = kwargs.get("critic_notes")
        if not c_notes: c_notes = ""

        sections_list = curated_content.get("sections") or []
        used_plots = [s.get("plot_path") for s in sections_list if s.get("plot_path")]

        payload = {
            "title": curated_content.get("title", kwargs.get("title")),
            "generated_at": "",
            "overview": curated_content.get("overview"),
            "sections": sections_list,
            "conclusion": curated_content.get("conclusion"),
            "change_log": curated_content.get("change_log"),
            "critic_notes": c_notes,
            "critic_decision": kwargs.get("critic_decision"),
            "plots": used_plots
        }
        template = self._env.get_template(self.template_name)
        md = template.render(**payload)
        out_path = os.path.join(self.out_dir, self.out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        return {"report_path": out_path, "report_markdown": md}
