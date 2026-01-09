from __future__ import annotations
import os
from utils import utils
from datetime import datetime
from typing import Any, Dict
from jinja2 import Environment, FileSystemLoader, select_autoescape
from .base import Agent



class ReportAgent(Agent):
    def __init__(
        self,
        name: str = "Report",
        templates_dir: str = "templates",
        template_name: str = "report.md.j2",
        out_dir: str = "data/output",
        out_name: str = "report.md",
    ) -> None:
        super().__init__(name)
        self.templates_dir = templates_dir
        self.template_name = template_name
        self.out_dir = out_dir
        self.out_name = utils.add_timestamp_suffix(filename=out_name)
        self._env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=select_autoescape(enabled_extensions=(".html", ".xml")),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        os.makedirs(self.out_dir, exist_ok=True)
        template = self._env.get_template(self.template_name)
        payload = {
            "title": kwargs.get("title", "Data Analysis Report"),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "overview": kwargs.get(
                "overview", "This report summarizes the analysis and visualizations."
            ),
            "analysis": kwargs.get("analysis", "(no analysis)"),
            "plots": kwargs.get("plots", []),
            "critic_decision": kwargs.get("critic_decision"),
            "critic_notes": kwargs.get("critic_notes"),
        }
        md = template.render(**payload)
        out_path = os.path.join(self.out_dir, self.out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)

        return {"report_path": out_path, "report_markdown": md}
