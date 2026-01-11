from typing import Any, Dict, Optional, Tuple
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from .base import Agent


class CriticAgent(Agent):
    LABELS = ("ACCEPT", "REJECT", "RERUN", "AMBIGUOUS")

    def __init__(
            self,
            name: str = "Gemini-Critic",
            model_name: str = "gemini-2.5-flash",
            max_report_chars: int = 30000,
    ) -> None:
        super().__init__(name)
        self.name = name
        self.max_report_chars = max_report_chars
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            max_retries=2,
            temperature=0.0
        )

    # -------- helpers --------
    def _read_report(
            self, report_path: Optional[str], report_markdown: Optional[str]
    ) -> str:
        if report_markdown and report_markdown.strip():
            return report_markdown
        if not report_path:
            raise ValueError("Critic requires 'report_path' or 'report_markdown'.")
        if not os.path.isfile(report_path):
            raise FileNotFoundError(f"Report file not found: {report_path}")
        with open(report_path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_prompt(self, md: str) -> str:
        md_short = md[: self.max_report_chars]
        return (
            "You are a strict quality reviewer of analytical reports.\n"
            "Read the following Markdown report and provide a decision along with constructive feedback.\n\n"
            "POSSIBLE DECISIONS:\n"
            "- ACCEPT: structure is complete, content is coherent, formatting is correct.\n"
            "- REJECT: report is broken, missing sections, missing images, or logically inconsistent.\n"
            "- RERUN: report looks okay but specific data/plots seem wrong and need regeneration.\n"
            "- AMBIGUOUS: cannot determine from the provided text.\n\n"
            "OUTPUT FORMAT:\n"
            "You must strictly follow this format:\n"
            "DECISION: [One word from the list above]\n"
            "FEEDBACK: [Short explanation of why you made this decision. If REJECT, list specifically what is missing.]\n\n"
            "Report Content:\n"
            f"{md_short}\n\n"
            "Your Assessment:"
        )

    def _parse_response(self, text: str) -> Tuple[str, str]:
        text = text.strip()
        decision = "AMBIGUOUS"
        feedback = text
        pattern = r"DECISION:\s*([A-Z]+)\s*\n?FEEDBACK:\s*(.*)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            raw_decision = match.group(1).upper().strip()
            feedback_text = match.group(2).strip()
            clean_decision = re.sub(r"[^A-Z]", "", raw_decision)
            aliases = {
                "OK": "ACCEPT", "YES": "ACCEPT", "NO": "REJECT",
                "REGENERATE": "RERUN", "RE-RUN": "RERUN"
            }
            if clean_decision in self.LABELS:
                decision = clean_decision
            elif clean_decision in aliases:
                decision = aliases[clean_decision]
            feedback = feedback_text
        else:
            first_line = text.split('\n')[0].upper()
            for label in self.LABELS:
                if label in first_line:
                    decision = label
                    feedback = text.replace(label, "", 1).strip()
                    break

        return decision, feedback

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        report_path = kwargs.get("report_path")
        report_markdown = kwargs.get("report_markdown")

        md = self._read_report(report_path, report_markdown)
        prompt_content = self._build_prompt(md)
        messages = [HumanMessage(content=prompt_content)]
        response = self.llm.invoke(messages)
        raw_content = response.content
        decision, feedback = self._parse_response(raw_content)
        return {
            "critic_llm_decision": decision,
            "critic_llm_feedback": feedback,
            "critic_llm_raw": raw_content,
        }