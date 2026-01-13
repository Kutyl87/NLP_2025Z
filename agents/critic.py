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
            response_prefix: str = ""  # New param to handle 'vis_' or 'rep_' prefixes automatically
    ) -> None:
        super().__init__(name)
        self.name = name
        self.max_report_chars = max_report_chars
        self.response_prefix = response_prefix
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            max_retries=2,
            temperature=0.0
        )

    def _read_report(
            self, report_path: Optional[str], report_markdown: Optional[str]
    ) -> str:
        if report_markdown and report_markdown.strip():
            return report_markdown
        if not report_path:
            raise ValueError(f"[{self.name}] Requires 'report_path' or 'report_markdown'.")
        if not os.path.isfile(report_path):
            raise FileNotFoundError(f"[{self.name}] Report file not found: {report_path}")
        with open(report_path, "r", encoding="utf-8") as f:
            return f.read()

    def _parse_response(self, text: str) -> Tuple[str, str]:
        """
        Robust parsing logic that handles Markdown formatting (**Decision**)
        and introductory text, fixing the 'AMBIGUOUS' issue.
        """
        text = text.strip()
        decision = "AMBIGUOUS"
        feedback = text

        # 1. Clean markdown tokens
        clean_text = text.replace("**", "").replace("##", "").replace("__", "")

        # 2. Relaxed Regex: Look for "DECISION" anywhere
        pattern = r"DECISION\s*:?\s*([A-Z]+)"
        match = re.search(pattern, clean_text, re.IGNORECASE)

        if match:
            raw_decision = match.group(1).upper().strip()
            aliases = {
                "OK": "ACCEPT", "YES": "ACCEPT", "NO": "REJECT",
                "REGENERATE": "RERUN", "RE-RUN": "RERUN"
            }

            if raw_decision in self.LABELS:
                decision = raw_decision
            elif raw_decision in aliases:
                decision = aliases[raw_decision]

            # 3. Extract Feedback separately
            feedback_pattern = r"FEEDBACK\s*:?\s*(.*)"
            fb_match = re.search(feedback_pattern, clean_text, re.DOTALL | re.IGNORECASE)
            if fb_match:
                feedback = fb_match.group(1).strip()
            else:
                # If explicit feedback tag missing, strip the decision line and use the rest
                feedback = re.sub(pattern, "", clean_text, count=1, flags=re.IGNORECASE).strip()
        else:
            # 4. Fallback: Scan intro for keywords
            intro = clean_text[:200].upper()
            for label in self.LABELS:
                if re.search(rf"\b{label}\b", intro):
                    decision = label
                    feedback = clean_text
                    break

        return decision, feedback

    def _build_prompt(self, md: str) -> str:
        """
        Default prompt for the generic Critic.
        Subclasses should override this for specific focus areas.
        """
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

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        report_path = kwargs.get("report_path")
        report_markdown = kwargs.get("report_markdown")

        md = self._read_report(report_path, report_markdown)
        prompt_content = self._build_prompt(md)

        messages = [HumanMessage(content=prompt_content)]
        response = self.llm.invoke(messages)
        raw_content = response.content

        decision, feedback = self._parse_response(raw_content)

        # Automatically prefix keys based on the agent type (e.g. 'vis_critic_llm_decision')
        p = self.response_prefix
        return {
            f"{p}critic_llm_decision": decision,
            f"{p}critic_llm_feedback": feedback,
            f"{p}critic_llm_raw": raw_content,
        }


class CriticVisAgent(CriticAgent):
    def __init__(
            self,
            name: str = "Gemini-Critic-Vis",
            model_name: str = "gemini-2.5-flash",
            max_report_chars: int = 30000,
    ) -> None:
        # Initialize with specific prefix for Parallel Orchestrator
        super().__init__(name, model_name, max_report_chars, response_prefix="vis_")

    def _build_prompt(self, md: str) -> str:
        md_short = md[: self.max_report_chars]
        return (
            "You are a strict quality reviewer of analytical reports.\n"
            "Read the following Markdown report and provide a decision along with constructive feedback.\n"
            "Focus only on section 2 -- **Structured Analysis (The Core)**. Another Critic is reviewing the executive summary and conclusion.\n\n"
            "POSSIBLE DECISIONS:\n"
            "- ACCEPT: structure (of this section) is complete, content is coherent, formatting is correct.\n"
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


class CriticRepAgent(CriticAgent):
    def __init__(
            self,
            name: str = "Gemini-Critic-Rep",
            model_name: str = "gemini-2.5-flash",
            max_report_chars: int = 30000,
    ) -> None:
        # Initialize with specific prefix for Parallel Orchestrator
        super().__init__(name, model_name, max_report_chars, response_prefix="rep_")

    def _build_prompt(self, md: str) -> str:
        md_short = md[: self.max_report_chars]
        return (
            "You are a strict quality reviewer of analytical reports.\n"
            "Read the following Markdown report and provide a decision along with constructive feedback."
            "The second section (**Structured Analysis (The Core)**) is under another critic's review, so assume it will be correct.\n\n"
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


# class CriticAgent(Agent):
#     LABELS = ("ACCEPT", "REJECT", "RERUN", "AMBIGUOUS")

#     def __init__(
#             self,
#             name: str = "Gemini-Critic",
#             model_name: str = "gemini-2.5-flash",
#             max_report_chars: int = 30000,
#     ) -> None:
#         super().__init__(name)
#         self.name = name
#         self.max_report_chars = max_report_chars
#         self.llm = ChatGoogleGenerativeAI(
#             model=model_name,
#             max_retries=2,
#             temperature=0.0
#         )

#     def _read_report(
#             self, report_path: Optional[str], report_markdown: Optional[str]
#     ) -> str:
#         if report_markdown and report_markdown.strip():
#             return report_markdown
#         if not report_path:
#             raise ValueError("Critic requires 'report_path' or 'report_markdown'.")
#         if not os.path.isfile(report_path):
#             raise FileNotFoundError(f"Report file not found: {report_path}")
#         with open(report_path, "r", encoding="utf-8") as f:
#             return f.read()

#     def _build_prompt(self, md: str) -> str:
#         md_short = md[: self.max_report_chars]
#         return (
#             "You are a strict quality reviewer of analytical reports.\n"
#             "Read the following Markdown report and provide a decision along with constructive feedback.\n\n"
#             "POSSIBLE DECISIONS:\n"
#             "- ACCEPT: structure is complete, content is coherent, formatting is correct.\n"
#             "- REJECT: report is broken, missing sections, missing images, or logically inconsistent.\n"
#             "- RERUN: report looks okay but specific data/plots seem wrong and need regeneration.\n"
#             "- AMBIGUOUS: cannot determine from the provided text.\n\n"
#             "OUTPUT FORMAT:\n"
#             "You must strictly follow this format:\n"
#             "DECISION: [One word from the list above]\n"
#             "FEEDBACK: [Short explanation of why you made this decision. If REJECT, list specifically what is missing.]\n\n"
#             "Report Content:\n"
#             f"{md_short}\n\n"
#             "Your Assessment:"
#         )

#     def _parse_response(self, text: str) -> Tuple[str, str]:
#         text = text.strip()
#         decision = "AMBIGUOUS"
#         feedback = text
#         pattern = r"DECISION:\s*([A-Z]+)\s*\n?FEEDBACK:\s*(.*)"
#         match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
#         if match:
#             raw_decision = match.group(1).upper().strip()
#             feedback_text = match.group(2).strip()
#             clean_decision = re.sub(r"[^A-Z]", "", raw_decision)
#             aliases = {
#                 "OK": "ACCEPT", "YES": "ACCEPT", "NO": "REJECT",
#                 "REGENERATE": "RERUN", "RE-RUN": "RERUN"
#             }
#             if clean_decision in self.LABELS:
#                 decision = clean_decision
#             elif clean_decision in aliases:
#                 decision = aliases[clean_decision]
#             feedback = feedback_text
#         else:
#             first_line = text.split('\n')[0].upper()
#             for label in self.LABELS:
#                 if label in first_line:
#                     decision = label
#                     feedback = text.replace(label, "", 1).strip()
#                     break

#         return decision, feedback

#     def run(self, **kwargs: Any) -> Dict[str, Any]:
#         report_path = kwargs.get("report_path")
#         report_markdown = kwargs.get("report_markdown")

#         md = self._read_report(report_path, report_markdown)
#         prompt_content = self._build_prompt(md)
#         messages = [HumanMessage(content=prompt_content)]
#         response = self.llm.invoke(messages)
#         raw_content = response.content
#         decision, feedback = self._parse_response(raw_content)
#         return {
#             "critic_llm_decision": decision,
#             "critic_llm_feedback": feedback,
#             "critic_llm_raw": raw_content,
#         }
