from __future__ import annotations
from typing import Any, Dict, Optional
import os
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from .base import Agent


class CriticAgent(Agent):
    LABELS = ("ACCEPT", "REJECT", "RERUN", "AMBIGUOUS")
    _pipe = None

    def __init__(
        self,
        name: str = "LLM-Critic",
        model_id: str = "google/flan-t5-small",
        max_report_chars: int = 4000,
    ) -> None:
        super().__init__(name)
        self.model_id = model_id
        self.max_report_chars = max_report_chars

    @classmethod
    def _get_pipe(cls, model_id: str):
        if cls._pipe is None:
            tok = AutoTokenizer.from_pretrained(model_id)
            mod = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            cls._pipe = pipeline("text2text-generation", model=mod, tokenizer=tok)
        return cls._pipe

    # -------- helpers --------
    def _read_report(
        self, report_path: Optional[str], report_markdown: Optional[str]
    ) -> str:
        if report_markdown and report_markdown.strip():
            return report_markdown
        if not report_path:
            raise ValueError("LLM Critic requires 'report_path' or 'report_markdown'.")
        if not os.path.isfile(report_path):
            raise FileNotFoundError(f"Report file not found: {report_path}")
        with open(report_path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_prompt(self, md: str) -> str:
        md_short = md[: self.max_report_chars]
        return (
            "You are a strict quality reviewer of analytical reports.\n"
            "Read the following Markdown report and answer with ONE WORD ONLY from this set:\n"
            "ACCEPT, REJECT, RERUN, AMBIGUOUS\n\n"
            "Guidelines:\n"
            "- ACCEPT: structure is complete and content is coherent.\n"
            "- REJECT: report is clearly broken (missing major sections or images referenced but missing).\n"
            "- RERUN: report seems mostly fine but results/plots likely need to be regenerated or verified.\n"
            "- AMBIGUOUS: cannot determine from the provided text.\n\n"
            "Return ONLY the label (no punctuation, no extra text).\n\n"
            "Report:\n"
            f"{md_short}\n\n"
            "Answer:"
        )

    def _parse_label(self, text: str) -> str:
        t = text.strip().upper()
        first = re.split(r"[^A-Z]+", t, maxsplit=1)[0] if t else ""
        aliases = {
            "OK": "ACCEPT",
            "YES": "ACCEPT",
            "NO": "REJECT",
            "REGENERATE": "RERUN",
            "RE-RUN": "RERUN",
            "UNCLEAR": "AMBIGUOUS",
            "UNKNOWN": "AMBIGUOUS",
        }
        cand = aliases.get(first, first)
        return cand if cand in self.LABELS else "AMBIGUOUS"

    # -------- main --------
    def run(self, **kwargs: Any) -> Dict[str, Any]:
        report_path = kwargs.get("report_path")
        report_markdown = kwargs.get("report_markdown")

        md = self._read_report(report_path, report_markdown)
        prompt = self._build_prompt(md)

        pipe = self._get_pipe(self.model_id)
        out = pipe(
            prompt,
            max_new_tokens=8,
            temperature=0.0,
            num_beams=4,
            no_repeat_ngram_size=3,
        )
        raw = out[0]["generated_text"]
        decision = self._parse_label(raw)

        return {
            "critic_llm_decision": decision,
            "critic_llm_raw": raw,
        }
