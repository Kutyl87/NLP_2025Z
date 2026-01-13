from __future__ import annotations
from typing import Any, Dict, List, Tuple
import os
import pandas as pd
import numpy as np
from .base import Agent


class AnalystAgent(Agent):
    def __init__(
        self,
        name: str = "Analyst",
        input_path: str = "data/input/winequality-red.csv",
        out_dir: str = "data/output",
        max_hists: int = 10,
        max_pairs: int = 10,
        corr_threshold: float = 0.6,
    ) -> None:
        super().__init__(name)
        self.input_path = input_path
        self.out_dir = out_dir
        self.max_hists = max_hists
        self.max_pairs = max_pairs
        self.corr_threshold = corr_threshold

    def _load(self, path: str) -> pd.DataFrame:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext in [".csv", ".txt"]:
            return pd.read_csv(path, sep=None, engine="python", quoting=3)
        elif ext in [".xls", ".xlsx"]:
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [
            str(c).strip().strip('"').strip("'").replace(" ", "_").lower()
            for c in df.columns
        ]
        df.drop_duplicates(inplace=True)
        df.dropna(axis=1, how="all", inplace=True)
        for col in df.columns:
            if df[col].dtype.kind in "ifc":
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())
            else:
                if df[col].isna().any():
                    mode = df[col].mode()
                    df[col] = df[col].fillna(
                        mode.iloc[0] if not mode.empty else "unknown"
                    )
        return df

    def _viz_plan(self, df: pd.DataFrame) -> Dict[str, Any]:
        numerics = df.select_dtypes(include=["number"])
        num_cols: List[str] = numerics.columns.tolist()
        variances = numerics.var().sort_values(ascending=False)
        hists = variances.index.tolist()[: self.max_hists]

        pairs: List[Tuple[str, str]] = []
        if len(num_cols) >= 2:
            corr = numerics.corr().abs()
            for i, a in enumerate(num_cols):
                for j, b in enumerate(num_cols):
                    if j <= i:
                        continue
                    val = corr.loc[a, b]
                    if np.isfinite(val) and val >= self.corr_threshold:
                        pairs.append((a, b))
            pairs = pairs[: self.max_pairs]

        heatmap = len(num_cols) >= 3
        return {"hists": hists, "pairs": pairs, "heatmap": heatmap}

    def _insights_text(
        self, df_raw: pd.DataFrame, df: pd.DataFrame, plan: Dict[str, Any]
    ) -> str:
        n_raw = df_raw.shape[0]
        n = df.shape[0]
        removed = n_raw - n
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        miss_before = df_raw.isna().sum().sum()
        miss_after = df.isna().sum().sum()
        bullets = [
            f"- Cleaned dataset: {n} rows ({removed} removed), {df.shape[1]} columns; missing values handled ({miss_before} → {miss_after}).",
            f"- Numeric columns: {', '.join(num_cols)}.",
            (
                "- Visualization plan: "
                + "; ".join(
                    [f"histograms={plan['hists']}"]
                    + ([f"pairs={plan['pairs']}"] if plan["pairs"] else [])
                    + [f"heatmap={'yes' if plan['heatmap'] else 'no'}"]
                )
            ),
        ]
        return "\n".join(bullets)

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        data_path = kwargs.get("data_path", self.input_path)
        os.makedirs(self.out_dir, exist_ok=True)

        df_raw = self._load(data_path)
        df = self._clean(df_raw)

        cleaned_path = os.path.join(self.out_dir, "cleaned.csv")
        df.to_csv(cleaned_path, index=False)

        plan = self._viz_plan(df)
        analysis = self._insights_text(df_raw, df, plan)

        return {
            "analysis": analysis,
            "data_path": cleaned_path,
            "viz_plan": plan,
        }


# class AnalystParallelAgent(AnalystAgent):
#     def __init__(
#         self,
#         name: str = "Analyst",
#         input_path: str = "data/input/winequality-red.csv",
#         out_dir: str = "data/output",
#         max_hists: int = 10,
#         max_pairs: int = 10,
#         corr_threshold: float = 0.6,
#     ) -> None:
#         super().__init__(name, input_path, out_dir, max_hists, max_pairs, corr_threshold)

#     def run(self, **kwargs: Any) -> Dict[str, Any]:
#         data_path = kwargs.get("data_path", self.input_path)
#         os.makedirs(self.out_dir, exist_ok=True)

#         df_raw = self._load(data_path)
#         df = self._clean(df_raw)

#         cleaned_path = os.path.join(self.out_dir, "cleaned.csv")
#         df.to_csv(cleaned_path, index=False)

#         plan = self._viz_plan(df)
#         analysis = self._insights_text(df_raw, df, plan)

#         return {
#             "analysis": analysis,
#             "data_path": cleaned_path,
#             "viz_plan": plan,
#         }
