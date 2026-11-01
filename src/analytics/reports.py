from __future__ import annotations

import pandas as pd
from typing import Any


def format_table(df: pd.DataFrame, title: str) -> str:
    """Format a dataframe into a clean text table with title."""
    header = f"\n{'='*70}\n  {title}\n{'='*70}"
    if df.empty:
        return f"{header}\n  [No Data Available]\n"
    return f"{header}\n{df.to_string(index=False)}\n"


def generate_full_research_report(
    direction_df: pd.DataFrame,
    size_df: pd.DataFrame,
    session_df: pd.DataFrame,
    dow_df: pd.DataFrame,
    fvg_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    uncond_targets_df: pd.DataFrame,
    path_targets_df: pd.DataFrame,
    mfe_mae_summary: dict[str, Any],
) -> str:
    """
    Combine all research analyses into a unified formatted report.
    """
    output = []
    output.append("\n" + "#"*70)
    output.append("  ORDER BLOCK STATISTICAL RESEARCH & EDGE-DISCOVERY REPORT")
    output.append("#"*70)

    output.append(format_table(direction_df, "A. DIRECTION ANALYSIS (BULLISH VS BEARISH)"))
    output.append(format_table(size_df, "B. ORDER BLOCK SIZE & RANGE ANALYSIS"))
    output.append(format_table(session_df, "C1. SESSION ANALYSIS"))
    output.append(format_table(dow_df, "C2. DAY OF WEEK ANALYSIS"))
    output.append(format_table(fvg_df, "D. FVG & DISPLACEMENT CHARACTERISTICS"))
    output.append(format_table(dist_df, "E. DISTANCE FROM DAY CLOSE ANALYSIS"))
    output.append(format_table(uncond_targets_df, "F1. UNCONDITIONAL TARGET REACH PROBABILITY (P(MFE >= kR))"))
    output.append(format_table(path_targets_df, "F2. PATH-DEPENDENT TARGET-BEFORE-STOP ANALYSIS (vs -1.0R)"))

    output.append(f"\n{'='*70}\n  G. MFE / MAE RELATIONSHIP & ASYMMETRY\n{'='*70}")
    for k, v in mfe_mae_summary.items():
        if isinstance(v, float):
            output.append(f"  {k:28s}: {v:10.4f}")
        else:
            output.append(f"  {k:28s}: {v}")

    output.append("\n" + "#"*70 + "\n")
    return "\n".join(output)
