"""
analytics.py
------------
KPI calculations and data aggregations for the BloodBalance dashboard.

All functions accept a filtered or full DataFrame and return plain Python
dicts or DataFrames — no Streamlit or Plotly imports here, keeping the
analytics layer fully testable.
"""

import pandas as pd
import numpy as np


# ── KPI cards ─────────────────────────────────────────────────────────────────
def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Compute headline KPIs from the (optionally filtered) blood-bank DataFrame.

    Returns a dict with the following keys:
      total_records      : int
      total_states       : int
      total_districts    : int
      coord_count        : int   — records with valid lat/lon
      coord_pct          : float — % of records with valid coords
      with_email         : int   — records that have an email address
      with_24x7          : int   — records with 24x7 service
      category_counts    : dict  — {category: count}
    """
    if df is None or len(df) == 0:
        return {
            "total_records": 0,
            "total_states": 0,
            "total_districts": 0,
            "coord_count": 0,
            "coord_pct": 0.0,
            "with_email": 0,
            "with_24x7": 0,
            "category_counts": {},
        }

    total = len(df)
    coord_count = int(df["has_valid_coords"].sum()) if "has_valid_coords" in df.columns else 0
    email_count = int(df["email"].notna().sum() if "email" in df.columns else 0)
    # Count non-empty email strings
    if "email" in df.columns:
        email_count = int(df["email"].notna().sum())
    service_24 = int((df["service_time"] == "24x7").sum()) if "service_time" in df.columns else 0

    cat_counts = (
        df["category"].value_counts().to_dict() if "category" in df.columns else {}
    )

    return {
        "total_records": total,
        "total_states": int(df["state"].nunique()) if "state" in df.columns else 0,
        "total_districts": int(df["district"].nunique()) if "district" in df.columns else 0,
        "coord_count": coord_count,
        "coord_pct": round(coord_count / total * 100, 1) if total > 0 else 0.0,
        "with_email": email_count,
        "with_24x7": service_24,
        "category_counts": cat_counts,
    }


# ── Distribution aggregations ─────────────────────────────────────────────────
def state_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return blood-bank counts by state, sorted descending."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["state", "count"])
    result = df["state"].value_counts().reset_index()
    result.columns = ["state", "count"]
    return result.sort_values("count", ascending=True)  # ascending for horizontal bar


def district_distribution(
    df: pd.DataFrame, state: str = "All", top_n: int = 20
) -> pd.DataFrame:
    """Return top-N blood-bank counts by district."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["district", "count"])
    sub = df if state == "All" else df[df["state"] == state]
    result = sub["district"].value_counts().head(top_n).reset_index()
    result.columns = ["district", "count"]
    return result.sort_values("count", ascending=True)


def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Return count by category (Government / Private / Charity / Unknown)."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["category", "count"])
    result = df["category"].value_counts().reset_index()
    result.columns = ["category", "count"]
    return result


def service_time_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Return count by normalised service-time bucket."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["service_time", "count"])
    result = df["service_time"].value_counts().reset_index()
    result.columns = ["service_time", "count"]
    return result


def blood_component_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame with columns:
      feature, Yes, No, Unknown
    One row each for Blood Component and Apheresis.
    """
    rows = []
    for col, label in [("blood_component", "Blood Component Available"),
                       ("apheresis", "Apheresis Available")]:
        if col in df.columns:
            vc = df[col].value_counts()
            rows.append({
                "Feature": label,
                "Yes": int(vc.get("Yes", 0)),
                "No": int(vc.get("No", 0)),
                "Unknown": int(vc.get("Unknown", 0)),
            })
    return pd.DataFrame(rows)


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return column-level completeness summary sorted by completeness ascending.

    Columns: col_name, missing, total, pct_missing, pct_complete
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # Drop internal/derived cols from quality report
    skip_cols = {
        "sr_no", "has_valid_coords", "service_time_raw",
        "contact_1", "contact_2", "contact_3", "contact_4",
        "contact_5", "contact_6", "contact_7",
    }
    cols = [c for c in df.columns if c not in skip_cols]

    total = len(df)
    records = []
    for col in cols:
        missing = int(df[col].isna().sum())
        # Also count empty strings as missing for string columns
        if df[col].dtype == object:
            missing += int((df[col] == "").sum())
        missing = min(missing, total)
        records.append({
            "Column": col,
            "Missing": missing,
            "Present": total - missing,
            "% Complete": round((total - missing) / total * 100, 1),
        })

    result = pd.DataFrame(records)
    return result.sort_values("% Complete", ascending=True).reset_index(drop=True)


def top_states_table(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top-N states by record count as a formatted table."""
    dist = state_distribution(df).sort_values("count", ascending=False).head(n)
    return dist.rename(columns={"state": "State", "count": "Blood Banks Listed"})
