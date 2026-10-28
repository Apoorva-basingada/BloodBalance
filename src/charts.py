"""
charts.py
---------
Reusable Plotly chart functions for BloodBalance.

Every function accepts a DataFrame (already filtered) and returns a
plotly.graph_objs.Figure. No Streamlit imports here.

Colour palette (deep red / crimson themed — appropriate for blood-bank data):
  Primary   : #C0392B  (crimson)
  Secondary : #E74C3C  (red)
  Accent    : #F39C12  (amber)
  Neutral   : #2C3E50  (dark blue-grey)
  BG        : #1A1A2E  (near-black navy)
  Surface   : #16213E
  Surface2  : #0F3460
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Design tokens ──────────────────────────────────────────────────────────────
CRIMSON   = "#C0392B"
RED       = "#E74C3C"
AMBER     = "#F39C12"
TEAL      = "#1ABC9C"
BLUE      = "#3498DB"
PURPLE    = "#9B59B6"
DARK_BG   = "#1A1A2E"
SURFACE   = "#16213E"
SURFACE2  = "#0F3460"
TEXT      = "#ECF0F1"
GRID      = "rgba(255,255,255,0.08)"

CATEGORY_COLORS = {
    "Government": "#3498DB",
    "Private": CRIMSON,
    "Charity": AMBER,
    "Unknown": "#7F8C8D",
}

SERVICE_COLORS = {
    "24x7": TEAL,
    "Daytime": AMBER,
    "Other / Unknown": "#7F8C8D",
    "Unknown": "#7F8C8D",
}

COMPONENT_COLORS = {
    "Yes": TEAL,
    "No": CRIMSON,
    "Unknown": "#7F8C8D",
}


def _base_layout(title: str, height: int = 450) -> dict:
    """Return a dict of common layout properties for a dark-themed chart."""
    return dict(
        title=dict(text=title, font=dict(color=TEXT, size=16, family="Inter, sans-serif")),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT, family="Inter, sans-serif", size=12),
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    )


def empty_figure(message: str = "No data available for current filters.") -> go.Figure:
    """Return a blank figure with a centred message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=14, color=TEXT),
    )
    fig.update_layout(**_base_layout(""), height=300)
    return fig


# ── State distribution ─────────────────────────────────────────────────────────
def state_bar_chart(df_state: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of blood banks per state.
    df_state: DataFrame with columns ['state', 'count'], sorted ascending.
    """
    if df_state is None or len(df_state) == 0:
        return empty_figure("No state data to display.")

    fig = px.bar(
        df_state,
        x="count",
        y="state",
        orientation="h",
        color="count",
        color_continuous_scale=[[0, SURFACE2], [0.5, RED], [1.0, CRIMSON]],
        labels={"count": "Blood Banks", "state": "State"},
        text="count",
    )
    fig.update_traces(
        textposition="outside",
        textfont_size=10,
        marker_line_width=0,
    )
    fig.update_coloraxes(showscale=False)
    layout = _base_layout("Blood Banks by State", height=max(500, len(df_state) * 22))
    layout["xaxis"]["title"] = "Number of Blood Banks Listed"
    layout["yaxis"]["title"] = ""
    fig.update_layout(**layout)
    return fig


# ── District distribution ──────────────────────────────────────────────────────
def district_bar_chart(df_dist: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """
    Horizontal bar chart of blood banks per district (top N).
    """
    if df_dist is None or len(df_dist) == 0:
        return empty_figure("No district data to display.")

    data = df_dist.tail(top_n)  # already sorted ascending; take last N = highest
    fig = px.bar(
        data,
        x="count",
        y="district",
        orientation="h",
        color="count",
        color_continuous_scale=[[0, SURFACE2], [0.5, "#E74C3C"], [1.0, AMBER]],
        labels={"count": "Blood Banks", "district": "District"},
        text="count",
    )
    fig.update_traces(
        textposition="outside",
        textfont_size=10,
        marker_line_width=0,
    )
    fig.update_coloraxes(showscale=False)
    layout = _base_layout(f"Top {min(top_n, len(data))} Districts by Blood Bank Count",
                          height=max(400, len(data) * 24))
    layout["xaxis"]["title"] = "Number of Blood Banks Listed"
    layout["yaxis"]["title"] = ""
    fig.update_layout(**layout)
    return fig


# ── Category donut ─────────────────────────────────────────────────────────────
def category_donut_chart(df_cat: pd.DataFrame) -> go.Figure:
    """Donut pie chart of blood-bank categories."""
    if df_cat is None or len(df_cat) == 0:
        return empty_figure("No category data to display.")

    colors = [CATEGORY_COLORS.get(c, "#7F8C8D") for c in df_cat["category"]]
    fig = go.Figure(go.Pie(
        labels=df_cat["category"],
        values=df_cat["count"],
        hole=0.52,
        marker=dict(colors=colors, line=dict(color=DARK_BG, width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout("Blood Banks by Category", height=400),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig


# ── Service time bar ───────────────────────────────────────────────────────────
def service_time_chart(df_svc: pd.DataFrame) -> go.Figure:
    """Bar chart of normalised service-time buckets."""
    if df_svc is None or len(df_svc) == 0:
        return empty_figure("No service time data to display.")

    colors = [SERVICE_COLORS.get(s, AMBER) for s in df_svc["service_time"]]
    fig = go.Figure(go.Bar(
        x=df_svc["service_time"],
        y=df_svc["count"],
        marker_color=colors,
        text=df_svc["count"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
    ))
    layout = _base_layout("Service Time Distribution", height=380)
    layout["yaxis"]["title"] = "Blood Banks"
    layout["xaxis"]["title"] = "Service Time (normalised)"
    fig.update_layout(**layout)
    return fig


# ── Blood component grouped bar ────────────────────────────────────────────────
def blood_component_chart(df_comp: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart showing Yes/No/Unknown counts for
    Blood Component and Apheresis.
    """
    if df_comp is None or len(df_comp) == 0:
        return empty_figure("No component data to display.")

    fig = go.Figure()
    for status, color in COMPONENT_COLORS.items():
        if status in df_comp.columns:
            fig.add_trace(go.Bar(
                name=status,
                x=df_comp["Feature"],
                y=df_comp[status],
                marker_color=color,
                text=df_comp[status],
                textposition="outside",
                hovertemplate=f"<b>%{{x}}</b><br>{status}: %{{y}}<extra></extra>",
            ))

    fig.update_layout(
        **_base_layout("Blood Component & Apheresis Availability", height=420),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    fig.update_layout(yaxis_title="Number of Blood Banks")
    return fig


# ── Map ────────────────────────────────────────────────────────────────────────
def map_chart(df: pd.DataFrame) -> go.Figure:
    """
    Scatter map of blood bank locations using Plotly's built-in
    open-street-map tiles (no API key required).

    Falls back to a state distribution bar chart if valid coordinates
    are missing or too few.
    """
    if df is None or len(df) == 0:
        return empty_figure("No data for current filters.")

    map_df = df[df.get("has_valid_coords", pd.Series(False, index=df.index))].copy() \
        if "has_valid_coords" in df.columns else pd.DataFrame()

    if len(map_df) < 5:
        # Fallback: state bar chart
        from src.analytics import state_distribution
        dist = state_distribution(df)
        return state_bar_chart(dist)

    # Hover text
    map_df = map_df.copy()
    map_df["hover"] = (
        "<b>" + map_df["bank_name"].fillna("Unknown") + "</b><br>"
        + "State: " + map_df["state"].fillna("") + "<br>"
        + "District: " + map_df["district"].fillna("") + "<br>"
        + "City: " + map_df["city"].fillna("") + "<br>"
        + "Category: " + map_df["category"].fillna("") + "<br>"
        + "Service: " + map_df["service_time"].fillna("") + "<br>"
        + "Phone: " + map_df["contact_no"].fillna("")
    )

    color_map = {
        "Government": "#3498DB",
        "Private": "#E74C3C",
        "Charity": "#F39C12",
        "Unknown": "#95A5A6",
    }

    fig = px.scatter_map(
        map_df,
        lat="latitude",
        lon="longitude",
        color="category",
        color_discrete_map=color_map,
        hover_name="bank_name",
        custom_data=["hover"],
        zoom=4,
        center={"lat": 20.5, "lon": 78.9},
        height=600,
        map_style="open-street-map",
    )
    fig.update_traces(
        marker=dict(size=6, opacity=0.75),
        hovertemplate="%{customdata[0]}<extra></extra>",
    )
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(
            text=f"Blood Bank Locations — {len(map_df):,} mapped",
            font=dict(color=TEXT, size=16),
        ),
        legend=dict(
            bgcolor="rgba(26,26,46,0.8)",
            bordercolor=GRID,
            orientation="v",
        ),
    )
    return fig


# ── Missing value completeness chart ──────────────────────────────────────────
def missing_value_chart(df_mv: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of data completeness (% complete per column).
    Columns sorted from least complete at top.
    """
    if df_mv is None or len(df_mv) == 0:
        return empty_figure("No data quality metrics available.")

    colors = [
        "#E74C3C" if pct < 50 else AMBER if pct < 80 else TEAL
        for pct in df_mv["% Complete"]
    ]
    fig = go.Figure(go.Bar(
        x=df_mv["% Complete"],
        y=df_mv["Column"],
        orientation="h",
        marker_color=colors,
        text=[f"{p:.0f}%" for p in df_mv["% Complete"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Complete: %{x:.1f}%<br>Present: %{customdata[0]:,}<extra></extra>",
        customdata=df_mv[["Present"]].values,
    ))
    layout = _base_layout("Data Completeness by Column", height=max(450, len(df_mv) * 22))
    layout["xaxis"]["title"] = "% Records with Value Present"
    layout["xaxis"]["range"] = [0, 115]
    layout["yaxis"]["title"] = ""
    fig.update_layout(**layout)
    return fig


# ── Coord coverage pie ─────────────────────────────────────────────────────────
def coord_coverage_chart(df: pd.DataFrame) -> go.Figure:
    """Small donut chart showing coordinate coverage."""
    if df is None or len(df) == 0:
        return empty_figure()
    valid = int(df["has_valid_coords"].sum()) if "has_valid_coords" in df.columns else 0
    total = len(df)
    invalid = total - valid
    fig = go.Figure(go.Pie(
        labels=["Valid Coordinates", "Missing / Invalid"],
        values=[valid, invalid],
        hole=0.55,
        marker=dict(colors=[TEAL, CRIMSON], line=dict(color=DARK_BG, width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value:,} records<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout("Coordinate Coverage", height=340),
        showlegend=False,
    )
    return fig


# ── Transfusion EDA charts ─────────────────────────────────────────────────────
def transfusion_distribution_charts(df: pd.DataFrame) -> go.Figure:
    """
    2x2 grid of histograms for the four RFM features in the transfusion dataset.
    """
    if df is None or len(df) == 0:
        return empty_figure("No transfusion data.")

    cols = ["recency_months", "frequency_times", "monetary_cc", "time_months"]
    labels = ["Recency (months)", "Frequency (donations)", "Monetary (cc)", "Time (months)"]
    colors_list = [CRIMSON, AMBER, TEAL, BLUE]

    fig = make_subplots(rows=2, cols=2, subplot_titles=labels)
    for i, (col, color) in enumerate(zip(cols, colors_list)):
        r, c = divmod(i, 2)
        if col in df.columns:
            fig.add_trace(
                go.Histogram(
                    x=df[col],
                    marker_color=color,
                    opacity=0.8,
                    name=labels[i],
                    showlegend=False,
                    hovertemplate=f"{labels[i]}: %{{x}}<br>Count: %{{y}}<extra></extra>",
                ),
                row=r + 1, col=c + 1,
            )

    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        height=500,
        title=dict(text="RFM Feature Distributions", font=dict(color=TEXT, size=16)),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    for axis in ["xaxis", "yaxis", "xaxis2", "yaxis2", "xaxis3", "yaxis3", "xaxis4", "yaxis4"]:
        fig.update_layout(**{axis: dict(gridcolor=GRID, zerolinecolor=GRID)})
    return fig


def transfusion_target_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing class imbalance in the transfusion target."""
    if df is None or len(df) == 0:
        return empty_figure()
    vc = df["donated"].value_counts().reset_index()
    vc.columns = ["donated", "count"]
    vc["label"] = vc["donated"].map({1: "Donated (March 2007)", 0: "Did Not Donate"})
    fig = go.Figure(go.Bar(
        x=vc["label"],
        y=vc["count"],
        marker_color=[TEAL, CRIMSON],
        text=vc["count"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
    ))
    layout = _base_layout("Target Class Distribution (Class Imbalance Present)", height=380)
    layout["yaxis"]["title"] = "Count"
    fig.update_layout(**layout)
    return fig


def transfusion_feature_importance_chart(feature_names: list, importances: list) -> go.Figure:
    """Horizontal bar chart of logistic regression feature coefficients."""
    df_fi = pd.DataFrame({"Feature": feature_names, "Coefficient": importances})
    df_fi = df_fi.sort_values("Coefficient", ascending=True)
    colors = [TEAL if v >= 0 else CRIMSON for v in df_fi["Coefficient"]]
    fig = go.Figure(go.Bar(
        x=df_fi["Coefficient"],
        y=df_fi["Feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in df_fi["Coefficient"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Coefficient: %{x:.3f}<extra></extra>",
    ))
    layout = _base_layout("Logistic Regression Coefficients (Scaled Features)", height=350)
    layout["xaxis"]["title"] = "Coefficient Value"
    fig.update_layout(**layout)
    return fig


def transfusion_scatter_chart(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of Recency vs Frequency coloured by donation outcome."""
    if df is None or len(df) == 0:
        return empty_figure()
    df = df.copy()
    df["Outcome"] = df["donated"].map({1: "Donated", 0: "Did Not Donate"})
    fig = px.scatter(
        df,
        x="recency_months",
        y="frequency_times",
        color="Outcome",
        color_discrete_map={"Donated": TEAL, "Did Not Donate": CRIMSON},
        opacity=0.6,
        labels={"recency_months": "Recency (months)", "frequency_times": "Frequency"},
        title="Recency vs Frequency by Donation Outcome",
    )
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT),
        height=420,
        xaxis=dict(gridcolor=GRID),
        yaxis=dict(gridcolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# convenience re-export so dashboard can do `from src.charts import state_bar_chart`
__all__ = [
    "state_bar_chart",
    "district_bar_chart",
    "category_donut_chart",
    "service_time_chart",
    "blood_component_chart",
    "map_chart",
    "missing_value_chart",
    "coord_coverage_chart",
    "transfusion_distribution_charts",
    "transfusion_target_chart",
    "transfusion_feature_importance_chart",
    "transfusion_scatter_chart",
    "empty_figure",
]
