"""
BloodBalance — Interactive Blood Bank Analytics Dashboard
=========================================================
Streamlit application entry point.

Run with:
    streamlit run dashboard/app.py

from the project root directory.
"""

import os
import sys
import io

import streamlit as st
import pandas as pd
import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
# Allow imports from the project root when launched from any directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_blood_banks, load_transfusion, get_filtered_df
from src.analytics import (
    compute_kpis,
    state_distribution,
    district_distribution,
    category_breakdown,
    service_time_breakdown,
    blood_component_breakdown,
    missing_value_summary,
    top_states_table,
)
from src.charts import (
    state_bar_chart,
    district_bar_chart,
    category_donut_chart,
    service_time_chart,
    blood_component_chart,
    map_chart,
    missing_value_chart,
    coord_coverage_chart,
    transfusion_distribution_charts,
    transfusion_target_chart,
    transfusion_feature_importance_chart,
    transfusion_scatter_chart,
    empty_figure,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BloodBalance — Indian Blood Bank Analytics",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%);
    background-attachment: fixed;
}

/* Header */
.bb-header {
    background: linear-gradient(90deg, #8B0000 0%, #C0392B 40%, #E74C3C 100%);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(192,57,43,0.35);
}
.bb-header h1 {
    color: #FFFFFF;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.bb-header p {
    color: rgba(255,255,255,0.85);
    font-size: 1rem;
    margin: 0;
}

/* KPI cards */
.kpi-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.kpi-card {
    flex: 1;
    min-width: 140px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #E74C3C;
    line-height: 1.1;
    display: block;
}
.kpi-label {
    font-size: 0.78rem;
    color: rgba(236,240,241,0.7);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 6px;
    display: block;
}

/* Disclaimer box */
.disclaimer {
    background: rgba(243,156,18,0.12);
    border: 1px solid rgba(243,156,18,0.4);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #F39C12;
    margin-bottom: 16px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D0D1A 0%, #1A1A2E 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: rgba(255,255,255,0.6);
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #C0392B, #E74C3C) !important;
    color: white !important;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 48px 24px;
    color: rgba(236,240,241,0.5);
    font-size: 1rem;
}

/* Metric override */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 12px;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #ECF0F1;
    border-left: 3px solid #E74C3C;
    padding-left: 10px;
    margin: 16px 0 12px 0;
}

/* Data quality badge */
.quality-badge-good { color: #1ABC9C; font-weight: 600; }
.quality-badge-warn { color: #F39C12; font-weight: 600; }
.quality-badge-bad  { color: #E74C3C; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading blood bank data...")
def get_blood_banks():
    return load_blood_banks()


@st.cache_data(show_spinner="Loading transfusion data...")
def get_transfusion():
    return load_transfusion()


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="bb-header">
  <h1>🩸 BloodBalance</h1>
  <p>Interactive Blood Bank Analytics &amp; Decision-Support Dashboard &nbsp;·&nbsp;
     India Blood Bank Directory (National Health Portal) &nbsp;·&nbsp; Educational Use Only</p>
</div>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Preparing data..."):
    df_full = get_blood_banks()

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Filters")

    states = ["All"] + sorted(df_full["state"].dropna().unique().tolist())
    sel_state = st.selectbox("State", states, key="sel_state")

    # District options depend on selected state
    if sel_state == "All":
        districts = ["All"] + sorted(df_full["district"].dropna().unique().tolist())
    else:
        districts = ["All"] + sorted(
            df_full[df_full["state"] == sel_state]["district"].dropna().unique().tolist()
        )
    sel_district = st.selectbox("District", districts, key="sel_district")

    categories = ["All"] + sorted(df_full["category"].dropna().unique().tolist())
    sel_category = st.selectbox("Category", categories, key="sel_category")

    components = ["All"] + sorted(df_full["blood_component"].dropna().unique().tolist())
    sel_component = st.selectbox("Blood Component Available", components, key="sel_component")

    search_query = st.text_input("Search (name / city / address)", key="search_q", placeholder="e.g. AIIMS, Mumbai...")

    if st.button("🔄 Reset Filters", use_container_width=True):
        st.session_state["sel_state"] = "All"
        st.session_state["sel_district"] = "All"
        st.session_state["sel_category"] = "All"
        st.session_state["sel_component"] = "All"
        st.session_state["search_q"] = ""
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:rgba(255,255,255,0.4)'>Data: National Health Portal India<br>"
        "Last verified: 2024<br>Not real-time inventory.</small>",
        unsafe_allow_html=True,
    )

# ── Apply filters ──────────────────────────────────────────────────────────────
df = get_filtered_df(
    df_full,
    state=sel_state,
    district=sel_district,
    category=sel_category,
    blood_component=sel_component,
    search=search_query,
)

# Filter status banner
if len(df) == 0:
    st.error("⚠️ No records match the current filters. Try adjusting or resetting the filters.")
elif len(df) < len(df_full):
    st.info(f"Showing **{len(df):,}** of **{len(df_full):,}** blood banks based on selected filters.")

# ── KPIs ───────────────────────────────────────────────────────────────────────
kpis = compute_kpis(df)

col1, col2, col3, col4, col5, col6 = st.columns(6)
metrics = [
    (col1, "Total Blood Banks",  f"{kpis['total_records']:,}", "🏥"),
    (col2, "States Covered",     str(kpis["total_states"]),    "🗺️"),
    (col3, "Districts Covered",  str(kpis["total_districts"]), "📍"),
    (col4, "Mapped Locations",   f"{kpis['coord_pct']}%",       "🛰️"),
    (col5, "24x7 Service",       f"{kpis['with_24x7']:,}",     "🕐"),
    (col6, "Have Email",         f"{kpis['with_email']:,}",    "📧"),
]
for col, label, value, icon in metrics:
    with col:
        st.metric(f"{icon} {label}", value)

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_overview, tab_map, tab_charts, tab_directory, tab_quality, tab_donor = st.tabs([
    "📊 Overview",
    "🗺️ Map",
    "📈 Charts",
    "📋 Directory",
    "🔬 Data Quality",
    "🧬 Donor Analysis",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown('<div class="section-header">Category Breakdown</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        df_cat = category_breakdown(df)
        st.plotly_chart(category_donut_chart(df_cat), use_container_width=True)
    with col_b:
        st.markdown('<div class="section-header">Service Time</div>', unsafe_allow_html=True)
        df_svc = service_time_breakdown(df)
        st.plotly_chart(service_time_chart(df_svc), use_container_width=True)

    st.markdown('<div class="section-header">Top States at a Glance</div>', unsafe_allow_html=True)
    top_df = top_states_table(df, n=10)
    if len(top_df) > 0:
        st.dataframe(
            top_df.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.markdown('<div class="empty-state">No data to display.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>Important:</b> This dashboard displays directory listing data from the National Health Portal.
    It does not show real-time blood inventory, current stock levels, or live availability.
    Always contact your nearest blood bank directly to confirm availability.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.markdown(
        f"**{kpis['coord_count']:,}** of **{kpis['total_records']:,}** "
        f"records in current filter have valid coordinates ({kpis['coord_pct']}%)."
    )
    if len(df) == 0:
        st.markdown('<div class="empty-state">No data for the current filters.</div>', unsafe_allow_html=True)
    else:
        fig_map = map_chart(df)
        st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("""
    <div class="disclaimer">
    Map uses OpenStreetMap tiles via Plotly (no API key required).
    Coordinates sourced from the NHP directory listing — not independently verified.
    Do not use for navigation.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_charts:
    if len(df) == 0:
        st.markdown('<div class="empty-state">No records match current filters. Adjust filters to see charts.</div>', unsafe_allow_html=True)
    else:
        # State distribution
        st.markdown('<div class="section-header">Blood Banks by State</div>', unsafe_allow_html=True)
        df_state = state_distribution(df)
        st.plotly_chart(state_bar_chart(df_state), use_container_width=True)

        st.markdown("---")

        # District distribution with top-N selector
        st.markdown('<div class="section-header">Blood Banks by District</div>', unsafe_allow_html=True)
        col_ctrl, _ = st.columns([1, 3])
        with col_ctrl:
            top_n = st.slider("Top N Districts", min_value=5, max_value=50, value=20, step=5)
        df_dist = district_distribution(df, top_n=top_n)
        st.plotly_chart(district_bar_chart(df_dist, top_n=top_n), use_container_width=True)

        st.markdown("---")

        # Blood component and apheresis
        st.markdown('<div class="section-header">Blood Component & Apheresis</div>', unsafe_allow_html=True)
        df_comp = blood_component_breakdown(df)
        st.plotly_chart(blood_component_chart(df_comp), use_container_width=True)

        st.caption(
            "Blood Component: whether the bank lists any blood components as available. "
            "Apheresis: whether apheresis service is listed. "
            "'Unknown' includes missing entries and the 29 raw '2' values in the Apheresis column."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DIRECTORY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_directory:
    st.markdown(f"**{len(df):,} records** match current filters.")

    if len(df) == 0:
        st.markdown('<div class="empty-state">No records match the current filters.</div>', unsafe_allow_html=True)
    else:
        # Select columns for display
        display_cols = [
            "bank_name", "state", "district", "city",
            "category", "blood_component", "apheresis",
            "service_time", "contact_no", "mobile", "email", "address",
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        col_labels = {
            "bank_name": "Blood Bank Name",
            "state": "State",
            "district": "District",
            "city": "City",
            "category": "Category",
            "blood_component": "Blood Component",
            "apheresis": "Apheresis",
            "service_time": "Service Time",
            "contact_no": "Contact",
            "mobile": "Mobile",
            "email": "Email",
            "address": "Address",
        }
        display_df = df[display_cols].rename(columns=col_labels)

        # Pagination
        PAGE_SIZE = 50
        total_pages = max(1, (len(display_df) - 1) // PAGE_SIZE + 1)
        col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
        with col_p2:
            page = st.number_input(
                f"Page (1–{total_pages})",
                min_value=1, max_value=total_pages, value=1, step=1,
            )
        start = (page - 1) * PAGE_SIZE
        end = start + PAGE_SIZE
        st.dataframe(display_df.iloc[start:end], use_container_width=True, hide_index=True)

        st.caption(
            f"Showing rows {start + 1}–{min(end, len(display_df))} of {len(display_df):,}. "
            "Contact details are from the directory listing and have not been independently verified."
        )

        # Download button
        st.markdown("---")
        csv_buffer = io.BytesIO()
        display_df.to_csv(csv_buffer, index=False, encoding="utf-8")
        csv_buffer.seek(0)
        fname = "bloodbalance_filtered.csv"
        st.download_button(
            label="⬇️ Download Filtered Records as CSV",
            data=csv_buffer,
            file_name=fname,
            mime="text/csv",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA QUALITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_quality:
    st.markdown('<div class="section-header">Column Completeness (Full Dataset)</div>', unsafe_allow_html=True)
    df_mv = missing_value_summary(df_full)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.plotly_chart(missing_value_chart(df_mv), use_container_width=True)
    with col_b:
        st.plotly_chart(coord_coverage_chart(df_full), use_container_width=True)

    st.markdown('<div class="section-header">Completeness Table</div>', unsafe_allow_html=True)
    st.dataframe(df_mv, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Data Quality Notes</div>', unsafe_allow_html=True)
    st.markdown("""
| Issue | How Handled |
|---|---|
| **Encoding** | Raw CSV uses `latin-1` encoding. Loaded with explicit encoding; saved as UTF-8. |
| **Column names** | Leading/trailing whitespace stripped; renamed to `snake_case`. |
| **Service Time** | 50+ inconsistent variants normalised to: `24x7`, `Daytime`, `Other / Unknown`. Original values preserved in `service_time_raw` column. |
| **Apheresis '2' values** | 29 records had value `2` (meaning unclear). Treated as `Unknown`, not Yes or No. |
| **Category NaN** | 79 records had no category. Filled as `Unknown`. |
| **Missing coords** | 82 records lack latitude/longitude. Excluded from map. No coordinates invented. |
| **Invalid coords** | Validated against India's bounding box (lat 6–37.5°N, lon 68–97.5°E). |
| **Record count** | Cleaning pipeline asserts that all 2,820 raw records are preserved. No silent deletions. |
| **Duplicates** | Zero duplicate `Sr No` values found in raw data. |
| **Helpline/Fax/Website** | >80% missing. Included in dataset but not highlighted in KPIs to avoid misleading users. |
    """)

    st.markdown('<div class="section-header">Data Source</div>', unsafe_allow_html=True)
    st.markdown("""
- **Primary dataset:** Indian Blood Bank Directory — National Health Portal (NHP), Government of India
- **Format:** CSV, 2,820 records, 34 columns
- **Encoding:** latin-1 (original)
- **Last verified for this project:** 2024
- **Not real-time:** This is a directory listing. It does not reflect current blood stock, live availability, or verified contact status.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DONOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_donor:
    st.markdown("""
    <div class="disclaimer">
    🔬 <b>Dataset Notice:</b> This section uses the <b>UCI Blood Transfusion Service Center</b>
    dataset (I-Cheng Yeh, 2008) — historical blood-donation records from a mobile unit in
    <b>Hsin-Chu City, Taiwan, targeting March 2007</b>.<br><br>
    This dataset is <b>completely separate</b> from the Indian blood-bank directory.
    The model below does <b>not</b> predict Indian blood-bank stock levels, current donor
    availability, or real-world shortages. It is an <b>educational classification exercise</b>
    demonstrating RFM-based donation prediction on a historical research dataset.
    </div>
    """, unsafe_allow_html=True)

    try:
        df_tr = get_transfusion()

        # ── EDA ──────────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records", f"{len(df_tr):,}")
        c2.metric("Features", "4 (RFM + Time)")
        c3.metric("Donated (Class 1)", f"{int(df_tr['donated'].sum()):,}")
        c4.metric("Did Not Donate (Class 0)", f"{int((df_tr['donated'] == 0).sum()):,}")

        st.plotly_chart(transfusion_distribution_charts(df_tr), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(transfusion_target_chart(df_tr), use_container_width=True)
        with col_b:
            st.plotly_chart(transfusion_scatter_chart(df_tr), use_container_width=True)

        st.caption(
            "Note: Monetary (cc) and Frequency are perfectly correlated (each donation = 250 cc). "
            "This multicollinearity is a known property of this dataset."
        )

        # ── ML model ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">Logistic Regression Model (Educational)</div>', unsafe_allow_html=True)

        st.markdown("""
        **Goal:** Predict whether a donor donated blood in March 2007 based on their RFM history.

        **Methodology:**
        - Train / test split: 80% / 20% (stratified to preserve class ratio)
        - Features scaled with `StandardScaler`
        - Class imbalance handled with `class_weight='balanced'`
        - Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
        - **Why not accuracy alone?** The dataset is imbalanced (76% class 0 / 24% class 1).
          A dummy classifier that always predicts "did not donate" would achieve 76% accuracy —
          yet be completely useless. We therefore report ROC-AUC and the full classification report.
        """)

        with st.spinner("Training model..."):
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import (
                classification_report,
                roc_auc_score,
                confusion_matrix,
            )

            feature_cols = ["recency_months", "frequency_times", "monetary_cc", "time_months"]
            X = df_tr[feature_cols].values
            y = df_tr["donated"].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            )
            model.fit(X_train_s, y_train)

            y_pred = model.predict(X_test_s)
            y_prob = model.predict_proba(X_test_s)[:, 1]

            roc_auc = roc_auc_score(y_test, y_prob)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)

        # Metrics display
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{report['accuracy']:.2%}")
        m2.metric("Precision (Class 1)", f"{report['1']['precision']:.2%}")
        m3.metric("Recall (Class 1)", f"{report['1']['recall']:.2%}")
        m4.metric("F1-Score (Class 1)", f"{report['1']['f1-score']:.2%}")
        m5.metric("ROC-AUC", f"{roc_auc:.3f}")

        st.markdown("""
        > **Interpretation:** ROC-AUC significantly above 0.5 indicates the model is learning
        > meaningful patterns. A balanced class_weight ensures neither class dominates training.
        > Recall for class 1 is more practically important than raw accuracy in this context.
        """)

        # Confusion matrix
        col_cm, col_fi = st.columns(2)
        with col_cm:
            import plotly.graph_objects as go
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=["Predicted: No", "Predicted: Yes"],
                y=["Actual: No", "Actual: Yes"],
                colorscale=[[0, "#16213E"], [0.5, "#C0392B"], [1, "#E74C3C"]],
                text=cm.astype(str),
                texttemplate="%{text}",
                showscale=False,
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            ))
            fig_cm.update_layout(
                title=dict(text="Confusion Matrix", font=dict(color="#ECF0F1", size=14)),
                paper_bgcolor="#1A1A2E",
                plot_bgcolor="#16213E",
                font=dict(color="#ECF0F1"),
                height=320,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_fi:
            # Feature importance (coefficients after scaling)
            coeffs = model.coef_[0].tolist()
            labels = ["Recency", "Frequency", "Monetary", "Time"]
            st.plotly_chart(
                transfusion_feature_importance_chart(labels, coeffs),
                use_container_width=True,
            )

        st.markdown("""
        **Model Limitations:**
        - Frequency and Monetary are perfectly correlated in this dataset — multicollinearity
          affects coefficient interpretation.
        - The dataset is from Taiwan (2007) and a single mobile donation vehicle — not
          generalisable to modern Indian blood-donation behaviour.
        - This model is for educational demonstration only. Do not use it for any
          operational or medical decision making.
        """)

    except ImportError as e:
        st.error(f"scikit-learn is required for the Donor Analysis tab. Please run: `pip install scikit-learn`\n\nError: {e}")
    except Exception as e:
        st.error(f"An error occurred loading the Donor Analysis: {e}")
