# 🩸 BloodBalance — Interactive Blood Bank Analytics Dashboard

**BloodBalance** is an interactive data visualization and analytics platform for the Indian Blood Bank Directory dataset.
It is built as a college Data Visualization project to demonstrate how publicly available health data can be organized,
cleaned, and explored through a professional dashboard.

---

## Problem Statement

Blood-bank information in India is often spread across fragmented directory records.
There is no unified, easily navigable analytical view of where blood banks are located,
what services they list, or how complete the available data is.

**BloodBalance** addresses this by presenting the National Health Portal blood-bank directory
as an interactive, filterable, and downloadable dashboard — making the data accessible to
students, researchers, and healthcare analysts.

> ⚠️ **Important:** This dashboard displays directory listing data only.
> It does **not** show real-time blood inventory, current stock levels, or live donor availability.
> Always contact the blood bank directly to confirm availability.

---

## Intended Users

- Students and researchers exploring blood-bank distribution across India
- Healthcare data analysts studying regional coverage and data completeness
- People exploring publicly listed blood-bank locations and contact details
- College project evaluators assessing data visualization techniques

---

## Datasets

### Primary: Indian Blood Bank Directory
| Property | Value |
|---|---|
| Source | National Health Portal (NHP), Government of India |
| File | `data/raw/blood_banks_raw.csv` |
| Rows | 2,820 |
| Columns | 34 |
| States | 36 |
| Districts | 599 |
| Coordinate coverage | ~97% |
| Encoding | latin-1 (original), UTF-8 (cleaned) |
| Type | Directory listing — not real-time inventory |

### Supplementary: UCI Blood Transfusion Dataset
| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository (I-Cheng Yeh, 2008) |
| File | `data/raw/transfusion_raw.csv` |
| Rows | 748 |
| Columns | 5 |
| Region | Hsin-Chu City, Taiwan |
| Target | Donation in March 2007 (binary) |
| Purpose | Educational classification exercise (kept separate from directory) |

---

## Project Structure

```
bloodbalance/
├── data/
│   ├── raw/              # Unmodified source files
│   └── cleaned/          # Cleaned UTF-8 outputs (auto-generated)
├── src/
│   ├── data_loader.py    # Loading, cleaning, filtering
│   ├── analytics.py      # KPIs and aggregations
│   └── charts.py         # Reusable Plotly chart functions
├── dashboard/
│   └── app.py            # Streamlit app (6 tabs)
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Blood bank EDA (Colab-ready)
│   └── 02_transfusion_analysis.ipynb # Donor analysis + ML (Colab-ready)
├── tests/
│   └── test_pipeline.py  # Pytest unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dashboard Features

| Tab | Contents |
|---|---|
| 📊 Overview | KPI cards, category donut chart, service time chart, top states table |
| 🗺️ Map | Interactive Plotly map (OpenStreetMap, no API key needed) with hover details |
| 📈 Charts | State bar, district bar (top-N), blood component & apheresis grouped bar |
| 📋 Directory | Paginated searchable table, filtered CSV download |
| 🔬 Data Quality | Column completeness chart, coordinate coverage, cleaning decisions |
| 🧬 Donor Analysis | Transfusion EDA, logistic regression with ROC-AUC, confusion matrix |

**Sidebar filters** (affect all tabs consistently):
- State
- District
- Category (Government / Private / Charity)
- Blood Component Available
- Free-text search (name / city / address)
- Reset button

---

## Data Cleaning Decisions

| Issue | Decision |
|---|---|
| Raw encoding (latin-1) | Loaded with explicit encoding; saved as UTF-8 |
| Column whitespace | Stripped and renamed to `snake_case` |
| Service Time (50+ variants) | Normalised to `24x7`, `Daytime`, `Other/Unknown`. Raw preserved in `service_time_raw` |
| Apheresis value `2` (29 records) | Treated as `Unknown` — meaning unclear |
| Category NaN (79 records) | Filled as `Unknown` |
| Missing coordinates (82 records) | Excluded from map; no coordinates invented |
| Invalid coordinates | Validated against India bounding box (6–37.5°N, 68–97.5°E) |
| Record count | Cleaning asserts all 2,820 raw records are preserved |

---

## Setup & Running

### Prerequisites
- Python 3.9+
- pip

### 1. Install dependencies

```bash
cd bloodbalance
pip install -r requirements.txt
```

### 2. Run the data pipeline (generates cleaned files)

```bash
python src/data_loader.py
```

### 3. Run tests

```bash
python -m pytest tests/ -v
```

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Then open `http://localhost:8501` in your browser.

---

## Google Colab Instructions

### Option A — Upload files manually
1. Open the notebook in Colab.
2. In Cell 2, uncomment the `files.upload()` option.
3. Upload `blood_banks_raw.csv` or `transfusion_raw.csv` when prompted.

### Option B — Google Drive
1. Copy `data/raw/` to your Google Drive.
2. In Cell 2, uncomment the Drive mount option and update `DATA_PATH`.

### Option C — GitHub (if repo is public)
1. Set `DATA_PATH` to the raw GitHub URL of your CSV file.

---

## Analytical Questions Addressed

1. Which Indian states and districts have the most listed blood banks?
2. What proportion of blood banks list blood component services?
3. How many operate 24x7 versus daytime only?
4. How are blood banks distributed geographically across India?
5. How complete is the directory data, and which fields are most incomplete?
6. What predicts whether a historical donor donated blood in a given period? (Educational — Taiwan data)

---

## Limitations

- **Not real-time:** The directory reflects a snapshot, not current status.
- **Contact details unverified:** Phone numbers and emails may be outdated.
- **Coordinate precision:** Some coordinates may be approximate (city-level, not building-level).
- **No inventory data:** Legitimate real-time blood stock data was not available as a public dataset.
- **Transfusion model:** Taiwan 2007 data — results do not generalise to modern Indian donation behaviour.
- **Category field:** 79 records have no category listed.

---

## AI Assistance Disclosure

This project was developed with AI assistance (Google Deepmind Antigravity). The AI:
- Inspected and profiled the provided datasets
- Generated the data cleaning pipeline, analytics module, chart functions, and dashboard
- Wrote the unit tests and Colab notebooks

All findings and KPIs are calculated from the actual dataset — no values were hardcoded or fabricated.
The developer verified outputs and is responsible for the final project.

---

## Technologies Used

- **Python 3.9+**
- **Streamlit** — dashboard framework
- **Plotly** — interactive visualizations and maps
- **Pandas** — data manipulation
- **scikit-learn** — logistic regression model (donor analysis tab)
- **NumPy** — numerical operations
