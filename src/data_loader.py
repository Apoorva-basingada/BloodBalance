"""
data_loader.py
--------------
Loads, cleans, and saves the BloodBalance datasets.

Cleaning decisions are documented inline. Raw files are never modified.

Datasets:
  - blood_banks_raw.csv  : Indian blood-bank directory (2,820 records, 34 cols)
  - transfusion_raw.csv  : UCI Blood Transfusion dataset (748 records, 5 cols)
"""

import os
import re
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

BLOOD_BANKS_RAW = os.path.join(RAW_DIR, "blood_banks_raw.csv")
TRANSFUSION_RAW = os.path.join(RAW_DIR, "transfusion_raw.csv")

BLOOD_BANKS_CLEAN = os.path.join(CLEAN_DIR, "blood_banks_clean.csv")
TRANSFUSION_CLEAN = os.path.join(CLEAN_DIR, "transfusion_clean.csv")


# ── Service-time normalisation map ────────────────────────────────────────────
# The raw column contains 50+ inconsistent text variants of "24x7" plus
# some daytime-hour strings. Bucketed into three categories.
def _normalise_service_time(val):
    """Map raw service-time text to one of three canonical buckets."""
    if pd.isna(val):
        return "Unknown"
    s = str(val).strip().lower().replace(" ", "").replace("\\t", "")
    # Any variant of "24" with x/*/hours/hrs → 24x7
    if re.search(r"24", s):
        return "24x7"
    # Specific hour-range patterns (e.g. "10amto5pm", "9:00to17:00")
    if re.search(r"\d{1,2}(am|pm|:|h)", s) or "to" in s or "-" in s:
        return "Daytime"
    return "Other / Unknown"


def _normalise_apheresis(val):
    """
    Normalise Apheresis column.
    Raw values: YES, NO, 2 (unclear — treated as Unknown).
    """
    if pd.isna(val):
        return "Unknown"
    s = str(val).strip().upper()
    if s == "YES":
        return "Yes"
    if s == "NO":
        return "No"
    return "Unknown"  # covers the 29 '2' values in raw data


def _normalise_blood_component(val):
    """Normalise Blood Component Available to Yes / No / Unknown."""
    if pd.isna(val):
        return "Unknown"
    s = str(val).strip().upper()
    if s == "YES":
        return "Yes"
    if s == "NO":
        return "No"
    return "Unknown"


def _clean_phone(val):
    """Return phone string stripped of whitespace, or empty string."""
    if pd.isna(val):
        return ""
    return str(val).strip()


# ── Blood-bank directory ───────────────────────────────────────────────────────
def load_blood_banks(force_clean: bool = False) -> pd.DataFrame:
    """
    Load the cleaned blood-bank directory DataFrame.

    If the cleaned CSV already exists and force_clean is False, load it
    directly (faster). Otherwise run the full cleaning pipeline and save.

    Returns
    -------
    pd.DataFrame  with snake_case column names, normalised categories.
    """
    if os.path.exists(BLOOD_BANKS_CLEAN) and not force_clean:
        return pd.read_csv(BLOOD_BANKS_CLEAN, low_memory=False)

    # ── 1. Read raw file ──────────────────────────────────────────────────────
    df = pd.read_csv(BLOOD_BANKS_RAW, encoding="latin-1", on_bad_lines="skip")

    raw_count = len(df)  # 2820

    # ── 2. Strip leading/trailing whitespace from column names and strings ────
    df.columns = [c.strip() for c in df.columns]
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # ── 3. Rename columns to clean snake_case ────────────────────────────────
    rename_map = {
        "Sr No": "sr_no",
        "Blood Bank Name": "bank_name",
        "State": "state",
        "District": "district",
        "City": "city",
        "Address": "address",
        "Pincode": "pincode",
        "Contact No": "contact_no",
        "Mobile": "mobile",
        "Helpline": "helpline",
        "Fax": "fax",
        "Email": "email",
        "Website": "website",
        "Nodal Officer": "nodal_officer",
        "Contact Nodal Officer": "nodal_contact",
        "Mobile Nodal Officer": "nodal_mobile",
        "Email Nodal Officer": "nodal_email",
        "Qualification Nodal Officer": "nodal_qualification",
        "Category": "category",
        "Blood Component Available": "blood_component",
        "Apheresis": "apheresis",
        "Service Time": "service_time",
        "License #": "license_no",
        "Date License Obtained": "license_date",
        "Date of Renewal": "renewal_date",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Contact.No.1": "contact_1",
        "Contact.No.2": "contact_2",
        "Contact.No.3": "contact_3",
        "Contact.No.4": "contact_4",
        "Contact.No.5": "contact_5",
        "Contact.No.6": "contact_6",
        "Contact.No.7": "contact_7",
    }
    df.rename(columns=rename_map, inplace=True)

    # ── 4. Normalise categorical columns ─────────────────────────────────────
    # service_time: 50+ raw variants → 3 canonical buckets
    df["service_time_raw"] = df["service_time"].copy()
    df["service_time"] = df["service_time"].apply(_normalise_service_time)

    # apheresis: YES/NO/2 → Yes/No/Unknown
    df["apheresis"] = df["apheresis"].apply(_normalise_apheresis)

    # blood_component: YES/NO → Yes/No/Unknown
    df["blood_component"] = df["blood_component"].apply(_normalise_blood_component)

    # category: fill NaN as 'Unknown' (79 records)
    df["category"] = df["category"].fillna("Unknown")

    # ── 5. Coordinate validation ──────────────────────────────────────────────
    # Mark records with valid lat/lon (non-null, non-zero, within India bounds)
    def _valid_coord(lat, lon):
        try:
            lat, lon = float(lat), float(lon)
            return (
                6.0 <= lat <= 37.5
                and 68.0 <= lon <= 97.5
                and lat != 0.0
                and lon != 0.0
            )
        except (TypeError, ValueError):
            return False

    df["has_valid_coords"] = df.apply(
        lambda r: _valid_coord(r["latitude"], r["longitude"]), axis=1
    )

    # ── 6. Clean phone fields ─────────────────────────────────────────────────
    for col in ["contact_no", "mobile", "contact_1"]:
        df[col] = df[col].apply(_clean_phone)

    # ── 7. Verify no records silently discarded ───────────────────────────────
    clean_count = len(df)
    assert clean_count == raw_count, (
        f"Record count mismatch: raw={raw_count}, clean={clean_count}"
    )

    # ── 8. Save cleaned CSV (UTF-8) ───────────────────────────────────────────
    os.makedirs(CLEAN_DIR, exist_ok=True)
    df.to_csv(BLOOD_BANKS_CLEAN, index=False, encoding="utf-8")
    print(f"[data_loader] Blood banks cleaned: {clean_count} records saved.")

    return df


# ── Transfusion dataset ────────────────────────────────────────────────────────
def load_transfusion(force_clean: bool = False) -> pd.DataFrame:
    """
    Load the UCI Blood Transfusion dataset.

    Source: UCI Machine Learning Repository
            Blood Transfusion Service Center (I-Cheng Yeh, 2008)
    Context: Taiwan mobile blood-donation vehicle, March 2007 target window.
    This dataset is SEPARATE from the Indian blood-bank directory.

    Returns
    -------
    pd.DataFrame with renamed columns.
    """
    if os.path.exists(TRANSFUSION_CLEAN) and not force_clean:
        return pd.read_csv(TRANSFUSION_CLEAN)

    df = pd.read_csv(TRANSFUSION_RAW)

    # Rename to clean snake_case
    df.columns = [c.strip() for c in df.columns]
    df.rename(
        columns={
            "Recency (months)": "recency_months",
            "Frequency (times)": "frequency_times",
            "Monetary (c.c. blood)": "monetary_cc",
            "Time (months)": "time_months",
            "whether he/she donated blood in March 2007": "donated",
        },
        inplace=True,
    )

    os.makedirs(CLEAN_DIR, exist_ok=True)
    df.to_csv(TRANSFUSION_CLEAN, index=False)
    print(f"[data_loader] Transfusion cleaned: {len(df)} records saved.")
    return df


# ── Filtering helper ──────────────────────────────────────────────────────────
def get_filtered_df(
    df: pd.DataFrame,
    state: str = "All",
    district: str = "All",
    category: str = "All",
    blood_component: str = "All",
    search: str = "",
) -> pd.DataFrame:
    """
    Apply sidebar filters to the blood-bank DataFrame.

    All filters are additive (AND logic). Empty results return an empty
    DataFrame with the same columns (never None).
    """
    filtered = df.copy()

    if state and state != "All":
        filtered = filtered[filtered["state"] == state]
    if district and district != "All":
        filtered = filtered[filtered["district"] == district]
    if category and category != "All":
        filtered = filtered[filtered["category"] == category]
    if blood_component and blood_component != "All":
        filtered = filtered[filtered["blood_component"] == blood_component]
    if search and search.strip():
        q = search.strip().lower()
        mask = (
            filtered["bank_name"].str.lower().str.contains(q, na=False)
            | filtered["city"].str.lower().str.contains(q, na=False)
            | filtered["district"].str.lower().str.contains(q, na=False)
            | filtered["address"].str.lower().str.contains(q, na=False)
        )
        filtered = filtered[mask]

    return filtered.reset_index(drop=True)


# ── Standalone execution: regenerate cleaned files ────────────────────────────
if __name__ == "__main__":
    print("Running BloodBalance data pipeline...")
    bb = load_blood_banks(force_clean=True)
    tr = load_transfusion(force_clean=True)
    print(f"\nBlood banks : {len(bb)} records, {len(bb.columns)} columns")
    print(f"Transfusion : {len(tr)} records, {len(tr.columns)} columns")
    print(f"\nBlood component distribution:\n{bb['blood_component'].value_counts()}")
    print(f"\nService time distribution:\n{bb['service_time'].value_counts()}")
    print(f"\nCoords valid: {bb['has_valid_coords'].sum()} / {len(bb)}")
    print("\nPipeline complete.")
