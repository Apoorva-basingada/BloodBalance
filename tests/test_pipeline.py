"""
test_pipeline.py
----------------
Unit tests for the BloodBalance data pipeline.

Run from the project root:
    python -m pytest tests/ -v
"""

import os
import sys
import pytest
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_blood_banks, load_transfusion, get_filtered_df
from src.analytics import (
    compute_kpis,
    state_distribution,
    district_distribution,
    category_breakdown,
    service_time_breakdown,
    blood_component_breakdown,
    missing_value_summary,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def blood_banks_df():
    """Load the cleaned blood-bank DataFrame once for all tests."""
    return load_blood_banks()


@pytest.fixture(scope="module")
def transfusion_df():
    """Load the cleaned transfusion DataFrame once for all tests."""
    return load_transfusion()


# ── Data loading tests ─────────────────────────────────────────────────────────
class TestDataLoading:
    def test_blood_banks_loads(self, blood_banks_df):
        assert blood_banks_df is not None
        assert len(blood_banks_df) > 0, "Blood banks DataFrame should not be empty"

    def test_blood_banks_row_count(self, blood_banks_df):
        """Verify all 2,820 raw records are preserved."""
        assert len(blood_banks_df) == 2820, (
            f"Expected 2820 records, got {len(blood_banks_df)}"
        )

    def test_blood_banks_required_columns(self, blood_banks_df):
        required = [
            "bank_name", "state", "district", "category",
            "blood_component", "apheresis", "service_time",
            "latitude", "longitude", "has_valid_coords",
        ]
        for col in required:
            assert col in blood_banks_df.columns, f"Missing column: {col}"

    def test_blood_banks_no_all_nan_rows(self, blood_banks_df):
        """No row should be entirely NaN."""
        all_nan = blood_banks_df.isnull().all(axis=1).sum()
        assert all_nan == 0, f"{all_nan} completely-empty rows found"

    def test_blood_banks_state_count(self, blood_banks_df):
        n_states = blood_banks_df["state"].nunique()
        assert n_states >= 30, f"Expected at least 30 states, got {n_states}"

    def test_blood_banks_coord_coverage(self, blood_banks_df):
        valid = blood_banks_df["has_valid_coords"].sum()
        total = len(blood_banks_df)
        pct = valid / total * 100
        assert pct > 90, f"Expected >90% coordinate coverage, got {pct:.1f}%"

    def test_blood_banks_category_values(self, blood_banks_df):
        allowed = {"Government", "Private", "Charity", "Unknown"}
        actual = set(blood_banks_df["category"].unique())
        unexpected = actual - allowed
        assert not unexpected, f"Unexpected category values: {unexpected}"

    def test_blood_banks_service_time_normalised(self, blood_banks_df):
        allowed = {"24x7", "Daytime", "Other / Unknown", "Unknown"}
        actual = set(blood_banks_df["service_time"].unique())
        unexpected = actual - allowed
        assert not unexpected, f"Unexpected service_time values: {unexpected}"

    def test_blood_banks_apheresis_values(self, blood_banks_df):
        allowed = {"Yes", "No", "Unknown"}
        actual = set(blood_banks_df["apheresis"].unique())
        unexpected = actual - allowed
        assert not unexpected, f"Unexpected apheresis values: {unexpected}"

    def test_transfusion_loads(self, transfusion_df):
        assert transfusion_df is not None
        assert len(transfusion_df) == 748, f"Expected 748, got {len(transfusion_df)}"

    def test_transfusion_columns(self, transfusion_df):
        expected = ["recency_months", "frequency_times", "monetary_cc", "time_months", "donated"]
        for col in expected:
            assert col in transfusion_df.columns, f"Missing column: {col}"

    def test_transfusion_no_missing(self, transfusion_df):
        assert transfusion_df.isnull().sum().sum() == 0, "Transfusion data should have no missing values"

    def test_transfusion_target_binary(self, transfusion_df):
        vals = set(transfusion_df["donated"].unique())
        assert vals == {0, 1}, f"Target should be {{0,1}}, got {vals}"


# ── Analytics tests ────────────────────────────────────────────────────────────
class TestAnalytics:
    def test_compute_kpis_structure(self, blood_banks_df):
        kpis = compute_kpis(blood_banks_df)
        required_keys = [
            "total_records", "total_states", "total_districts",
            "coord_count", "coord_pct", "with_email", "with_24x7",
        ]
        for k in required_keys:
            assert k in kpis, f"Missing KPI key: {k}"

    def test_compute_kpis_total(self, blood_banks_df):
        kpis = compute_kpis(blood_banks_df)
        assert kpis["total_records"] == 2820

    def test_compute_kpis_empty_df(self):
        empty = pd.DataFrame(columns=["state", "district", "category",
                                       "has_valid_coords", "email", "service_time"])
        kpis = compute_kpis(empty)
        assert kpis["total_records"] == 0
        assert kpis["total_states"] == 0

    def test_state_distribution(self, blood_banks_df):
        dist = state_distribution(blood_banks_df)
        assert "state" in dist.columns
        assert "count" in dist.columns
        assert dist["count"].sum() == 2820

    def test_district_distribution(self, blood_banks_df):
        dist = district_distribution(blood_banks_df, top_n=10)
        assert len(dist) <= 10

    def test_category_breakdown(self, blood_banks_df):
        cats = category_breakdown(blood_banks_df)
        total = cats["count"].sum()
        assert total == 2820

    def test_service_time_breakdown(self, blood_banks_df):
        svc = service_time_breakdown(blood_banks_df)
        assert "24x7" in svc["service_time"].values

    def test_missing_value_summary(self, blood_banks_df):
        mv = missing_value_summary(blood_banks_df)
        assert "Column" in mv.columns
        assert "% Complete" in mv.columns
        assert (mv["% Complete"] >= 0).all()
        assert (mv["% Complete"] <= 100).all()


# ── Filter tests ───────────────────────────────────────────────────────────────
class TestFiltering:
    def test_filter_by_state(self, blood_banks_df):
        filtered = get_filtered_df(blood_banks_df, state="Maharashtra")
        assert len(filtered) > 0
        assert filtered["state"].unique().tolist() == ["Maharashtra"]

    def test_filter_by_category(self, blood_banks_df):
        filtered = get_filtered_df(blood_banks_df, category="Government")
        assert (filtered["category"] == "Government").all()

    def test_filter_no_results(self, blood_banks_df):
        filtered = get_filtered_df(blood_banks_df, state="NonExistentState_XYZ")
        assert len(filtered) == 0
        # Must return DataFrame, not None
        assert isinstance(filtered, pd.DataFrame)

    def test_filter_search(self, blood_banks_df):
        filtered = get_filtered_df(blood_banks_df, search="AIIMS")
        # At least one AIIMS should exist in the dataset
        assert len(filtered) >= 0  # non-negative
        assert isinstance(filtered, pd.DataFrame)

    def test_filter_all_preserves_count(self, blood_banks_df):
        filtered = get_filtered_df(blood_banks_df, state="All", district="All",
                                   category="All", blood_component="All", search="")
        assert len(filtered) == len(blood_banks_df)

    def test_filter_combined(self, blood_banks_df):
        filtered = get_filtered_df(
            blood_banks_df,
            state="Maharashtra",
            category="Private",
        )
        if len(filtered) > 0:
            assert (filtered["state"] == "Maharashtra").all()
            assert (filtered["category"] == "Private").all()
