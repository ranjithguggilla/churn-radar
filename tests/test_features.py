from __future__ import annotations

import pandas as pd

from churn_radar.explain import recommended_action, risk_drivers, risk_tier
from churn_radar.features import TARGET, customers_to_frame


def test_clean_data_has_no_missing_values(clean_df):
    assert len(clean_df) == 7043
    assert not clean_df.isna().any().any()
    assert set(clean_df[TARGET].unique()) == {0, 1}
    assert "customerID" not in clean_df.columns


def test_blank_total_charges_become_zero(clean_df):
    new_customers = clean_df[clean_df["tenure"] == 0]
    assert len(new_customers) > 0
    assert (new_customers["TotalCharges"] == 0).all()
    assert (new_customers["avg_monthly_spend"] == new_customers["MonthlyCharges"]).all()


def test_inference_frame_matches_training_columns(clean_df, customer):
    frame = customers_to_frame([customer])
    assert list(frame.columns) == [c for c in clean_df.columns if c != TARGET]
    pd.testing.assert_series_equal(frame.dtypes, clean_df.drop(columns=[TARGET]).dtypes)


def test_derived_features(customer):
    row = customers_to_frame([customer]).iloc[0]
    assert row["TotalCharges"] == 190.0
    assert row["avg_monthly_spend"] == 95.0
    assert row["tenure_bucket"] == "0-6m"
    # phone, fiber, TV, movies
    assert row["num_services"] == 4


def test_explicit_total_charges_is_kept(customer):
    row = customers_to_frame([{**customer, "TotalCharges": 150.0}]).iloc[0]
    assert row["TotalCharges"] == 150.0
    assert row["avg_monthly_spend"] == 75.0


def test_risk_tiers_and_actions():
    assert risk_tier(0.9) == "high"
    assert risk_tier(0.4) == "medium"
    assert risk_tier(0.1) == "low"
    assert "48 hours" in recommended_action(0.9)
    assert recommended_action(0.1).startswith("Standard")


def test_risk_drivers(customer, loyal_customer):
    assert len(risk_drivers(customer)) == 5
    assert risk_drivers(loyal_customer) == []
