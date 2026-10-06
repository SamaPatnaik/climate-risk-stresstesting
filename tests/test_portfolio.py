"""M2 tests. Unit tests use a small fake division table; the last test needs M1 outputs."""
import numpy as np
import pandas as pd
import pytest

from src.assumptions import load_assumptions, values
from src.hazard import PROCESSED
from src.portfolio import generate_portfolio, mortgage_weights, summarize

P = values("portfolio")


@pytest.fixture
def div():
    return pd.DataFrame({
        "cduid": ["5901", "5915", "5959"],
        "owner_households": [20_000, 600_000, 1_000],
        "pct_owners_with_mortgage": [55.0, 60.0, 60.0],
        "median_dwelling_value": [400_000, 1_000_000, 200_000],
    })


def test_every_assumption_is_documented():
    for section, params in load_assumptions().items():
        for name, spec in params.items():
            assert {"value", "placeholder"} <= spec.keys(), f"{section}.{name}"
            assert spec.get("source") or section == "scenarios", f"{section}.{name} has no source"


def test_scenario_multipliers_ordered():
    m = values("scenarios")
    assert 1 <= m["orderly"] <= m["disorderly"] < m["current_policies"] <= m["hot_house"]


def test_reproducible(div):
    a = generate_portfolio(div, P)
    b = generate_portfolio(div, P)
    pd.testing.assert_frame_equal(a, b)
    c = generate_portfolio(div, P, seed=P["seed"] + 1)
    assert not np.allclose(a["balance"], c["balance"])


def test_allocation_matches_mortgage_weights(div):
    loans = generate_portfolio(div, P)
    assert len(loans) == P["n_loans"]
    share = loans["cduid"].value_counts(normalize=True).reindex(div["cduid"], fill_value=0)
    w = mortgage_weights(div)
    expected = (w / w.sum()).to_numpy()
    se = np.sqrt(expected * (1 - expected) / P["n_loans"])
    assert (np.abs(share.to_numpy() - expected) < 4 * se + 1e-12).all()


def test_balances(div):
    loans = generate_portfolio(div, P)
    assert (loans["balance"] > 0).all()
    assert loans["balance"].mean() == pytest.approx(P["balance_mean"], rel=0.02)
    # higher dwelling value -> higher mean balance
    m = loans.groupby("cduid")["balance"].mean()
    assert m["5959"] < m["5901"] < m["5915"]


def test_insurance_and_lgd(div):
    loans = generate_portfolio(div, P)
    assert loans["insured"].mean() == pytest.approx(P["insured_share"], abs=0.01)
    assert (loans.loc[loans["insured"], "lgd_base"] == P["lgd_base_insured"]).all()
    assert (loans.loc[~loans["insured"], "lgd_base"] >= P["lgd_floor"]).all()
    assert (loans["pd_base"] == P["pd_base"]).all()


def test_summary_consistent(div):
    loans = generate_portfolio(div, P)
    s = summarize(loans, div["cduid"])
    assert len(s) == len(div)
    assert s["n_loans"].sum() == len(loans)
    assert s["ead_total"].sum() == pytest.approx(loans["balance"].sum())
    assert s["lgd_base_wavg"].between(0, 1).all()


@pytest.mark.skipif(not (PROCESSED / "divisions.parquet").exists(),
                    reason="run `python -m src.hazard` first")
def test_real_divisions_have_portfolio_inputs():
    div = pd.read_parquet(PROCESSED / "divisions.parquet")
    for col in ["owner_households", "pct_owners_with_mortgage", "median_dwelling_value"]:
        assert div[col].notna().all() and (div[col] > 0).all(), col
    assert div["pct_owners_with_mortgage"].between(0, 100).all()
