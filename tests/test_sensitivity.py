"""M5 tests on a small synthetic book; the last test checks consistency with M4 outputs."""
import numpy as np
import pandas as pd
import pytest

from src.hazard import PROCESSED
from src.model import PORTFOLIO_ID
from src.sensitivity import (apply_override, base_config, evaluate, tornado, window_hazard)


@pytest.fixture(scope="module")
def book():
    rng = np.random.default_rng(1)
    cds = ["A", "B"]
    n = 1500
    insured = rng.random(n) < 0.25
    loans = pd.DataFrame({"loan_id": np.arange(n), "cduid": rng.choice(cds, n),
                          "balance": rng.lognormal(np.log(300_000), 0.5, n),
                          "insured": insured, "pd_base": 0.0026,
                          "lgd_base": np.where(insured, 0.0, 0.2)})
    years = np.arange(1990, 2026)
    share = {"A": np.where(years >= 2015, 0.02, 0.002), "B": np.where(years == 2023, 0.05, 0.0)}
    yearly = pd.DataFrame([{"cduid": c, "year": y, "burn_share": share[c][i]}
                           for c in cds for i, y in enumerate(years)])
    div = pd.DataFrame({"cduid": cds, "vulnerability": 1.0})
    return loans, div, yearly


def test_window_hazard_reference_fixed_and_mean(book):
    _, _, yearly = book
    y_all, s_all = window_hazard(yearly, 2.0)
    y_rec, s_rec = window_hazard(yearly, 2.0, (2015, 2025))
    assert set(y_rec["year"]) == set(range(2015, 2026))
    # worst CD-year (B 2023, 0.05) maps to 1 / headroom = 0.5 in both windows (ref = 0.1)
    assert y_all["hazard"].max() == pytest.approx(0.5)
    assert y_rec["hazard"].max() == pytest.approx(0.5)
    assert s_rec["A"] == pytest.approx(0.02 / 0.1)
    assert s_all["A"] == pytest.approx((25 * 0.002 + 11 * 0.02) / 36 / 0.1)


def test_override_does_not_mutate_base():
    cfg = base_config()
    before = cfg["model"]["beta_pd"]["median"]
    c2 = apply_override(cfg, "beta_pd_median", before + 1)
    assert cfg["model"]["beta_pd"]["median"] == before
    assert c2["model"]["beta_pd"]["median"] == before + 1
    with pytest.raises(KeyError):
        apply_override(cfg, "not_a_parameter", 1)


def test_uplift_linear_in_observed_pd(book):
    # PD_0 scales with observed PD and nothing caps, so the uplift scales exactly
    loans, div, yearly = book
    cfg = base_config()
    a = evaluate(cfg, div, loans, yearly)
    b = evaluate(apply_override(cfg, "pd_base", 2 * cfg["portfolio"]["pd_base"]), div, loans, yearly)
    assert b["el_uplift"] == pytest.approx(2 * a["el_uplift"])


def test_directions(book):
    loans, div, yearly = book
    cfg = base_config()
    base = evaluate(cfg, div, loans, yearly)["el_uplift"]
    assert base > 0
    assert evaluate(apply_override(cfg, "hot_house_multiplier", 1.0), div, loans,
                    yearly)["el_uplift"] == pytest.approx(0.0)
    assert evaluate(apply_override(cfg, "hazard_headroom", 3.0), div, loans, yearly)["el_uplift"] < base
    assert evaluate(apply_override(cfg, "beta_pd_median", 3.0), div, loans, yearly)["el_uplift"] > base
    # recent seasons burned more in A, so the 2015-2025 window raises the uplift
    assert evaluate(apply_override(cfg, "fire_window", [2015, 2025]), div, loans,
                    yearly)["el_uplift"] > base


def test_tornado_table(book):
    loans, div, yearly = book
    specs = {"beta_pd_median": {"low": 1.0, "high": 3.0},
             "fire_window": {"low": [1990, 2025], "high": [2015, 2025]}}
    t = tornado(div, loans, yearly, specs, n_runs=200, seed=3)
    assert list(t["el_uplift_swing"]) == sorted(t["el_uplift_swing"], reverse=True)
    w = t.set_index("parameter").loc["fire_window"]
    # one-sided: the low end is the base window, so it equals the base exactly
    assert w["el_uplift_low"] == w["el_uplift_base"]
    assert w["es_uplift_low"] == w["es_uplift_base"]
    assert (t["es_uplift_swing"] >= 0).all()


@pytest.mark.skipif(not (PROCESSED / "results.parquet").exists(), reason="run `python -m src.simulate` first")
def test_base_matches_m4_analytic():
    div = pd.read_parquet(PROCESSED / "divisions.parquet")
    loans = pd.read_parquet(PROCESSED / "loans.parquet")
    yearly = pd.read_parquet(PROCESSED / "burn_by_year.parquet")
    res = pd.read_parquet(PROCESSED / "results.parquet")
    bc = res[res["cduid"] == PORTFOLIO_ID].set_index("scenario")["el_analytic"]
    e = evaluate(base_config(), div, loans, yearly)
    assert e["el_uplift"] == pytest.approx(bc["hot_house"] - bc["historical"], rel=1e-9)
