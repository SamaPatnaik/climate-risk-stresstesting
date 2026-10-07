"""M4 tests on a small synthetic book (high PD so tails are populated and runs are fast)."""
import numpy as np
import pandas as pd
import pytest

from src.model import (PORTFOLIO_ID, REFERENCE_SCENARIO, calibrate_pd, deterministic_el,
                       expected_el_over_seasons)
from src.simulate import n_tail, simulate, summarize_runs, tail_stats

SCEN = {REFERENCE_SCENARIO: 1.0, "orderly": 1.2, "hot_house": 2.0}
MP = {"beta_pd": {"median": 2.0, "sigma": 0.5}, "beta_lgd": {"median": 0.15, "sigma": 0.5},
      "climate_lgd_applies_to_insured": False}
BPD = 2.0 * np.exp(0.125)
BLGD = 0.15 * np.exp(0.125)
CDS = ["A", "B", "S"]  # S is a low-count CD
N_RUNS = 600
MIN_LOANS = 100


@pytest.fixture(scope="module")
def book():
    rng = np.random.default_rng(0)
    counts = {"A": 1500, "B": 1000, "S": 40}
    cd = np.repeat(list(counts), list(counts.values()))
    n = len(cd)
    insured = rng.random(n) < 0.25
    loans = pd.DataFrame({"loan_id": np.arange(n), "cduid": cd,
                          "balance": rng.lognormal(np.log(300_000), 0.5, n),
                          "insured": insured, "pd_base": 0.05,
                          "lgd_base": np.where(insured, 0.0, 0.2)})
    # 4 seasons: 1990 no fire anywhere; 2023 a bad year hitting A and B together
    yearly = pd.DataFrame({
        "cduid": np.tile(CDS, 4),
        "year": np.repeat([1990, 2003, 2017, 2023], 3),
        "hazard": [0.0, 0.0, 0.0,
                   0.10, 0.00, 0.02,
                   0.25, 0.05, 0.05,
                   0.50, 0.30, 0.10]})
    div = (yearly.groupby("cduid")["hazard"].mean().rename("hazard_score").reset_index()
           .assign(vulnerability=1.0))
    lx = calibrate_pd(loans, div, BPD)
    cd_losses, runs = simulate(lx, CDS, yearly, SCEN, MP, N_RUNS, seed=7, chunk=64)
    analytic = expected_el_over_seasons(lx, div, yearly, SCEN, BPD, BLGD)
    res = summarize_runs(cd_losses, CDS, analytic, MIN_LOANS)
    return dict(lx=lx, yearly=yearly, div=div, cd_losses=cd_losses, runs=runs, res=res)


def test_reproducible_and_chunk_invariant(book):
    again, runs2 = simulate(book["lx"], CDS, book["yearly"], SCEN, MP, N_RUNS, seed=7, chunk=25)
    pd.testing.assert_frame_equal(book["runs"], runs2)
    for s in SCEN:
        np.testing.assert_allclose(book["cd_losses"][s], again[s])


def test_mc_mean_matches_analytic(book):
    # E[loss] = season-averaged EL at E[beta] (bilinear in independent betas)
    res = book["res"]
    z = (res["el_mean"] - res["el_analytic"]) / res["el_mean_se"]
    assert (z.abs() < 4).all(), res.assign(z=z)[["cduid", "scenario", "z"]]


def test_mc_uplift_matches_analytic(book):
    # uplift has a much smaller SE (common random numbers), so this is the sharper check
    res = book["res"]
    ref = res[res["scenario"] == REFERENCE_SCENARIO].set_index("cduid")["el_analytic"]
    r = res[res["scenario"] != REFERENCE_SCENARIO]
    z = (r["el_uplift"] - (r["el_analytic"] - r["cduid"].map(ref))) / r["el_uplift_se"]
    assert (z.abs() < 4).all(), r.assign(z=z)[["cduid", "scenario", "z"]]


def test_jensen_gap(book):
    # EL is quadratic in H, so season-averaged E[EL] exceeds EL at the mean hazard
    # when hazard varies across seasons, and the two agree when it doesn't
    lx, div, yearly = book["lx"], book["div"], book["yearly"]
    avg = expected_el_over_seasons(lx, div, yearly, SCEN, BPD, BLGD).set_index(["scenario", "cduid"])
    at_mean = deterministic_el(lx, div, SCEN, BPD, BLGD).set_index(["scenario", "cduid"])
    assert (avg.loc["hot_house", "el"] > at_mean.loc["hot_house", "el"]).loc[["A", "B"]].all()
    flat = yearly.assign(hazard=yearly["cduid"].map(div.set_index("cduid")["hazard_score"]))
    avg_flat = expected_el_over_seasons(lx, div, flat, SCEN, BPD, BLGD)
    np.testing.assert_allclose(avg_flat["el"], deterministic_el(lx, div, SCEN, BPD, BLGD)["el"])


def test_portfolio_tails_from_run_totals_not_sums(book):
    res = book["res"].set_index(["scenario", "cduid"])
    for s, L in book["cd_losses"].items():
        t = tail_stats(L.sum(axis=1))
        assert res.loc[(s, PORTFOLIO_ID), "loss_p99"] == pytest.approx(t["loss_p99"])
        assert res.loc[(s, PORTFOLIO_ID), "es_99"] == pytest.approx(t["es_99"])
        # diversification: portfolio ES <= sum of CD ES (empirical top-k ES is subadditive)
        cd_es = tail_stats(L)["es_99"].sum()
        assert t["es_99"] <= cd_es + 1e-6


def test_tail_ordering(book):
    r = book["res"].dropna(subset=["loss_p99"])
    assert (r["loss_p95"] <= r["loss_p99"] + 1e-9).all()
    assert (r["loss_p99"] <= r["es_99"] + 1e-9).all()
    assert (r["el_mean"] <= r["es_99"] + 1e-9).all()


def test_common_random_numbers_monotone_across_scenarios(book):
    L = book["cd_losses"]
    assert (L[REFERENCE_SCENARIO] <= L["orderly"] + 1e-9).all()
    assert (L["orderly"] <= L["hot_house"] + 1e-9).all()


def test_fire_season_bootstrap(book):
    runs, L = book["runs"], book["cd_losses"]
    assert set(runs["year"]) == {1990, 2003, 2017, 2023}  # every season drawn, nothing else
    # no-fire season: hazard 0 everywhere, so all scenarios give identical losses
    calm = (runs["year"] == 1990).to_numpy()
    np.testing.assert_allclose(L["hot_house"][calm], L[REFERENCE_SCENARIO][calm])
    # bad season raises losses in both affected CDs together
    bad = (runs["year"] == 2023).to_numpy()
    assert (L["hot_house"][bad].mean(axis=0)[:2] > L["hot_house"][calm].mean(axis=0)[:2]).all()


def test_uplift_vs_reference_and_no_climate(book):
    r = book["res"].set_index(["scenario", "cduid"])
    assert r.loc[(REFERENCE_SCENARIO, PORTFOLIO_ID), "el_uplift"] == pytest.approx(0.0)
    assert r.loc[("hot_house", PORTFOLIO_ID), "el_uplift"] > 0
    # vs no-climate is larger: it also counts historical hazard
    assert (r.loc[("hot_house", PORTFOLIO_ID), "el_uplift_vs_no_climate"]
            > r.loc[("hot_house", PORTFOLIO_ID), "el_uplift"])


def test_tail_uplift_columns(book):
    r = book["res"].set_index(["scenario", "cduid"])
    for col in ["loss_p95", "loss_p99", "es_99"]:
        assert r.loc[(REFERENCE_SCENARIO, PORTFOLIO_ID), f"{col}_uplift"] == pytest.approx(0.0)
        assert r.loc[("hot_house", PORTFOLIO_ID), f"{col}_uplift"] == pytest.approx(
            r.loc[("hot_house", PORTFOLIO_ID), col] - r.loc[(REFERENCE_SCENARIO, PORTFOLIO_ID), col])
    # common random numbers: each run's loss is non-decreasing in m, so tail quantiles are too
    assert r.loc[("hot_house", PORTFOLIO_ID), "es_99_uplift"] >= 0


def test_low_count_tails_suppressed(book):
    r = book["res"]
    small = r[r["cduid"] == "S"]
    assert small["tail_suppressed"].all()
    assert small[["loss_p95", "loss_p99", "es_99", "loss_p95_uplift", "loss_p99_uplift",
                  "es_99_uplift"]].isna().all().all()
    assert small["el_mean"].notna().all()
    big = r[r["cduid"].isin(["A", "B", PORTFOLIO_ID])]
    assert not big["tail_suppressed"].any()
    assert big[["loss_p95", "loss_p99", "es_99"]].notna().all().all()
    # S's loans still count in the BC portfolio
    bc_loans = r.loc[r["cduid"] == PORTFOLIO_ID, "n_loans"].iloc[0]
    assert bc_loans == r.loc[(r["cduid"] != PORTFOLIO_ID) & (r["scenario"] == REFERENCE_SCENARIO),
                             "n_loans"].sum()


def test_es_definition():
    x = np.arange(1, 201, dtype=float)  # worst 1% of 200 = top 2 -> mean(199, 200)
    t = tail_stats(x)
    assert t["es_99"] == pytest.approx(199.5)
    assert t["loss_p99"] <= t["es_99"]


def test_n_tail_float_safe():
    # (1 - 0.99) * 1000 = 10.000000000000009; naive ceil gives 11
    assert n_tail(1000) == 10
    assert n_tail(200) == 2
    assert n_tail(50) == 1
