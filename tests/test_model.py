"""M3 tests: hand-computed worked example, model identities, and tail suppression."""
import numpy as np
import pandas as pd
import pytest

from src.model import (PORTFOLIO_ID, REFERENCE_SCENARIO, calibrate_pd, deterministic_el,
                       expected_loss, lognormal_mean, loan_el, stress_hazard, stressed_lgd,
                       stressed_pd, suppress_small_tails)

# ---------------------------------------------------------------------------
# Worked example (computed by hand; README uses the same numbers)
#
#   One uninsured loan in a CD with hazard score h = 0.31, vulnerability v = 1
#   EAD = $300,000   PD_base = 0.26%   LGD_base = 20%
#   Scenario: current_policies, m = 1.6     beta_PD = 2.0   beta_LGD = 0.15
#
#   H   = min(1, 0.31 x 1.6)                = 0.496
#   PD  = 0.0026 x (1 + 2.0 x 0.496)
#       = 0.0026 x 1.992                    = 0.0051792   (0.52%, ~2x baseline)
#   LGD = min(1, 0.20 + 0.15 x 0.496 x 1)
#       = 0.20 + 0.0744                     = 0.2744
#   EL  = 0.0051792 x 0.2744 x 300,000
#       = 0.0051792 x 82,320                = $426.35
#
#   Reference (historical, m = 1): H = 0.31
#     PD = 0.0026 x 1.62 = 0.004212, LGD = 0.2465 -> EL = $311.48
#   Climate uplift vs historical = 426.35 - 311.48 = $114.87  (+37%)
#   Secondary, vs no wildfire (H = 0): EL = 0.0026 x 0.20 x 300,000 = $156.00
#     -> uplift vs no climate = $270.35
#   (PD_base here is used as given; in the pipeline it is first calibrated to
#    the no-climate intercept PD_0, see test_pd_calibration_matches_observed_at_reference.)
#
#   Same loan, insured: LGD stays 0 (insurer absorbs loss), so EL = $0
#   despite PD rising to 0.0051792.
#
#   Cap case: h = 0.8, m = 2.0 -> h x m = 1.6, H = min(1, 1.6) = 1
#   PD = 0.0026 x 3 = 0.0078   LGD = 0.20 + 0.15 = 0.35
#   EL = 0.0078 x 0.35 x 300,000 = $819.00
# ---------------------------------------------------------------------------
EX = dict(h=0.31, v=1.0, ead=300_000, pd_base=0.0026, lgd_base=0.20,
          m=1.6, beta_pd=2.0, beta_lgd=0.15)


def test_worked_example_step_by_step():
    H = stress_hazard(EX["h"], EX["m"])
    assert H == pytest.approx(0.496)
    pd_ = stressed_pd(EX["pd_base"], H, EX["beta_pd"])
    assert pd_ == pytest.approx(0.0051792)
    lgd = stressed_lgd(EX["lgd_base"], H, EX["v"], EX["beta_lgd"], insured=False)
    assert lgd == pytest.approx(0.2744)
    el = expected_loss(pd_, lgd, EX["ead"])
    assert el == pytest.approx(426.351744)
    el_base = expected_loss(EX["pd_base"], EX["lgd_base"], EX["ead"])
    assert el_base == pytest.approx(156.0)
    assert el - el_base == pytest.approx(270.351744)


def test_worked_example_insured():
    H = stress_hazard(EX["h"], EX["m"])
    lgd = stressed_lgd(0.0, H, EX["v"], EX["beta_lgd"], insured=True, applies_to_insured=False)
    assert lgd == 0.0
    assert expected_loss(stressed_pd(EX["pd_base"], H, EX["beta_pd"]), lgd, EX["ead"]) == 0.0


def test_worked_example_cap():
    H = stress_hazard(0.8, 2.0)
    assert H == 1.0
    pd_ = stressed_pd(EX["pd_base"], H, EX["beta_pd"])
    lgd = stressed_lgd(EX["lgd_base"], H, 1.0, EX["beta_lgd"])
    assert pd_ == pytest.approx(0.0078)
    assert lgd == pytest.approx(0.35)
    assert expected_loss(pd_, lgd, EX["ead"]) == pytest.approx(819.0)


def test_worked_example_through_loan_el():
    loans = pd.DataFrame({"hazard_score": [EX["h"], EX["h"]], "vulnerability": [1.0, 1.0],
                          "pd_base": [0.0026, 0.0026], "lgd_base": [0.20, 0.0],
                          "insured": [False, True], "balance": [300_000.0, 300_000.0]})
    el = loan_el(loans, EX["m"], EX["beta_pd"], EX["beta_lgd"])
    np.testing.assert_allclose(el, [426.351744, 0.0])


# --- model identities -------------------------------------------------------

def test_no_climate_identities():
    # m = 0 or beta = 0 reproduces the baseline
    assert stressed_pd(0.0026, stress_hazard(0.5, 0.0), 2.0) == pytest.approx(0.0026)
    assert stressed_pd(0.0026, 0.7, 0.0) == pytest.approx(0.0026)
    assert stressed_lgd(0.20, 0.7, 1.0, 0.0) == pytest.approx(0.20)


def test_monotone_in_hazard():
    H = np.linspace(0, 1, 11)
    el = expected_loss(stressed_pd(0.0026, H, 2.0), stressed_lgd(0.20, H, 1.0, 0.15), 1.0)
    assert (np.diff(el) > 0).all()


def test_caps():
    assert stressed_pd(0.6, 1.0, 5.0) == 1.0
    assert stressed_lgd(0.9, 1.0, 1.0, 0.5) == 1.0
    assert stress_hazard(3.0, 2.0) == 1.0


def test_lognormal_mean():
    assert lognormal_mean({"median": 2.0, "sigma": 0.5}) == pytest.approx(2.0 * np.exp(0.125))


# --- aggregation ------------------------------------------------------------

@pytest.fixture
def small_book():
    div = pd.DataFrame({"cduid": ["5901", "5915"], "hazard_score": [0.31, 0.0],
                        "vulnerability": [1.0, 1.0]})
    loans = pd.DataFrame({"loan_id": [0, 1, 2], "cduid": ["5901", "5901", "5915"],
                          "balance": [300_000.0, 300_000.0, 500_000.0],
                          "insured": [False, True, False], "pd_base": 0.0026,
                          "lgd_base": [0.20, 0.0, 0.20]})
    return loans, div


SCEN = {REFERENCE_SCENARIO: 1.0, "orderly": 1.2, "current_policies": 1.6}


def test_deterministic_el_aggregation(small_book):
    loans, div = small_book
    res = deterministic_el(loans, div, SCEN, 2.0, 0.15)
    cp = res[res["scenario"] == "current_policies"].set_index("cduid")
    # 5901: worked-example loan + insured loan (EL 0); 5915: h = 0 so EL = baseline
    assert cp.loc["5901", "el"] == pytest.approx(426.351744)
    assert cp.loc["5915", "el"] == pytest.approx(0.0026 * 0.20 * 500_000)
    assert cp.loc["5915", "el_uplift"] == pytest.approx(0.0)
    assert cp.loc[PORTFOLIO_ID, "el"] == pytest.approx(cp.loc[["5901", "5915"], "el"].sum())
    assert cp.loc[PORTFOLIO_ID, "n_loans"] == 3
    # higher multiplier -> higher portfolio EL
    bc = res[res["cduid"] == PORTFOLIO_ID].set_index("scenario")["el"]
    assert bc[REFERENCE_SCENARIO] < bc["orderly"] < bc["current_policies"]


def test_uplift_measured_against_reference(small_book):
    # worked-example loan at m = 1: H = 0.31, PD = 0.0026 x 1.62 = 0.004212,
    # LGD = 0.20 + 0.15 x 0.31 = 0.2465, EL = 0.004212 x 0.2465 x 300,000 = $311.48
    # -> current_policies uplift vs historical = 426.35 - 311.48 = $114.87
    loans, div = small_book
    res = deterministic_el(loans, div, SCEN, 2.0, 0.15)
    r = res.set_index(["scenario", "cduid"])
    el_ref = 0.0026 * (1 + 2.0 * 0.31) * (0.20 + 0.15 * 0.31) * 300_000
    assert el_ref == pytest.approx(311.4774)
    assert r.loc[(REFERENCE_SCENARIO, "5901"), "el"] == pytest.approx(el_ref)
    assert r.loc[(REFERENCE_SCENARIO, "5901"), "el_uplift"] == pytest.approx(0.0)
    assert r.loc[("current_policies", "5901"), "el_uplift"] == pytest.approx(426.351744 - el_ref)
    # secondary column: uplift vs H = 0 is unchanged from the worked example
    assert r.loc[("current_policies", "5901"), "el_uplift_vs_no_climate"] == pytest.approx(270.351744)


def test_reference_scenario_required(small_book):
    loans, div = small_book
    with pytest.raises(AssertionError):
        deterministic_el(loans, div, {"orderly": 1.2}, 2.0, 0.15)


def test_pd_calibration_matches_observed_at_reference(small_book):
    loans, div = small_book
    lx = calibrate_pd(loans, div, beta_pd_mean=2.0)
    # loan-count-weighted mean PD at m = 1 equals the observed 0.26%
    pd_ref = stressed_pd(lx["pd_base"], stress_hazard(lx["hazard_score"], 1.0), 2.0)
    assert pd_ref.mean() == pytest.approx(0.0026)
    assert (lx["pd_observed"] == 0.0026).all()
    # hand check: hazards (0.31, 0.31, 0) -> factor = mean(1.62, 1.62, 1) = 1.41333
    assert lx["pd_base"].iloc[0] == pytest.approx(0.0026 / (4.24 / 3))


# --- tail suppression ---------------------------------------------------------

def test_suppress_small_tails():
    res = pd.DataFrame({"cduid": ["5957", "5959", "5915", PORTFOLIO_ID],
                        "n_loans": [3, 99, 49_192, 3],  # BC row never suppressed
                        "el_mean": [1.0, 2.0, 3.0, 4.0],
                        "loss_p95": [5, 6, 7, 8], "loss_p99": [9, 10, 11, 12],
                        "es_99": [13, 14, 15, 16]})
    out = suppress_small_tails(res, min_loans=100).set_index("cduid")
    assert out.loc[["5957", "5959"], ["loss_p95", "loss_p99", "es_99"]].isna().all().all()
    assert out.loc[["5915", PORTFOLIO_ID], ["loss_p95", "loss_p99", "es_99"]].notna().all().all()
    assert out["el_mean"].notna().all()  # mean EL always reported
    assert out["tail_suppressed"].tolist() == [True, True, False, False]
