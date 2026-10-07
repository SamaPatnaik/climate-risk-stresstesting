"""M3: deterministic climate-stressed credit loss model.

Per loan, for a scenario with hazard multiplier m:
    H   = min(1, h x m)                               stressed hazard
    PD  = min(1, PD_0 x (1 + beta_pd x H))
    LGD = min(1, LGD_base + beta_lgd x H x v)         uninsured loans only by default
    EL  = PD x LGD x EAD
PD_0 is the no-climate intercept, calibrated so the portfolio PD at the reference
scenario (historical climate, m = 1) equals the observed pd_base (see
calibrate_pd). Climate uplift is EL minus EL at m = 1; EL at H = 0 is kept as
a secondary column.

EL is bilinear in (beta_pd, beta_lgd), which are independent, so E over betas =
EL at E[beta]. But EL is QUADRATIC in H (the PD and LGD uplifts multiply), so
E over fire seasons of EL > EL at the mean hazard (Jensen). The Monte Carlo
mean therefore converges to expected_el_over_seasons(...), not to
deterministic_el(...) at the mean hazard; the gap is the extra loss from fire
risk arriving in concentrated bad seasons. PD alone is linear in H, so the PD
calibration at the mean hazard is exact.

Usage (from project root):
    python -m src.model
"""
import numpy as np
import pandas as pd

from src.assumptions import values
from src.hazard import PROCESSED

TAIL_COLS = ["loss_p95", "loss_p99", "es_99"]
PORTFOLIO_ID = "BC"
REFERENCE_SCENARIO = "historical"


def stress_hazard(h, m):
    return np.minimum(1.0, np.asarray(h, dtype=float) * m)


def stressed_pd(pd_base, H, beta_pd):
    return np.minimum(1.0, np.asarray(pd_base) * (1 + beta_pd * np.asarray(H)))


def stressed_lgd(lgd_base, H, vulnerability, beta_lgd, insured=False, applies_to_insured=False):
    """Insured loans keep LGD_base unless applies_to_insured (insurer absorbs the shortfall)."""
    stressed = np.minimum(1.0, np.asarray(lgd_base) + beta_lgd * np.asarray(H) * vulnerability)
    if applies_to_insured:
        return stressed
    return np.where(insured, lgd_base, stressed)


def expected_loss(pd_, lgd, ead):
    return np.asarray(pd_) * np.asarray(lgd) * np.asarray(ead)


def lognormal_mean(spec: dict) -> float:
    """E[X] for X ~ lognormal with given median and log-sd: median x exp(sigma^2 / 2)."""
    return spec["median"] * np.exp(spec["sigma"] ** 2 / 2)


def attach_hazard(loans: pd.DataFrame, div: pd.DataFrame) -> pd.DataFrame:
    """Add each loan's CD hazard_score and vulnerability (no-op if already attached)."""
    if "hazard_score" in loans.columns:
        return loans.copy()
    out = loans.merge(div[["cduid", "hazard_score", "vulnerability"]], on="cduid", how="left")
    assert out["hazard_score"].notna().all(), "loans reference unknown cduid"
    return out


def calibrate_pd(loans: pd.DataFrame, div: pd.DataFrame, beta_pd_mean: float) -> pd.DataFrame:
    """Replace observed pd_base with the no-climate intercept PD_0.

    PD_0 = pd_observed / mean_loans(1 + E[beta_pd] x h), so the loan-count-weighted
    mean PD at m = 1 equals the observed rate. Keeps pd_observed for reference.
    """
    out = attach_hazard(loans, div)
    factor = (1 + beta_pd_mean * out["hazard_score"]).mean()
    out["pd_observed"] = out["pd_base"]
    out["pd_base"] = out["pd_base"] / factor
    return out


def loan_el(loans: pd.DataFrame, m: float, beta_pd: float, beta_lgd: float,
            applies_to_insured: bool = False) -> np.ndarray:
    """Per-loan EL for one scenario. `loans` must already carry hazard_score and vulnerability."""
    H = stress_hazard(loans["hazard_score"], m)
    pd_ = stressed_pd(loans["pd_base"], H, beta_pd)
    lgd = stressed_lgd(loans["lgd_base"], H, loans["vulnerability"], beta_lgd,
                       loans["insured"], applies_to_insured)
    return expected_loss(pd_, lgd, loans["balance"])


def deterministic_el(loans: pd.DataFrame, div: pd.DataFrame, multipliers: dict,
                     beta_pd: float, beta_lgd: float, applies_to_insured: bool = False) -> pd.DataFrame:
    """EL per (CD, scenario) plus a portfolio row (cduid = "BC").

    `multipliers` must include the reference scenario; el_uplift is measured against it.
    """
    assert REFERENCE_SCENARIO in multipliers, f"missing reference scenario {REFERENCE_SCENARIO!r}"
    lx = attach_hazard(loans, div)
    lx["el_no_climate"] = expected_loss(lx["pd_base"], lx["lgd_base"], lx["balance"])
    rows = []
    for scenario, m in multipliers.items():
        lx["el"] = loan_el(lx, m, beta_pd, beta_lgd, applies_to_insured)
        g = lx.groupby("cduid").agg(n_loans=("loan_id", "size"), ead_total=("balance", "sum"),
                                    el=("el", "sum"), el_no_climate=("el_no_climate", "sum"))
        g.loc[PORTFOLIO_ID] = g.sum()
        g["scenario"], g["multiplier"] = scenario, m
        rows.append(g)
    out = pd.concat(rows).rename_axis("cduid").reset_index()
    return add_uplift_columns(out, "el")


def expected_el_over_seasons(loans: pd.DataFrame, div: pd.DataFrame, yearly: pd.DataFrame,
                             multipliers: dict, beta_pd: float, beta_lgd: float,
                             applies_to_insured: bool = False) -> pd.DataFrame:
    """E[EL] under the fire-season bootstrap: mean over years (equal weights) of
    deterministic_el using that year's hazard h_y in every CD. Same columns as deterministic_el."""
    base = loans.drop(columns=["hazard_score", "vulnerability"], errors="ignore")
    per_year = []
    for _, g in yearly.groupby("year"):
        div_y = div.drop(columns="hazard_score").merge(
            g[["cduid", "hazard"]].rename(columns={"hazard": "hazard_score"}), on="cduid")
        per_year.append(deterministic_el(base, div_y, multipliers, beta_pd, beta_lgd,
                                         applies_to_insured))
    allyears = pd.concat(per_year)
    out = (allyears.groupby(["cduid", "scenario"], sort=False)
           .agg(n_loans=("n_loans", "first"), ead_total=("ead_total", "first"),
                el=("el", "mean"), el_no_climate=("el_no_climate", "first"),
                multiplier=("multiplier", "first"))
           .reset_index())
    return add_uplift_columns(out, "el")


def add_uplift_columns(res: pd.DataFrame, el_col: str) -> pd.DataFrame:
    """Uplift vs the reference scenario (primary) and vs H = 0 (secondary), per CD."""
    ref = res.loc[res["scenario"] == REFERENCE_SCENARIO].set_index("cduid")[el_col]
    out = res.copy()
    out["el_ref"] = out["cduid"].map(ref)
    out["el_uplift"] = out[el_col] - out["el_ref"]
    out["el_uplift_pct"] = np.where(out["el_ref"] > 0, out["el_uplift"] / out["el_ref"] * 100, np.nan)
    out["el_uplift_vs_no_climate"] = out[el_col] - out["el_no_climate"]
    out["el_rate_bps"] = out[el_col] / out["ead_total"] * 1e4
    return out


def suppress_small_tails(results: pd.DataFrame, min_loans: int) -> pd.DataFrame:
    """Blank tail measures for CDs with n_loans < min_loans; the BC portfolio row is never blanked.

    Adds `tail_suppressed` so the API/frontend can say why a value is missing.
    """
    out = results.copy()
    small = (out["n_loans"] < min_loans) & (out["cduid"] != PORTFOLIO_ID)
    out["tail_suppressed"] = small
    # tail measures and anything derived from them (e.g. loss_p99_uplift)
    cols = [c for c in out.columns if any(c.startswith(t) for t in TAIL_COLS)]
    out[cols] = out[cols].astype(float)
    out.loc[small, cols] = np.nan
    return out


def main() -> None:
    div = pd.read_parquet(PROCESSED / "divisions.parquet")
    loans = pd.read_parquet(PROCESSED / "loans.parquet")
    mp = values("model")
    bpd, blgd = lognormal_mean(mp["beta_pd"]), lognormal_mean(mp["beta_lgd"])
    lx = calibrate_pd(loans, div, bpd)
    print(f"PD calibration: observed {lx['pd_observed'].iloc[0]:.4%} -> "
          f"no-climate intercept PD_0 {lx['pd_base'].iloc[0]:.4%}")
    res = deterministic_el(lx, div, values("scenarios"), bpd, blgd,
                           mp["climate_lgd_applies_to_insured"])

    bc = res[res["cduid"] == PORTFOLIO_ID].set_index("scenario")
    print("\nBC portfolio, deterministic EL at E[beta] (ILLUSTRATIVE scenarios, reference = historical):")
    print(bc[["multiplier", "el", "el_uplift", "el_uplift_pct", "el_no_climate",
              "el_rate_bps"]].to_string(float_format=lambda x: f"{x:,.2f}"))

    hh = res[(res["scenario"] == "hot_house") & (res["cduid"] != PORTFOLIO_ID)]
    hh = hh.merge(div[["cduid", "cd_name", "hazard_score"]], on="cduid")
    print("\nTop CDs by EL uplift vs historical, hot_house:")
    print(hh.sort_values("el_uplift", ascending=False).head(8)[
        ["cduid", "cd_name", "n_loans", "hazard_score", "el_uplift", "el_uplift_pct"]].to_string(
        index=False, float_format=lambda x: f"{x:,.4f}" if abs(x) < 1 else f"{x:,.2f}"))


if __name__ == "__main__":
    main()
