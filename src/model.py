"""M3: deterministic climate-stressed credit loss model.

Per loan, for a scenario with hazard multiplier m:
    H   = min(1, h x m)                               stressed hazard
    PD  = min(1, PD_base x (1 + beta_pd x H))
    LGD = min(1, LGD_base + beta_lgd x H x v)         uninsured loans only by default
    EL  = PD x LGD x EAD
The climate uplift is EL minus EL with H = 0 (PD_base, LGD_base).

With independent betas and caps not binding, EL is linear in beta_pd, beta_lgd
and their product, so E[EL] = EL evaluated at E[beta_pd], E[beta_lgd]. M4's
Monte Carlo mean should converge to deterministic_el(...) at the beta means.

Usage (from project root):
    python -m src.model
"""
import numpy as np
import pandas as pd

from src.assumptions import values
from src.hazard import PROCESSED

TAIL_COLS = ["loss_p95", "loss_p99", "es_99"]
PORTFOLIO_ID = "BC"


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
    """Add each loan's CD hazard_score and vulnerability."""
    out = loans.merge(div[["cduid", "hazard_score", "vulnerability"]], on="cduid", how="left")
    assert out["hazard_score"].notna().all(), "loans reference unknown cduid"
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
    """EL per (CD, scenario) plus a portfolio row (cduid = "BC")."""
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
    out["el_uplift"] = out["el"] - out["el_no_climate"]
    out["el_rate_bps"] = out["el"] / out["ead_total"] * 1e4
    return out


def suppress_small_tails(results: pd.DataFrame, min_loans: int) -> pd.DataFrame:
    """Blank tail measures for CDs with n_loans < min_loans; the BC portfolio row is never blanked.

    Adds `tail_suppressed` so the API/frontend can say why a value is missing.
    """
    out = results.copy()
    small = (out["n_loans"] < min_loans) & (out["cduid"] != PORTFOLIO_ID)
    out["tail_suppressed"] = small
    cols = [c for c in TAIL_COLS if c in out.columns]
    out[cols] = out[cols].astype(float)
    out.loc[small, cols] = np.nan
    return out


def main() -> None:
    div = pd.read_parquet(PROCESSED / "divisions.parquet")
    loans = pd.read_parquet(PROCESSED / "loans.parquet")
    mp = values("model")
    res = deterministic_el(loans, div, values("scenarios"),
                           lognormal_mean(mp["beta_pd"]), lognormal_mean(mp["beta_lgd"]),
                           mp["climate_lgd_applies_to_insured"])

    bc = res[res["cduid"] == PORTFOLIO_ID].set_index("scenario")
    print("BC portfolio, deterministic EL at E[beta] (ILLUSTRATIVE scenarios):")
    print(bc[["multiplier", "el", "el_no_climate", "el_uplift", "el_rate_bps"]].to_string(
        float_format=lambda x: f"{x:,.2f}"))

    hh = res[(res["scenario"] == "hot_house") & (res["cduid"] != PORTFOLIO_ID)]
    hh = hh.merge(div[["cduid", "cd_name", "hazard_score"]], on="cduid")
    hh["uplift_pct"] = hh["el_uplift"] / hh["el_no_climate"] * 100
    print("\nTop CDs by EL uplift, hot_house:")
    print(hh.sort_values("el_uplift", ascending=False).head(8)[
        ["cduid", "cd_name", "n_loans", "hazard_score", "el_uplift", "uplift_pct"]].to_string(
        index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
