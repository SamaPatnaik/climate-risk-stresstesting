"""M4: Monte Carlo credit loss with fire-season bootstrap.

Each run r draws, once and shared across all scenarios (common random numbers):
  - a historical fire season y_r (uniform over the window years); every CD takes
    its own burn hazard h_{cd, y_r} from that year, so bad seasons hit many CDs at once
  - beta_pd_r, beta_lgd_r from their lognormal distributions
  - one uniform U_{i,r} per loan; loan i defaults in scenario s if U_{i,r} < PD_{i,r,s}
Loss_{i,r,s} = default x LGD x EAD. Because PD and LGD rise with m and the draws are
shared, every run's loss is non-decreasing across scenarios.

Risk measures:
  - division rows: mean, p95, p99, ES_99 of that division's run losses; tails are
    suppressed below reporting.min_loans_for_tail (tail_suppressed = True)
  - portfolio row (cduid = "BC"): computed from the per-run portfolio total, never by
    summing division tails (VaR is not additive; ES is subadditive)
ES_99 = mean of the worst ceil(1% x n_runs) runs.

Outputs (data/processed/):
  results.parquet / results.json   one row per (CD or BC, scenario)
  portfolio_runs.parquet           one row per (run, scenario): year, betas, BC loss

Usage (from project root):
    python -m src.simulate [--runs 1000]
"""
import argparse
import math

import numpy as np
import pandas as pd

from src.assumptions import values
from src.hazard import PROCESSED
from src.model import (PORTFOLIO_ID, REFERENCE_SCENARIO, TAIL_COLS, add_uplift_columns, calibrate_pd,
                       deterministic_el, expected_el_over_seasons, lognormal_mean,
                       suppress_small_tails)

ES_ALPHA = 0.99
CHUNK_RUNS = 25  # runs simulated at once; peak memory ~ 5 x CHUNK_RUNS x n_loans x 8 bytes


def n_tail(n: int, alpha: float = ES_ALPHA) -> int:
    """Number of worst runs averaged by ES_alpha: ceil((1 - alpha) n), at least 1.

    Rounded before ceil: (1 - 0.99) x 1000 is 10.000000000000009 in floating point.
    """
    return max(1, math.ceil(round((1 - alpha) * n, 9)))


def tail_stats(losses: np.ndarray, alpha: float = ES_ALPHA) -> dict:
    """p95, p99 and ES_alpha (mean of the worst n_tail(n) runs) along axis 0."""
    n = losses.shape[0]
    k = n_tail(n, alpha)
    worst = np.sort(losses, axis=0)[n - k:]
    return {"loss_p95": np.quantile(losses, 0.95, axis=0),
            "loss_p99": np.quantile(losses, 0.99, axis=0),
            "es_99": worst.mean(axis=0)}


def draw_run_inputs(rng: np.random.Generator, n_runs: int, years: np.ndarray, mp: dict) -> pd.DataFrame:
    bp, bl = mp["beta_pd"], mp["beta_lgd"]
    return pd.DataFrame({
        "run": np.arange(n_runs),
        "year": rng.choice(years, size=n_runs, replace=True),
        "beta_pd": rng.lognormal(np.log(bp["median"]), bp["sigma"], n_runs),
        "beta_lgd": rng.lognormal(np.log(bl["median"]), bl["sigma"], n_runs),
    })


def simulate(lx: pd.DataFrame, cduids: list, yearly: pd.DataFrame, scenarios: dict, mp: dict,
             n_runs: int, seed: int, chunk: int = CHUNK_RUNS):
    """Simulate run losses per CD.

    lx: calibrated loans (calibrate_pd) with cduid, balance, pd_base, lgd_base, insured, vulnerability.
    yearly: burn_by_year with cduid, year, hazard.
    Returns (cd_losses {scenario: (n_runs, n_cd) array}, runs DataFrame).
    """
    rng = np.random.default_rng(seed)
    h_table = yearly.pivot(index="year", columns="cduid", values="hazard").reindex(columns=cduids)
    assert h_table.notna().all().all(), "burn_by_year missing CD-years"
    years = h_table.index.to_numpy()
    runs = draw_run_inputs(rng, n_runs, years, mp)
    year_pos = pd.Index(years).get_indexer(runs["year"])
    h = h_table.to_numpy()  # (n_years, n_cd)

    cd_pos = pd.Index(cduids).get_indexer(lx["cduid"])
    assert (cd_pos >= 0).all(), "loans reference unknown cduid"
    n_cd, n_loans = len(cduids), len(lx)
    ead = lx["balance"].to_numpy()
    pd0 = lx["pd_base"].to_numpy()
    lgd0 = lx["lgd_base"].to_numpy()
    v = lx["vulnerability"].to_numpy()
    lgd_fixed = lx["insured"].to_numpy() & (not mp["climate_lgd_applies_to_insured"])

    out = {s: np.zeros((n_runs, n_cd)) for s in scenarios}
    for start in range(0, n_runs, chunk):
        r = slice(start, min(start + chunk, n_runs))
        c = r.stop - r.start
        u = rng.random((c, n_loans))
        bpd = runs["beta_pd"].to_numpy()[r, None]
        blgd = runs["beta_lgd"].to_numpy()[r, None]
        h_run = h[year_pos[r]]                       # (c, n_cd): this run's fire season
        bins = (np.arange(c)[:, None] * n_cd + cd_pos[None, :]).ravel()
        for s, m in scenarios.items():
            H = np.minimum(1.0, h_run * m)[:, cd_pos]  # (c, n_loans)
            pd_ = np.minimum(1.0, pd0 * (1 + bpd * H))
            lgd = np.where(lgd_fixed, lgd0, np.minimum(1.0, lgd0 + blgd * H * v))
            loss = (u < pd_) * lgd * ead
            out[s][r] = np.bincount(bins, weights=loss.ravel(), minlength=c * n_cd).reshape(c, n_cd)
    return out, runs


def summarize_runs(cd_losses: dict, cduids: list, analytic: pd.DataFrame,
                   min_loans: int) -> pd.DataFrame:
    """Per (CD, scenario) and BC rows; BC tails come from per-run portfolio totals."""
    meta = analytic.set_index(["scenario", "cduid"])
    L_ref = cd_losses[REFERENCE_SCENARIO]
    rows = []
    for s, L in cd_losses.items():
        n_runs = L.shape[0]
        total = L.sum(axis=1)
        diff = L - L_ref  # per-run uplift; common random numbers make its SE small
        for ids, losses, d in [(cduids, L, diff), ([PORTFOLIO_ID], total[:, None],
                                                   diff.sum(axis=1)[:, None])]:
            t = tail_stats(losses)
            df = pd.DataFrame({"cduid": ids, "scenario": s,
                               "el_mean": losses.mean(axis=0),
                               "el_mean_se": losses.std(axis=0, ddof=1) / np.sqrt(n_runs),
                               "el_uplift_se": d.std(axis=0, ddof=1) / np.sqrt(n_runs),
                               **t})
            rows.append(df)
    res = pd.concat(rows, ignore_index=True)
    keep = ["n_loans", "ead_total", "multiplier", "el_no_climate", "el"]
    res = res.join(meta[keep], on=["scenario", "cduid"]).rename(columns={"el": "el_analytic"})
    res["n_loans"] = res["n_loans"].astype(int)
    res = add_uplift_columns(res, "el_mean")
    res = add_tail_uplift_columns(res)
    return suppress_small_tails(res, min_loans)


def add_tail_uplift_columns(res: pd.DataFrame) -> pd.DataFrame:
    """loss_p95_uplift, loss_p99_uplift, es_99_uplift: tail measure minus the same CD's
    tail measure in the reference scenario (both from the same runs)."""
    out = res.copy()
    ref = out[out["scenario"] == REFERENCE_SCENARIO].set_index("cduid")
    for col in TAIL_COLS:
        out[f"{col}_uplift"] = out[col] - out["cduid"].map(ref[col])
    return out


def run_pipeline(n_runs: int | None = None):
    div = pd.read_parquet(PROCESSED / "divisions.parquet")
    loans = pd.read_parquet(PROCESSED / "loans.parquet")
    yearly = pd.read_parquet(PROCESSED / "burn_by_year.parquet")
    mp, sim, scen = values("model"), values("simulation"), values("scenarios")
    n_runs = n_runs or sim["n_runs"]

    bpd, blgd = lognormal_mean(mp["beta_pd"]), lognormal_mean(mp["beta_lgd"])
    lx = calibrate_pd(loans, div, bpd)
    applies = mp["climate_lgd_applies_to_insured"]
    analytic = expected_el_over_seasons(lx, div, yearly, scen, bpd, blgd, applies)
    at_mean_h = deterministic_el(lx, div, scen, bpd, blgd, applies)

    cduids = div["cduid"].tolist()
    cd_losses, runs = simulate(lx, cduids, yearly, scen, mp, n_runs, sim["seed"])
    res = summarize_runs(cd_losses, cduids, analytic, values("reporting")["min_loans_for_tail"])
    # EL at the mean hazard (M3 figure); el_analytic - this = loss from fires arriving in bad seasons
    res = res.merge(at_mean_h[["cduid", "scenario", "el"]].rename(columns={"el": "el_at_mean_hazard"}),
                    on=["cduid", "scenario"])
    port = pd.concat([runs.assign(scenario=s, loss=L.sum(axis=1)) for s, L in cd_losses.items()],
                     ignore_index=True)
    return res, port, lx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=None, help="default: simulation.n_runs")
    args = ap.parse_args()

    res, port, lx = run_pipeline(args.runs)
    res.to_parquet(PROCESSED / "results.parquet", index=False)
    res.to_json(PROCESSED / "results.json", orient="records", indent=1)
    port.to_parquet(PROCESSED / "portfolio_runs.parquet", index=False)

    m = lambda x: f"{x / 1e6:,.2f}"  # noqa: E731  ($M)
    bc = res[res["cduid"] == PORTFOLIO_ID].set_index("scenario")
    print(f"PD_0 = {lx['pd_base'].iloc[0]:.4%} (observed {lx['pd_observed'].iloc[0]:.2%} at m = 1)")
    print(f"\nBC portfolio, {port['run'].nunique():,} runs ($M; ILLUSTRATIVE scenarios, "
          f"reference = historical):")
    money = ["el_mean", "el_mean_se", "el_analytic", "el_uplift", "loss_p99", "loss_p99_uplift",
             "es_99", "es_99_uplift", "el_no_climate"]
    print(bc[["multiplier", "el_mean", "el_mean_se", "el_analytic", "el_uplift", "el_uplift_pct",
              "loss_p99", "loss_p99_uplift", "es_99", "es_99_uplift", "el_no_climate"]].to_string(
        formatters={c: m for c in money}, float_format=lambda x: f"{x:,.2f}"))

    z = (bc["el_mean"] - bc["el_analytic"]) / bc["el_mean_se"]
    print(f"\nMC mean vs season-averaged E[EL], z-scores: {z.round(2).to_dict()}")
    zu = ((bc["el_uplift"] - (bc["el_analytic"] - bc.loc["historical", "el_analytic"]))
          / bc["el_uplift_se"]).drop("historical")
    print(f"MC uplift vs analytic uplift, z-scores: {zu.round(2).to_dict()}")
    print(f"Season-averaged E[EL] minus EL at mean hazard ($M): "
          f"{((bc['el_analytic'] - bc['el_at_mean_hazard']) / 1e6).round(3).to_dict()}")

    hh = port[port["scenario"] == "hot_house"]
    k = n_tail(len(hh))
    worst = hh.nlargest(k, "loss")
    print(f"\nFire seasons in the worst {k} hot_house runs (ES_99 tail): "
          f"{worst['year'].value_counts().to_dict()}")
    sup = res[res["tail_suppressed"]]["cduid"].unique().tolist()
    print(f"Tail measures suppressed (n_loans < threshold) for CDs: {sup}")


if __name__ == "__main__":
    main()
