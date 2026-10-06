"""M2: synthetic BC mortgage portfolio.

Loans are allocated to census divisions (CDs) by a multinomial draw weighted by
owner households with a mortgage. Balances are lognormal, with each CD's mean
scaled by its median dwelling value. Parameters come from assumptions.yaml.

Outputs (data/processed/):
  loans.parquet              one row per loan
  portfolio_summary.parquet  one row per CD

Usage (from project root):
    python -m src.portfolio
"""
import numpy as np
import pandas as pd

from src.assumptions import values
from src.hazard import PROCESSED


def mortgage_weights(div: pd.DataFrame) -> pd.Series:
    """Estimated owner households with a mortgage per CD."""
    return div["owner_households"] * div["pct_owners_with_mortgage"] / 100


def cd_mean_balance(div: pd.DataFrame, weights: pd.Series, p: dict) -> pd.Series:
    """Mean balance per CD; the mortgage-weighted average across CDs equals balance_mean."""
    value_ratio = div["median_dwelling_value"] / np.average(div["median_dwelling_value"], weights=weights)
    scale = value_ratio ** p["balance_value_elasticity"]
    scale = scale / np.average(scale, weights=weights)
    return p["balance_mean"] * scale


def generate_portfolio(div: pd.DataFrame, p: dict, seed: int | None = None) -> pd.DataFrame:
    """One row per loan: loan_id, cduid, balance (EAD), insured, pd_base, lgd_base."""
    rng = np.random.default_rng(p["seed"] if seed is None else seed)
    div = div.reset_index(drop=True)
    w = mortgage_weights(div)

    counts = rng.multinomial(p["n_loans"], (w / w.sum()).to_numpy())
    cduid = np.repeat(div["cduid"].to_numpy(), counts)

    # Lognormal with E[balance] = CD mean: mu = ln(mean) - sigma^2 / 2
    sigma = p["balance_sigma"]
    mu = np.repeat(np.log(cd_mean_balance(div, w, p).to_numpy()) - sigma**2 / 2, counts)
    balance = rng.lognormal(mu, sigma)

    insured = rng.random(p["n_loans"]) < p["insured_share"]
    lgd_uninsured = max(p["lgd_base_uninsured"], p["lgd_floor"])
    return pd.DataFrame({
        "loan_id": np.arange(p["n_loans"]),
        "cduid": cduid,
        "balance": balance,
        "insured": insured,
        "pd_base": p["pd_base"],
        "lgd_base": np.where(insured, p["lgd_base_insured"], lgd_uninsured),
    })


def summarize(loans: pd.DataFrame, cduids: pd.Series) -> pd.DataFrame:
    """Per-CD loan count, EAD and EAD-weighted baseline PD/LGD; CDs with no loans get 0 / NaN."""
    loans = loans.assign(pd_x_ead=loans["pd_base"] * loans["balance"],
                         lgd_x_ead=loans["lgd_base"] * loans["balance"])
    g = loans.groupby("cduid").agg(n_loans=("loan_id", "size"), ead_total=("balance", "sum"),
                                   pd_x_ead=("pd_x_ead", "sum"), lgd_x_ead=("lgd_x_ead", "sum"),
                                   insured_share=("insured", "mean"))
    g["pd_base_wavg"] = g["pd_x_ead"] / g["ead_total"]
    g["lgd_base_wavg"] = g["lgd_x_ead"] / g["ead_total"]
    g = g.drop(columns=["pd_x_ead", "lgd_x_ead"]).reindex(cduids)
    g[["n_loans", "ead_total"]] = g[["n_loans", "ead_total"]].fillna(0)
    return g.astype({"n_loans": int}).rename_axis("cduid").reset_index()


def main() -> None:
    div = pd.read_parquet(PROCESSED / "divisions.parquet")
    loans = generate_portfolio(div, values("portfolio"))
    summary = summarize(loans, div["cduid"])
    summary["tail_reportable"] = summary["n_loans"] >= values("reporting")["min_loans_for_tail"]
    loans.to_parquet(PROCESSED / "loans.parquet", index=False)
    summary.to_parquet(PROCESSED / "portfolio_summary.parquet", index=False)

    out = summary.merge(div[["cduid", "cd_name", "hazard_score"]], on="cduid")
    out["ead_share"] = out["ead_total"] / out["ead_total"].sum()
    out["avg_balance"] = out["ead_total"] / out["n_loans"]
    cols = ["cduid", "cd_name", "n_loans", "avg_balance", "ead_share", "hazard_score"]
    print(out.sort_values("ead_total", ascending=False)[cols].to_string(
        index=False, float_format=lambda x: f"{x:,.3f}" if x < 10 else f"{x:,.0f}"))
    print(f"\n{len(loans):,} loans | total EAD ${loans['balance'].sum() / 1e9:,.1f}B | "
          f"mean ${loans['balance'].mean():,.0f} | median ${loans['balance'].median():,.0f} | "
          f"insured {loans['insured'].mean():.1%}")
    hw = np.average(out["hazard_score"], weights=out["ead_total"])
    print(f"EAD-weighted hazard {hw:.3f} vs unweighted mean {out['hazard_score'].mean():.3f}")


if __name__ == "__main__":
    main()
