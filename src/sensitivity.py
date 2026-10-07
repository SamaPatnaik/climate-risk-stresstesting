"""M5: one-at-a-time sensitivity analysis (tornado chart).

Each parameter in assumptions.yaml `sensitivity:` moves to its low and high value with
everything else at base. Metrics, for the BC portfolio, Hot House vs historical:
  - el_uplift: season-averaged analytic E[EL] uplift (exact, no simulation noise)
  - es_uplift: ES_99 uplift from the Monte Carlo (same seed for every evaluation, so
    differences between bars come from the parameter, not from resampling)

Outputs:
  data/processed/sensitivity.parquet / .json   one row per parameter
  out/tornado.png

Usage (from project root):
    python -m src.sensitivity [--runs 1000]
"""
import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.assumptions import values
from src.hazard import PROCESSED, hazard_reference, normalize_hazard
from src.model import (PORTFOLIO_ID, REFERENCE_SCENARIO, calibrate_pd, expected_el_over_seasons,
                       lognormal_mean)
from src.simulate import simulate, tail_stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
TARGET = "hot_house"
LABELS = {  # ASCII only: Windows consoles (cp1252) can't print Greek letters
    "beta_pd_median": "Beta_PD median",
    "beta_lgd_median": "Beta_LGD median",
    "pd_base": "Observed PD (BC arrears)",
    "lgd_base_uninsured": "LGD, uninsured",
    "hot_house_multiplier": "Hot House multiplier",
    "hazard_headroom": "Hazard scale (headroom)",
    "fire_window": "Fire-season window",
}


def base_config() -> dict:
    return {"model": values("model"), "portfolio": values("portfolio"),
            "scenarios": values("scenarios"), "window": None}


def apply_override(cfg: dict, name: str, value) -> dict:
    """Copy of cfg with one sensitivity parameter set to value."""
    c = copy.deepcopy(cfg)
    if name == "beta_pd_median":
        c["model"]["beta_pd"]["median"] = value
    elif name == "beta_lgd_median":
        c["model"]["beta_lgd"]["median"] = value
    elif name in ("pd_base", "lgd_base_uninsured"):
        c["portfolio"][name] = value
    elif name == "hot_house_multiplier":
        c["scenarios"][TARGET] = value
    elif name == "hazard_headroom":
        c["model"]["hazard_headroom"] = value
    elif name == "fire_window":
        c["window"] = tuple(value)
    else:
        raise KeyError(f"unknown sensitivity parameter {name!r}")
    return c


def base_value(cfg: dict, name: str):
    return {"beta_pd_median": cfg["model"]["beta_pd"]["median"],
            "beta_lgd_median": cfg["model"]["beta_lgd"]["median"],
            "pd_base": cfg["portfolio"]["pd_base"],
            "lgd_base_uninsured": cfg["portfolio"]["lgd_base_uninsured"],
            "hot_house_multiplier": cfg["scenarios"][TARGET],
            "hazard_headroom": cfg["model"]["hazard_headroom"],
            "fire_window": list(cfg["window"]) if cfg["window"] else None}[name]


def window_hazard(yearly: pd.DataFrame, headroom: float, window=None):
    """Yearly hazard h_y for the window's seasons, and each CD's mean h over them.

    The reference (headroom x worst CD-year) uses all years in `yearly`, so changing the
    window changes which seasons are sampled, not the hazard scale.
    """
    ref = hazard_reference(yearly["burn_share"], headroom)
    lo, hi = window or (yearly["year"].min(), yearly["year"].max())
    y = yearly[yearly["year"].between(lo, hi)].copy()
    y["hazard"] = normalize_hazard(y["burn_share"], ref)
    return y, y.groupby("cduid")["hazard"].mean().rename("hazard_score")


def evaluate(cfg: dict, div: pd.DataFrame, loans: pd.DataFrame, yearly: pd.DataFrame,
             n_runs: int = 0, seed: int = 0) -> dict:
    """BC Hot House vs historical: analytic EL uplift and (if n_runs) MC ES_99 uplift."""
    mp, port = cfg["model"], cfg["portfolio"]
    y, score = window_hazard(yearly, mp["hazard_headroom"], cfg["window"])
    d = div.drop(columns="hazard_score", errors="ignore").merge(score.reset_index(), on="cduid")

    lx = loans.drop(columns=["hazard_score", "vulnerability", "pd_observed"], errors="ignore").copy()
    lx["pd_base"] = port["pd_base"]
    lx["lgd_base"] = np.where(lx["insured"], port["lgd_base_insured"],
                              max(port["lgd_base_uninsured"], port["lgd_floor"]))
    bpd, blgd = lognormal_mean(mp["beta_pd"]), lognormal_mean(mp["beta_lgd"])
    lx = calibrate_pd(lx, d, bpd)
    scen = {REFERENCE_SCENARIO: cfg["scenarios"][REFERENCE_SCENARIO], TARGET: cfg["scenarios"][TARGET]}
    applies = mp["climate_lgd_applies_to_insured"]

    an = expected_el_over_seasons(lx, d, y, scen, bpd, blgd, applies)
    bc = an[an["cduid"] == PORTFOLIO_ID].set_index("scenario")["el"]
    out = {"el_ref": bc[REFERENCE_SCENARIO], "el_target": bc[TARGET],
           "el_uplift": bc[TARGET] - bc[REFERENCE_SCENARIO]}
    if n_runs:
        cd_losses, _ = simulate(lx, d["cduid"].tolist(), y, scen, mp, n_runs, seed)
        es = {s: tail_stats(L.sum(axis=1))["es_99"] for s, L in cd_losses.items()}
        out.update(es_ref=es[REFERENCE_SCENARIO], es_target=es[TARGET],
                   es_uplift=es[TARGET] - es[REFERENCE_SCENARIO])
    return out


def tornado(div, loans, yearly, specs: dict, n_runs: int, seed: int, cfg: dict | None = None):
    """One row per parameter: low/high input, metric at low/high, swing. Sorted by EL swing."""
    cfg = cfg or base_config()
    base = evaluate(cfg, div, loans, yearly, n_runs, seed)
    rows = []
    for name, spec in specs.items():
        bv = base_value(cfg, name)
        res = {}
        for side in ("low", "high"):
            v = spec[side]
            same = (list(v) if isinstance(v, (list, tuple)) else v) == bv or (
                name == "fire_window" and bv is None
                and tuple(v) == (yearly["year"].min(), yearly["year"].max()))
            res[side] = base if same else evaluate(apply_override(cfg, name, v), div, loans,
                                                   yearly, n_runs, seed)
        row = {"parameter": name, "label": LABELS.get(name, name),
               "base_value": str(bv if bv is not None else
                                 [int(yearly["year"].min()), int(yearly["year"].max())]),
               "low_value": str(spec["low"]), "high_value": str(spec["high"])}
        for metric in ("el_uplift", "es_uplift"):
            if metric not in base:
                continue
            row[f"{metric}_base"] = base[metric]
            row[f"{metric}_low"] = res["low"][metric]
            row[f"{metric}_high"] = res["high"][metric]
            row[f"{metric}_swing"] = abs(res["high"][metric] - res["low"][metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("el_uplift_swing", ascending=False, ignore_index=True)


# Validated diverging pair (blue <-> red) on a light surface; text stays in ink colours.
BELOW, ABOVE = "#2a78d6", "#e34948"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def _panel(ax, t: pd.DataFrame, metric: str, title: str) -> None:
    t = t.sort_values(f"{metric}_swing", ascending=True, ignore_index=True)  # biggest on top
    base = t[f"{metric}_base"].iloc[0] / 1e6
    for i, r in t.iterrows():
        for side in ("low", "high"):
            v = r[f"{metric}_{side}"] / 1e6
            if np.isclose(v, base):
                continue
            ax.barh(i, v - base, left=base, height=0.6, color=ABOVE if v > base else BELOW,
                    edgecolor=SURFACE, linewidth=2)
            ax.annotate(r[f"{side}_value"], (v, i), xytext=(4 if v > base else -4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=MUTED,
                        ha="left" if v > base else "right")
    ax.axvline(base, color=INK, linewidth=1)
    ax.set_yticks(range(len(t)), t["label"], color=INK)
    ax.set_xlabel("$M", color=MUTED)
    ax.set_title(f"{title}\nbase ${base:,.2f}M", fontsize=10, color=INK, loc="left")
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.margins(x=0.25)


def plot_tornado(t: pd.DataFrame, path: Path) -> None:
    metrics = [("el_uplift", "Mean EL uplift, Hot House vs historical")]
    if "es_uplift_base" in t:
        metrics.append(("es_uplift", "ES 99% uplift, Hot House vs historical"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 4.8), facecolor=SURFACE)
    for ax, (metric, title) in zip(np.atleast_1d(axes), metrics):
        ax.set_facecolor(SURFACE)
        _panel(ax, t, metric, title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BELOW), plt.Rectangle((0, 0), 1, 1, color=ABOVE)]
    fig.legend(handles, ["lowers the uplift", "raises the uplift"], loc="lower center", ncol=2,
               frameon=False, fontsize=9)
    fig.suptitle("BC wildfire stress test: one-at-a-time sensitivity (ILLUSTRATIVE scenarios)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=None, help="MC runs per evaluation; 0 = analytic only")
    args = ap.parse_args()
    n_runs = values("simulation")["n_runs"] if args.runs is None else args.runs
    OUT.mkdir(exist_ok=True)

    div = pd.read_parquet(PROCESSED / "divisions.parquet")
    loans = pd.read_parquet(PROCESSED / "loans.parquet")
    yearly = pd.read_parquet(PROCESSED / "burn_by_year.parquet")
    specs = values("sensitivity")
    t = tornado(div, loans, yearly, specs, n_runs, values("simulation")["seed"])

    t.to_parquet(PROCESSED / "sensitivity.parquet", index=False)
    t.to_json(PROCESSED / "sensitivity.json", orient="records", indent=1)
    plot_tornado(t, OUT / "tornado.png")

    cols = ["label", "base_value", "low_value", "high_value", "el_uplift_low", "el_uplift_high",
            "el_uplift_swing"] + (["es_uplift_low", "es_uplift_high", "es_uplift_swing"]
                                  if "es_uplift_base" in t else [])
    print(f"Base: EL uplift ${t['el_uplift_base'].iloc[0] / 1e6:,.3f}M"
          + (f", ES_99 uplift ${t['es_uplift_base'].iloc[0] / 1e6:,.3f}M" if "es_uplift_base" in t else ""))
    print(t[cols].to_string(index=False, formatters={
        c: (lambda x: f"{x / 1e6:,.3f}") for c in cols if "uplift" in c}))
    print(f"Wrote sensitivity.parquet/.json and {OUT / 'tornado.png'}")


if __name__ == "__main__":
    main()
