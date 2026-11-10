"""M6: read-only API over the precomputed stress-test outputs.

Nothing is simulated per request: every endpoint serves files written by the
pipeline (src.hazard -> src.portfolio -> src.simulate -> src.sensitivity).

Endpoints (all GET):
  /api/health
  /api/scenarios                      scenario ids, labels, multipliers (ILLUSTRATIVE)
  /api/regions                        GeoJSON of BC census divisions + hazard + loan counts
  /api/risk?scenario=&region=         results rows; both filters optional; region = CDUID or "BC"
  /api/losses?scenario=               per-run BC portfolio losses (histogram input)
  /api/sensitivity                    tornado table
  /api/assumptions                    every parameter with value, source, placeholder flag

Run locally (from project root):
    uvicorn api.main:app --reload
Env:
    CORS_ORIGINS   comma-separated allowed origins (default "*"; read-only public data)
    DATA_DIR       directory with the processed files (default data/processed)
"""
import json
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from src.assumptions import load_assumptions
from src.model import PORTFOLIO_ID, REFERENCE_SCENARIO

ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = ("Scenario multipliers are stylized and illustrative, not NGFS model output. "
              "The loan portfolio is synthetic.")


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe records (NaN -> null, numpy scalars -> Python)."""
    return json.loads(df.to_json(orient="records"))


class Store:
    """All API data, loaded once at startup."""

    def __init__(self, data_dir: Path):
        a = load_assumptions()
        self.assumptions = a
        self.scenarios = [{"id": k, "label": v.get("label", k), "multiplier": v["value"],
                           "is_reference": k == REFERENCE_SCENARIO} for k, v in a["scenarios"].items()]
        self.scenario_ids = [s["id"] for s in self.scenarios]

        results = pd.read_parquet(data_dir / "results.parquet")
        self.results = results
        self.region_ids = set(results["cduid"])

        summary = pd.read_parquet(data_dir / "portfolio_summary.parquet").set_index("cduid")
        geo = json.loads((data_dir / "divisions.geojson").read_text(encoding="utf-8"))
        for feat in geo["features"]:
            props = feat["properties"]
            row = summary.loc[props["cduid"]]
            props["n_loans"] = int(row["n_loans"])
            props["ead_total"] = float(row["ead_total"])
            props["tail_reportable"] = bool(row["tail_reportable"])
        self.regions = geo

        self.losses = pd.read_parquet(data_dir / "portfolio_runs.parquet")
        self.sensitivity = _records(pd.read_parquet(data_dir / "sensitivity.parquet"))

    def check_scenario(self, scenario: str | None) -> None:
        if scenario is not None and scenario not in self.scenario_ids:
            raise HTTPException(422, f"unknown scenario {scenario!r}; expected one of {self.scenario_ids}")


def create_app(data_dir: Path | None = None) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get("DATA_DIR", ROOT / "data" / "processed"))
    store = Store(data_dir)

    app = FastAPI(title="BC Wildfire Mortgage Stress Test API",
                  description=DISCLAIMER, version="0.1.0")
    origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET"],
                       allow_headers=["*"])
    app.add_middleware(GZipMiddleware, minimum_size=1000)  # GeoJSON ~0.5 MB -> much smaller

    @app.get("/api/health")
    def health():
        return {"status": "ok", "n_regions": len(store.region_ids) - 1,
                "scenarios": store.scenario_ids}

    @app.get("/api/scenarios")
    def scenarios():
        return {"reference_scenario": REFERENCE_SCENARIO, "illustrative": True,
                "disclaimer": DISCLAIMER, "scenarios": store.scenarios}

    @app.get("/api/regions")
    def regions():
        return store.regions

    @app.get("/api/risk")
    def risk(scenario: str | None = Query(None, description="scenario id; omit for all"),
             region: str | None = Query(None, description=f"CDUID or {PORTFOLIO_ID!r}; omit for all")):
        store.check_scenario(scenario)
        if region is not None and region not in store.region_ids:
            raise HTTPException(404, f"unknown region {region!r}")
        df = store.results
        if scenario is not None:
            df = df[df["scenario"] == scenario]
        if region is not None:
            df = df[df["cduid"] == region]
        return {"reference_scenario": REFERENCE_SCENARIO, "disclaimer": DISCLAIMER,
                "rows": _records(df)}

    @app.get("/api/losses")
    def losses(scenario: str = Query(..., description="scenario id")):
        store.check_scenario(scenario)
        df = store.losses[store.losses["scenario"] == scenario]
        return {"scenario": scenario, "portfolio": PORTFOLIO_ID, "n_runs": len(df),
                "runs": _records(df[["run", "year", "beta_pd", "beta_lgd", "loss"]])}

    @app.get("/api/sensitivity")
    def sensitivity():
        return {"target_scenario": "hot_house", "reference_scenario": REFERENCE_SCENARIO,
                "metrics": {"el_uplift": "season-averaged analytic mean EL uplift ($)",
                            "es_uplift": "Monte Carlo ES 99% uplift ($)"},
                "rows": store.sensitivity}

    @app.get("/api/assumptions")
    def assumptions():
        return store.assumptions

    return app


app = create_app()
