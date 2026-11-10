"""M6 API tests against the real processed outputs (skipped until the pipeline has run)."""
import math

import pytest
from fastapi.testclient import TestClient

from src.hazard import PROCESSED

NEEDED = ["results.parquet", "portfolio_summary.parquet", "divisions.geojson",
          "portfolio_runs.parquet", "sensitivity.parquet"]
pytestmark = pytest.mark.skipif(not all((PROCESSED / f).exists() for f in NEEDED),
                                reason="run the pipeline through src.sensitivity first")


@pytest.fixture(scope="module")
def client():
    from api.main import create_app
    return TestClient(create_app(PROCESSED))


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["n_regions"] == 29


def test_scenarios_labelled_illustrative(client):
    j = client.get("/api/scenarios").json()
    assert j["illustrative"] is True and "not NGFS model output" in j["disclaimer"]
    ids = [s["id"] for s in j["scenarios"]]
    assert ids[0] == "historical" and j["reference_scenario"] == "historical"
    assert {"orderly", "disorderly", "current_policies", "hot_house"} <= set(ids)


def test_regions_geojson(client):
    j = client.get("/api/regions").json()
    assert j["type"] == "FeatureCollection" and len(j["features"]) == 29
    props = {f["properties"]["cduid"]: f["properties"] for f in j["features"]}
    assert props["5915"]["n_loans"] > 40_000           # Greater Vancouver
    assert props["5957"]["tail_reportable"] is False   # Stikine, 3 loans


def test_risk_filters(client):
    all_rows = client.get("/api/risk").json()["rows"]
    assert len(all_rows) == 5 * 30
    one = client.get("/api/risk", params={"scenario": "hot_house", "region": "BC"}).json()["rows"]
    assert len(one) == 1 and one[0]["cduid"] == "BC" and one[0]["el_uplift"] > 0
    by_scen = client.get("/api/risk", params={"scenario": "orderly"}).json()["rows"]
    assert len(by_scen) == 30 and {r["scenario"] for r in by_scen} == {"orderly"}


def test_suppressed_tails_are_null_not_nan(client):
    rows = client.get("/api/risk", params={"region": "5957"}).json()["rows"]
    assert all(r["tail_suppressed"] for r in rows)
    for r in rows:
        assert r["es_99"] is None and r["loss_p99_uplift"] is None
        assert r["el_mean"] is not None and math.isfinite(r["el_mean"])


def test_errors(client):
    assert client.get("/api/risk", params={"scenario": "net_zero_2099"}).status_code == 422
    assert client.get("/api/risk", params={"region": "9999"}).status_code == 404
    assert client.get("/api/losses", params={"scenario": "nope"}).status_code == 422
    assert client.get("/api/losses").status_code == 422  # scenario is required


def test_losses(client):
    j = client.get("/api/losses", params={"scenario": "hot_house"}).json()
    assert j["n_runs"] == len(j["runs"]) == 1000
    assert {"run", "year", "loss"} <= set(j["runs"][0])


def test_sensitivity_and_assumptions(client):
    s = client.get("/api/sensitivity").json()
    assert {r["parameter"] for r in s["rows"]} >= {"fire_window", "hazard_headroom", "pd_base"}
    a = client.get("/api/assumptions").json()
    assert a["portfolio"]["pd_base"]["value"] == 0.0026
    assert "source" in a["portfolio"]["insured_share"]


def test_cors_and_gzip(client):
    r = client.get("/api/regions", headers={"Origin": "https://example.org",
                                            "Accept-Encoding": "gzip"})
    assert r.headers.get("access-control-allow-origin") in ("*", "https://example.org")
    assert r.headers.get("content-encoding") == "gzip"
