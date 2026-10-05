"""M1 tests. Unit tests always run; output tests skip until `python -m src.hazard` has run."""
import json

import pandas as pd
import pytest

from src.hazard import EXPECTED_BC_CDS, PROCESSED, normalize_hazard

DIV = PROCESSED / "divisions.parquet"
YEARLY = PROCESSED / "burn_by_year.parquet"
GEOJSON = PROCESSED / "divisions.geojson"

needs_outputs = pytest.mark.skipif(
    not (DIV.exists() and YEARLY.exists() and GEOJSON.exists()),
    reason="run `python -m src.hazard` first",
)


def test_normalize_hazard_scales_to_max():
    h = normalize_hazard(pd.Series([0.0, 0.002, 0.008]))
    assert h.tolist() == [0.0, 0.25, 1.0]


def test_normalize_hazard_rejects_all_zero():
    with pytest.raises(ValueError):
        normalize_hazard(pd.Series([0.0, 0.0]))


@pytest.fixture(scope="module")
def div():
    return pd.read_parquet(DIV)


@needs_outputs
def test_divisions_shape_and_keys(div):
    assert len(div) == EXPECTED_BC_CDS
    assert div["cduid"].is_unique
    assert div["cduid"].str.fullmatch(r"59\d\d").all()


@needs_outputs
def test_no_nulls(div):
    cols = ["land_area_km2", "annual_burn_share", "ever_burned_share", "hazard_score",
            "vulnerability", "owner_households", "private_households"]
    assert div[cols].notna().all().all(), div[cols].isna().sum()


@needs_outputs
def test_hazard_range(div):
    assert div["hazard_score"].between(0, 1).all()
    assert div["hazard_score"].max() == pytest.approx(1.0)
    assert div["annual_burn_share"].between(0, 1).all()
    assert div["ever_burned_share"].between(0, 1).all()
    # sum of yearly burned area >= area burned at least once (reburns counted twice)
    n_years = div["window_end"] - div["window_start"] + 1
    assert (div["annual_burned_km2"] * n_years >= div["ever_burned_km2"] * (1 - 1e-6)).all()


@needs_outputs
def test_households(div):
    assert (div["owner_households"] > 0).all()
    assert (div["owner_households"] <= div["private_households"]).all()
    # BC 2021: ~2.0M private households; loose bounds catch wrong-ID / wrong-level joins
    assert 1.5e6 < div["private_households"].sum() < 2.5e6


@needs_outputs
def test_yearly_consistent_with_annual(div):
    yearly = pd.read_parquet(YEARLY)
    n_years = int(div["window_end"].iloc[0] - div["window_start"].iloc[0] + 1)
    assert len(yearly) == EXPECTED_BC_CDS * n_years
    assert not yearly.duplicated(["cduid", "year"]).any()
    annual = yearly.groupby("cduid")["burned_km2"].sum() / n_years
    merged = div.set_index("cduid")["annual_burned_km2"]
    pd.testing.assert_series_equal(annual.sort_index(), merged.sort_index(),
                                   check_names=False, rtol=1e-9)


@needs_outputs
def test_geojson():
    assert GEOJSON.stat().st_size < 2e6
    gj = json.loads(GEOJSON.read_text())
    assert len(gj["features"]) == EXPECTED_BC_CDS
    xs, ys = [], []
    for feat in gj["features"]:
        coords = feat["geometry"]["coordinates"]
        polys = coords if feat["geometry"]["type"] == "MultiPolygon" else [coords]
        for poly in polys:
            for x, y in poly[0]:
                xs.append(x)
                ys.append(y)
    # BC lon/lat envelope => confirms EPSG:4326, not Albers metres
    assert -140 < min(xs) and max(xs) < -113
    assert 48 < min(ys) and max(ys) < 60.1
