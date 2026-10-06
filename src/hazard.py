"""M1: wildfire hazard table per BC census division (CD).

Outputs (data/processed/):
  divisions.parquet     one row per CD: land area, burn shares, hazard score h,
                        vulnerability placeholder, owner households
  burn_by_year.parquet  one row per (CD, year) incl. zero-burn years; used later
                        for the fire-season bootstrap and window sensitivity
  divisions.geojson     simplified EPSG:4326 boundaries + key fields for the web map

Usage (from project root):
    python -m src.hazard [--start 1990] [--end 2025] [--refresh]
"""
import argparse
import io
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

ALBERS = "EPSG:3005"
WGS84 = "EPSG:4326"
FIRE_LAYER = "WHSE_LAND_AND_NATURAL_RESOURCE.PROT_HISTORICAL_FIRE_POLYS_SP"
WFS_URL = f"https://openmaps.gov.bc.ca/geo/pub/{FIRE_LAYER}/ows"
PAGE_SIZE = 10_000  # server CountDefault
FIRES_CACHE = RAW / "bc_fire_perimeters_historical.gpkg"
CD_ZIP = RAW / "lcd_000b21a_e.zip"
CENSUS_ZIP = RAW / "98-401-X2021004_eng_CSV.zip"
CENSUS_CSV = "98-401-X2021004_English_CSV_data.csv"

BC_PRUID = "59"
BC_DGUID_PREFIX = "2021A000359"  # 2021A0003 = CD level, 59 = BC
EXPECTED_BC_CDS = 29
# Census Profile 2021 characteristic IDs (25% sample data)
CHAR_HOUSEHOLDS = 1414  # Total - Private households by tenure
CHAR_OWNER = 1415       # Owner

SIMPLIFY_OVERLAY_M = 50   # CD simplification for intersection only
SIMPLIFY_WEB_M = 500      # CD simplification for the web GeoJSON
MIN_WEB_PART_KM2 = 1.0    # drop islands < 1 km2 from the web GeoJSON (~0.03% of area)
DEFAULT_VULNERABILITY = 1.0  # PLACEHOLDER: uniform until a vulnerability index is added


def download_fires(refresh: bool = False) -> gpd.GeoDataFrame:
    """Fetch all historical fire perimeters via paged WFS, cached as GeoPackage."""
    if FIRES_CACHE.exists() and not refresh:
        return gpd.read_file(FIRES_CACHE)

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": f"pub:{FIRE_LAYER}",
        "outputFormat": "application/json",
        "srsName": ALBERS,
        "sortBy": "OBJECTID",  # stable ordering is required for paging
        "count": PAGE_SIZE,
    }
    features, start, total = [], 0, None
    while total is None or start < total:
        params["startIndex"] = start
        for attempt in range(3):
            try:
                r = requests.get(WFS_URL, params=params, timeout=600)
                r.raise_for_status()
                page = r.json()
                break
            except (requests.RequestException, ValueError) as e:
                print(f"  page at {start} failed (attempt {attempt + 1}): {e}")
                if attempt == 2:
                    raise
        total = page["numberMatched"]
        features.extend(page["features"])
        start += PAGE_SIZE
        print(f"  downloaded {len(features):,} / {total:,}")

    fires = gpd.GeoDataFrame.from_features(features, crs=ALBERS)
    assert len(fires) == total, f"expected {total} features, got {len(fires)}"
    assert fires["OBJECTID"].is_unique, "duplicate OBJECTIDs: paging is unstable"
    fires.to_file(FIRES_CACHE, driver="GPKG")
    return fires


def load_cds() -> gpd.GeoDataFrame:
    """BC census divisions in BC Albers. land_area_km2 is StatCan LANDAREA (excludes water)."""
    cds = gpd.read_file(f"zip://{CD_ZIP}", where=f"PRUID='{BC_PRUID}'")
    assert len(cds) == EXPECTED_BC_CDS, f"expected {EXPECTED_BC_CDS} BC CDs, got {len(cds)}"
    cds = cds.to_crs(ALBERS).rename(columns={"CDUID": "cduid", "CDNAME": "cd_name"})
    cds["land_area_km2"] = cds["LANDAREA"].astype(float)
    return cds[["cduid", "cd_name", "land_area_km2", "geometry"]]


def load_owner_households() -> pd.DataFrame:
    """Owner and total private households per BC CD from Census Profile 2021."""
    cols = ["DGUID", "CHARACTERISTIC_ID", "CHARACTERISTIC_NAME", "C1_COUNT_TOTAL"]
    keep = []
    with zipfile.ZipFile(CENSUS_ZIP) as z, z.open(CENSUS_CSV) as f:
        reader = pd.read_csv(io.TextIOWrapper(f, encoding="latin-1"), usecols=cols,
                             dtype={"DGUID": str}, chunksize=500_000)
        for chunk in reader:
            mask = (chunk["DGUID"].str.startswith(BC_DGUID_PREFIX)
                    & chunk["CHARACTERISTIC_ID"].isin([CHAR_HOUSEHOLDS, CHAR_OWNER]))
            keep.append(chunk[mask])
    df = pd.concat(keep)

    # Guard against the IDs meaning something else in some CD.
    names = df.groupby("CHARACTERISTIC_ID")["CHARACTERISTIC_NAME"].unique()
    assert all(n.strip() == "Owner" for n in names[CHAR_OWNER]), names[CHAR_OWNER]
    assert all("by tenure" in n for n in names[CHAR_HOUSEHOLDS]), names[CHAR_HOUSEHOLDS]

    df["cduid"] = df["DGUID"].str[-4:]
    wide = df.pivot(index="cduid", columns="CHARACTERISTIC_ID", values="C1_COUNT_TOTAL")
    wide = wide.rename(columns={CHAR_HOUSEHOLDS: "private_households", CHAR_OWNER: "owner_households"})
    return wide.reset_index()


def burn_by_year(fires: gpd.GeoDataFrame, cds: gpd.GeoDataFrame,
                 start: int, end: int) -> pd.DataFrame:
    """Burned km2 per (CD, year), clipped to CDs, with zero rows for no-burn years."""
    fires = fires[fires["FIRE_YEAR"].between(start, end)].copy()
    fires["geometry"] = fires.geometry.make_valid()
    # One (multi)polygon per year so same-year overlaps aren't double counted.
    by_year = fires[["FIRE_YEAR", "geometry"]].dissolve(by="FIRE_YEAR").reset_index()

    cds_simple = cds[["cduid", "geometry"]].copy()
    cds_simple["geometry"] = cds_simple.geometry.simplify(SIMPLIFY_OVERLAY_M, preserve_topology=True)
    inter = gpd.overlay(cds_simple, by_year, how="intersection", keep_geom_type=True)
    inter["burned_km2"] = inter.geometry.area / 1e6
    burned = inter.groupby(["cduid", "FIRE_YEAR"])["burned_km2"].sum()

    full = pd.MultiIndex.from_product([cds["cduid"], range(start, end + 1)],
                                      names=["cduid", "year"])
    out = burned.rename_axis(["cduid", "year"]).reindex(full, fill_value=0.0).reset_index()
    out = out.merge(cds[["cduid", "land_area_km2"]], on="cduid")
    out["burn_share"] = (out["burned_km2"] / out["land_area_km2"]).clip(upper=1.0)
    return out[["cduid", "year", "burned_km2", "burn_share"]]


def ever_burned_km2(fires: gpd.GeoDataFrame, cds: gpd.GeoDataFrame,
                    start: int, end: int) -> pd.Series:
    """Area burned at least once in the window, per CD."""
    fires = fires[fires["FIRE_YEAR"].between(start, end)]
    union = gpd.GeoDataFrame(geometry=[fires.geometry.make_valid().unary_union], crs=ALBERS)
    cds_simple = cds[["cduid", "geometry"]].copy()
    cds_simple["geometry"] = cds_simple.geometry.simplify(SIMPLIFY_OVERLAY_M, preserve_topology=True)
    inter = gpd.overlay(cds_simple, union, how="intersection", keep_geom_type=True)
    return inter.assign(a=inter.geometry.area / 1e6).groupby("cduid")["a"].sum()


def normalize_hazard(annual_share: pd.Series) -> pd.Series:
    """h = share / max(share): highest-hazard CD = 1, ratios between CDs preserved."""
    mx = annual_share.max()
    if mx <= 0:
        raise ValueError("no burned area in window; cannot normalize hazard")
    return annual_share / mx


def build_divisions(start: int, end: int | None, refresh: bool = False):
    fires = download_fires(refresh)
    end = end or int(fires["FIRE_YEAR"].max())
    n_years = end - start + 1
    cds = load_cds()

    yearly = burn_by_year(fires, cds, start, end)
    annual = yearly.groupby("cduid")["burned_km2"].sum() / n_years
    ever = ever_burned_km2(fires, cds, start, end)

    div = cds.drop(columns="geometry").set_index("cduid")
    div["annual_burned_km2"] = annual
    div["ever_burned_km2"] = ever.reindex(div.index).fillna(0.0)
    div["annual_burn_share"] = div["annual_burned_km2"] / div["land_area_km2"]
    div["ever_burned_share"] = (div["ever_burned_km2"] / div["land_area_km2"]).clip(upper=1.0)
    div["hazard_score"] = normalize_hazard(div["annual_burn_share"])
    div["vulnerability"] = DEFAULT_VULNERABILITY
    div["window_start"], div["window_end"] = start, end
    div = div.reset_index().merge(load_owner_households(), on="cduid", how="left")
    return div, yearly, cds


def write_geojson(div: pd.DataFrame, cds: gpd.GeoDataFrame, path: Path) -> None:
    web = cds[["cduid", "geometry"]].copy()
    web["geometry"] = web.geometry.simplify(SIMPLIFY_WEB_M, preserve_topology=True)
    # ~28k coastal island parts dominate file size; display only, hazard uses full geometry
    parts = web.explode(index_parts=False)
    parts = parts[parts.geometry.area >= MIN_WEB_PART_KM2 * 1e6]
    web = parts.dissolve(by="cduid").reset_index()
    web = web.merge(div[["cduid", "cd_name", "hazard_score", "annual_burn_share",
                         "owner_households"]], on="cduid").to_crs(WGS84)
    path.unlink(missing_ok=True)  # GeoJSON driver won't overwrite
    web.to_file(path, driver="GeoJSON", COORDINATE_PRECISION=5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1990)
    ap.add_argument("--end", type=int, default=None, help="default: latest year in data")
    ap.add_argument("--refresh", action="store_true", help="re-download fires")
    args = ap.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    div, yearly, cds = build_divisions(args.start, args.end, args.refresh)
    div.to_parquet(PROCESSED / "divisions.parquet", index=False)
    yearly.to_parquet(PROCESSED / "burn_by_year.parquet", index=False)
    write_geojson(div, cds, PROCESSED / "divisions.geojson")

    cols = ["cduid", "cd_name", "annual_burn_share", "hazard_score", "owner_households"]
    print(div.sort_values("hazard_score", ascending=False)[cols].to_string(index=False))
    size_mb = (PROCESSED / "divisions.geojson").stat().st_size / 1e6
    print(f"Wrote divisions.parquet, burn_by_year.parquet, divisions.geojson ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
