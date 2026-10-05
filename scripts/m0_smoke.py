"""M0 smoke test: BC wildfire burn share by census division.

Downloads historical fire perimeters (BC WFS, paged), reads 2021 census
division boundaries, intersects them in BC Albers (EPSG:3005) and computes,
per census division (CD):
  - annual_burn_share: mean over years of (area burned that year / land area)
  - ever_burned_share: share of land area burned at least once in the window
Writes out/m0_burn_share.csv and out/m0_burn_share.png.

Usage (from project root):
    python scripts/m0_smoke.py [--start 1990] [--end 2025] [--refresh]
"""
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "out"

ALBERS = "EPSG:3005"
FIRE_LAYER = "WHSE_LAND_AND_NATURAL_RESOURCE.PROT_HISTORICAL_FIRE_POLYS_SP"
WFS_URL = f"https://openmaps.gov.bc.ca/geo/pub/{FIRE_LAYER}/ows"
PAGE_SIZE = 10_000  # server CountDefault
FIRES_CACHE = RAW / "bc_fire_perimeters_historical.gpkg"
CD_ZIP = RAW / "lcd_000b21a_e.zip"
BC_PRUID = "59"
EXPECTED_BC_CDS = 29
SIMPLIFY_M = 50  # tolerance (metres) for CD geometry used in intersection only


def download_fires(refresh: bool = False) -> gpd.GeoDataFrame:
    """Fetch all historical fire perimeters via paged WFS, cached as GeoPackage."""
    if FIRES_CACHE.exists() and not refresh:
        print(f"Using cached fires: {FIRES_CACHE.name}")
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
    print(f"Cached {len(fires):,} fires -> {FIRES_CACHE.name}")
    return fires


def load_cds() -> gpd.GeoDataFrame:
    cds = gpd.read_file(f"zip://{CD_ZIP}", where=f"PRUID='{BC_PRUID}'")
    assert len(cds) == EXPECTED_BC_CDS, f"expected {EXPECTED_BC_CDS} BC CDs, got {len(cds)}"
    cds = cds.to_crs(ALBERS)
    cds["land_area_km2"] = cds.geometry.area / 1e6
    return cds[["CDUID", "CDNAME", "LANDAREA", "land_area_km2", "geometry"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1990)
    ap.add_argument("--end", type=int, default=None, help="default: latest year in data")
    ap.add_argument("--refresh", action="store_true", help="re-download fires")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    fires = download_fires(args.refresh)
    end = args.end or int(fires["FIRE_YEAR"].max())
    fires = fires[fires["FIRE_YEAR"].between(args.start, end)].copy()
    n_years = end - args.start + 1
    print(f"Window {args.start}-{end} ({n_years} yrs): {len(fires):,} perimeters")

    fires["geometry"] = fires.geometry.make_valid()
    fires = fires[fires.geometry.geom_type.isin(["Polygon", "MultiPolygon", "GeometryCollection"])]

    # One (multi)polygon per year so same-year overlaps aren't double counted.
    by_year = fires[["FIRE_YEAR", "geometry"]].dissolve(by="FIRE_YEAR").reset_index()
    reported_ha = fires["FIRE_SIZE_HECTARES"].sum()
    dissolved_ha = by_year.geometry.area.sum() / 1e4
    print(f"Sanity: reported {reported_ha:,.0f} ha vs dissolved-by-year {dissolved_ha:,.0f} ha "
          f"(ratio {dissolved_ha / reported_ha:.2f})")

    cds = load_cds()
    cds_simple = cds[["CDUID", "geometry"]].copy()
    cds_simple["geometry"] = cds_simple.geometry.simplify(SIMPLIFY_M, preserve_topology=True)

    # Annual burn: area burned per (CD, year), averaged over all years in window.
    inter = gpd.overlay(cds_simple, by_year, how="intersection", keep_geom_type=True)
    inter["burned_km2"] = inter.geometry.area / 1e6
    per_cd_year = inter.groupby(["CDUID", "FIRE_YEAR"])["burned_km2"].sum()
    annual_km2 = per_cd_year.groupby("CDUID").sum() / n_years  # zero-burn years count

    # Ever burned: union across years, then intersect.
    ever = gpd.GeoDataFrame(geometry=[by_year.geometry.unary_union], crs=ALBERS)
    ever_inter = gpd.overlay(cds_simple, ever, how="intersection", keep_geom_type=True)
    ever_km2 = ever_inter.assign(a=ever_inter.geometry.area / 1e6).groupby("CDUID")["a"].sum()

    res = cds.drop(columns="geometry").set_index("CDUID")
    res["annual_burned_km2"] = annual_km2.reindex(res.index).fillna(0.0)
    res["ever_burned_km2"] = ever_km2.reindex(res.index).fillna(0.0)
    res["annual_burn_share"] = res["annual_burned_km2"] / res["land_area_km2"]
    res["ever_burned_share"] = res["ever_burned_km2"] / res["land_area_km2"]
    res = res.reset_index()

    assert res["annual_burn_share"].between(0, 1).all()
    assert res["ever_burned_share"].between(0, 1).all()
    area_diff = (res["land_area_km2"] / res["LANDAREA"] - 1).abs().max()
    print(f"Max |computed land area / StatCan LANDAREA - 1| = {area_diff:.1%}")

    res.to_csv(OUT / "m0_burn_share.csv", index=False)
    cols = ["CDUID", "CDNAME", "land_area_km2", "annual_burn_share", "ever_burned_share"]
    print(res.sort_values("annual_burn_share", ascending=False)[cols].to_string(
        index=False, float_format=lambda x: f"{x:,.4f}"))

    plot = cds.merge(res[["CDUID", "annual_burn_share", "ever_burned_share"]], on="CDUID")
    plot["geometry"] = plot.geometry.simplify(500)  # display only
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, col, title in [
        (axes[0], "annual_burn_share", f"Mean annual share burned, {args.start}-{end}"),
        (axes[1], "ever_burned_share", f"Share burned at least once, {args.start}-{end}"),
    ]:
        plot.plot(column=col, ax=ax, cmap="OrRd", legend=True, edgecolor="grey", linewidth=0.3)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle("BC census divisions: historical wildfire burn share (M0 smoke test)")
    fig.tight_layout()
    fig.savefig(OUT / "m0_burn_share.png", dpi=150)
    print(f"Wrote {OUT / 'm0_burn_share.csv'} and {OUT / 'm0_burn_share.png'}")


if __name__ == "__main__":
    main()
