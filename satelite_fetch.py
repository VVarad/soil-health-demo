#!/usr/bin/env python3
"""Sources:
https://developers.google.com/earth-engine/guides/auth
https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2
https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A2
https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY
https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path


DEFAULT_PROJECT = "soil-health-kolhapur-demo-482731"
SERVICE_ACCOUNT_EMAIL = (
    "satellite-reader@soil-health-kolhapur-demo-482731.iam.gserviceaccount.com"
)
SERVICE_ACCOUNT_KEY_FILE = "/home/varad/.config/soil-health/earth-engine-reader-7c91a0b3.json"

FEATURE_NAMES = (
    "temp", "SI", "precipitation", "sand05", "silt05", "sand515", "TGSI",
    "NDVI_Kharif", "NDVI_Rabi", "NDVI_Zaid", "pH_0-5", "pH_5-15",
    "NIRv_Kharif", "NIRv_Rabi", "NIRv_Zaid",
)
OPTICAL_BANDS = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
COLLECTIONS = {
    "landsat8": "LANDSAT/LC08/C02/T1_L2",
    "landsat9": "LANDSAT/LC09/C02/T1_L2",
    "temperature": "MODIS/061/MOD11A2",
    "rainfall": "UCSB-CHG/CHIRPS/DAILY",
    "sand": "projects/soilgrids-isric/sand_mean",
    "silt": "projects/soilgrids-isric/silt_mean",
    "soil_ph": "projects/soilgrids-isric/phh2o_mean",
}


def season_windows(year: int) -> dict[str, tuple[str, str]]:
    """Agricultural year; Earth Engine end dates are exclusive."""
    return {
        "Kharif": (f"{year}-07-01", f"{year}-11-01"),
        "Rabi": (f"{year}-11-01", f"{year + 1}-04-01"),
        "Zaid": (f"{year + 1}-04-01", f"{year + 1}-07-01"),
    }


def initialize_earth_engine(project: str, authenticate: bool):
    """Lazy import keeps dry runs/imports independent of installed EE packages."""
    if not project or project == DEFAULT_PROJECT:
        raise ValueError(
            "Configure EE_PROJECT or --project "
            "to an Earth Engine-enabled project you are authorized to use."
        )
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError("Install earthengine-api in a separate environment first.") from exc

    email = os.environ.get("EE_SERVICE_ACCOUNT")
    key_file = os.environ.get("EE_KEY_FILE")
    if bool(email) != bool(key_file):
        raise ValueError("Set both EE_SERVICE_ACCOUNT and EE_KEY_FILE, or neither.")
    if email and key_file:
        if authenticate:
            raise ValueError("Choose service-account authentication OR --authenticate.")
        credentials = ee.ServiceAccountCredentials(email, str(Path(key_file).expanduser()))
        ee.Initialize(credentials=credentials, project=project)
    else:
        if authenticate:
            ee.Authenticate()
        ee.Initialize(project=project)
    ee.data.setDeadline(120000)  # Milliseconds; fail instead of waiting indefinitely.
    return ee


def prepare_landsat(image):
    """Mask invalid observations, scale reflectance, and compute optical indices."""
    # QA_PIXEL bits 0..5: fill, dilated cloud, cirrus, cloud, shadow, snow.
    clear = image.select("QA_PIXEL").bitwiseAnd(63).eq(0)
    clear = clear.And(image.select("QA_RADSAT").eq(0))
    reflectance = (
        image.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
        .multiply(0.0000275).add(-0.2).rename(OPTICAL_BANDS).updateMask(clear)
    )
    # normalizedDifference also masks negative reflectance operands.
    ndvi = reflectance.normalizedDifference(["NIR", "RED"]).rename("NDVI")
    nirv = ndvi.multiply(reflectance.select("NIR")).rename("NIRv")
    return reflectance.addBands([ndvi, nirv]).copyProperties(image, ["system:time_start"])


def safe_composite(ee, collection, bands, reducer="mean"):
    """An empty collection produces masked bands, never invented zero values."""
    empty = ee.Image.constant([0] * len(bands)).rename(bands).updateMask(ee.Image(0))
    selected = collection.select(bands)
    reduced = selected.median() if reducer == "median" else selected.mean()
    return ee.Image(ee.Algorithms.If(collection.size().gt(0), reduced, empty))


def normalized_ratio(image, positive: str, negative: str, name: str):
    """Explicit ratio to preserve valid negative reflectance, guarding division."""
    a, b = image.select(positive), image.select(negative)
    denominator = a.add(b)
    return a.subtract(b).divide(denominator).updateMask(denominator.abs().gt(1e-8)).rename(name)


def fetch_features(ee, latitude: float, longitude: float, year: int) -> dict:
    """Fetch real data for one coordinate; no application imports or model calls."""
    point = ee.Geometry.Point([longitude, latitude])  # EE expects longitude first.
    roi = point.buffer(1000)
    start, end = f"{year}-07-01", f"{year + 1}-07-01"
    seasons = season_windows(year)

    # Landsat, rather than Sentinel-2, matches the current upstream notebook.
    landsat = (
        ee.ImageCollection(COLLECTIONS["landsat8"])
        .merge(ee.ImageCollection(COLLECTIONS["landsat9"]))
        .filterBounds(roi).filterDate(start, end).map(prepare_landsat)
    )
    annual = safe_composite(ee, landsat, OPTICAL_BANDS)
    optical = annual.addBands([
        normalized_ratio(annual, "RED", "BLUE", "SI"),
        normalized_ratio(annual, "SWIR1", "NIR", "TGSI"),
    ])
    seasonal_counts = {}
    for season, (begin, finish) in seasons.items():
        images = landsat.filterDate(begin, finish)
        median = safe_composite(ee, images, ["NDVI", "NIRv"], reducer="median")
        optical = optical.addBands(median.rename([f"NDVI_{season}", f"NIRv_{season}"]))
        count = ee.Image(ee.Algorithms.If(
            images.size().gt(0), images.select("NDVI").count(), ee.Image.constant(0)
        )).unmask(0).rename(f"valid_observations_{season}")
        optical = optical.addBands(count)
        seasonal_counts[season] = images.size()

    def prepare_temperature(image):
        qa = image.select("QC_Day")
        # Good mandatory QA, good data quality, reported LST error <= 1 K.
        good = qa.bitwiseAnd(3).eq(0).And(qa.rightShift(2).bitwiseAnd(3).eq(0))
        good = good.And(qa.rightShift(6).bitwiseAnd(3).eq(0))
        return image.select("LST_Day_1km").multiply(0.02).subtract(273.15).updateMask(good).rename("temp")

    modis = (ee.ImageCollection(COLLECTIONS["temperature"])
             .filterBounds(roi).filterDate(start, end).map(prepare_temperature))
    temperature = safe_composite(ee, modis, ["temp"])
    chirps = (ee.ImageCollection(COLLECTIONS["rainfall"])
              .filterBounds(roi).filterDate(start, end))
    # Mean DAILY rainfall over the period, not total annual rainfall.
    rainfall = safe_composite(ee, chirps, ["precipitation"])

    # SoilGrids are static modelled soil maps, NOT direct satellite measurements.
    sand = ee.Image(COLLECTIONS["sand"]).select(
        ["sand_0-5cm_mean", "sand_5-15cm_mean"], ["sand05", "sand515"])
    silt = ee.Image(COLLECTIONS["silt"]).select(["silt_0-5cm_mean"], ["silt05"])
    ph = ee.Image(COLLECTIONS["soil_ph"]).select(
        ["phh2o_0-5cm_mean", "phh2o_5-15cm_mean"], ["pH_0-5", "pH_5-15"]
    ).divide(10.0)
    soil = sand.addBands([silt, ph])  # Texture remains in g/kg.

    def at_point(image, scale):
        # UTM zone 43N covers Kolhapur/Sangli. Sampling coarse products at their
        # own scale avoids suggesting that all features have 30 m information.
        return image.reduceRegion(
            reducer=ee.Reducer.first(), geometry=point, crs="EPSG:32643",
            scale=scale, maxPixels=100000, tileScale=2,
        )

    # One explicit server request. No assets, Drive exports, or training jobs.
    response = ee.Dictionary({
        "optical": at_point(optical, 30),
        "temperature": at_point(temperature, 1000),
        "rainfall": at_point(rainfall, 5566),
        "soil_maps": at_point(soil, 250),
        "landsat_scene_counts": ee.Dictionary(seasonal_counts),
        "modis_scene_count": modis.size(),
        "chirps_day_count": chirps.size(),
    }).getInfo()

    values = {}
    for group in ("optical", "temperature", "rainfall", "soil_maps"):
        values.update(response.get(group) or {})
    features = {name: values.get(name) for name in FEATURE_NAMES}
    missing = [name for name, value in features.items() if value is None]
    return {
        "status": "partial" if missing else "complete",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "latitude": latitude, "longitude": longitude,
        "period": {"start_inclusive": start, "end_exclusive": end},
        "features": features, "missing_features": missing,
        "quality": {
            "landsat_scene_counts": response["landsat_scene_counts"],
            "valid_optical_observations": {
                season: values.get(f"valid_observations_{season}") for season in seasons
            },
            "modis_scene_count": response["modis_scene_count"],
            "chirps_day_count": response["chirps_day_count"],
        },
        "sources": COLLECTIONS,
        "units": {"temp": "daytime land surface temperature, deg C",
                  "precipitation": "mean daily rainfall, mm/day",
                  "sand05": "g/kg", "sand515": "g/kg", "silt05": "g/kg",
                  "pH_0-5": "pH", "pH_5-15": "pH",
                  "SI_TGSI_NDVI_NIRv": "dimensionless"},
        "model_compatible_without_revalidation": False,
        "note": "Environmental inputs only; this request does not measure or predict nutrients.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lat", type=float, default=16.7050, help="Latitude; defaults to Kolhapur city centre")
    parser.add_argument("--lon", type=float, default=74.2433)
    parser.add_argument("--year", type=int, default=2023, help="Agricultural year starts July 1")
    parser.add_argument("--project", default=os.environ.get("EE_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--fetch", action="store_true", help="Actually contact Earth Engine")
    parser.add_argument("--authenticate", action="store_true", help="Explicitly authorize OAuth login")
    parser.add_argument("--output", type=Path, help="Optional NEW JSON file; existing files are never overwritten")
    args = parser.parse_args(argv)

    if not (math.isfinite(args.lat) and math.isfinite(args.lon)
            and 15 <= args.lat <= 18.5 and 72.5 <= args.lon <= 76.5):
        parser.error("Use a finite coordinate near Kolhapur/Sangli (lat 15..18.5, lon 72.5..76.5).")
    if not 2013 <= args.year < date.today().year:
        parser.error("Choose a completed agricultural year from 2013 onwards.")
    if args.authenticate and not args.fetch:
        parser.error("--authenticate requires --fetch.")
    if args.output and not args.fetch:
        parser.error("--output requires --fetch; request plans are printed only.")
    if args.output and args.output.exists():
        parser.error("Output already exists; choose a new path to avoid changing existing files.")

    if not args.fetch:
        print(json.dumps({
            "mode": "request_plan_only_no_network",
            "project": args.project,
            "project_configuration_required": args.project == DEFAULT_PROJECT,
            "latitude": args.lat, "longitude": args.lon,
            "seasons": season_windows(args.year), "datasets": COLLECTIONS,
            "output_feature_names": FEATURE_NAMES,
            "authentication": "OAuth or service account; no API key required",
        }, indent=2))
        return 0

    try:
        ee = initialize_earth_engine(args.project, args.authenticate)
        result = fetch_features(ee, args.lat, args.lon, args.year)
        payload = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            # Exclusive create protects any existing app/data/model files.
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(payload + "\n")
        print(payload)
        return 0 if result["status"] == "complete" else 2
    except Exception as exc:
        # Never replace a failed request with invented observations or print keys.
        print(f"Fetch failed ({type(exc).__name__}). Check project access, credentials, "
              "dataset access, connectivity and output path. No fallback data generated.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
