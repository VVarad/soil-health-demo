"""
Train the micronutrient models (S, Zn, Fe) for Kolhapur and Sangli.

Inputs per field: basic soil test (N, P, K, OC, pH, EC) + satellite/soil-map features.
Outputs written for the app:
  data/fields.csv     one row per field: location, inputs, lab S/Zn/Fe, honest predictions
  models/<nutr>.joblib  final models trained on all fields (used for custom input)
  data/metrics.json   accuracy summary

"Honest predictions" are out-of-fold: each field is predicted by a model that never saw
its village (5-fold GroupKFold by village).

Needs the raw Soil Health Card data from the upstream project:
  python train.py --shc ../Soil-Health-Mapping-PanIndia/shc_data/AEZS/AGRI_2023-24
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

SATELLITE = ['temp', 'SI', 'precipitation', 'sand05', 'silt05', 'sand515', 'TGSI',
             'NDVI_Kharif', 'NDVI_Rabi', 'NDVI_Zaid', 'pH_0-5', 'pH_5-15',
             'NIRv_Kharif', 'NIRv_Rabi', 'NIRv_Zaid']
BASIC_TEST = ['N', 'P', 'K', 'OC', 'pH', 'EC']
FEATURES = SATELLITE + BASIC_TEST
# Soil Health Card critical limits (mg/kg): below = deficient
TARGETS = {'S': 10.0, 'Zn': 0.6, 'Fe': 4.5}


def rf():
    return RandomForestRegressor(n_estimators=200, max_depth=16, min_samples_leaf=3,
                                 max_features=0.6, random_state=42, n_jobs=-1)


def main(shc_dir):
    d = pd.concat([pd.read_csv(os.path.join(shc_dir, f)) for f in ["AEZ_6.csv", "AEZ_19.csv"]])
    d = d[d["district"].str.upper().isin(["KOLHAPUR", "SANGLI"])].copy()
    d["district"] = d["district"].str.title()
    d["village"] = d["village"].astype(str).str.strip().str.title()
    d["SI"] = (d.RED - d.BLUE) / (d.RED + d.BLUE)
    d["TGSI"] = (d.SWIR1 - d.NIR) / (d.SWIR1 + d.NIR)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + list(TARGETS))
    d = d.reset_index(drop=True)
    groups = d["district"] + "|" + d["village"]
    print(f"{len(d)} fields, {groups.nunique()} villages")

    os.makedirs("models", exist_ok=True)
    metrics = {}
    for t, cut in TARGETS.items():
        cv = GroupKFold(n_splits=5)
        d[f"pred_{t}"] = cross_val_predict(rf(), d[FEATURES], d[t], cv=cv, groups=groups)
        sat_only = cross_val_predict(rf(), d[SATELLITE], d[t], cv=cv, groups=groups)
        lab_def, pred_def = d[t] < cut, d[f"pred_{t}"] < cut
        metrics[t] = {
            "r2": round(r2_score(d[t], d[f"pred_{t}"]), 2),
            "r2_satellite_only": round(r2_score(d[t], sat_only), 2),
            "status_accuracy": round(float((lab_def == pred_def).mean()), 3),
            "baseline_accuracy": round(float(max(lab_def.mean(), 1 - lab_def.mean())), 3),
            "deficient_share": round(float(lab_def.mean()), 3),
            "cutoff": cut,
        }
        joblib.dump(rf().fit(d[FEATURES], d[t]), f"models/{t}.joblib", compress=3)
        print(t, metrics[t])

    cols = ["district", "village", "latitude", "longitude"] + FEATURES + list(TARGETS) + \
           [f"pred_{t}" for t in TARGETS]
    d[cols].round(5).to_csv("data/fields.csv", index=False)
    json.dump({"fields": len(d), "villages": int(groups.nunique()), "targets": metrics},
              open("data/metrics.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shc", default="../Soil-Health-Mapping-PanIndia/shc_data/AEZS/AGRI_2023-24")
    main(ap.parse_args().shc)
