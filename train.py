import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

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

TARGETS = {'S': 10.0, 'Zn': 0.6, 'Fe': 4.5}


def rf():
    return RandomForestRegressor(n_estimators=200, max_depth=16, min_samples_leaf=3,
                                 max_features=0.6, random_state=42, n_jobs=-1)


def rebuild_regressors(shc_dir, output):
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
        joblib.dump(rf().fit(d[FEATURES], d[t]), output / "models" / f"{t}.joblib", compress=3)
        print(t, metrics[t])

    cols = ["district", "village", "latitude", "longitude"] + FEATURES + list(TARGETS) + \
           [f"pred_{t}" for t in TARGETS]
    data = d[cols].round(5).copy()
    return data, {"fields": len(d), "villages": int(groups.nunique()), "targets": metrics}


def main(shc_dir, jobs=4):
    from classifier_study import train_classifiers

    root = Path(__file__).resolve().parent
    # Complete training in a staging directory before replacing any serving files.
    with TemporaryDirectory(dir=root) as tmp:
        staging = Path(tmp)
        (staging / "models").mkdir()
        (staging / "data").mkdir()
        if shc_dir is not None:
            data, metrics = rebuild_regressors(shc_dir, staging)
        else:
            data = pd.read_csv(root / "data/fields.csv")
            metrics = json.loads((root / "data/metrics.json").read_text())
        train_classifiers(data, FEATURES, TARGETS, metrics, staging, jobs)
        data.to_csv(staging / "data/fields.csv", index=False)
        (staging / "data/metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        for artifact in staging.glob("*/*"):
            artifact.replace(root / artifact.relative_to(staging))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shc", type=Path, help="Also rebuild the RF regressors from the original raw SHC CSVs.")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel evaluation folds.")
    args = parser.parse_args()
    main(args.shc.resolve() if args.shc else None, args.jobs)
