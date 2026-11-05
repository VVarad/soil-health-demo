"""Train village-held-out CatBoost models and compare deficiency classifiers."""
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from joblib import Parallel, delayed, parallel_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict

# Depth, minimum samples per leaf, and fraction of candidate features.
RF_SETTINGS = [(16, 3, .6), (None, 1, .35), (None, 1, .6), (None, 2, .85),
               (12, 2, .35), (12, 4, .85), (8, 4, .6), (8, 8, .35)]


def catboost_path(root, fold, target):
    return Path(root) / "models" / f"catboost_fold{fold}_{target}.cbm"


def forest(settings):
    depth, leaf, fraction = settings
    return RandomForestClassifier(n_estimators=200, max_depth=depth, min_samples_leaf=leaf,
                                  max_features=fraction, random_state=42, n_jobs=1)


def evaluate_fold(X, labels, groups, train, test, fold, target, root):
    x_train, y_train = X.iloc[train], labels[train]
    inner = list(GroupKFold(3).split(x_train, groups=groups[train]))
    scores = []
    for settings in RF_SETTINGS:
        pred = cross_val_predict(forest(settings), x_train, y_train, cv=inner)
        scores.append(np.mean(pred == y_train))
    tuned = forest(RF_SETTINGS[np.argmax(scores)]).fit(x_train, y_train)
    rf_pred = tuned.predict(X.iloc[test])

    cat = CatBoostClassifier(depth=5, iterations=400, learning_rate=.05, random_seed=42,
                             loss_function="Logloss", verbose=False, allow_writing_files=False,
                             thread_count=1)
    cat.fit(x_train, y_train.astype(int))
    cat.save_model(str(catboost_path(root, fold, target)))
    return test, rf_pred, cat.predict_proba(X.iloc[test])[:, 1]


def train_classifiers(data, features, cutoffs, metrics, root, jobs):
    X = data[features]
    groups = (data.district + "|" + data.village).to_numpy()
    splits = list(GroupKFold(5).split(X, groups=groups))
    folds = np.zeros(len(X), dtype=int)
    for fold, (_, test) in enumerate(splits, 1):
        folds[test] = fold
    data["village_fold"] = folds
    for t, cutoff in cutoffs.items():
        labels = data[t].to_numpy() < cutoff
        tuned_predictions = np.zeros(len(X), dtype=bool)
        scores = np.zeros(len(X))
        with parallel_config(backend="loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=jobs)(
                delayed(evaluate_fold)(X, labels, groups, train, test, fold, t, root)
                for fold, (train, test) in enumerate(splits, 1))
        for test, rf_pred, cat_score in results:
            tuned_predictions[test], scores[test] = rf_pred, cat_score
        data[f"catboost_score_{t}"] = scores
        entry = metrics["targets"][t]
        entry["status_accuracy"] = float(np.mean((data[f"pred_{t}"] < cutoff) == labels))
        entry["tuned_rf_accuracy"] = float(np.mean(tuned_predictions == labels))
        entry["catboost_accuracy"] = float(np.mean((scores > .5) == labels))
        print(t, entry["status_accuracy"], entry["tuned_rf_accuracy"], entry["catboost_accuracy"], flush=True)
