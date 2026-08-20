"""
Four drop-in replacements for kirana_analysis.py.

Each block below replaces the function of the same name. Nothing else in the
module changes, and no call signature breaks: the two functions that gain a
return value gain it as an extra dict key, and the one that gains an argument
gains it with a default.

  1. choose_threshold / suggest_threshold  — stop reporting the 0.50 fallback
                                             as a chosen operating point
  2. _one_run                              — carry the floor flag through
  3. fit_deployment_models                 — honour the pooling setting, carry
                                             the flag, record the dummy map
  4. predict_new_shop                      — exact dummy matching

The matching app.py in this bundle already expects all four.
"""

# ============================================================ 1. threshold
#
# The bug: suggest_threshold initialises best = 0.50 and best_prec = -1.0, and
# only overwrites them when a cut-off clears the recall floor. When nothing on
# the grid reaches 70% recall — which is the expected case for a model sitting
# near chance — it returns 0.50 having selected nothing, and the dashboard then
# labels that "Cut-off chosen on training folds". The number is presented as a
# decision when it is an admission.
#
# Replace suggest_threshold with these two functions.


def choose_threshold(y_oof, proba_oof, min_recall: float = 0.70) -> dict:
    """
    Pick an operating point on out-of-fold training predictions.

    Choosing the threshold by staring at the test-set confusion matrix is the
    same leak as screening on the full sample. This picks the cut-off that
    maximises precision subject to a recall floor, using data the test set
    never saw.

    Returns the threshold together with whether the floor was actually
    reachable. When it is not, the returned 0.50 is a fallback and callers must
    say so rather than presenting it as a selection.
    """
    y_oof = np.asarray(y_oof)
    proba_oof = np.asarray(proba_oof)
    grid = np.arange(0.05, 0.96, 0.01)

    best, best_prec, floor_met = 0.50, -1.0, False
    best_recall = 0.0
    for t in grid:
        m = metrics_at(y_oof, proba_oof, float(t))
        if m["recall"] >= min_recall and m["precision"] > best_prec:
            best, best_prec, best_recall = float(t), m["precision"], m["recall"]
            floor_met = True

    if not floor_met:
        # Report what the fallback actually achieves, so the caller can show
        # the reader how far short of the floor the model falls.
        m = metrics_at(y_oof, proba_oof, 0.50)
        best_prec, best_recall = m["precision"], m["recall"]

    return {
        "threshold": best,
        "floor_met": floor_met,
        "min_recall": float(min_recall),
        "oof_precision": float(best_prec),
        "oof_recall": float(best_recall),
    }


def suggest_threshold(y_oof, proba_oof, min_recall: float = 0.70) -> float:
    """Backwards-compatible wrapper returning the threshold alone."""
    return choose_threshold(y_oof, proba_oof, min_recall)["threshold"]


# ============================================================== 2. _one_run
#
# Only the `if full:` block changes: it now calls choose_threshold and stores
# the flag alongside the value. Everything above and below is untouched.


def _one_run(X, y, seed: int, test_size: float, merge_rare: bool, full: bool) -> dict:
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # The merge map is learned on the training rows only.
    if merge_rare:
        keep = fit_rare_merge(X_tr_raw)
        X_tr_raw = apply_rare_merge(X_tr_raw, keep)
        X_te_raw = apply_rare_merge(X_te_raw, keep)
    else:
        keep = {}

    X_tr = encode(X_tr_raw)
    X_te = encode(X_te_raw, columns=X_tr.columns.tolist())

    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    cv = StratifiedKFold(n_splits=_n_splits(y_tr), shuffle=True, random_state=seed)

    results = {}
    for name, (est, grid) in _grids(pos_weight).items():
        search = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        proba = best.predict_proba(X_te)[:, 1]

        entry = {
            "cv_auc": float(search.best_score_),
            "test_auc": float(roc_auc_score(y_te, proba)),
            "test_ap": float(average_precision_score(y_te, proba)),
            "best_params": search.best_params_,
        }

        if full:
            fpr, tpr, _ = roc_curve(y_te, proba)
            prec, rec, _ = precision_recall_curve(y_te, proba)
            # Out-of-fold training predictions: the honest place to pick a
            # threshold, and a second read on how the model generalises.
            oof = cross_val_predict(
                best, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1
            )[:, 1]
            chosen = choose_threshold(y_tr.values, oof)
            entry.update(
                {
                    "proba": proba,
                    "roc": (fpr, tpr),
                    "pr": (rec, prec),
                    "oof_proba": oof,
                    "oof_y": y_tr.values,
                    "suggested_threshold": chosen["threshold"],
                    "suggested_threshold_floor_met": chosen["floor_met"],
                    "suggested_threshold_detail": chosen,
                    "at_050": metrics_at(y_te, proba, 0.50),
                    "importances": _importances(best, X_tr.columns.tolist()),
                }
            )
            entry["at_suggested"] = metrics_at(
                y_te, proba, entry["suggested_threshold"]
            )

        results[name] = entry

    return {
        "results": results,
        "y_test": y_te.values,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_encoded_cols": int(X_tr.shape[1]),
        "train_events": int(y_tr.sum()),
        "test_events": int(y_te.sum()),
        "majority_accuracy": float((y_te == 0).mean()),
        "n_splits": int(cv.get_n_splits()),
        "rare_merge_applied": bool(merge_rare),
    }


# =================================================== 3. fit_deployment_models
#
# Two problems. The app calls deployment(X, y) without passing the sidebar's
# pooling setting, so Score a Shop always pooled regardless of what the rest of
# the page was doing and the banner could read "reported specification" while
# the scorer ran a different encoding. And the dummy-column map is rebuilt in
# predict_new_shop by prefix matching, which is fragile; it is recorded here
# instead, where the levels are already known.


def fit_deployment_models(
    X: pd.DataFrame, y: pd.Series, merge_rare: bool = True
) -> dict:
    """Fit deployment models on all resolved observations."""
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].astype(int)

    keep = fit_rare_merge(X) if merge_rare else {}
    X_merged = apply_rare_merge(X, keep) if merge_rare else X
    encoded = encode(X_merged)
    cat_cols, bin_cols = split_column_kinds(X)

    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    cv = StratifiedKFold(n_splits=_n_splits(y), shuffle=True, random_state=SEED)

    fitted, thresholds, floor_met = {}, {}, {}
    for name, (estimator, grid) in _grids(pos_weight).items():
        search = GridSearchCV(estimator, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(encoded, y)
        fitted[name] = search.best_estimator_
        oof = cross_val_predict(
            search.best_estimator_, encoded, y, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        chosen = choose_threshold(y.values, oof)
        thresholds[name] = chosen["threshold"]
        floor_met[name] = chosen["floor_met"]

    levels = {c: sorted(X_merged[c].astype(str).unique().tolist()) for c in cat_cols}

    # Recorded rather than reconstructed. get_dummies names a column
    # f"{col}_{level}", and two predictors where one name is a prefix of the
    # other ("Sales Trend" and "Sales Trend Detail") make prefix matching
    # ambiguous at exactly the moment it is used to decide whether an answer
    # reached the model.
    dummy_map = {
        c: [f"{c}_{lvl}" for lvl in levels[c] if f"{c}_{lvl}" in encoded.columns]
        for c in cat_cols
    }

    return {
        "models": fitted,
        "columns": encoded.columns.tolist(),
        "cat_cols": cat_cols,
        "bin_cols": bin_cols,
        "rare_keep": keep,
        "levels": levels,
        "dummy_map": dummy_map,
        "thresholds": thresholds,
        "threshold_floor_met": floor_met,
        "merge_rare": bool(merge_rare),
        "n_fit": int(len(y)),
        "base_rate": float(y.mean()),
    }


# ======================================================= 4. predict_new_shop
#
# Uses the recorded dummy map, falling back to the old prefix scan only for a
# bundle fitted before this change.


def predict_new_shop(bundle: dict, record: dict) -> dict:
    """
    Predict failure probability for one new shop record.

    The two column kinds are handled separately. Casting the binary
    multi-response indicators to string turned 1 into "1", get_dummies made
    "Col: Option_1" out of it, reindex found no match and filled zero — so
    every multi-response signal silently disappeared while the function still
    returned a confident-looking number.
    """
    cat_cols = bundle["cat_cols"]
    bin_cols = bundle["bin_cols"]

    row: dict[str, object] = {}
    for c in cat_cols:
        v = str(record.get(c, MISSING_LABEL)).strip()
        row[c] = MISSING_LABEL if v.lower() in {"", "nan", "none"} else v
    for c in bin_cols:
        row[c] = int(pd.to_numeric(record.get(c, 0), errors="coerce") or 0)

    raw = pd.DataFrame([row])
    raw = apply_rare_merge(raw, bundle.get("rare_keep", {}))
    encoded = encode(raw, columns=bundle["columns"])

    # Every categorical column must light up exactly one dummy. If none fires,
    # the value was unseen at fit time and the model is being handed an
    # all-zero block rather than an answer.
    dummy_map = bundle.get("dummy_map")
    unmatched = []
    for c in cat_cols:
        if dummy_map is not None:
            cols = dummy_map.get(c, [])
        else:  # bundle predates the recorded map
            cols = [col for col in bundle["columns"] if col.startswith(f"{c}_")]
        if not any(encoded[col].iloc[0] == 1 for col in cols if col in encoded.columns):
            unmatched.append(f"{c}={row[c]!r}")

    probs = {
        name: float(model.predict_proba(encoded)[:, 1][0])
        for name, model in bundle["models"].items()
    }
    return {
        "probabilities": probs,
        "thresholds": bundle.get("thresholds", {}),
        "threshold_floor_met": bundle.get("threshold_floor_met", {}),
        "unmatched": unmatched,
    }
