
# from __future__ import annotations

# import warnings

# import numpy as np
# import pandas as pd
# from scipy import stats
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     average_precision_score,
#     confusion_matrix,
#     precision_recall_curve,
#     roc_auc_score,
#     roc_curve,
# )
# from sklearn.model_selection import (
#     GridSearchCV,
#     StratifiedKFold,
#     cross_val_predict,
#     train_test_split,
# )

# try:
#     from xgboost import XGBClassifier

#     HAS_XGB = True
# except ImportError:  # the app degrades to two models rather than failing
#     HAS_XGB = False


# # ---------------------------------------------------------------- schema

# PREFERRED_SHEET = "Data 400"

# ID_COLS = ["Survey ID", "Shop Name"]

# # Fields that only exist once the outcome is known. Dropped before modelling
# # so neither target can leak back into the predictors.
# OUTCOME_COLS = ["Owner's Expectation", "Consideration of Closure", "Outcome Status"]

# REQUIRED_COLS = ["Outcome Status", "Owner's Expectation"]

# FAILURE_STATUSES = [
#     "Permanently closed",
#     "Temporarily closed but expected to reopen",
#     "Changed into another type of business",
# ]
# UNRESOLVED_STATUS = "Could not be verified"

# BELIEF_HIGH_RISK = ["Unlikely", "Very unlikely"]
# BELIEF_MISSING = {"", "nan", "none", "<na>", "not reported", "no response", "na", "n/a"}

# MIN_CATEGORY_N = 10  # categories smaller than this are merged before testing
# MISSING_LABEL = "Not reported"
# SEED = 42


# class WorkbookError(ValueError):
#     """Raised when an uploaded workbook does not match the expected schema."""


# # ------------------------------------------------------------------ data


# def _clean_name(name: object) -> str:
#     """Normalise a column header: strip, collapse spaces, straighten quotes."""
#     s = str(name).strip()
#     s = s.replace("\u2019", "'").replace("\u2018", "'")
#     s = s.replace("\u201c", '"').replace("\u201d", '"')
#     s = s.replace("\u00a0", " ")
#     return " ".join(s.split())


# def list_sheets(path_or_buffer) -> list[str]:
#     """Sheet names in the workbook, so the app can offer a picker."""
#     try:
#         return pd.ExcelFile(path_or_buffer).sheet_names
#     except Exception:  # unreadable file; the caller reports it
#         return []


# def load_raw(path_or_buffer, sheet: str | None = None) -> pd.DataFrame:
#     """
#     Read the survey workbook.

#     The sheet is resolved rather than hardcoded: an uploaded workbook that
#     happens to name its sheet something else should still load.
#     """
#     sheets = list_sheets(path_or_buffer)
#     if sheet is None:
#         sheet = PREFERRED_SHEET if PREFERRED_SHEET in sheets else (sheets[0] if sheets else 0)
#     df = pd.read_excel(path_or_buffer, sheet_name=sheet)
#     df.columns = [_clean_name(c) for c in df.columns]

#     missing = [c for c in REQUIRED_COLS if c not in df.columns]
#     if missing:
#         raise WorkbookError(
#             "The workbook is missing required column(s): "
#             + ", ".join(missing)
#             + ". Found: "
#             + ", ".join(map(str, df.columns[:25]))
#             + ("…" if len(df.columns) > 25 else "")
#         )

#     if "Survey ID" in df.columns and df["Survey ID"].duplicated().any():
#         n = int(df["Survey ID"].duplicated().sum())
#         warnings.warn(f"{n} duplicated Survey ID value(s) in the workbook.", stacklevel=2)

#     return df


# def build_targets(df: pd.DataFrame, unresolved_as_failure: bool = False) -> pd.DataFrame:
#     """
#     Attach the two targets.

#     unresolved_as_failure supports the sensitivity check the missingness
#     demands: a shop that could not be verified is plausibly one that
#     disappeared, so dropping those rows may bias the base rate downward.
#     Set it True to recode them as failures and see how far the results move.
#     """
#     out = df.copy()
#     status = out["Outcome Status"].astype(str).str.strip()

#     observed = pd.Series(np.nan, index=out.index, dtype="float")
#     observed[status.isin(FAILURE_STATUSES)] = 1.0
#     observed[~status.isin(FAILURE_STATUSES) & (status != UNRESOLVED_STATUS)] = 0.0
#     if unresolved_as_failure:
#         observed[status == UNRESOLVED_STATUS] = 1.0
#     out["target_observed"] = observed

#     # A blank expectation is not evidence of confidence. It stays NaN and is
#     # dropped downstream, exactly as an unresolved outcome is.
#     raw_expect = out["Owner's Expectation"]
#     expect = raw_expect.astype(str).str.strip()
#     belief = pd.Series(np.nan, index=out.index, dtype="float")
#     answered = ~(raw_expect.isna() | expect.str.lower().isin(BELIEF_MISSING))
#     belief[answered] = expect[answered].isin(BELIEF_HIGH_RISK).astype(float)
#     out["target_belief"] = belief

#     return out


# # ------------------------------------------------------------ predictors


# def multi_response_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
#     """Columns storing several answers joined with semicolons."""
#     return [c for c in candidates if df[c].astype(str).str.contains(";", na=False).any()]


# def build_predictors(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
#     """
#     Returns the predictor frame plus the names of the single-response and
#     multi-response-derived variables. Multi-response items become one binary
#     indicator per option, which keeps the encoded matrix far narrower than
#     treating every observed combination as its own category.
#     """
#     drop = set(ID_COLS + OUTCOME_COLS + ["target_observed", "target_belief"])
#     candidates = [c for c in df.columns if c not in drop]

#     multi = multi_response_columns(df, candidates)
#     single = [c for c in candidates if c not in multi]

#     X = df[single].copy()
#     for c in X.columns:
#         col = X[c].astype(str).str.strip()
#         blank = X[c].isna() | col.str.lower().isin({"nan", "<na>", "none", ""})
#         X[c] = col.mask(blank, MISSING_LABEL).astype(str)

#     derived: list[str] = []
#     for col in multi:
#         exploded = (
#             df[col]
#             .fillna("")
#             .astype(str)
#             .str.split(";")
#             .apply(
#                 lambda parts: [
#                     p.strip()
#                     for p in (parts if isinstance(parts, list) else [])
#                     if p.strip() and p.strip().lower() != "nan"
#                 ]
#             )
#         )
#         options = sorted({opt for row in exploded for opt in row})
#         for opt in options:
#             name = f"{col}: {opt}"
#             X[name] = exploded.apply(lambda row, o=opt: int(o in row)).astype(int)
#             derived.append(name)

#     return X, single, derived


# def split_column_kinds(X: pd.DataFrame) -> tuple[list[str], list[str]]:
#     """
#     Categorical (text) columns and binary indicator columns, separately.

#     Tested on dtype rather than on `== object`: pandas 3 stores text in a
#     dedicated string dtype, so the old object check silently classified every
#     categorical column as numeric and one-hot encoded nothing.
#     """
#     binary = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
#     cat = [c for c in X.columns if c not in binary]
#     return cat, binary


# # --------------------------------------------------- rare-category merge


# def fit_rare_merge(X: pd.DataFrame, min_n: int = MIN_CATEGORY_N) -> dict[str, list[str]]:
#     """
#     Learn which categories are frequent enough to keep, per column.

#     Fitted on training data only. Applying a merge learned from the full
#     sample would let the test rows influence the encoding.
#     """
#     cat_cols, _ = split_column_kinds(X)
#     keep: dict[str, list[str]] = {}
#     for col in cat_cols:
#         counts = X[col].astype(str).value_counts()
#         keep[col] = sorted(counts[counts >= min_n].index.tolist())
#     return keep


# def apply_rare_merge(X: pd.DataFrame, keep: dict[str, list[str]]) -> pd.DataFrame:
#     out = X.copy()
#     for col, kept in keep.items():
#         if col not in out.columns:
#             continue
#         s = out[col].astype(str)
#         out[col] = s.where(s.isin(kept), "Other (merged)")
#     return out


# def encode(X: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
#     """One-hot the categorical columns, leaving binary indicators untouched."""
#     cat_cols, _ = split_column_kinds(X)
#     enc = pd.get_dummies(X, columns=cat_cols, drop_first=False)
#     enc = enc.astype(float)
#     if columns is not None:
#         enc = enc.reindex(columns=columns, fill_value=0.0)
#     return enc


# # --------------------------------------------------- chi-square screen


# def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
#     denom = n * (min(r, c) - 1)
#     return float(np.sqrt(chi2 / denom)) if denom > 0 and np.isfinite(chi2) else np.nan


# def cramers_v_bias_corrected(chi2: float, n: int, r: int, c: int) -> float:
#     """
#     Bergsma (2013) bias correction. Cramer's V is biased upward at small n
#     and with many categories, which is exactly this dataset's situation.
#     """
#     if not np.isfinite(chi2) or n <= 1:
#         return np.nan
#     phi2 = chi2 / n
#     phi2c = max(0.0, phi2 - ((r - 1) * (c - 1)) / (n - 1))
#     rc = r - (r - 1) ** 2 / (n - 1)
#     cc = c - (c - 1) ** 2 / (n - 1)
#     denom = min(rc - 1, cc - 1)
#     return float(np.sqrt(phi2c / denom)) if denom > 0 else np.nan


# def _merge_rare(series: pd.Series, min_n: int = MIN_CATEGORY_N) -> pd.Series:
#     counts = series.value_counts()
#     rare = counts[counts < min_n].index
#     if len(rare) == 0:
#         return series
#     return series.where(~series.isin(rare), "Other (merged)")


# def screen_predictors(X: pd.DataFrame, y: pd.Series, yates: bool = False) -> pd.DataFrame:
#     """
#     Chi-square test of independence per predictor, with Cramer's V and BH q.

#     yates=False by default. The continuity correction deflates chi2 on 2x2
#     tables, and the same chi2 is the numerator of the effect size, so leaving
#     it on understates both. Set yates=True to reproduce the conservative
#     p-values if a reviewer asks for them.
#     """
#     mask = y.notna()
#     X, y = X.loc[mask], y.loc[mask].astype(int)

#     rows = []
#     for col in X.columns:
#         s = _merge_rare(X[col].astype(str))
#         table = pd.crosstab(s, y)
#         if table.shape[0] < 2 or table.shape[1] < 2:
#             continue

#         chi2, p, dof, expected = stats.chi2_contingency(table, correction=yates)
#         small_expected = float((expected < 5).mean())
#         n = int(table.values.sum())
#         r, c = table.shape

#         v = cramers_v(chi2, n, r, c)
#         v_corr = cramers_v_bias_corrected(chi2, n, r, c)
#         test = "Chi-square"
#         chi2_out: float = float(chi2)
#         dof_out: float = float(dof)

#         # Fisher's exact for 2x2 tables where the chi-square approximation
#         # fails. When it fires, chi2 and dof belong to a test that was not
#         # used for the p-value, so they are blanked rather than reported.
#         if table.shape == (2, 2) and small_expected > 0.20:
#             _, p = stats.fisher_exact(table.values)
#             test = "Fisher exact"
#             chi2_out = np.nan
#             dof_out = np.nan

#         rows.append(
#             {
#                 "variable": col,
#                 "chi2": chi2_out,
#                 "dof": dof_out,
#                 "p": float(p),
#                 "cramers_v": v,
#                 "cramers_v_corrected": v_corr,
#                 "test": test,
#                 "n": n,
#                 "pct_expected_below_5": small_expected,
#                 "n_categories": int(r),
#                 "low_cell_warning": small_expected > 0.20,
#             }
#         )

#     cols = [
#         "variable", "chi2", "dof", "p", "cramers_v", "cramers_v_corrected", "test",
#         "n", "pct_expected_below_5", "n_categories", "low_cell_warning",
#     ]
#     if not rows:  # an empty frame used to blow up on sort_values("p")
#         res = pd.DataFrame(columns=cols)
#         res["q"] = pd.Series(dtype=float)
#         res["significant_raw"] = pd.Series(dtype=bool)
#         res["survives_bh"] = pd.Series(dtype=bool)
#         return res

#     res = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
#     res["q"] = benjamini_hochberg(res["p"].values)
#     res["significant_raw"] = res["p"] < 0.05
#     res["survives_bh"] = res["q"] < 0.05
#     return res


# def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
#     p = np.asarray(p, dtype=float)
#     n = len(p)
#     if n == 0:
#         return np.array([], dtype=float)
#     order = np.argsort(p)
#     ranked = p[order] * n / (np.arange(n) + 1)
#     ranked = np.minimum.accumulate(ranked[::-1])[::-1]
#     q = np.empty(n)
#     q[order] = np.clip(ranked, 0, 1)
#     return q


# # --------------------------------------------------------------- models


# def _grids(pos_weight: float) -> dict:
#     """
#     Estimators and their grids.

#     n_jobs stays at 1 on the estimators; only the search parallelises.
#     Nesting -1 inside -1 oversubscribes threads, which on a 1 GB Streamlit
#     Community Cloud container is a plausible way to get the app killed.
#     """
#     grids = {
#         "Logistic Regression": (
#             LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED),
#             {"C": [0.01, 0.1, 1.0]},
#         ),
#         "Random Forest": (
#             RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1),
#             {"n_estimators": [300], "max_depth": [3, 5, None], "min_samples_leaf": [1, 5]},
#         ),
#     }
#     if HAS_XGB:
#         grids["XGBoost"] = (
#             XGBClassifier(
#                 eval_metric="logloss",
#                 random_state=SEED,
#                 scale_pos_weight=pos_weight,
#                 n_jobs=1,
#             ),
#             {"n_estimators": [200], "max_depth": [2, 3], "learning_rate": [0.05, 0.1]},
#         )
#     return grids


# def _n_splits(y_train: pd.Series, requested: int = 5) -> int:
#     """
#     Fold count that cannot leave a fold with zero positives.

#     With five folds and eight positives, a fold with none returns NaN AUC and
#     GridSearchCV propagates it silently.
#     """
#     pos = int(np.sum(y_train == 1))
#     neg = int(np.sum(y_train == 0))
#     return max(2, min(requested, pos, neg))


# def metrics_at(y_true, proba, threshold: float) -> dict:
#     y_true = np.asarray(y_true)
#     proba = np.asarray(proba)
#     if len(y_true) == 0:
#         return {
#             "threshold": threshold, "accuracy": np.nan, "precision": 0.0, "recall": 0.0,
#             "specificity": 0.0, "f1": 0.0, "tn": 0, "fp": 0, "fn": 0, "tp": 0,
#         }
#     pred = (proba >= threshold).astype(int)
#     tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
#     recall = tp / (tp + fn) if (tp + fn) else 0.0
#     precision = tp / (tp + fp) if (tp + fp) else 0.0
#     return {
#         "threshold": threshold,
#         "accuracy": (tp + tn) / len(y_true),
#         "precision": precision,
#         "recall": recall,
#         "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
#         "f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
#         "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
#     }


# def suggest_threshold(y_oof, proba_oof, min_recall: float = 0.70) -> float:
#     """
#     Pick an operating point on out-of-fold training predictions.

#     Choosing the threshold by staring at the test-set confusion matrix is the
#     same leak as screening on the full sample. This picks the cut-off that
#     maximises precision subject to a recall floor, using data the test set
#     never saw.
#     """
#     y_oof = np.asarray(y_oof)
#     proba_oof = np.asarray(proba_oof)
#     grid = np.arange(0.05, 0.96, 0.01)
#     best, best_prec = 0.50, -1.0
#     for t in grid:
#         m = metrics_at(y_oof, proba_oof, float(t))
#         if m["recall"] >= min_recall and m["precision"] > best_prec:
#             best, best_prec = float(t), m["precision"]
#     return best


# def _importances(est, columns: list[str]) -> pd.DataFrame:
#     """Coefficients as odds ratios, or tree importances, whichever applies."""
#     if hasattr(est, "coef_"):
#         coef = np.ravel(est.coef_)
#         out = pd.DataFrame(
#             {"feature": columns, "coefficient": coef, "odds_ratio": np.exp(coef)}
#         )
#         out["magnitude"] = out["coefficient"].abs()
#     elif hasattr(est, "feature_importances_"):
#         out = pd.DataFrame(
#             {"feature": columns, "importance": np.asarray(est.feature_importances_)}
#         )
#         out["magnitude"] = out["importance"]
#     else:
#         return pd.DataFrame(columns=["feature", "magnitude"])
#     return out.sort_values("magnitude", ascending=False).reset_index(drop=True)


# def _one_run(X, y, seed: int, test_size: float, merge_rare: bool, full: bool) -> dict:
#     X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
#         X, y, test_size=test_size, stratify=y, random_state=seed
#     )

#     # The merge map is learned on the training rows only.
#     if merge_rare:
#         keep = fit_rare_merge(X_tr_raw)
#         X_tr_raw = apply_rare_merge(X_tr_raw, keep)
#         X_te_raw = apply_rare_merge(X_te_raw, keep)
#     else:
#         keep = {}

#     X_tr = encode(X_tr_raw)
#     X_te = encode(X_te_raw, columns=X_tr.columns.tolist())

#     pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
#     cv = StratifiedKFold(n_splits=_n_splits(y_tr), shuffle=True, random_state=seed)

#     results = {}
#     for name, (est, grid) in _grids(pos_weight).items():
#         search = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
#         search.fit(X_tr, y_tr)
#         best = search.best_estimator_
#         proba = best.predict_proba(X_te)[:, 1]

#         entry = {
#             "cv_auc": float(search.best_score_),
#             "test_auc": float(roc_auc_score(y_te, proba)),
#             "test_ap": float(average_precision_score(y_te, proba)),
#             "best_params": search.best_params_,
#         }

#         if full:
#             fpr, tpr, _ = roc_curve(y_te, proba)
#             prec, rec, _ = precision_recall_curve(y_te, proba)
#             # Out-of-fold training predictions: the honest place to pick a
#             # threshold, and a second read on how the model generalises.
#             oof = cross_val_predict(
#                 best, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1
#             )[:, 1]
#             entry.update(
#                 {
#                     "proba": proba,
#                     "roc": (fpr, tpr),
#                     "pr": (rec, prec),
#                     "oof_proba": oof,
#                     "oof_y": y_tr.values,
#                     "suggested_threshold": suggest_threshold(y_tr.values, oof),
#                     "at_050": metrics_at(y_te, proba, 0.50),
#                     "importances": _importances(best, X_tr.columns.tolist()),
#                 }
#             )
#             entry["at_suggested"] = metrics_at(y_te, proba, entry["suggested_threshold"])

#         results[name] = entry

#     return {
#         "results": results,
#         "y_test": y_te.values,
#         "n_train": int(len(y_tr)),
#         "n_test": int(len(y_te)),
#         "n_encoded_cols": int(X_tr.shape[1]),
#         "train_events": int(y_tr.sum()),
#         "test_events": int(y_te.sum()),
#         "majority_accuracy": float((y_te == 0).mean()),
#         "n_splits": int(cv.get_n_splits()),
#         "rare_merge_applied": bool(merge_rare),
#     }


# def run_models(
#     X: pd.DataFrame,
#     y: pd.Series,
#     test_size: float = 0.30,
#     n_repeats: int = 10,
#     merge_rare: bool = True,
# ) -> dict:
#     """
#     Stratified split, tuned by grid search inside stratified CV on the
#     training portion, test set scored once.

#     Repeated across n_repeats seeds. One split with this few test events
#     cannot distinguish a real difference between models from the luck of the
#     draw, so the repeats table is the figure worth reporting; the seed-SEED
#     run is kept in full for the curves and the confusion matrices.
#     """
#     mask = y.notna()
#     X, y = X.loc[mask], y.loc[mask].astype(int)

#     if int(y.sum()) < 4:
#         raise WorkbookError(
#             f"Only {int(y.sum())} failure events available. Too few to split and model."
#         )

#     primary = _one_run(X, y, SEED, test_size, merge_rare, full=True)

#     rows = []
#     for i in range(n_repeats):
#         seed = SEED + i
#         run = primary if seed == SEED else _one_run(X, y, seed, test_size, merge_rare, full=False)
#         for name, r in run["results"].items():
#             rows.append(
#                 {
#                     "seed": seed, "model": name, "cv_auc": r["cv_auc"],
#                     "test_auc": r["test_auc"], "test_ap": r["test_ap"],
#                 }
#             )

#     repeats = pd.DataFrame(rows)
#     summary = (
#         repeats.groupby("model")[["cv_auc", "test_auc", "test_ap"]]
#         .agg(["mean", "std", "min", "max"])
#         .round(3)
#     )

#     # Encoded width with and without the merge, so the "width problem" panel
#     # can report both rather than only the inflated number.
#     width_unmerged = int(encode(X).shape[1])

#     primary.update(
#         {
#             "repeats": repeats,
#             "repeat_summary": summary,
#             "n_repeats": n_repeats,
#             "n_encoded_cols_unmerged": width_unmerged,
#             "base_rate": float(y.mean()),
#         }
#     )
#     return primary


# def agreement(df: pd.DataFrame) -> dict:
#     """How far the observed outcome and the owner's own expectation line up."""
#     sub = df.dropna(subset=["target_observed", "target_belief"])
#     obs = sub["target_observed"].astype(int)
#     bel = sub["target_belief"].astype(int)
#     table = pd.crosstab(obs, bel)
#     for i in (0, 1):
#         if i not in table.index:
#             table.loc[i] = 0
#         if i not in table.columns:
#             table[i] = 0
#     table = table.sort_index().sort_index(axis=1)

#     # correction=False: phi is an effect size, and Yates' correction is not
#     # part of its definition.
#     chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
#     n = int(table.values.sum())
#     phi = float(np.sqrt(chi2 / n)) if n else np.nan
#     if (obs.corr(bel) or 0) < 0:
#         phi = -phi

#     return {
#         "table": table,
#         "phi": phi,
#         "p": float(p),
#         "agreement_rate": float((obs == bel).mean()) if len(obs) else np.nan,
#         "failed_unanticipated": int(((obs == 1) & (bel == 0)).sum()),
#         "n_failures": int(obs.sum()),
#         "n": int(len(sub)),
#         "n_dropped": int(len(df) - len(sub)),
#     }


# # ------------------------------------------------------ deployment helpers


# def fit_deployment_models(X: pd.DataFrame, y: pd.Series, merge_rare: bool = True) -> dict:
#     """Fit deployment models on all resolved observations."""
#     mask = y.notna()
#     X = X.loc[mask].copy()
#     y = y.loc[mask].astype(int)

#     keep = fit_rare_merge(X) if merge_rare else {}
#     X_merged = apply_rare_merge(X, keep) if merge_rare else X
#     encoded = encode(X_merged)
#     cat_cols, bin_cols = split_column_kinds(X)

#     pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
#     cv = StratifiedKFold(n_splits=_n_splits(y), shuffle=True, random_state=SEED)

#     fitted, thresholds = {}, {}
#     for name, (estimator, grid) in _grids(pos_weight).items():
#         search = GridSearchCV(estimator, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
#         search.fit(encoded, y)
#         fitted[name] = search.best_estimator_
#         oof = cross_val_predict(
#             search.best_estimator_, encoded, y, cv=cv, method="predict_proba", n_jobs=-1
#         )[:, 1]
#         thresholds[name] = suggest_threshold(y.values, oof)

#     return {
#         "models": fitted,
#         "columns": encoded.columns.tolist(),
#         "cat_cols": cat_cols,
#         "bin_cols": bin_cols,
#         "rare_keep": keep,
#         "levels": {c: sorted(X_merged[c].astype(str).unique().tolist()) for c in cat_cols},
#         "thresholds": thresholds,
#         "n_fit": int(len(y)),
#         "base_rate": float(y.mean()),
#     }


# def predict_new_shop(bundle: dict, record: dict) -> dict:
#     """
#     Predict failure probability for one new shop record.

#     The two column kinds are handled separately. Casting the binary
#     multi-response indicators to string turned 1 into "1", get_dummies made
#     "Col: Option_1" out of it, reindex found no match and filled zero — so
#     every multi-response signal silently disappeared while the function still
#     returned a confident-looking number.
#     """
#     cat_cols = bundle["cat_cols"]
#     bin_cols = bundle["bin_cols"]

#     row: dict[str, object] = {}
#     for c in cat_cols:
#         v = str(record.get(c, MISSING_LABEL)).strip()
#         row[c] = MISSING_LABEL if v.lower() in {"", "nan", "none"} else v
#     for c in bin_cols:
#         row[c] = int(pd.to_numeric(record.get(c, 0), errors="coerce") or 0)

#     raw = pd.DataFrame([row])
#     raw = apply_rare_merge(raw, bundle.get("rare_keep", {}))
#     encoded = encode(raw, columns=bundle["columns"])

#     # Every categorical column must light up exactly one dummy. If none fires,
#     # the value was unseen at fit time and the model is being handed an
#     # all-zero block rather than an answer.
#     unmatched = []
#     for c in cat_cols:
#         hits = [col for col in bundle["columns"] if col.startswith(f"{c}_") and encoded[col].iloc[0] == 1]
#         if not hits:
#             unmatched.append(f"{c}={row[c]!r}")

#     probs = {
#         name: float(model.predict_proba(encoded)[:, 1][0])
#         for name, model in bundle["models"].items()
#     }
#     return {
#         "probabilities": probs,
#         "thresholds": bundle.get("thresholds", {}),
#         "unmatched": unmatched,
#     }
"""
Verify two figures the thesis reports against what the data actually holds.
Run from the same directory as app.py and KiranaPasal.xlsx:

    python check_numbers.py
"""

import kirana_analysis as ka

df = ka.build_targets(
    ka.load_raw("KiranaPasal.xlsx", ka.PREFERRED_SHEET),
    unresolved_as_failure=False,
)

# 1. Belief-target denominator.
#    Thesis: "answered for every record", 101 of 400 = 25.25%.
answered = int(df["target_belief"].notna().sum())
flagged = int(df["target_belief"].sum())
print(f"records                : {len(df)}")
print(f"belief answered        : {answered}  ({len(df) - answered} blank)")
print(f"belief flagged         : {flagged}")
print(f"  over answered        : {flagged / answered:.2%}")
print(f"  over all records     : {flagged / len(df):.2%}")
if answered != len(df):
    print("  >>> THESIS SAYS 'answered for every record' — THAT IS WRONG.")
    print(f"  >>> Correct figure is {flagged}/{answered} = {flagged / answered:.2%}")

print()

# 2. Agreement base.
#    Thesis: 69.8% agreement, phi = 0.041, 32 of 46 failures unanticipated.
ag = ka.agreement(df)
print(f"agreement base n       : {ag['n']}  ({ag['n_dropped']} dropped)")
print(f"agreement rate         : {ag['agreement_rate']:.1%}   (thesis: 69.8%)")
print(f"phi                    : {ag['phi']:.3f}       (thesis: 0.041)")
print(f"failures unanticipated : {ag['failed_unanticipated']} of {ag['n_failures']}"
      f"   (thesis: 32 of 46)")
if ag["n_dropped"]:
    print("  >>> State the base explicitly in the thesis: 'on the n shops with both")
    print("  >>> a resolved outcome and an answered expectation'.")

print()

# 3. Encoded width under the reported specification.
#    Thesis: 266 encoded columns against 270 training rows.
X, _, _ = ka.build_predictors(df)
out = ka.run_models(X, df["target_observed"], n_repeats=1, merge_rare=False)
print(f"predictor variables    : {X.shape[1]}      (thesis: 88)")
print(f"encoded columns        : {out['n_encoded_cols']}     (thesis: 266)")
print(f"training rows          : {out['n_train']}     (thesis: 270)")
print(f"test failures          : {out['test_events']} of {out['n_test']}"
      f"   (thesis: 14 of 117)")
for name, r in out["results"].items():
    print(f"  {name:<20} test AUC {r['test_auc']:.3f}")