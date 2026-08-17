"""
Analysis core for the kirana pasal failure-risk dashboard.

Everything the dashboard shows is computed here, from the survey workbook,
so no figure in the app is typed in by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:  # the app degrades to two models rather than failing
    HAS_XGB = False

SHEET = "Data 400"

ID_COLS = ["Survey ID", "Shop Name"]

# Fields that only exist once the outcome is known. Dropped before modelling
# so neither target can leak back into the predictors.
OUTCOME_COLS = ["Owner\u2019s Expectation", "Consideration of Closure", "Outcome Status"]

# Outcome statuses coded as failure.
FAILURE_STATUSES = [
    "Permanently closed",
    "Temporarily closed but expected to reopen",
    "Changed into another type of business",
]
UNRESOLVED_STATUS = "Could not be verified"

# Owner expectation levels coded as high risk on the belief target.
BELIEF_HIGH_RISK = ["Unlikely", "Very unlikely"]

MIN_CATEGORY_N = 10  # categories smaller than this are merged before testing
SEED = 42


# ---------------------------------------------------------------- data


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    status = out["Outcome Status"].astype(str).str.strip()

    observed = pd.Series(np.nan, index=out.index, dtype="float")
    observed[status.isin(FAILURE_STATUSES)] = 1.0
    observed[~status.isin(FAILURE_STATUSES) & (status != UNRESOLVED_STATUS)] = 0.0
    out["target_observed"] = observed  # NaN where unresolved

    expect = out["Owner\u2019s Expectation"].astype(str).str.strip()
    out["target_belief"] = expect.isin(BELIEF_HIGH_RISK).astype(float)
    return out


def multi_response_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Columns storing several answers joined with semicolons."""
    return [c for c in candidates if df[c].astype(str).str.contains(";", na=False).any()]


def build_predictors(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Returns the predictor frame plus the names of the single-response and
    multi-response-derived variables. Multi-response items become one binary
    indicator per option, which keeps the encoded matrix far narrower than
    treating every observed combination as its own category.
    """
    drop = set(ID_COLS + OUTCOME_COLS + ["target_observed", "target_belief"])
    candidates = [c for c in df.columns if c not in drop]

    multi = multi_response_columns(df, candidates)
    single = [c for c in candidates if c not in multi]

    X = df[single].copy()
    for c in X.columns:
        X[c] = X[c].astype(str).str.strip().replace({"nan": "Not reported", "": "Not reported"})

    derived: list[str] = []
    for col in multi:
        exploded = (
            df[col]
            .fillna("")
            .astype(str)
            .str.split(";")
            .apply(
                lambda parts: [
                    p.strip()
                    for p in (parts if isinstance(parts, list) else [])
                    if p.strip() and p.strip().lower() != "nan"
                ]
            )
        )
        options = sorted({opt for row in exploded for opt in row})
        for opt in options:
            name = f"{col}: {opt}"
            X[name] = exploded.apply(lambda row, o=opt: int(o in row))
            derived.append(name)

    return X, single, derived


# ------------------------------------------------- chi-square screen


def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    denom = n * (min(r, c) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan


def _merge_rare(series: pd.Series) -> pd.Series:
    counts = series.value_counts()
    rare = counts[counts < MIN_CATEGORY_N].index
    if len(rare) == 0:
        return series
    return series.where(~series.isin(rare), "Other (merged)")


def screen_predictors(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Chi-square test of independence per predictor, with Cramer's V and BH q."""
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask].astype(int)

    rows = []
    for col in X.columns:
        s = _merge_rare(X[col].astype(str))
        table = pd.crosstab(s, y)
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue

        chi2, p, dof, expected = stats.chi2_contingency(table)
        small_expected = float((expected < 5).mean())
        test = "Chi-square"

        # Fisher's exact for 2x2 tables where the chi-square approximation fails
        if table.shape == (2, 2) and small_expected > 0.20:
            _, p = stats.fisher_exact(table.values)
            test = "Fisher exact"

        rows.append(
            {
                "variable": col,
                "chi2": chi2,
                "dof": int(dof),
                "p": p,
                "cramers_v": cramers_v(chi2, int(table.values.sum()), *table.shape),
                "test": test,
                "pct_expected_below_5": small_expected,
                "n_categories": int(table.shape[0]),
                "low_cell_warning": small_expected > 0.20,
            }
        )

    res = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    res["q"] = benjamini_hochberg(res["p"].values)
    res["significant_raw"] = res["p"] < 0.05
    res["survives_bh"] = res["q"] < 0.05
    return res


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    return q


# ----------------------------------------------------------- models


def _grids(pos_weight: float) -> dict:
    grids = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
            {"C": [0.01, 0.1, 1.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
            {"n_estimators": [300], "max_depth": [3, 5, None], "min_samples_leaf": [1, 5]},
        ),
    }
    if HAS_XGB:
        grids["XGBoost"] = (
            XGBClassifier(
                eval_metric="logloss",
                random_state=SEED,
                scale_pos_weight=pos_weight,
                n_jobs=-1,
            ),
            {"n_estimators": [200], "max_depth": [2, 3], "learning_rate": [0.05, 0.1]},
        )
    return grids


def metrics_at(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_models(X: pd.DataFrame, y: pd.Series, test_size: float = 0.30) -> dict:
    """
    70:30 stratified split. Encoding and tuning happen inside the training
    folds only; the test set is scored once.
    """
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask].astype(int)

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )

    X_tr = pd.get_dummies(X_tr_raw, drop_first=False)
    X_te = pd.get_dummies(X_te_raw, drop_first=False).reindex(columns=X_tr.columns, fill_value=0)

    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results = {}
    for name, (est, grid) in _grids(pos_weight).items():
        search = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_tr, y_tr)
        proba = search.best_estimator_.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        results[name] = {
            "cv_auc": float(search.best_score_),
            "test_auc": float(roc_auc_score(y_te, proba)),
            "best_params": search.best_params_,
            "proba": proba,
            "roc": (fpr, tpr),
            "at_050": metrics_at(y_te, proba, 0.50),
        }

    return {
        "results": results,
        "y_test": y_te.values,
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "n_encoded_cols": X_tr.shape[1],
        "train_events": int(y_tr.sum()),
        "test_events": int(y_te.sum()),
        "majority_accuracy": float((y_te == 0).mean()),
    }


def agreement(df: pd.DataFrame) -> dict:
    """How far the observed outcome and the owner's own expectation line up."""
    sub = df.dropna(subset=["target_observed"])
    obs = sub["target_observed"].astype(int)
    bel = sub["target_belief"].astype(int)
    table = pd.crosstab(obs, bel)
    for i in (0, 1):
        if i not in table.index:
            table.loc[i] = 0
        if i not in table.columns:
            table[i] = 0
    table = table.sort_index().sort_index(axis=1)

    chi2, p, _, _ = stats.chi2_contingency(table)
    phi = float(np.sqrt(chi2 / table.values.sum()))
    if (obs.corr(bel) or 0) < 0:
        phi = -phi

    return {
        "table": table,
        "phi": phi,
        "p": float(p),
        "agreement_rate": float((obs == bel).mean()),
        "failed_unanticipated": int(((obs == 1) & (bel == 0)).sum()),
        "n_failures": int(obs.sum()),
        "n": int(len(sub)),
    }

# ----------------------------------------------------------- deployment helpers

def fit_deployment_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """Fit deployment models on all resolved observations."""
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].astype(int)
    encoded = pd.get_dummies(X, drop_first=False)
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))

    fitted = {}
    for name, (estimator, grid) in _grids(pos_weight).items():
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        search = GridSearchCV(estimator, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(encoded, y)
        fitted[name] = search.best_estimator_

    return {
        "models": fitted,
        "columns": encoded.columns.tolist(),
        "feature_names": X.columns.tolist(),
    }


def predict_new_shop(bundle: dict, record: dict) -> dict:
    """Predict failure probability for one new shop record."""
    row = {c: record.get(c, "Not reported") for c in bundle["feature_names"]}
    raw = pd.DataFrame([row])
    for c in raw.columns:
        raw[c] = raw[c].astype(str).str.strip().replace({"nan": "Not reported", "": "Not reported"})
    encoded = pd.get_dummies(raw, drop_first=False).reindex(
        columns=bundle["columns"], fill_value=0
    )
    return {
        name: float(model.predict_proba(encoded)[:, 1][0])
        for name, model in bundle["models"].items()
    }
