"""B3 — Nested LASSO selection + LDA, after Vashkevich & Rushkevich (2021)."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_B3(X_train: np.ndarray, y_train: np.ndarray,
             *, cfg: dict[str, Any], seed: int):
    from sparc_als.utils.seeds import set_all_seeds
    set_all_seeds(seed)
    lasso = LassoCV(cv=cfg["selection"]["inner_cv_folds"],
                    n_alphas=cfg["selection"]["alphas"],
                    random_state=seed)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    lasso.fit(X_s, y_train)
    selected = np.flatnonzero(np.abs(lasso.coef_) > 1e-8)
    if len(selected) == 0:
        selected = np.array([0])
    lda = Pipeline([("scale", StandardScaler()),
                    ("lda", LinearDiscriminantAnalysis(solver=cfg["classifier"]["solver"]))])
    lda.fit(X_train[:, selected], y_train)
    return lda, selected
