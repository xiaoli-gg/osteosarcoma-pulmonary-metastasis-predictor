from __future__ import annotations

import numpy as np
import pandas as pd


class StackingFusionPredictionPipeline:
    """End-to-end fixed-feature stacking fusion prediction pipeline."""

    def __init__(
        self,
        feature_names,
        clinical_features,
        radiomics_features,
        deep_learning_features,
        feature_name_mapping,
        clinical_preprocessor,
        radiomics_preprocessor,
        deep_learning_preprocessor,
        clinical_model,
        radiomics_model,
        deep_learning_model,
        stacking_meta_learner,
        cutoff=0.357,
    ):
        self.feature_names = list(feature_names)
        self.clinical_features = list(clinical_features)
        self.radiomics_features = list(radiomics_features)
        self.deep_learning_features = list(deep_learning_features)
        self.feature_name_mapping = dict(feature_name_mapping)
        self.clinical_preprocessor = clinical_preprocessor
        self.radiomics_preprocessor = radiomics_preprocessor
        self.deep_learning_preprocessor = deep_learning_preprocessor
        self.clinical_model = clinical_model
        self.radiomics_model = radiomics_model
        self.deep_learning_model = deep_learning_model
        self.stacking_meta_learner = stacking_meta_learner
        self.cutoff = float(cutoff)

    def _normalize_columns(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        X = X.copy()
        rename = {old: new for old, new in self.feature_name_mapping.items() if old in X.columns and new not in X.columns}
        if rename:
            X = X.rename(columns=rename)
        return X

    def validate_input(self, X):
        X = self._normalize_columns(X)
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError("Missing required feature columns: " + ", ".join(missing))
        X = X[self.feature_names].copy()
        for c in self.feature_names:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[self.feature_names].isna().any().any():
            bad = [c for c in self.feature_names if X[c].isna().any()]
            raise ValueError("Non-numeric or missing values found in columns: " + ", ".join(bad))
        return X

    def single_modality_probabilities(self, X):
        X = self.validate_input(X)
        Xc = self.clinical_preprocessor.transform(X[self.clinical_features])
        Xr = self.radiomics_preprocessor.transform(X[self.radiomics_features])
        Xd = self.deep_learning_preprocessor.transform(X[self.deep_learning_features])
        return pd.DataFrame(
            {
                "P_clinical": self.clinical_model.predict_proba(Xc)[:, 1],
                "P_radiomics": self.radiomics_model.predict_proba(Xr)[:, 1],
                "P_deep": self.deep_learning_model.predict_proba(Xd)[:, 1],
            },
            index=X.index,
        )

    def predict_proba(self, X):
        probs = self.single_modality_probabilities(X)
        p = self.stacking_meta_learner.predict_proba(probs[["P_clinical", "P_radiomics", "P_deep"]])[:, 1]
        return np.asarray(p, dtype=float)

    def predict(self, X):
        return (self.predict_proba(X) >= self.cutoff).astype(int)
