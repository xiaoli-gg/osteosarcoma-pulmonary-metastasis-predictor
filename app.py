from __future__ import annotations

import io
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
sys.path.insert(0, str(MODEL_DIR))
import stacking_pipeline  # noqa: F401

MODEL_PATH = MODEL_DIR / "stacking_fusion_prediction_pipeline.joblib"

FEATURES = [
    "Age",
    "ALP",
    "Monocyte",
    "Neutrophil",
    "MLR",
    "wavelet-LLH_glszm_GrayLevelNonUniformity",
    "wavelet-LHL_glszm_SizeZoneNonUniformityNormalized",
    "wavelet-HHH_glcm_ClusterShade",
    "wavelet-HHH_glszm_GrayLevelNonUniformityNormalized",
    "wavelet-HHH_glszm_SizeZoneNonUniformityNormalized",
    "wavelet-HHH_glszm_ZoneVariance",
    "Feature_508",
    "Feature_738",
    "Feature_879",
]


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


def classify(prob):
    return "High risk" if prob >= 0.357 else "Low risk"


def validate_uploaded(df):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    X = df[FEATURES].copy()
    bad = []
    for c in FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isna().any():
            bad.append(c)
    if bad:
        raise ValueError("Missing or non-numeric values detected in: " + ", ".join(bad))
    return X


def main():
    st.set_page_config(page_title="Osteosarcoma Pulmonary Metastasis Risk", layout="wide")
    st.title("Stacking Fusion Pulmonary Metastasis Risk Predictor")
    st.warning("Research use only. This tool should not be used as a standalone clinical decision-making system.")

    pipeline = load_pipeline()
    st.caption("Fixed cutoff: 0.357. Predicted probability >= 0.357 is classified as high risk.")

    tab_manual, tab_batch = st.tabs(["Manual input", "CSV batch upload"])

    with tab_manual:
        st.subheader("Manual input")
        values = {}
        groups = {
            "Clinical features": FEATURES[:5],
            "Radiomics features": FEATURES[5:11],
            "Deep learning features": FEATURES[11:],
        }
        for group_name, feats in groups.items():
            st.markdown(f"**{group_name}**")
            cols = st.columns(3)
            for i, feature in enumerate(feats):
                with cols[i % 3]:
                    values[feature] = st.number_input(feature, value=0.0, format="%.6f")
        if st.button("Predict", type="primary"):
            X = pd.DataFrame([values], columns=FEATURES)
            prob = float(pipeline.predict_proba(X)[0])
            risk = classify(prob)
            st.metric("Predicted probability", f"{prob:.3f}")
            st.metric("Risk classification", risk)

    with tab_batch:
        st.subheader("CSV batch upload")
        st.write("Upload a CSV file containing the 14 required feature columns. No patient identifiers or outcome columns are required.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                X = validate_uploaded(df)
                probs = pipeline.predict_proba(X)
                out = X.copy()
                out["predicted_probability"] = probs
                out["cutoff"] = 0.357
                out["risk_classification"] = ["high risk" if p >= 0.357 else "low risk" for p in probs]
                st.dataframe(out[["predicted_probability", "cutoff", "risk_classification"]], use_container_width=True)
                buf = io.StringIO()
                out.to_csv(buf, index=False)
                st.download_button("Download predictions CSV", data=buf.getvalue(), file_name="stacking_fusion_predictions.csv", mime="text/csv")
            except Exception as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
