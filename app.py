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
STYLE_PATH = APP_DIR / "styles.css"
CUTOFF = 0.357

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

FEATURE_GROUPS = {
    "Clinical features": FEATURES[:5],
    "Radiomics features": FEATURES[5:11],
    "Deep learning features": FEATURES[11:],
}


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


def classify(prob):
    return "High risk" if prob >= CUTOFF else "Low risk"


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


def load_css():
    if STYLE_PATH.exists():
        st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def result_card(prob: float) -> None:
    risk = classify(prob)
    is_high = risk == "High risk"
    status_class = "risk-high" if is_high else "risk-low"
    st.markdown(
        f"""
        <div class="result-card {status_class}">
            <div class="result-label">Prediction Result</div>
            <div class="result-main">{risk}</div>
            <div class="result-grid">
                <div>
                    <span class="metric-label">Risk Probability</span>
                    <strong>{prob * 100:.1f}%</strong>
                </div>
                <div>
                    <span class="metric-label">Threshold</span>
                    <strong>{CUTOFF:.3f}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_inputs() -> tuple[pd.DataFrame, bool]:
    st.sidebar.markdown('<div class="sidebar-title">Patient Data Input</div>', unsafe_allow_html=True)
    values = {}
    for group_name, feats in FEATURE_GROUPS.items():
        st.sidebar.markdown(f'<div class="sidebar-group">{group_name}</div>', unsafe_allow_html=True)
        for feature in feats:
            values[feature] = st.sidebar.number_input(feature, value=0.0, format="%.6f", key=f"manual_{feature}")
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    submitted = st.sidebar.button("Predict", type="primary", use_container_width=True)
    return pd.DataFrame([values], columns=FEATURES), submitted


def model_information() -> None:
    with st.expander("Model Information", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Clinical model**")
            st.write("XGBoost")
            st.caption("Inputs: Age, ALP, Monocyte, Neutrophil, MLR")
        with c2:
            st.markdown("**Radiomics model**")
            st.write("Logistic Regression")
            st.caption("Inputs: six locked CT radiomics features")
        with c3:
            st.markdown("**Deep learning model**")
            st.write("Logistic Regression")
            st.caption("Inputs: Feature_508, Feature_738, Feature_879")
        st.info(
            "The final stacking fusion model combines P_clinical, P_radiomics, and P_deep. "
            "This interface is for research use only and is not a standalone clinical decision-making system."
        )


def contribution_chart(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    probs = pipeline.single_modality_probabilities(X).iloc[0]
    plot_df = pd.DataFrame(
        {
            "Modality": ["Clinical", "Radiomics", "Deep learning"],
            "Probability": [probs["P_clinical"], probs["P_radiomics"], probs["P_deep"]],
        }
    )
    return plot_df


def render_prediction_view(pipeline, X: pd.DataFrame) -> None:
    prob = float(pipeline.predict_proba(X)[0])
    risk = classify(prob)
    st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
    result_card(prob)

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<div class="panel-title">Model Contributions</div>', unsafe_allow_html=True)
        contrib = contribution_chart(pipeline, X)
        st.bar_chart(contrib.set_index("Modality"), height=260)
        st.caption("Bars show the three single-modality predicted probabilities used by the stacking fusion model.")

    with right:
        st.markdown('<div class="panel-title">Detailed Data</div>', unsafe_allow_html=True)
        details = X.T.reset_index()
        details.columns = ["Feature", "Value"]
        st.dataframe(details, use_container_width=True, height=300)
        summary = pd.DataFrame(
            [
                {"Item": "Final probability", "Value": f"{prob:.3f}"},
                {"Item": "Threshold", "Value": f"{CUTOFF:.3f}"},
                {"Item": "Risk group", "Value": risk},
            ]
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)


def render_batch_view(pipeline) -> None:
    st.markdown('<div class="section-title">CSV Batch Prediction</div>', unsafe_allow_html=True)
    st.write("Upload a CSV file containing the 14 required feature columns. Patient identifiers and outcome columns are not required.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        return
    try:
        df = pd.read_csv(uploaded)
        X = validate_uploaded(df)
        probs = pipeline.predict_proba(X)
        out = X.copy()
        out["predicted_probability"] = probs
        out["cutoff"] = CUTOFF
        out["risk_classification"] = ["high risk" if p >= CUTOFF else "low risk" for p in probs]
        st.dataframe(out[["predicted_probability", "cutoff", "risk_classification"]], use_container_width=True)
        buf = io.StringIO()
        out.to_csv(buf, index=False)
        st.download_button(
            "Download predictions CSV",
            data=buf.getvalue(),
            file_name="stacking_fusion_predictions.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(str(exc))


def main():
    st.set_page_config(page_title="Osteosarcoma Pulmonary Metastasis Risk", layout="wide")
    load_css()

    pipeline = load_pipeline()
    manual_X, submitted = sidebar_inputs()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Research use only</div>
            <h1>CT-based Osteosarcoma Pulmonary Metastasis Predictor</h1>
            <p>Stacking fusion pulmonary metastasis risk prediction</p>
            <div class="cutoff-pill">Fixed cutoff: 0.357</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning("Research use only. This tool should not be used as a standalone clinical decision-making system.")
    model_information()

    tab_predict, tab_batch = st.tabs(["Individual prediction", "CSV batch upload"])
    with tab_predict:
        if submitted:
            render_prediction_view(pipeline, manual_X)
        else:
            st.markdown(
                """
                <div class="empty-state">
                    <strong>Prediction Results</strong>
                    <p>Enter the 14 locked features in the sidebar and click Predict.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with tab_batch:
        render_batch_view(pipeline)


if __name__ == "__main__":
    main()
