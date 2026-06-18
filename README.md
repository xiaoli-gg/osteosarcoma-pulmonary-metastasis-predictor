# Stacking Fusion Pulmonary Metastasis Risk Predictor

This repository contains a Streamlit research-use app for pulmonary metastasis risk prediction in osteosarcoma using a fixed 14-feature stacking fusion model.

Research use only. This tool should not be used as a standalone clinical decision-making system.

## Folder Structure

```text
streamlit_deploy_stacking_fusion_model_v1/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── model/
│   ├── stacking_fusion_prediction_pipeline.joblib
│   ├── stacking_pipeline.py
│   ├── feature_schema.json
│   ├── cutoff.json
│   └── model_card.md
├── examples/
│   ├── example_single_input.csv
│   └── example_batch_input.csv
└── internal_check/
    ├── training_report.md
    ├── deployment_smoke_test_report.md
    └── upload_safety_check.md
```

## Required Input Features

The app requires exactly these 14 numeric features:

- `Age`
- `ALP`
- `Monocyte`
- `Neutrophil`
- `MLR`
- `wavelet-LLH_glszm_GrayLevelNonUniformity`
- `wavelet-LHL_glszm_SizeZoneNonUniformityNormalized`
- `wavelet-HHH_glcm_ClusterShade`
- `wavelet-HHH_glszm_GrayLevelNonUniformityNormalized`
- `wavelet-HHH_glszm_SizeZoneNonUniformityNormalized`
- `wavelet-HHH_glszm_ZoneVariance`
- `Feature_508`
- `Feature_738`
- `Feature_879`

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Create a GitHub repository.
2. Upload this folder's files to the repository root.
3. In Streamlit Cloud, create a new app from the repository.
4. Set the main file path to `app.py`.

## CSV Input Format

CSV files must contain the 14 required feature columns. Do not include patient identifiers or outcome columns. See `examples/example_single_input.csv` and `examples/example_batch_input.csv`.

## Output Interpretation

The model returns a predicted probability of pulmonary metastasis and a risk classification using cutoff `0.357`:

- predicted probability >= 0.357: high risk
- predicted probability < 0.357: low risk

## Limitations

- Retrospective study.
- Small validation cohorts.
- Deep learning features are exploratory.
- Pulmonary lesions were not routinely pathologically confirmed.
- Further multicenter prospective validation is needed.

## Citation

Citation placeholder.

## Contact

Contact placeholder.

## License

License placeholder.
