# Model Card

## Model Name

Stacking fusion prediction model for pulmonary metastasis risk prediction in osteosarcoma.

## Intended Use

Research use.

## Not Intended Use

This model should not be used as a standalone clinical decision-making system.

## Input

Fourteen locked numeric features: five clinical features, six radiomics features, and three deep learning features.

## Output

Predicted probability and high/low risk classification.

Cutoff: `0.357`.

## Training and Evaluation Data

- Training data: development cohort.
- Evaluation data: temporal validation cohort and external validation cohort.

## Deployment Model Performance

| Cohort | n | events | AUROC | AUPRC | Brier score | Accuracy | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| development | 100 | 40 | 0.974 | 0.961 | 0.087 | 0.900 | 0.925 | 0.883 |
| temporal_validation | 15 | 6 | 0.833 | 0.821 | 0.203 | 0.667 | 1.000 | 0.444 |
| external_validation | 28 | 11 | 0.813 | 0.810 | 0.190 | 0.750 | 0.727 | 0.765 |

## Limitations

- Retrospective study.
- Small validation cohorts.
- Deep learning features are exploratory.
- Pulmonary lesions were not routinely pathologically confirmed.
- Further multicenter prospective validation is needed.
