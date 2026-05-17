# Opportunistic Rotator Cuff Tear Screening from Chest CT

This repository contains reproducible analysis code for a radiomics and
deep-learning workflow for opportunistic rotator cuff tear screening from
routine chest CT.

The public code is organized to avoid common sources of optimistic bias:

- imputation, scaling, and LASSO feature selection are fitted inside each
  cross-validation training fold only;
- out-of-fold predictions preserve original patient IDs;
- the final deployable artifact stores the training imputer, scaler, selected
  features, and classifier together;
- external validation applies the saved training artifact and never refits
  preprocessing on external data;
- validation-set performance is not used to select the best deep-learning epoch
  in the MIL cross-validation script.

## Repository Layout

```text
config.py                    # ROI definitions and global constants
data_loader.py               # table loading and feature inference
feature_selection.py         # in-fold LASSO selection helpers
models.py                    # prespecified classifier definitions
train_radiomics.py           # radiomics CV and final artifact export
external_validate.py         # external validation from saved artifact
combat_harmonization.py      # optional training-set-only harmonization utility
evaluate/                    # ROC/PR, calibration, DCA, DeLong utilities
mil/                         # side-separated attention MIL model and CV
requirements.txt
```

## Installation

Python 3.10 or newer is required because the code uses modern type-hint syntax.

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Input Data

Radiomics tables should contain one row per patient. The loader recognizes
`ID` and `cat` by default; common alternatives such as `PatientID`, `Cat`, and
`Label` are also accepted.

```text
ID,cat,Supraspinatus_Vol,...,Vertebrae_C7_Contrast
A001,0,12.3,...,45.6
A002,1,10.8,...,42.1
```

MIL uses a patch metadata CSV:

```text
ID,Cat,patch_path,side,struct,cx,cy,cz
A001,0,/absolute/path/patch001.npy,left,scapula,100,200,50
A001,0,/absolute/path/patch002.npy,right,scapula,300,200,50
```

Patient identifiers should be pseudonymized before sharing.

## Radiomics Cross-Validation

```bash
python train_radiomics.py --data data/radiomics_train.xlsx --output results/radiomics_cv --n_splits 10 --nested_outer_splits 5 --nested_inner_splits 5 --k_list 5 8 10 12 18 --final_model CatBoost --final_k 10
```

Outputs:

- `cv_fold_metrics_descriptive.csv`: descriptive fold-level grid metrics;
- `cv_oof_predictions_descriptive_long.csv`: descriptive grid OOF predictions;
- `cv_summary_descriptive.csv`: descriptive grid summary. Do not use this alone
  as the selected-model performance claim;
- `nested_outer_fold_metrics.csv`: outer-fold metrics after inner-fold
  model/K selection;
- `nested_oof_predictions.csv`: unbiased internal OOF predictions from the
  nested protocol;
- `nested_inner_selections.csv`: model/K selected inside each outer fold;
- `radiomics_final_artifact.joblib`: final training artifact for external validation;
- `radiomics_final_artifact_summary.json`: human-readable artifact summary.

When reporting internal performance after model selection, use the nested CV
outputs. Use the descriptive grid only to show the model-search landscape.

## External Validation

```bash
python external_validate.py --artifact results/radiomics_cv/radiomics_final_artifact.joblib --data data/radiomics_external.xlsx --output results/external_center_1 --threshold 0.5
```

The external script checks that all training features are present and never
fits imputation, scaling, or feature selection on the external cohort.
Artifacts should be used with the code version that created them; regenerate
the artifact after changing preprocessing code.

## MIL Cross-Validation

```bash
python -m mil --meta data/patch_metadata.csv --out results/mil --n_folds 5 --epochs 50 --max_inst 24 --eval_every 0
```

The MIL script uses deterministic validation patch sampling and evaluates the
fixed final epoch for each fold. If early stopping or architecture selection is
introduced, use a nested validation split and report that procedure explicitly.
The script saves `training_log_ss_attnmil.csv`; rows with `Evaluated=True`
contain validation AUC values. Use this file to document loss trajectories and
justify the fixed epoch count. The MIL head assumes the
patient label is positive when at least one side is positive. Side-specific
labels require a side-specific objective instead of max pooling.

## Harmonization

`combat_harmonization.py` provides a conservative training-set-only
location/scale harmonization utility for sensitivity analyses. It supports
covariates such as age or sex so their effects are preserved rather than
absorbed into batch correction. Outcome labels (`cat`, `Label`) can be supplied
only for labelled explanatory analyses by setting
`allow_outcome_covariates=True`; they must not be used to transform validation
or external data for predictive performance estimation.

The code has been smoke-tested with Python 3.12 and PyTorch 2.5 in the local
development environment. `torch` is constrained to the 2.x series in
`requirements.txt`.

## Reproducibility Checklist

- Keep raw data and generated model weights out of Git unless explicitly
  de-identified and permitted.
- Use the same fixed seed reported in the manuscript.
- Do not choose bootstrap seeds based on favorable confidence intervals.
- Use a prespecified threshold or report threshold selection transparently.
- Report external validation using artifacts fitted only on the training cohort.
- Do not interpret descriptive grid CV as final performance after selecting the
  best model from that same grid; use nested CV or external validation.
