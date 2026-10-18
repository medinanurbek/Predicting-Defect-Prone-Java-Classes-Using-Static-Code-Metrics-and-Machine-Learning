# Predicting Defect Prone Java Classes Using Static Code Metrics and Machine Learning

Research Methods and Tools project at Astana IT University, group SE-2409.

**Authors:** Kabysheva Zukhra, Nurbek Medina, Maksatova Akniyet, Nauryzbay Tomiris.

The project investigates how well static code metrics predict Java classes with recorded defects and which metrics contribute to prediction. Assignment 3 includes a completed computational pilot, a methodology report, and a plan for further project-level evaluation.

## Repository contents

| Path | Contents |
| --- | --- |
| `Assignment3_Practical/run_pipeline.py` | Data preparation, model fitting, scoring, importance analysis, and checks |
| `Assignment3_Practical/fetch_data.py` | Retrieve the five source CSVs from the versioned authors' archive |
| `Assignment3_Practical/data/` | Original data, provenance, SHA256 manifest, and upstream metadata |
| `Assignment3_Practical/outputs/` | Recorded experiment results, predictions, configurations, and verification |
| `Assignment3_Practical/START_DEMO.cmd` | Windows demonstration launcher |
| `Assignment3_Practical/DEMO_RU.txt` | Step-by-step demonstration guide in Russian |
| `Assignment3_Practical/README_RU.txt` | Research explanation and defence notes in Russian |
| `output/Assignment3_Kabysheva_Nurbek_Maksatova_Nauryzbay.docx` | Assignment 3 report |
| `output/Assignment3_Practical.zip` | Portable practical submission package |
| `docs/` | Earlier literature review and research-question assignments |

## Quick start on Windows

Install Python 3.12, download or clone this repository, and double-click:

```text
Assignment3_Practical\START_DEMO.cmd
```

The first launch installs the pinned dependencies and requires internet access. Subsequent runs use the local environment. Results from this launcher are written to `Assignment3_Practical/outputs_demo/`; the recorded `outputs/` files remain unchanged.

For a manual launch, open PowerShell in `Assignment3_Practical`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_pipeline.py --output outputs_demo
```

The source data are included. Optionally retrieve them again with:

```powershell
.\.venv\Scripts\python.exe fetch_data.py
```

Acquisition uses HTTP byte ranges to retrieve selected members of the upstream ZIP. It does not require downloading the complete archive. The pipeline verifies the source CSV SHA256 hashes before training.

## Completed pilot

The five original PROMISE tables contain **2,996 Java class records** and **20 static predictors**. The label is `bug > 0`; a zero label means no recorded defect, not proof that a class is fault-free.

| Partition | Projects | Retained records | Records with defects |
| --- | --- | ---: | ---: |
| Training | Ant 1.7, JEdit 4.3, POI 3.0 | 1,679 | 458 |
| Validation | Ivy 2.0 | 341 | 39 |
| Test | Camel 1.6 | 934 | 187 |

Exact predictor vectors shared across partitions are conservatively excluded from later partitions: 11 validation and 31 test records. Identical metrics do not necessarily imply identical source code, so this rule also changes the evaluation population.

The pilot compares Logistic Regression, Random Forest, and XGBoost using either all 20 predictors or six predefined CK predictors. A constant-probability dummy classifier provides a baseline. Imputation and LR standardisation use training data only. Class weights address imbalance. The pilot uses seed 42 and a fixed classification threshold of 0.5.

XGBoost with all 20 predictors was selected among the full-feature models by validation average precision, before test evaluation.

| Test configuration | Precision | Recall | F1 | ROC-AUC | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy | 0.000 | 0.000 | 0.000 | 0.500 | 0.200 |
| Logistic Regression all20 | 0.253 | 0.380 | 0.303 | 0.545 | 0.229 |
| Random Forest all20 | 0.346 | 0.096 | 0.151 | 0.578 | 0.264 |
| XGBoost all20 | 0.298 | 0.305 | 0.302 | 0.592 | 0.258 |
| Logistic Regression CK6 | 0.288 | 0.364 | 0.322 | 0.581 | 0.295 |
| Random Forest CK6 | 0.264 | 0.198 | 0.226 | 0.559 | 0.231 |
| XGBoost CK6 | 0.271 | 0.380 | 0.316 | 0.577 | 0.242 |

Selected XGBoost detected **57 of 187** positive test records and missed **130**. The observed transfer is limited. The dummy obtains about 80% accuracy while missing every positive record, illustrating why accuracy alone is inadequate.

Logistic Regression CK6 has the highest observed test AP, but the test outcomes were not used to change the validation-based selection. These are exploratory pilot results, not evidence of a universally best classifier.

## Evidence and reproducibility

- `outputs/metrics.csv`: all validation and test scores.
- `outputs/test_predictions.csv`: individual test probabilities, decisions, and labels.
- `outputs/selected_model_errors.csv`: false positives and false negatives for the selected model.
- `outputs/validation_permutation_importance.csv`: 20-repeat importance analysis on Ivy.
- `outputs/checks.txt`: seven input, partition, preprocessing, and scoring checks.
- `outputs/configurations.json` and `outputs/environment.json`: recorded settings and versions.
- `outputs/audit.json`: hashes, exclusions, and partition counts.
- `outputs/reproduction_check.json`: byte-identical metrics, predictions, importance, and split records from an earlier same-environment repeat.

The full nested leave-one-project-out comparison, repetition across seeds, overlap sensitivity analysis, and confirmation on an additional untouched project remain planned. The inspected Camel pilot must not be described as fresh confirmation after further model development.

## Data attribution

Rudolf Ferenc, Zoltán Tóth, Gergely Ladányi, István Siket, and Tibor Gyimóthy. **Unified Bug Dataset**, version 1.2 (2020). [Versioned Zenodo record](https://doi.org/10.5281/zenodo.3693686). The record specifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

This repository includes five extracted original PROMISE tables under the archive's `PROMISE/orig/defect/ck/` directory, along with derived technical records and outputs. Processing derives binary labels and hashed class identifiers and excludes specified overlaps. Java source code is not redistributed. Preserve the upstream attribution and notices when sharing the data. The dataset notice does not grant a separate licence to this repository's implementation code.

Background: Ferenc et al. (2020), [A public unified bug dataset for java and its assessment regarding metrics and bug prediction](https://doi.org/10.1007/s11219-020-09515-0). Metric definitions follow Jureczko and Madeyski, [Software product metrics used to build defect prediction models](https://madeyski.e-informatyka.pl/download/JureczkoMadeyskiSoftwareProductMetrics.pdf).
