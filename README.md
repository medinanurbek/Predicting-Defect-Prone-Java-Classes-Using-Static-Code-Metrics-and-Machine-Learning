# Predicting Defect Prone Java Classes Using Static Code Metrics and Machine Learning

Research Methods and Tools project at Astana IT University, group SE-2409.

**Authors:** Kabysheva Zukhra, Nurbek Medina, Maksatova Akniyet, Nauryzbay Tomiris.

The project investigates how well static code metrics predict Java classes with recorded defects and which metrics contribute to prediction. Assignment 4 contains the completed project-level benchmark, additional Synapse evaluation, research report and reproducibility package. The Assignment 3 pilot is retained as historical evidence.

## Run the complete Assignment 4 project from GitHub

On Windows, install **Git** and **Python 3.12**. Open PowerShell in the folder where you want to download the project, then run:

```powershell
$base = 'https://github.com/medinanurbek/'
$project = 'Predicting-Defect-Prone-Java-Classes-'
$project += 'Using-Static-Code-Metrics-and-Machine-Learning'
git clone ($base + $project + '.git') RMT_Assignment4
Set-Location .\RMT_Assignment4\Assignment3_Practical
.\START_ASSIGNMENT4.cmd
```

The folder keeps its original name `Assignment3_Practical` to preserve continuity with the earlier pilot. It contains the complete Assignment 4 implementation. The first launch creates the Python environment and installs pinned libraries; it requires internet access. The source datasets are included.

The launcher performs four steps: (1) the full five-project, three-seed nested benchmark; (2) Synapse evaluation with the original frozen configuration; (3) descriptive, threshold and sensitivity summaries; and (4) verification of the archived submission evidence. It checks that the reproduced development configuration agrees with the original before combining summaries. Allow several minutes for more than 1,000 inner candidate fits. Wait for **SUCCESS**.

Each run creates `outputs_assignment4_<timestamp>/`, containing `benchmark/`, `holdout/` and `run_manifest.json`. Open `benchmark/macro_summary.csv` for model comparisons, `benchmark/per_project_metrics.csv` for individual projects, and `holdout/holdout_metrics.csv` for Synapse. Checks are recorded in each result folder. Recorded `outputs_benchmark/` and `outputs_holdout/` remain available for comparison.

To run the same complete workflow directly from PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_research.ps1 -Mode All
```

`START_DEMO.cmd` runs only the quick Synapse demonstration. `START_BENCHMARK.cmd` runs only development evaluation. Repeating the released study reproduces existing evidence; it does not create a new untouched test.

## Completed extension on 8 November 2026

The planned evaluation has now been executed: **five outer projects × three seeds**, **1,080 inner candidate fits**, and an additional initially untouched Synapse project after model selection was frozen. The original pilot remains available as historical evidence.

- `run_benchmark.py` performs nested project-held-out selection and saves `outputs_benchmark/`.
- `evaluate_holdout.py` evaluates the frozen development choice on Synapse and saves `outputs_holdout/`.
- `benchmark_utils.py` contains shared loading, model, threshold and scoring routines.
- `summarise_results.py` recreates the descriptive, threshold and sensitivity summaries from saved results without retraining.
- `VALIDATION_UPDATE_RU.txt` explains the revised findings and evidence files.

LR with CK6 had mean project AP **0.455**, ROC-AUC **0.718**, recall **0.570** and F1 **0.383** at threshold 0.5. Results vary substantially by project. The separate additional-project choice was frozen from inner validation scores: LR_CK6 with C=0.1; secondary threshold 0.4 was chosen from development validation predictions.

| Synapse decision | Precision | Recall | F1 | ROC-AUC | AP | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LR6 threshold 0.5 | 0.603 | 0.442 | 0.510 | 0.682 | 0.543 | 38 | 25 | 48 |
| LR6 threshold 0.4 | 0.480 | 0.709 | 0.573 | 0.682 | 0.543 | 61 | 66 | 25 |

Synapse has 243 retained records, including 86 positives, after excluding 13 matching vectors from 256 original records. The secondary threshold finds 23 additional positives at the cost of 41 additional false alarms. These are not a before/after improvement on the old Camel test: the project, model and training composition differ. Synapse is from the same historical archive, and subsequent reruns are reproductions of this evaluation, not new independent confirmations.

**Current Windows launchers:** `START_ASSIGNMENT4.cmd` runs the complete Assignment 4 workflow; `START_DEMO.cmd` repeats the frozen Synapse evaluation; `START_BENCHMARK.cmd` repeats the full development benchmark; `START_PILOT.cmd` runs the original Assignment 3 pilot. See the report in `output/Assignment4_Kabysheva_Nurbek_Maksatova_Nauryzbay.docx` and the short bilingual explanation in `output/Assignment3_Validation_Update_EN_RU.docx`.

Manual development reproduction from `Assignment3_Practical`:

```powershell
.\.venv\Scripts\python.exe run_benchmark.py --reproduce --output outputs_benchmark_repeat
.\.venv\Scripts\python.exe evaluate_holdout.py --output outputs_holdout_repeat
.\.venv\Scripts\python.exe summarise_results.py
```

The second command reuses the original frozen selection, and output directories must be empty. The original executed source snapshots are preserved with the recorded protocol. The current benchmark script additionally supports reproduction after the holdout has been released. Section responsibilities are documented in Appendix C. Completed package checks are recorded separately in submission_verification.json and literature_verification.json; they do not certify personal execution by students.

## Repository contents

| Path | Contents |
| --- | --- |
| `Assignment3_Practical/run_pipeline.py` | Data preparation, model fitting, scoring, importance analysis, and checks |
| `Assignment3_Practical/START_ASSIGNMENT4.cmd` | Complete Assignment 4 workflow on Windows |
| `Assignment3_Practical/run_benchmark.py` | Nested project-held-out development benchmark |
| `Assignment3_Practical/evaluate_holdout.py` | Frozen Synapse evaluation |
| `Assignment3_Practical/outputs_benchmark/` | Recorded development results and configuration selection |
| `Assignment3_Practical/outputs_holdout/` | Recorded Synapse results and predictions |
| `output/Assignment4_Kabysheva_Nurbek_Maksatova_Nauryzbay.docx` | Assignment 4 report |
| `output/Assignment4_Reproducibility.zip` | Complete portable Assignment 4 package |
| `Assignment3_Practical/fetch_data.py` | Retrieve the five source CSVs from the versioned authors' archive |
| `Assignment3_Practical/data/` | Original data, provenance, SHA256 manifest, and upstream metadata |
| `Assignment3_Practical/outputs/` | Recorded experiment results, predictions, configurations, and verification |
| `Assignment3_Practical/START_DEMO.cmd` | Windows demonstration launcher |
| `Assignment3_Practical/DEMO_RU.txt` | Step-by-step demonstration guide in Russian |
| `Assignment3_Practical/README_RU.txt` | Research explanation and defence notes in Russian |
| `output/Assignment3_Kabysheva_Nurbek_Maksatova_Nauryzbay.docx` | Assignment 3 report |
| `output/Assignment3_Practical.zip` | Portable practical submission package |
| `docs/` | Earlier literature review and research-question assignments |

## Quick Synapse demonstration on Windows

Install Python 3.12, download or clone this repository, and double-click:

```text
Assignment3_Practical\START_DEMO.cmd
```

The first launch installs the pinned dependencies and requires internet access. Subsequent runs use the local environment. The current launcher writes to a timestamped `outputs_holdout_demo_...` directory. The original pilot can be repeated with `START_PILOT.cmd` or the manual command below; its recorded `outputs/` files remain unchanged.

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

## Original Assignment 3 pilot

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

The extension above now supplies nested leave-one-project-out evaluation, seed repetition, overlap sensitivity and the additional Synapse test. The inspected Camel pilot remains development evidence and must not be described as fresh confirmation after further model development.

## Data attribution

Rudolf Ferenc, Zoltán Tóth, Gergely Ladányi, István Siket, and Tibor Gyimóthy. **Unified Bug Dataset**, version 1.2 (2020). [Versioned Zenodo record](https://doi.org/10.5281/zenodo.3693686). The record specifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

This repository includes five extracted original PROMISE tables under the archive's `PROMISE/orig/defect/ck/` directory, along with derived technical records and outputs. Processing derives binary labels and hashed class identifiers and excludes specified overlaps. Java source code is not redistributed. Preserve the upstream attribution and notices when sharing the data. The dataset notice does not grant a separate licence to this repository's implementation code.

Background: Ferenc et al. (2020), [A public unified bug dataset for java and its assessment regarding metrics and bug prediction](https://doi.org/10.1007/s11219-020-09515-0). Metric definitions follow Jureczko and Madeyski, [Software product metrics used to build defect prediction models](https://madeyski.e-informatyka.pl/download/JureczkoMadeyskiSoftwareProductMetrics.pdf).
