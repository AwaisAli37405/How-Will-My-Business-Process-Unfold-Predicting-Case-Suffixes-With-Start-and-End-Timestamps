# How Will My Business Process Unfold? Predicting Case Suffixes With Start and End Timestamps

Supplementary material and reproduction package for the paper/thesis
**“How Will My Business Process Unfold? Predicting Case Suffixes With Start and End Timestamps”**
by **Muhammad Awais Ali**, **Marlon Dumas**, and **Fredrik Milani**
(arXiv: [2509.14536](https://arxiv.org/abs/2509.14536)), corresponding to
**Chapter 6 (Resource-Availability–Enhanced approach)** and the consolidated framework of the PhD thesis.

This repository extends the sweep-line, multi-model (MM) approach with **resource-availability
features** (the **MM<sub>avail</sub>** variant). Availability calendars are automatically discovered
from historical event logs and integrated into the prediction of inter-start and processing times,
while the sweep-line online phase predicts the suffixes of all ongoing cases in lockstep.

---

## 1. Reproducibility summary

| Item | Specification |
|---|---|
| **Datasets & versions** | (i) The 12 real-life and synthetic logs of Chapter 5 (BPI17W, BPI12W, INS, ACR, MP, CVS, CFS, CFM, P2P, GOV, WorkOrder, P2PFin); and (ii) **8 synthetic loan-application logs** generated with the Apromore simulator, one per resource-availability scenario (Table 27 of the thesis), each with 2,000 cases and Poisson inter-arrivals (mean 20 min). See [Section 4](#4-datasets). |
| **Preprocessing** | Logs are converted to activity-instance logs with start/end timestamps; **resource-availability calendars are discovered from the log** and encoded as the *time-until-next-on-duty* feature, alongside intra-case and resource-contention features. Input logs go in `online_simulation/input_files/`. |
| **Train/test split** | Strict **temporal split** at `t_cutoff` = 80% of total process duration. Cases completing before the cutoff train the models; cases completing after, plus cases ongoing at the cutoff, form the test set. |
| **Random seed** | Fixed seed via the `SEED` variable in each `dg_training.py` / `dg_predictiction.py` (**default: 42**); seeds `{42, 1, 2, 3, 4}` for the repeated-seed / variability analysis. |
| **Hyperparameter search** | Random search, **50 iterations**, 80/20 train/validation split, **early stopping (patience = 10)**, up to **200 epochs**. Search space in [Section 5](#5-hyperparameter-search-space) (Table 25 of the thesis). Default: layer size 100, n-gram 10, batch 64, Adam. |
| **Ablation** | Base **MM** (intra + contention features) vs. **MM<sub>avail</sub>** (+ resource-availability features), all else held constant — this isolates the contribution of the availability features (EQ1). |
| **Runtime environment** | Python **3.8**, Anaconda environment `lstm` from `environment.yml` (TensorFlow/Keras, pm4py). |
| **Hardware used** | NVIDIA RTX 3090 GPU, Intel Core i9 CPU, 64 GB RAM. Training ≈2.5 h (small logs) to 5–6 h (large logs); inference ≈15,000 events in ≈100 s (≈150 events/s) on BPI17W. |
| **Reproduce command** | See [Section 6](#6-step-by-step-reproduction). |

---

## 2. Repository structure

```
.../
├── offiline/
│   ├── Next_activity_predictor/   # model α
│   ├── Start_time_predictor/      # model β (inter-start time)
│   └── Dur_predictor/             # model γ (processing time)
└── online/
    └── online_simulation/         # sweep-line online phase (MM / MM_avail)
        ├── script_testing.py
        ├── dg_predictiction.py
        ├── models_spec.ini
        └── input_files/
```

Each predictor folder contains its own `dg_training.py`, `dg_predictiction.py`,
`models_spec.ini`, `environment.yml`, and `training.sh` / `testing.sh`.

---

## 3. Environment setup

```bash
conda env create -f environment.yml
conda activate lstm
```

---

## 4. Datasets

| Log(s) | Type | Notes |
|---|---|---|
| BPI17W, BPI12W | Real-life | 4TU.ResearchData |
| INS, ACR, MP, GOV, WorkOrder, P2PFin | Real-life | Supplementary archive |
| CVS, CFS, CFM, P2P | Synthetic | Supplementary archive |
| 8 availability scenarios (a–h) | Synthetic | Apromore-generated loan-application logs; 2,000 cases each; calendars vary by density (dense/sparse), intermittency, and homogeneity (Table 27) |

---

## 5. Hyperparameter search space

Random search, **50 iterations**, early stopping patience 10 (Table 25 of the thesis):

| Parameter | Search space |
|---|---|
| Batch size | {32, 64, 128} |
| Normalization | {lognorm, max} |
| Epochs | 200 (early stopping, patience = 10) |
| N-gram size (`N_size`) | {5, 10, 15, 20, 25} |
| BiLSTM layer size (`L_size`) | {50, 100, 150} |
| Activation | {selu, tanh} |
| Optimizer | {Nadam, Adam, SGD, Adagrad} |

**Default configuration:** layer size 100, n-gram 10, batch 64, 200 epochs, Adam.

---

## 6. Step-by-step reproduction

### 6.1 Offline phase — train the three predictors (with availability features)

Run inside each of `offiline/Next_activity_predictor/`, `offiline/Start_time_predictor/`, `offiline/Dur_predictor/`:

```bash
python ./dg_training.py -f <LOG>.csv -m lstm -e 50 -o rand_hpc
```

Resource-availability features are enabled through the configuration in `models_spec.ini`
(set the availability/inter feature set). Trained `.h5` models and the held-out test set are
written to the output folder.

### 6.2 Online phase — sweep-line suffix prediction

```bash
cd online/online_simulation
python ./script_testing.py
# or:
python ./dg_predictiction.py -a pred_sfx -c <LOG> -b "<LOG>.h5" -v "d_action" -r 1
```

Outputs control-flow (Damerau–Levenshtein) and temporal (inter-start MAE, processing-time MAE)
results for the predicted suffixes.

### 6.3 Ablation — effect of resource-availability features

Reproduce the EQ1 ablation by running the pipeline twice on the same log: once with the base
**MM** feature set and once with the **MM<sub>avail</sub>** feature set (toggle in `models_spec.ini`),
keeping the architecture, seed, and hyperparameters fixed. For the controlled study, run across
the 8 availability scenarios (a–h).

---

## 7. Determinism note

Set the `SEED` variable in all training/prediction scripts and reuse it across the three
predictors and the online phase. Note that an incorrect next-activity prediction propagates to
the start-/processing-time predictors and to the derived availability features; results are
therefore reported as aggregates over repeated runs.

---

## 8. Citation

```bibtex
@article{ali2026unfold,
  title   = {How Will My Business Process Unfold? Predicting Case Suffixes With Start and End Timestamps},
  author  = {Ali, Muhammad Awais and Dumas, Marlon and Milani, Fredrik},
  journal = {arXiv preprint arXiv:2509.14536},
  year    = {2026}
}
```
