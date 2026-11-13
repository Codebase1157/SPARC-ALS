# Manuscript ↔ code map

A section-by-section cross-reference, so every claim in the paper can
be audited against a file in this repository.

| Manuscript section                                 | File(s)                                                                            |
| -------------------------------------------------- | ---------------------------------------------------------------------------------- |
| §3 Materials (Table 2)                             | `configs/preprocessing.yaml`, `sparc_als/data/preprocess.py`                       |
| §3 Forced alignment and vowel nuclei               | `configs/forced_alignment.yaml`, `sparc_als/data/forced_alignment.py`              |
| §3 Speaker-disjoint splits                         | `splits/*.json`, `sparc_als/data/splits.py`                                        |
| §4.1 Encoder                                       | `sparc_als/models/encoder.py`                                                       |
| §4.1 Dual adapters, Equation (1) ASP               | `sparc_als/models/adapters.py`                                                      |
| §4.1 Supervised contrastive, Equation (2)          | `sparc_als/training/losses.py::sup_con_loss`                                       |
| §4.1 GRL + nuisance (Eq. 3)                        | `sparc_als/training/losses.py::adversarial_loss`, `cross_covariance_penalty`       |
| §4.1 λ_rev sigmoid schedule                        | `sparc_als/training/losses.py::grl_lambda`                                         |
| §4.2 VSGA, Equation (4)                            | `configs/graph_priors.yaml`, `sparc_als/models/vsga.py`                             |
| §4.3 NDT axes (Table 3)                            | `configs/acoustic_targets.yaml`, `sparc_als/utils/acoustic_features.py`,`sparc_als/models/ndt.py` |
| §4.3 Equations (5) and (6)                         | `sparc_als/models/ndt.py::uncertainty_weighted`, `sparc_als/utils/reference_stats.py` |
| §4.4 Counterfactual operators (Table 4)            | `configs/counterfactuals.yaml`, `sparc_als/training/counterfactuals.py`            |
| §4.4 L_cf                                          | `sparc_als/training/losses.py::cf_consistency_loss`                                |
| §4.5 Meta-adaptation                               | `sparc_als/training/meta_adapt.py`                                                  |
| §4.5 Training order and checkpointing              | `configs/training.yaml`, `sparc_als/training/checkpoints.py`                       |
| §5 E1 – E7                                         | `scripts/E{1..7}_*.py`, `run.sh`                                                   |
| §5 Baselines and controls on identical splits      | `configs/baselines/*.yaml`, `configs/controls/*.yaml`, `sparc_als/baselines/`      |
| §5 Statistical tests                               | `sparc_als/eval/{bootstrap,delong,mcnemar}.py`, `sparc_als/eval/paired_tests.py`   |
| §6 Per-speaker and per-utterance predictions       | `predictions/<corpus>/<experiment>/*_seed<seed>.csv`                                |
| §6 Calibration temperatures                        | `predictions/<corpus>/<experiment>/temperature_seed<seed>.json`                    |
