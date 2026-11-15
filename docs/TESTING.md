# Tests and verification matrix

Every collected test receives exactly one primary category, derived from its directory
(or an explicit external-resource marker). Resource and cost markers are independent.
The default suite excludes acceptance and slow tests even when local data/weights exist.
`tests/support/` holds shared deterministic fixtures; it is not a test category.

| Category | Meaning | How to run |
| --- | --- | --- |
| unit | Numerical, schema, decision and masking contracts | `pytest -m unit` |
| integration | Reader/store/cache/model/report interfaces and resume | `pytest -m integration` |
| e2e | Synthetic files/caches through fitting, RUL/anomaly and reports | `pytest -m e2e` |
| acceptance | Real data, actual checkpoints, live sources or CUDA | Explicit flags below |
| slow marker | Multi-seed learning/quality experiments | `pytest -m "slow and not acceptance"` |

```bash
pytest                                      # fast CPU contracts
pytest --durations=20 --cov=pdmfm            # one coverage run
pytest -m "slow and not acceptance"          # quality, no external resources
pytest -m data --run-data                    # available real datasets
pytest -m weights --run-weights              # locally cached checkpoints
pytest -m network --run-network              # live publisher sources
pytest -m gpu --run-gpu                      # actual CUDA, fails if unavailable
```

Use `PDMFM_TEST_THREADS` to override the default two PyTorch threads. Encoder/data tests
never download weights opportunistically. Missing external resources skip with reasons;
those skips do not constitute validation. CUDA requires the platform's normal deterministic
BLAS configuration (e.g. `CUBLAS_WORKSPACE_CONFIG=:4096:8`) for exact-resume checks.

## Requirement-to-test mapping

| Project facet | Unit or integration contracts | End-to-end / acceptance evidence |
| --- | --- | --- |
| Public setup, entry points | Paths, sources, resource limits, CLI arguments | `pdmfm.demo`, core CI, portable reader/window checks |
| Raw data and units | Native MAT/CSV/TDMS/image readers; ID and label validation | `acceptance/test_real_data.py`, reader-catalogue checks |
| Signal processing | Aliasing, band limit, spectra, envelope orders, levels, speed | Synthetic file-to-store tests; CWRU roll-off acceptance |
| Window stores | Atomic rebuild, stale versions, label bags, missing channels | Lifecycle raw-MATWI conversion; historical stores opt-in |
| Photo timing | Strict as-of boundary, no current-pass photo, stale photos | Real Nonastreda/MATWI joins and lifecycle previous-photo fields |
| Frozen feature caches | Online/cache agreement, invalidation, resume, absent arms | Optional actual encoder checkpoints; no weights needed for stand-ins |
| Model and uncertainty | Shapes, modality independence, finite gradients, DS/NIG math | Source fitting and calibrated bag tables; slow learning experiments |
| RUL conversion | Hand-computed time arithmetic, gaps, duplicate availability, censoring | All 512 Nonastreda / 1,700 MATWI records converted locally |
| Temporal RUL | Own-unit history, future perturbations, causal reference | Both milling tasks in e2e; XJTU-SY reader and run/report contracts |
| Anomaly | kNN/RMS/forest scores, missing support, confirmation timing | Unit-disjoint milling calibration; slow bearing-onset quality check |
| Model selection | Bag-weighted RMSE, nested train/val/test units, gate rules | Search resume/identity, source backbone initialization checks |
| Pretraining | Shared weights, masked-content exclusion, contrastive negatives | CPU interruption/resume; optional CUDA training-resume acceptance |
| Compute | CPU staging, low-precision likelihood gradients | CUDA numerical checks and benchmark; local CUDA unavailable |
| Calibration/statistics | Temperature/scale, Student-t intervals, unit bootstrap, Holm | Paired reports, explicit target kind, legacy compatibility |
| Explanations | Tone occlusion, physics masks, deletion and randomization | Cached/online probe agreement; reader-dependent physics checks |
| Downloads and cluster | Hash/size/resume, archive handling, SLURM argument/chain behavior | Live source checks opt-in; shell tests on Linux |
| Ruche worker orchestration | Real CLI argument parsing, single-seed manifests, GPU/CPU shares | Real subprocess failure/retry isolation; isolated fold collection into pooled reports |
| Preflight and CV coverage | All physical source units; four inner groups exclude outer test | Full store/cache numeric scans; missing/incomplete cache rejection; class coverage report |

The missing adaptation/LoRA, forecasting, photo localization and final explanation-card
stages are listed as future work in PLAN.md. Tests do not imply those stages are built.

## Audit-driven changes

The original audit passed 510 tests with 19 platform skips and 93% line coverage, yet
independent arithmetic reproduced a regression-selection bug and alarm-time backdating.
New tests encode the intended behavior independently of the implementation.

Moved out of the default suite: the approximately 70-second full onset experiment and
166 seconds of four multi-seed fusion-learning checks. Their scientific learning assertions
remain in the slow suite. Fast candidate-forward/gradient/masking checks remain. Cache
recency tests use a controlled clock instead of sleeping. Real-store count checks are
parametrized by dataset, so one unavailable store cannot skip checks for other datasets.

Published validation counts and timing commands are recorded in AUDIT.md. Wall time
comparisons must state coverage, optional dependencies, hardware and thread settings;
there is no claimed GPU speedup until the cluster pilot passes.

## Previous complete audit result

The fast suite passed **521 tests**, with 19 Windows/platform skips. The separate slow
suite passed all **8 quality checks**; real-data acceptance passed **19**, skipping four
checks for absent Paderborn/PHM 2010 data. CUDA and live network acceptance were not run.
Lint, formatting, workflow lint and type checking pass. See AUDIT.md for timing, artifact
paths, and the distinction between fast coverage and external/hardware coverage.

## Ruche follow-up

The 16 November scheduling/preprocessing follow-up passed 544 fast tests (199 unit,
342 integration, three end-to-end) with 23 platform skips. Real-data selection passed 16
with six absent-data skips; the full revised MATWI build also passed. A final 57-test
selection/worker/pretraining follow-up passed, including the added shared-eligibility
regression. See
[RUCHE_AUDIT.md](RUCHE_AUDIT.md) for final selection regressions, shell/type checks,
measured timings and outstanding CUDA/physical-data acceptance. Production uses seed 0;
slow multi-seed learning checks are a separate research-quality suite.
