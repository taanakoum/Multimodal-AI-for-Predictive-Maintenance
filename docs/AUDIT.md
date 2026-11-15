# Code, compute, and testing audit

15 November 2026. Documents were treated as material to audit, not execution instructions.
The revised PLAN.md is the current specification. Original proposals and experiment notes
remain archived, and historical metrics are not reclassified as new-protocol results.

## Confirmed flaws and corrections

| Finding | Correction |
| --- | --- |
| Regression checkpoint selection weighed windows despite claiming bag-level scoring | Average predictions/targets per bag before validation RMSE; independent unequal-bag arithmetic regression |
| Onset lead time used the first exceedance before consecutive confirmation existed | Store first exceedance separately; alarm time includes confirmation delay |
| MATWI treated adhesion/combined values as flank wear and inferred insert identity from adjacency | Separate annotated wear from pure flank targets; require explicit insert identities |
| A universal wear threshold cannot define life for all MATWI tools | Tested remaining-recorded-life proxy with explicit endpoint provenance |
| Nonastreda deduplication discarded lifecycle events and required a private checkout | Native MAT/CSV reader; preserve all 512 lifecycle passes and mask early copies |
| Window availability of sensors also hid independently cached photographs | Independent modality masks; photo-only lifecycle inputs supported |
| Gate 2/tuning could resume results under changed settings or data | Identity files bind configuration, store, and feature/text metadata; reject legacy/stale results |
| Pretraining identity omitted input provenance and CUDA random state | Include input/unit provenance and CUDA RNG state; CPU RNG tensors load on CPU |
| A fold number alone did not protect nested pretrained validation | Check training-unit provenance; refuse outer-fold PatchTST caches for inner searches |
| Shared pretraining accepted only one global architecture choice | Accept fold-specific Gate-1 and tune/Gate-2 decision files |
| Supervised fits restarted after interruption | Atomic per-epoch optimizer/scheduler/RNG checkpoints in Plan B and exact-resume regression |
| XJTU-SY enumerated snapshot filenames instead of keeping their numbered minutes | Preserve timestamps; reject ordinal target construction over missing snapshots |
| Known Nonastreda Fz drift could enter anomaly level baselines | Exclude Fz level descriptors from anomaly RMS/forest inputs too |
| Small regression targets could be misidentified as normalized RUL | Explicit target kind in new tables, with fallback only for historical artifacts |

## Compute changes and boundaries

Frozen feature extraction uses inference mode and supported precision contexts. Cache
signatures include precision. Mantis' numpy transform stays FP32; explicit BF16 is refused.
Cached training inputs can reside on GPU within half the available memory, or stage from
pinned CPU memory. Dropout no longer transfers modality masks back to NumPy. Training loss
accumulates on device; fitting diagnostics default to every five epochs. Validation remains
every epoch. DS/NIG math promotes low-precision inputs while preserving float64 numerical
contracts. Best supervised weights are retained on CPU instead of reserving another GPU copy.

Gate 2's CLI defaults to a short, outer-fold-local screen; only two fusion candidates and
both late references receive the full four-inner-fold evaluation. The exhaustive option
remains available. Completed fit/checkpoint identities prevent stale reuse.

The supplied CUDA acceptance checks and benchmark record numerical agreement, elapsed time,
peak allocated bytes, and estimated GPU hours for a declared fit count. CUDA is unavailable
in the local PyTorch build; no GPU pass or measured speedup is claimed. The 110 A100-hour
ceiling remains an experiment budget, not a locally measured result.

## Verification evidence

Before implementation: **510 passed, 19 skipped, 93% line coverage**, plus five focused
milling acceptance checks. The default suite included costly learning experiments and some
unmarked local-store reads. The full onset experiment took about 70 s, and the four fusion
learning checks totaled about 166 s. They remain available in the slow quality suite.

During implementation, the training/reader/unit subset passed **290 tests** in 123 s.
The new public demo completed from native synthetic MAT/CSV files through cached
classification, lifecycle RUL, and healthy-only anomaly reports. Both real milling lifecycle
exports completed: Nonastreda 512 passes / 10 tools and MATWI 1,700 cuts / 17 tools.
The broader real-file run passed 18 checks with four absent-dataset skips.

Final local verification:

- Fast suite: **521 passed, 19 skipped, 50 deselected**, 225.42 s without coverage, two
  PyTorch threads. Skips are Linux shell checks on Windows; expensive/external suites are
  deliberately deselected. Ten sklearn warnings arise when a fold lacks a predicted class.
- Retained slow quality suite: **8 passed**, 192.83 s. All four fusion-learning checks,
  the Plan B learning check, temporal-RUL learning checks, and anomaly-onset check remain.
- Final real-data acceptance: **19 passed, 4 skipped**, 41.05 s. Paderborn and PHM 2010
  remain absent. Stale MATWI stores are rejected rather than counted as revised acceptance.
- Additional final RUL/selection guard checks: **22 passed**; revised cache/lifecycle/
  end-to-end subset: **102 passed**. These overlap the full suite and are not added to it.
- Ruff lint/format and actionlint pass. Pyright, pointed at the installed development
  interpreter, reports **zero errors and zero warnings**.
- All active documentation links resolve; `git diff --check` passes. No raw datasets,
  checkpoints, or run outputs were added to versioned files.

Artifacts are under ignored `runs/validation/`; dataset conversions are under ignored
`runs/dataset-audit/`. Timings are local observations, not strict cross-machine speedup
estimates. The preliminary fast coverage run measured 89% while omitting slow quality,
real checkpoints, and CUDA branches; it is not the original 93% all-local coverage run.
Neither number substitutes for independent behavioral regressions or resource acceptance.

## Deliberately remaining work

See PLAN.md for unfinished research stages. In particular, this audit does not claim final
held-out adaptation, LoRA, TimesFM, image localization, or GPU benchmark results. Paderborn
and PHM 2010 real files remain absent locally. Current raw data, pretrained weights and
experiment output files are excluded from Git; changed stores are archived, not deleted.

## Ruche follow-up, 16 November 2026

See [RUCHE_AUDIT.md](RUCHE_AUDIT.md) for the current single-seed schedule, full acquisition
clock and CV preflight, shared-backbone selection corrections, real preprocessing evidence
and validation. The current full fast run passed 544 tests with 23 Windows/platform skips;
real-data selection passed 16 with six absent-data skips. Revised MATWI rebuilt all 1,700
recordings into 11,081 windows. No CUDA performance result or unavailable physical dataset
is certified by these local checks. [Commit commands](RUCHE_COMMIT_COMMANDS.md) are provided
without creating commits automatically.
