# Ruche, cross-validation and preprocessing follow-up

16 November 2026. The current [runbook](RUCHE_RUNBOOK.md) supersedes archived cluster
commands. [Commit commands](RUCHE_COMMIT_COMMANDS.md) split the changes into four reviewable
commits. No cluster jobs or commits were created during local validation.

## Corrections

- Production defaults are seed 0 for supervised source fits, shared pretraining, RUL,
  tuning and baselines. Five outer folds remain. Multi-seed legacy artifacts remain readable.
- GPU arrays allocate one GPU per element and run isolated dataset/fold tasks, with up to
  four active elements per array. Cached-model packing is configurable; two workers split
  eight CPU threads and 90% of the CUDA allocator capacity. Actual occupancy and throughput
  must determine whether one or two workers is better. Training batch sizes stay unchanged.
- Unique task directories prevent concurrent fold jobs from overwriting summaries and
  decisions. Real subprocess tests cover exit status, failure propagation, GPU-mask
  preservation, memory/thread shares and retry behavior. Five completed folds collect
  into one pooled report after their workers exit.
- Environment/module activation fails early. Compute jobs load weights offline. Separate
  commands cover fast tests, cached checkpoints, raw data and CUDA; fast tests do not imply
  external acceptance. GPU jobs log ten-second utilization/memory/power samples.
- MATWI was missing from Gate 1 despite participating in source training. It now contributes
  five tool-grouped wear-classification probes. Shared selection excludes candidates with
  no applicable arm on any scored source task: worst-rank penalties alone were insufficient
  to prevent a bearing-only encoder input from winning the cross-task average.
- Shared pretraining now merges tuning and Gate-2 reports per dataset before calculating
  agreement across sources. Gate 2 owns its evaluated fusion recipe. Missing keys from
  unrelated search files no longer veto otherwise agreed settings. Fine-tuning retains the
  shared backbone's width, head and fusion architecture, and logs effective overrides.
- A new CPU preflight checks complete source-unit coverage, all configured classes,
  bag/unit membership, numeric contents, photo causality, disjoint outer/inner splits and
  complete finite caches. Partial smoke stores and missing sources fail explicitly.
- MATWI date conversion now explicitly uses nanoseconds rather than assuming pandas'
  inferred integer datetime unit. The full acquisition clock must be monotonic and uniform
  within the stated jitter tolerance. Nonfinite sensor values and invalid rates fail before
  filtering/resampling. MATWI window policy is v3, invalidating old stores and their caches.

## Cross-validation findings

| Dataset | Train / validation / test units per outer fold |
| --- | --- |
| Nonastreda | 7 / 1 / 2 in every fold |
| MATWI | 9/4/4, 10/3/4, 11/3/3, 11/3/3, 10/4/3 |
| XJTU-SY | 9 / 3 / 3 in every fold |
| Paderborn source protocol | 9 / 3 / 3; structurally checked on synthetic bearings |

Each physical source unit is tested exactly once. The four inner validation groups cover
all non-test units exactly once, without any outer test unit entering an inner split.
Class/bag counts are reported for training, validation and test, not assumed from fold IDs.
Paderborn's three healthy source bearings cannot populate all five test groups: folds 4/5
lack healthy test examples and fold 3 lacks healthy validation. Report class coverage and
pooled out-of-fold metrics. CWRU/KAIST load splits measure condition shift. MATWI groups
mix both materials; they do not measure unseen-material generalization. Unit-bootstrap
intervals do not measure seed variability, and overlapping training pools do not make
five folds independent repeated experiments.

## Real preprocessing evidence

| Available store | Windows | Recordings / bags | Result |
| --- | --- | --- | --- |
| CWRU | 356 | 40 | Numeric/CV preflight passed; all ten classes at all four loads |
| Nonastreda | 4,241 | 378 | Numeric/CV/photo preflight passed; all ten tools |
| MATWI, revised rebuild | 11,081 | 1,700 | Full raw acquisition clocks and five channels checked; all 17 tools; numeric/CV/photo preflight passed |
| XJTU-SY | 9,216 | 9,216 | Numeric/CV preflight passed; all 15 bearings |
| KAIST | 4,378 | 45 | Numeric/photo preflight passed; held-out dataset, no source CV |

The full MATWI rebuild took approximately 349 seconds locally. Its 1,052 pure-flank-labelled
recordings comprise 475 sharp, 354 used and 223 dulled; 648 unlabelled recordings remain
available for lifecycle inputs. Revised outputs are preserved under ignored
`runs/preprocessing-audit/windows/matwi`; the existing external MATWI store was not replaced.
Rebuild the default MATWI store and regenerate matching features before production use.

Preflight results are in `runs/preprocessing-audit/all-stores.json`. Paderborn and PHM 2010
stores/raw files are absent locally, and their preflight entries fail explicitly. Nothing
here certifies their physical preprocessing. Remaining-recorded-operation MATWI targets
are still proxies to the last recording, not verified physical failure times. Nonastreda's
512-pass RUL lifecycle remains distinct from its 378-recording wear store.

## Validation and limits

- Full fast suite: **544 passed, 23 Windows/Linux-only skips**, 50 deselected, in 241 seconds.
  Passing categories: 199 unit, 342 integration, three synthetic end-to-end tests. The
  additional final Gate-1 eligibility change is covered by a **57-test follow-up selection/
  worker/pretraining suite**, all passed in 69 seconds. The strengthened missing-class and
  nonfinite-cache preflight regression also passed separately.
- Real-data selection: **16 passed, six absent-data skips**, in 64 seconds. Four skips are
  Paderborn/PHM reader checks; two are their parametrized channel-catalogue checks. The
  independent full MATWI rebuild/preflight above validates the revised store.
- Ruff lint/format, workflow lint and `pyright pdmfm` pass. The package type check has zero
  errors/warnings; this is the CI package scope, not a claim about all analysis/test scripts.
- Git Bash parsed every Slurm script. Mocked `sbatch` exercised array bounds, dependencies,
  chain submission and invalid/non-Slurm invocation; no real submission occurred. The Linux
  pytest shell suite remains a CI check because its Windows skip marker is intentional.
- The ten fast-suite warnings concern predictions of classes absent from a scoring group;
  they are not convergence failures. Per-part class coverage is therefore disclosed.
- Slow multi-seed synthetic quality experiments were not rerun for this scheduling follow-up.
  Their previous eight-test result is recorded in AUDIT.md. CUDA utilization, GPU allocator
  behavior and mixed-precision numerical agreement require the runbook's Ruche pilot;
  CPU-only PyTorch cannot establish those results or certify the 110 A100-hour ledger.

Ignored validation artifacts: `runs/validation/ruche-fast.xml`, `ruche-data.xml`,
`ruche-workers.xml`, `ruche-preflight.xml`, `ruche-final-selection.xml`,
`ruche-pyright.json`, and `ruche-shell-smoke.sh`. No raw data, checkpoints or generated
window/feature files were added to Git. Current commands and measured results are kept
separate from historical experiments and unsupported future research stages.
