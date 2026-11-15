# Implementation and evaluation plan

Updated 15 November 2026. This is the current specification; the original proposal and
its amendment history are preserved in [archive](archive/PLAN-2026-10-23.md).

## Objective and model

Build a reproducible research pipeline for diagnosis, wear, RUL, and healthy-only anomaly
scoring using frozen sensor/image encoders and small trainable heads. Demonstrate modality
fusion only where paired observations exist. Text supplies class descriptions; operating
context is excluded from predictive inputs because it carries dataset/label priors.

One shared expert per physical sensor family, a photo expert, and dataset-specific target
heads form Plan B. Gate 2 may select concat, Perceiver, MBT, or C2F fusion. Class opinions
use Dirichlet evidence; regression uses NIG/Student-t predictive distributions. Evidence
combination is a modeling assumption, not proof that correlated modalities are independent.
Validate uncertainty using held-out calibration and empirical coverage.

## Dataset and lifecycle contracts

Source datasets: CWRU, Paderborn artificial-damage/source-healthy bearings, XJTU-SY,
Nonastreda, and MATWI. Targets remain KAIST, Paderborn real-damage/target-healthy bearings,
and PHM 2010. Historical KAIST EDA already contains labeled baselines: disclose this prior
inspection and describe the target as excluded from training/search, not never inspected.
Acceptance checks may inspect target formats and inventory; no target model
score may choose architecture, hyperparameters, or training steps.

- High-rate signals use anti-aliased 12 kHz processing with a 5 kHz common band. Low-rate
  milling inputs remain 1 kHz; MATWI resampling uses measured CSV acquisition duration.
  Raw window shards remain float32; frozen feature banks may be float16.
- Photos must be available before the current pass starts. Predictions for milling RUL
  occur at pass end, after the whole current recording has been observed. Within-pass
  online prediction is not claimed.
- Nonastreda: preserve all 512 logical passes for RUL. Exact duplicate signals are
  available only at their latest group member. RUL ends at the first dulled pass followed
  by at least two dulled labels among the next three recorded passes. This retrospective
  confirmed-wear target is not independently measured mechanical failure.
- Nonastreda's primary operating clock counts interior missing runs as four passes, using
  the median duration of preceding passes. Leading gaps and photo pauses do not count.
  Always report recorded-only clock sensitivity. Censored and post-endpoint passes have
  no ordinary regression label. The legacy 378-pass view is retained for historical
  diagnosis/wear comparisons, not used to reconstruct life.
- MATWI: order actual sensor timestamps, sum inclusive recording durations, and label each
  pass end with remaining recorded operating time to the final recording. Report this as
  a proxy unless per-tool terminal evidence establishes failure. Wall-clock gaps are not
  operation. Preserve missing wear labels and uncertain insert identities for RUL.
- Only pure flank-wear annotations label MATWI flank regression/classes. Mixed wear and
  adhesion retain separate metadata. Multiple inserts require explicit identity; their
  chronological adjacency does not identify them. The 71/110 micrometer class boundaries
  are operational bins, not universal failure thresholds.
- XJTU-SY retains normalized RUL from an offline FPT rule. FPT may use the complete life
  as a label-construction operation; model inputs remain causal. Missing numbered
  snapshots must not silently compress time or ordinal RUL labels.

See [dataset evidence](DATASET_AUDIT.md) for unsupported tasks and unresolved access.

## Evaluation and compute

- Preserve five unit-grouped source folds. CWRU/KAIST load splits measure condition shift,
  never independent-unit generalization. Scores aggregate at bag level; regression
  checkpoint selection must use the same aggregation as reported evaluation.
- Audit all required source units and class/bag coverage before training. Paderborn has
  three healthy source bearings for five folds: outer test folds 4/5 and validation fold 3
  lack healthy examples. Report that limitation and pooled out-of-fold class metrics;
  fold-wise balanced accuracy does not always represent all three classes. MATWI groups
  mix both materials and do not establish unseen-material generalization.
- Fit normalizers on training units and calibration on validation units. Report pooled
  and equal-unit summaries; bootstrap physical units and correct paired tests.
  Target kind is explicit in new result tables; normalized-RUL reporting cannot be
  inferred from small regression values. Legacy tables retain their old fallback.
- Gate 1 chooses frozen encoders/arms per outer fold. A shared winner must have applicable
  inputs on every scored source task; a favorable mean rank cannot compensate for a
  missing milling arm. Gate 2 defaults to a ten-epoch screen
  of all fusion candidates on the first inner split of that outer fold. Retain its best
  two and both late references for full four-inner-split evaluation. `--screen-epochs 0`
  evaluates all candidates. The existing +0.01 gain / 60% paired-win rule is retained.
- Pretraining accepts fold-specific Gate-1 and Gate-2 choices. Across source datasets, only
  unanimous overrides apply; conflicting fusion choices use late experts and other conflicts
  retain the common defaults. Merge tuning and Gate-2 decisions per dataset before seeking
  agreement across sources; Gate 2 owns the fusion recipe it evaluated. Fine-tuning retains
  the shared backbone's width, head and fusion architecture, explicitly recording any
  override of a dataset-specific choice. An initialized model must
  carry training-unit provenance contained in the current training split. Outer-fold
  PatchTST caches are refused for nested searches; proper inner-split pretraining is
  additional future work, not silently transductive validation.
- Production evaluations and search fits use one fixed seed (0). Five outer folds remain;
  they are different test units, not repeated random seeds. Unit-bootstrap intervals measure
  unit sampling uncertainty, not training-seed variability. Reuse only identical complete
  run identities. CUDA RNG, optimizer, scheduler, and sampling states are checkpointed.
- Stage immutable feature inputs on GPU when they fit within half the worker's available
  allocation; otherwise
  use pinned CPU staging. CPU uses FP32. Supported CUDA uses BF16 under `auto`, while
  evidential math remains FP32. Mantis' numpy transform remains FP32.
- Run the supplied CUDA acceptance and cached-feature benchmark before full jobs. Keep a
  measured time ledger below 110 A100-hours; report device, precision, elapsed time, peak
  memory, fit count, and numerical agreement. CPU measurements do not establish GPU savings.
- Ruche source jobs use isolated dataset/fold directories and one-GPU array elements,
  with up to four elements active. Pilot one versus two cached-model workers per GPU;
  divide CPU threads and bound each worker's CUDA allocator share. Use one worker for
  large frozen encoders or if concurrent fits increase wall time/OOM. Preserve training
  batch sizes unless an explicit recipe change is revalidated. See the current runbook.

## Task outputs and acceptance

RUL uses causal histories of cached signal summaries, independently masked earlier photos,
and elapsed-operation gaps. Milling targets are uncapped seconds; report RMSE, MAE,
Student-t NLL, calibrated coverage/width, and final-1500-seconds performance separately.
Always compare training-mean, schedule, and mean-training-lifetime-minus-elapsed baselines.

Milling anomaly scores fit on training tools' sharp bags only. Calibration healthy bags
set a fixed 95th-percentile threshold; untouched test tools supply AUROC/AUPRC, false
alarms, and detection rate. Missing support is reported explicitly. Test-derived FAR@95
is descriptive ROC analysis, not an operational threshold. XJTU onset reports both first
exceedance and the time when consecutive observations actually confirm an alarm.

Fast CI covers arithmetic, causality, masking, file interfaces, identity/resumption,
calibration/reporting, and a synthetic end-to-end pipeline. Slow research experiments and
external/hardware acceptance run explicitly. See [test matrix](TESTING.md).

## Delivered versus future work

Implemented: native readers, window/feature stores, source model selection, experts/fusion,
shared pretraining, calibrated source evaluation, causal RUL, milling lifecycle conversion,
healthy-only anomaly baselines, leakage/physics checks, reporting, CPU demo, categorized CI.

Pending research: final GPU benchmark/full runs; target adaptation (zero-shot/linear/LoRA),
PHM 2010/Paderborn real-file acceptance; TimesFM; learned reconstruction anomaly scores;
photo segmentation/localization; full explanation cards; strict inner-split PatchTST
pretraining; Nonastreda leave-one-tool-out comparison. None is presented as completed.
