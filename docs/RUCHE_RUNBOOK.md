# Ruche execution runbook

Reviewed 16 November 2026. [PLAN.md](PLAN.md) defines the scientific protocol;
[DATASET_AUDIT.md](DATASET_AUDIT.md) and [eda/README.md](eda/README.md) explain dataset
limitations. Archived runbooks are historical and must not supply production commands.

## Cluster and environment

The [official cluster inventory](https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/01_cluster_overview/)
currently lists four 40 GB A100s per GPU node. Each command here uses one visible GPU;
independent folds use Slurm arrays. Allocating four GPUs to one Python process does not
make this code distributed. The wrappers request `gpua100`, eight CPUs per GPU, and at
most 24 hours. Confirm your live partition/QOS limits with `sinfo -s` and your account's
Slurm configuration before submitting; job options can override wrapper defaults.

On the login node, check `module avail anaconda miniconda` and configure `slurm/env.sh`
for your available module and environment. Install Python >=3.11, a CUDA-enabled PyTorch
>=2.4 wheel compatible with the compute-node driver, and this package's `[dev,encoders]`
extras. Do not assume a CPU-only environment copied from Windows supports CUDA.
Use the [official PyTorch installer](https://pytorch.org/get-started/locally/) for the wheel.
Keep raw/extracted/interim data under `$WORKDIR/pdmfm_data`, weights under
`$WORKDIR/hf_cache`, and submit from the repository root. Defaults are configurable in
`env.sh`; `--export=NONE` does not carry arbitrary login-shell overrides into jobs.

```bash
source slurm/env.sh
python -m pip install -e '.[dev,encoders]'
mkdir -p runs/logs runs/jobs
bash slurm/prefetch_weights.sh
sbatch slurm/checks.sbatch
sbatch slurm/job.sbatch python -m pytest -m weights --run-weights
sbatch slurm/job.sbatch python -m pytest -m gpu --run-gpu
```

Prefetch on the network-enabled login node. Extraction, model loading, preprocessing,
tests, probes, and training run on compute nodes. Jobs read cached weights offline.
A failed module load/activation now stops the job immediately. A passing fast test job
alone does not establish raw-data, checkpoint, or CUDA acceptance.

## Data preprocessing and acceptance

Inspect `python -m pdmfm.fetch plan` before downloading. Build each dataset in its own CPU
job, waiting for completion before preflight. `cpu_short` defaults to one hour: override
partition/time for larger builds using your live allocation limits. Do not run concurrent
writers for the same window store or encoder cache.

```bash
sbatch slurm/cpu.sbatch python -m pdmfm.windows build cwru nonastreda matwi
sbatch slurm/cpu.sbatch python -m pdmfm.windows build xjtu_sy paderborn
# Separate held-out targets; never include them in source pretraining/selection:
sbatch slurm/cpu.sbatch python -m pdmfm.windows build kaist phm2010
sbatch slurm/cpu.sbatch python -m pdmfm.lifecycle --dataset nonastreda --clock gap
sbatch slurm/cpu.sbatch python -m pdmfm.lifecycle --dataset matwi
sbatch slurm/cpu.sbatch python -m pytest -m data --run-data
# Submit this after the source builds finish successfully:
sbatch slurm/cpu.sbatch python -m pdmfm.preflight --datasets cwru nonastreda matwi xjtu_sy paderborn --out runs/preflight/source-windows.json
```

Preflight scans complete stores for finite signals/levels, valid window times, bag/unit
consistency, causal photo ages, all required physical source units, and disjoint outer and
inner folds. With an encoder it additionally requires fully extracted, finite feature
caches. It exits nonzero and records errors rather than silently dropping a source.
The acceptance suite can skip absent raw datasets; inspect its skip summary as well as
preflight. Paderborn and PHM 2010 remain physically unverified until real files are provided.

| Dataset | Preprocessing and interpretation to preserve |
| --- | --- |
| CWRU | Anti-alias resampling and common 5 kHz band; nominal load-context speed avoids per-file RPM fingerprints. Load shift is not unseen-bearing transfer. |
| Nonastreda | Native force at 1 kHz; wear store and all-512-pass lifecycle are distinct views. Previous-pass photos only. Confirmed dulled endpoints; gap-clock estimates have recorded-only sensitivity. |
| MATWI | All five channels; parse the sixth CSV column, validate the full monotonic uniform acquisition clock, then resample by measured duration to 1 kHz. Pure flank labels only; unidentified inserts stay unlabelled for wear. Missing wear does not remove lifecycle records. |
| XJTU-SY | Anti-aliased vibration, actual numbered minute chronology, offline 10%-reference/10-consecutive FPT with kurtosis fallback. Incomplete minute sequences are refused. Model histories stay causal. |
| Paderborn | Distinct artificial-source and real-damage-target bearings; never admit target bearings through a source-fold mask. Real-format acceptance still required. |
| KAIST | Common-band vibration, relative temperature; class-dependent current channels excluded. State-paired devices are not sample-synchronous. Target loads measure condition shift. |
| PHM 2010 | Only c1/c4/c6 have wear labels; cutter-disjoint evaluation. Wear is not evidence of an observed failure endpoint. |
| Optional NASA milling / Najafi / MBHM | Sparse wear / single motor / unresolved unit joins respectively. Not admitted as extra RUL or independent-transfer evidence. See the dataset audit. |

MATWI reader policy is now `clock-v3`: old stores must be rebuilt and features regenerated.
Changed stores are archived, not mixed with current cache identities. The timestamp fix
explicitly uses nanoseconds, including on pandas versions that infer microsecond datetimes.
MATWI life means remaining **recorded operating seconds to last recording**, not verified
physical failure RUL. Nonastreda's confirmed wear endpoint also requires its stated proxy
assumptions. Preprocessing quality cannot turn either dataset into verified failure data.

## Frozen features and GPU throughput pilot

Run single-process extraction first. Increase encoder batch size on an isolated pilot,
starting at 256 then 512/1024 if memory permits. `--chunk` limits queued windows/save
frequency and must be large enough to supply the desired batch; T arms multiply each
window into several encoder inputs. Compare windows/second and peak memory; reduce batch
size on OOM. Larger extraction batches do not change optimizer steps.

```bash
sbatch slurm/job.sbatch python -m pdmfm.features extract --dataset cwru nonastreda matwi xjtu_sy paderborn --encoder mantis --arms F --device cuda --batch-size 256 --chunk 1024
sbatch slurm/job.sbatch python -m pdmfm.features images --dataset nonastreda matwi --device cuda --precision auto
sbatch slurm/job.sbatch python -m pdmfm.features texts --dataset cwru nonastreda matwi xjtu_sy paderborn --device cuda
# After extraction finishes; repeat for every Gate-1 encoder/arm candidate:
sbatch slurm/cpu.sbatch python -m pdmfm.preflight --datasets cwru nonastreda matwi xjtu_sy paderborn --encoder mantis --arms F --out runs/preflight/mantis-F.json
sbatch slurm/job.sbatch python -m pdmfm.benchmark --dataset nonastreda --encoder mantis --arms F --fits 120
```

Cache extraction resumes completed chunks. Mantis' NumPy transform stays FP32. Supported
CUDA training uses automatic BF16 with FP32 evidential math. Immutable training inputs
are staged within half each worker's memory allowance, or pinned in host memory.
Keep optimizer batch sizes at the validated defaults (256 fusion/pretraining, 128 RUL).
Changing them to fill memory changes the learning recipe and needs new validation.

All GPU wrappers write `runs/logs/gpu-<job>-<task>.csv` with utilization, memory and power
samples every ten seconds. Cached-model workers also record per-task elapsed time and
exit status. Pilot otherwise equivalent folds with one and two workers per GPU, using
separate run IDs/manifest files. Select the setting with better completed fits/GPU-hour
and valid metrics; retain one worker if packing slows work or exceeds memory. Packing
splits eight CPU threads and 90% of CUDA allocator capacity among workers. The
[PyTorch allocator limit](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.set_per_process_memory_fraction.html)
does not account for every CUDA context/library allocation, so leave headroom. No CUDA
MPS service or global GPU changes are needed. High occupancy is measured, not guaranteed.

Count allocated GPU-hours once per array element, not once per packed worker. Four active
GPUs consume four hours per wall hour. Include extraction, all search candidates/inner
fits, pretraining, failed/restarted jobs, and final evaluation in the **110 A100-hour**
ledger. `benchmark --fits` is a pilot estimate, not a complete experiment budget.
Use `sacct -j JOB_ID --format=JobID,ElapsedRaw,AllocTRES,State,MaxRSS` after each array to
record actual allocated time and failed tasks; do not sum both parent and step records.

## Cross-validation and one-seed jobs

Use **seed 0 only** in production, including final evaluations. CLI defaults and generated
manifests now use one seed. Explicit legacy multi-seed results remain readable. Slow
synthetic learning tests may exercise multiple seeds to test scientific stability; those
are separate from the production Slurm schedule.

| Source | Outer test / validation / train units | Inner selection |
| --- | --- | --- |
| Nonastreda | 2 / 1 / 7 tools | Four disjoint validation groups from the 8 non-test tools |
| MATWI | 3-4 / 3-4 / 9-11 tools, both materials per group | Four groups from the non-test tools; not leave-one-material-out |
| XJTU-SY | 3 / 3 / 9 bearings | Four groups, each spanning the three conditions |
| Paderborn | 3 / 3 / 9 source bearings | Four groups; no real-damage target bearing |

Every source unit appears in outer test exactly once. Windows and histories never split
one physical unit across partitions. Paderborn has only three source healthy bearings:
folds 4/5 have no healthy test bearing, and fold 3 has no healthy validation bearing.
Report per-class/unit coverage and pooled out-of-fold metrics; do not claim every fold's
balanced accuracy measures all three classes. Five folds are not five independent random
repeats: training pools overlap. Unit-bootstrap intervals express physical-unit sampling
uncertainty and do not quantify training-seed variability. Small tool counts and material
confounding still limit generalization claims.

Gate 1 ranks encoders/arms using each outer fold's own validation data, now including
MATWI. Complete the full configured source task list. A pooled winner is descriptive;
use `per_fold_winner` for the final outer-test models. A shared winner needs an input arm on
every scored task: bearing-only E or 12 kHz S cannot serve the 1 kHz milling datasets.
T+E can serve milling through T. Unsupported candidates remain in the comparison table
but are ineligible for shared selection. Gate 2/tuning rotate four inner
splits of the non-test pool. Gate 2's ten-epoch screen retains two finalists plus both late
references; `--screen-epochs 0` requests exhaustive evaluation. Never fit scaling on
validation/test data. Checkpoint/interval calibration uses outer validation only. Nested
search refuses outer-fold PatchTST caches until inner-split pretraining exists.

## Source workflow and dependency barriers

The following is a one-candidate example; extract every candidate's caches before adding
it to Gate 1. Run each numbered stage only after its prerequisites finish successfully.
Do not start GPU training directly in the login shell.

1. Probe on CPU after preflight:

   ```bash
   sbatch slurm/cpu.sbatch python -m pdmfm.probe --run-id gate1 --candidates mantis:F
   ```

2. After Gate 1 completes, generate immutable Gate-2/tuning manifests on the login node,
   inspect the commands, then submit. The full source set is explicit:

   ```bash
   python -m pdmfm.jobs make --stage gate2 --run-id gate2 --datasets nonastreda matwi xjtu_sy paderborn --gate1 runs/gate1/gate1/report.json --out runs/jobs/gate2.json
   python -m pdmfm.jobs show runs/jobs/gate2.json
   bash slurm/submit.sh runs/jobs/gate2.json 2
   python -m pdmfm.jobs make --stage tune --run-id tune --datasets nonastreda matwi xjtu_sy paderborn --gate1 runs/gate1/gate1/report.json --out runs/jobs/tune.json
   bash slurm/submit.sh runs/jobs/tune.json 2
   ```

   Gate 2 and tuning are separate searches from common defaults, not a joint exhaustive
   hyperparameter/fusion optimization. Both use seed 0. Wait for both arrays to finish.

3. Pretrain fold-local shared backbones with the same Gate-1 choices. Supply every source
   decision explicitly; `shared_choices` applies only unanimous overrides, using late
   fusion/common defaults on disagreement. Templates expand to each fold's isolated files.

   ```bash
   python -m pdmfm.jobs make --stage pretrain --run-id pre --datasets nonastreda matwi xjtu_sy paderborn --gate1 runs/gate1/gate1/report.json --decisions 'runs/tune/tune-nonastreda-fold{fold}/best.json' 'runs/gate2/gate2-nonastreda-fold{fold}/decision.json' 'runs/tune/tune-matwi-fold{fold}/best.json' 'runs/gate2/gate2-matwi-fold{fold}/decision.json' 'runs/tune/tune-xjtu_sy-fold{fold}/best.json' 'runs/gate2/gate2-xjtu_sy-fold{fold}/decision.json' 'runs/tune/tune-paderborn-fold{fold}/best.json' 'runs/gate2/gate2-paderborn-fold{fold}/decision.json' --out runs/jobs/pre.json
   bash slurm/submit.sh runs/jobs/pre.json 2
   ```

   CWRU's fixed 0-2 hp source split is added automatically. Missing sources/caches fail
   the GPU worker preflight. No target unit or outer validation/test unit enters a shared
   fold's pretraining. Backbone architecture and training-unit provenance are checked
   when initializing the supervised model. Tuning and Gate-2 files merge per source before
   consensus; Gate 2 retains its evaluated fusion recipe. Fine-tuning takes the shared
   backbone's width, head and fusion, even when a source's standalone Gate-2 winner differs.
   Effective settings are logged; the shared method is not a separate per-source fusion
   winner experiment. Such an experiment would need its own matching pretraining.

4. After pretraining finishes, generate final per-dataset jobs. Run this pair once for
   each source dataset, substituting its name in `--datasets` and decision paths:

   ```bash
   python -m pdmfm.jobs make --stage planb --run-id final --datasets nonastreda --gate1 runs/gate1/gate1/report.json --decisions 'runs/tune/tune-nonastreda-fold{fold}/best.json' 'runs/gate2/gate2-nonastreda-fold{fold}/decision.json' --init 'runs/pretrain/pre-shared-fold{fold}/fold{fold}-seed{seed}/best.pt' --out runs/jobs/final-nonastreda.json
   bash slurm/submit.sh runs/jobs/final-nonastreda.json 2
   python -m pdmfm.jobs make --stage rul --run-id life --datasets nonastreda matwi xjtu_sy --gate1 runs/gate1/gate1/report.json --out runs/jobs/rul.json
   bash slurm/submit.sh runs/jobs/rul.json 2
   ```

   Each array element requests one GPU, runs up to two isolated processes, and the array
   caps itself at four active GPUs. Do not overlap unrelated arrays beyond your aggregate
   account limit. To enforce dependencies, pass e.g. `--dependency=afterok:12345` after
   the worker count in `submit.sh`. Generating decisions/initializations requires prior
   artifacts, so generation itself belongs after the relevant barrier. Each process owns
   `runs/<stage>/<run-id>-<dataset>-fold<k>/`; shared pretraining uses `shared` as dataset.
   Unique run IDs are mandatory across concurrent manifests. Retrying the same immutable
   manifest resumes failed tasks; successful tasks are skipped. Never submit the same
   manifest twice concurrently. Leave checkpoint/run identity guards enabled.

5. After all five folds for a dataset finish, pool final reports on CPU:

   ```bash
   sbatch slurm/cpu.sbatch python -m pdmfm.jobs collect runs/jobs/final-nonastreda.json --dataset nonastreda --out runs/collected/nonastreda
   sbatch slurm/cpu.sbatch python -m pdmfm.jobs collect runs/jobs/rul.json --dataset matwi --out runs/collected/matwi-rul
   sbatch slurm/cpu.sbatch python -m pdmfm.anomaly --dataset nonastreda --features mantis:F --run-id nona-anomaly
   bash slurm/status.sh
   ```

   Collection refuses missing/nonzero task status, incomplete five-fold coverage, or an
   unrelated existing output directory. It preserves the source manifest. Source workers
   never write a shared summary while another fold is running. Inspect unit-bootstrap
   pooled reports and lifecycle assumptions before presenting results. Anomaly calibration
   remains healthy training support plus healthy validation thresholds; test-derived ROC
   summaries do not set operational thresholds.

Target adaptation/LoRA, forecasting and final held-out research evaluations remain future
stages. Passing source execution checks does not mark those stages implemented.
