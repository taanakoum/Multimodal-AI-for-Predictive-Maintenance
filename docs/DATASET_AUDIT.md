# Dataset capability and provenance audit

15 November 2026. This audit combines publisher documentation, native reader inspection,
local inventories, synthetic format contracts, and opt-in real-file acceptance. Inspecting
target formats is not target model selection; no held-out model scores were used here.

## Capability matrix

| Dataset | Diagnosis/wear | Healthy-only anomaly | RUL | Independent evaluation unit |
| --- | --- | --- | --- | --- |
| Nonastreda | Visual wear class and measured flank wear | Worn-tool novelty relative to sharp training tools | Time to retrospectively confirmed dulled wear; estimated cutting-time proxy | 10 tools |
| MATWI | Pure flank labels only; adhesion/mixed annotation retained separately | Worn-tool novelty; some tools have no observed sharp state | Remaining recorded operation to final recording; physical terminal events unverified | 17 tool sets |
| XJTU-SY | Terminal failure mode is metadata, not a per-snapshot fault label | Onset relative to derived FPT; RMS comparisons partly circular | Run-to-failure bearing RUL; normalized FPT target implemented | 15 bearings, three settings |
| CWRU | Seeded bearing faults and sizes | Healthy-reference novelty; random windows from one recording are correlated | Unsupported: no lifecycle/failure time | Specimen across loads; load shift is not new-unit transfer |
| Paderborn | Artificial and naturally damaged bearings | Healthy versus damage; reserve target-healthy bearings | Unsupported: damage measurements are not a failure trajectory | Physical bearing |
| KAIST | Fault/severity under different loads | Healthy versus fault after target unlock | Unsupported: no run-to-failure trajectories | Repeated specimens; condition shift |
| PHM 2010 | Cutter/flute wear, only labeled cutters admitted | Would need a pre-registered healthy definition | Challenge is prognostic, but this repo currently implements wear transfer only; no invented failure endpoint | Cutter; local raw files absent |
| NASA milling | Sparse flank-wear regression | Not currently implemented | Incomplete sampled wear sequences need an explicit endpoint/censoring protocol | Case/insert set |
| Najafi thermal | Condition classification from pseudo-color images | Image novelty possible, not currently implemented | Unsupported: no lifecycle/failure timestamps | One motor; adjacent frames are correlated |
| MBHM | Aggregated bearing labels/text, optional research | Not currently implemented | No coherent lifecycle target supplied by the registry | Metadata lacks a reliable physical-unit join |

## Nonastreda conversion review and implementation

The [publisher record](https://data.mendeley.com/datasets/m892d2wtzh/1) supplies wear classes
and regression measurements, and lists RUL among possible experiments. It does not by
itself supply a measured mechanical-failure timestamp.

The separately reviewed RUL Model method is applicable as a *confirmed-wear* task:
choose the first dulled pass with at least two dulled labels in the next three recorded
passes; place the endpoint at its end. Confirmation uses future labels retrospectively.
The endpoint's confirmation time is not available to the predictor and must not be used
as an online alarm timestamp. Full-life smoothed wear is likewise a target/oracle, never
an ordinary predictive input.

The native conversion retains 512 passes across ten tools, with 132 duplicate groups
and 378 distinct force recordings. Earlier members keep photos and duration but their
signal is unavailable until the latest member. Cached legacy representatives may supply
identical signal features once that availability condition holds. This avoids rebuilding
life from the historical 378-pass deduplicated view.

Missing interior runs are estimated at four passes each, with each pass duration equal
to the median of preceding recorded pass durations. Leading missing operation is not
counted. Neither the synthetic photo pause nor inspection downtime contributes to RUL.
This is an estimated cutting clock, not an observed wall clock.

Both gap-aware and recorded-only conversions completed locally: **512 observations,
10 tools, 376 pass-end targets at/before the confirmed endpoint**, and no tools censored
under this endpoint rule. The largest difference is T4: endpoint about 3,657 s with
estimated missing operation versus 2,116 s from recorded passes only. Always report both.
Native ingestion preserves the earlier 378-pass view for historical wear comparisons.

## MATWI conversion: tested recorded-life proxy

The [publisher record](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi%3A10.48804%2FGK6LHH)
describes tool lives with paired sensors and images. The downloaded README documents
acquisition timestamps in the sixth sensor CSV column and acknowledges synchronization
errors. Wear annotations distinguish flank wear, adhesion, and combinations.

For tool recordings in acquisition order, let `d_i` be the measured inclusive recording
duration, `tau_i = sum(d_j, j <= i)`, and `RUL_i = tau_N - tau_i`. Predict at recording
end. Photographs must precede that recording's start. Wall-clock ordering determines
which photo is available; elapsed recorded operation determines the regression target.

Validation performed on all **1,700 recordings / 17 tools**:

- Streamed the roughly 9.88 GB of sensor CSVs during the feasibility audit to check sample
  counts and acquisition extents; the production converter needs only the first two and
  final timestamp rows.
- Unique recording IDs and chronological starts; no overlaps within a tool.
- Durations vary approximately 37-126 s, so one constant duration per cut is invalid.
- Nonnegative targets, exactly zero at each final recording, and adjacent target
  differences equal the next recording duration. Missing wear labels do not remove RUL.
- A fixed five-fold, unit-disjoint CPU baseline used training tools' mean lifetime minus
  elapsed operation. Mean per-tool MAE was **1,569.7 s**, versus **1,698.1 s** for constant
  training-mean RUL and **2,765.5 s** for zero RUL. This is conversion/evaluation feasibility,
  not neural model performance or validation of the physical failure endpoint.

Do not force a universal 110 micrometer endpoint: Sets 6, 7, and 9 never reach it, while
Set 15's first labeled value is already 120 micrometers. The revised reader uses pure
flank measurements for flank supervision and leaves multiple-insert wear unknown unless
explicit insert identity is available. Set 17's alternating photographs alone do not
establish that identity. All its recordings remain eligible for recorded-life RUL.

A small first/middle/last sensor sample found near-flat force channels in Set 3. This is
not a full quality diagnosis or a reason to discard the entire tool. Retain missing/quality
information and report sensor-only versus image/combined results; do not interpret sensor
acquisition defects as physical tool-failure ground truth.

## Other datasets: verified facts and limitations

**CWRU.** The [official dataset](https://engineering.case.edu/bearingdatacenter/welcome)
contains seeded faults under different loads, rather than failure histories. Local checks
confirm 40 records, file-variable/RPM quirks, and removal of the recorder roll-off shortcut.
Normal data and fault data have different native sampling rates; common-band processing
remains necessary. Healthy support/test windows from a shared recording are descriptive,
transductive novelty evaluation, not independent new-asset testing.

**XJTU-SY.** The [publisher page](https://biaowang.tech/xjtu-sy-bearing-datasets/)
documents 15 complete bearing lives, two accelerometers at 25.6 kHz, and minute-spaced
1.28 s snapshots. Local inventory and lifetimes pass acceptance. The audit found a reader
bug: files 1, 2, 10 were assigned minutes 1, 2, 3. The reader now retains actual numbered
minutes and window construction refuses incomplete ordinal timelines. FPT labels are
retrospective; RMS onset metrics are not independent evidence when RMS defines FPT.

**KAIST.** Local inventory, device offsets, and current spectra pass acceptance. The two
devices are not synchronous; features are paired by machine state, not exact sample time.
Two current channels are absent for all outer-race conditions and remain excluded from
model inputs. Absolute thermal levels identify sessions; slow channels remain relative.
No RUL endpoint or physical new-unit generalization is supported by these load splits.
[Publisher record](https://data.mendeley.com/datasets/ztmf3m7h5x/6).

**Paderborn.** The registry's old homepage returned 404; it now points to the
[current official data center](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter).
The official description supports bearing-condition measurements with vibration/current,
and documents non-commercial data terms. Synthetic format and bearing split tests pass;
raw files are absent locally, so channel rates/layout remain explicitly unverified on
real data. Source/target bearings and source/target healthy references stay separate.

**PHM 2010.** The [official challenge](https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/)
is prognostic, but a challenge title does not define the target for this implementation.
The reader admits wear labels only for c1/c4/c6 and rejects invalid wear schemas. Raw
files are absent locally: inventory, units, spindle assumptions and any new RUL endpoint
require real-file acceptance before claiming transfer results.

**NASA milling.** Local acceptance confirms 167 runs across 16 cases with 21 missing
wear measurements. The six channels are sampled at 250 Hz and include RMS-processed
vibration/acoustic envelopes. They cannot support high-frequency bearing spectra or
bearing-geometry physics tests. Case-disjoint sparse wear regression is the supported use.

**Najafi.** Local image inventory passes acceptance. It is one motor with 11 conditions,
not eleven independent machines. Random frame splits can exploit acquisition similarity.
There are no paired high-rate sensors or validated life endpoints.
[Publisher record](https://data.mendeley.com/datasets/m4sbt8hbvk/3).

**MBHM.** Read-only inspection of the local SQLite metadata found `condition`, `file_info`,
and `label_note`. Condition fields include dataset, bearing-model code, channel, rpm and
load; they do not identify the physical bearing or lifecycle. It includes CWRU, PU and
XJTU-derived windows, so mixing it into pretraining could include held-out machines.
Keep it outside admitted source training until an auditable per-unit/source-record join
exists. Aggregation does not turn those windows into independent data.
[Publisher metadata](https://huggingface.co/datasets/SIA-IDE/MBHM/blob/main/metadata.sqlite).

## Local acceptance status

The broader opt-in real-data run completed **18 passed, 4 skipped** before the store-count
check was split into individual dataset cases. Skips were the absent Paderborn and PHM
2010 raw files. The changed MATWI window store must be rebuilt; a stale store is rejected,
not counted as acceptance of the new reader. No GPU or target model score was used.

## EDA cross-check

The seven original summaries and nine figures in [eda/](eda/README.md) were reviewed
against the changes. They support retaining force-level descriptors, the common bearing
band limit, relative slow channels, and the strong acquisition/schedule baselines. The
Nonastreda summary is a historical 378-pass view, not the new 512-pass lifecycle. XJTU's
headline FPT fields use an earlier short rule: selected-rule fields and the implemented
kurtosis fallback must be used instead. Historical KAIST EDA already reports labeled
baselines; future reports must disclose that exposure rather than claiming never-inspected
target labels. No new target fit was performed during this audit.

New MATWI and Nonastreda lifecycle summaries were added without replacing old evidence.
MATWI currently supplies **1,052 pure-flank supervised recording labels**; all 1,700
recordings retain recorded-life RUL labels independently of wear-type availability.
