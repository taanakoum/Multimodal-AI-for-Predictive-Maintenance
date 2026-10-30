# Project 14 — Multimodal PdM Foundation Model: Implementation Plan

Oct 23, 2026 · @Mohammad Taan Al Akoum

> **Amendments since this copy** (reasons in `PROJECT_LOG.md`, decision log; evidence in
> `docs/EDA_REPORT.md`):
>
> - Every sensor channel is band-limited to a common 5 kHz after resampling, and the E arm's
>   band is 1-5 kHz (was 1-5.5 kHz). Recorder roll-offs above 5 kHz identify the source and,
>   on CWRU, the class.
> - The E arm's shaft speed is measured from the vibration of each recording (nominal for
>   XJTU-SY's short snapshots).
> - XJTU-SY RUL labels use an FPT whose healthy reference is the first 10 % of life (at least
>   10 snapshots), with 3σ and 10 consecutive exceedances, and a kurtosis fallback. The
>   3-snapshot rule raised false alarms at 0.6 % of life.
> - Window shards are float32, not float16: float16 rounding is white noise up to Nyquist and
>   would undo the band limit.
> - The own PatchTST is pretrained once per outer fold, so Gate 1 never scores a model on
>   units it was pretrained on.
> - Gate 1 drops arms a dataset lacks (E without bearings, S below 12 kHz). A candidate left
>   with none takes the worst rank on that task.
> - Nonastreda passes are separated by a photo pause on each tool's clock, so every window of
>   a pass, its first included, receives the previous pass's photos.
> - The `Sample` record below is realized as the window store (`pdmfm.windows`): one
>   manifest row per window, with signals, levels and slow summaries as arrays.
>   `pdmfm.schema` keeps only `ImageItem` and the as-of join.
> - Level tokens (tested 2026-10-29, after the code audit) stay as planned: four
>   descriptors relative to the healthy baseline. Two additions were tried on the Nonastreda
>   inner-validation tools and remain options only. The force channels' window mean
>   (`--force-mean`; Fx and Fy means follow flank wear, Spearman ±0.67, not the recording
>   block) lowered the force expert (macro-F1 0.54 against 0.66). Absolute descriptors
>   (`--levels both`, which the EDA suggested) made no difference. Level and slow
>   descriptors are standardized with training-row statistics, which is neutral. Slow
>   channels are always relative (absolute temperature identifies the session, EDA F3).
> - Wear and RUL intervals are calibrated like the classes: one scale factor on the
>   predictive scale, fitted by NLL on the inner-validation units (decision D, 2026-10-29).
> - (2026-10-30) Photo age is counted in passes; the stale-photo test (photos 2 and 3
>   passes old) is part of every Nonastreda table. Photo-lag and augmentation training,
>   modality dropout in Plan B, a free linear head and a low-rank W_t were tested and stay
>   options (`PROJECT_LOG.md`, week 5).
> - (2026-10-30) RUL is also scored after the first prediction time, where the prognosis
>   matters. Results are pooled out of fold with 95 % intervals from resampling test units
>   (the plan's "bootstrap CIs"); seeds are averaged, not counted as independent data.
>   Hyperparameters are searched on the inner-validation units only (`pdmfm.tune`).
> - (2026-10-30) The KIT CNC set named under "Facts" holds no tool pictures and no wear
>   labels (checked on its record). The paired image-sensor milling set is MATWI
>   (KU Leuven; 17 tools; decision E).
> - Plan B (stage 4): one expert per channel family (force, vibration, current), one for the
>   slow channels and one for the photos. Context text is not an expert: it never names the
>   state, so its opinion would be vacuous. Persistence is the previous bag's true label (or
>   wear) of the same unit, with the training majority (or mean) for a unit's first bag; it
>   assumes the last inspection is known, so it is a strong reference, not a deployable model.

## Summary

The plan builds a multimodal PdM model from frozen pretrained encoders: a sensor foundation model, DINOv2 for images and Qwen3-Embedding for text. Their outputs are cached as tokens and fused by a small transformer picked through a fusion search. Evidential heads, whose classes are text descriptions of faults, give a prediction plus an uncertainty. The fusion stage is pretrained with masked-token modeling, cross-modal contrastive learning and text anchoring on four source datasets. It is then adapted to three held-out datasets with few-shot linear probes and LoRA. Schedule: 13 weeks plus 1 buffer week, from 2026-10-26 to 2027-01-31.

| Decision | Choice |
| --- | --- |
| Direction | Plan A. Plan B (one expert per modality, combined by Dempster-Shafer) is built first as baseline and fallback |
| Data | Public data only; rotating machinery and milling |
| Source datasets (pretraining and model selection) | CWRU, Paderborn artificial damage, Nonastreda, XJTU-SY; Najafi thermal and MBHM only if their week-1 checks pass |
| Held-out targets (never seen before final runs) | KAIST, Paderborn real damage, PHM 2010 |
| Tasks | Fault class and severity; wear class and flank wear (µm); RUL; anomaly detection from healthy data only |
| Code | New repository, with `omnifuser` installed as an editable dependency; no OmniFuser GPU runs |
| Explanations | Template cards; a small Qwen model may only rephrase them |

What the audit changed (details in the audit section):

1. Model choices (Gates 1 and 2) use only validation units of the source datasets. The three targets are no longer used to pick anything.
2. Signals are resampled to one rate per family: 12 kHz for vibration, current and acoustic emission (AE), and the native 1 kHz for Nonastreda force. Every resampling step is anti-aliased.
3. Signal level is put back as an input. The sensor encoders normalize each input and so discard energy, so each channel gets level tokens measured against the machine's own healthy baseline.
4. LoRA targets the sensor encoder, where the parameters are, not the 3 M-parameter fusion stage.
5. Pretraining is described honestly: it is self-supervised (masked tokens, contrastive learning) plus supervised text anchoring on source labels.
6. Cross-validation uses five unit-grouped outer folds shared by all source datasets, with one pretraining run per fold.
7. Anomaly detection (with false-alarm rates) is added as a task. TimesFM is added as a forecasting baseline for RUL. AnomalyGPT is not adopted.
8. The physics check on explanations now uses frequency-band occlusion, so it works with any encoder.

## Context, goals and constraints

OmniFuser (force signals plus tool photos, Nonastreda) is implemented and passes its 137 tests on CPU. Only smoke tests have run; there are no GPU results. It cannot meet Project 14 by itself: it is trained from scratch on 378 passes of one process, with no pretraining, no text, no transfer and no XAI. Project 14 has three goals:

1. **Architecture.** A unified backbone with self-supervised pretraining that fuses high-frequency sensors with low-frequency images and text.
2. **Generalization.** Few-shot learning and PEFT on unseen assets, for fault classification and RUL.
3. **Trust.** Calibrated uncertainty and faithful, human-readable explanations of cross-modal attention.

These data facts shape every option:

| Fact | Consequence |
| --- | --- |
| No public dataset pairs vibration, thermal images and maintenance text on the same machine (searched Mendeley, IEEE DataPort, AssetOpsBench, MaintNet) | The corpus is modality-incomplete: each dataset brings what it has, so fusion must accept any subset of modalities |
| Sensor signals paired with photos exist only for milling: Nonastreda (force with tool, chip and workpiece photos) | Image↔sensor fusion is shown on milling. Thermal sets (Najafi motor) are unpaired and train only the image pathway and the label space |
| Real logs paired with sensors: only AssetOpsBench (chillers, 15-minute data). MaintNet is text only | Text has three honest roles: machine and operating context, fault-description prototypes (a shared label space), and domain knowledge. VLT-style prompts built from signal statistics carry no new information and are kept only as an ablation |
| Rates from 1 kHz to 64 kHz, 1 to 7 channels per dataset, photos once per pass, text once per machine | One tokenizer per signal family, plus an as-of join that attaches each slow item with its age |
| Known leaks: CWRU and KAIST use one specimen per fault condition; Nonastreda has a post-pass photo, a run-number shortcut and Fz drift | Splits by physical unit where units exist, otherwise by operating condition, stated as such. Every foundation claim is tested on held-out datasets |

## Options considered and ranking

Plan A ranks first because it is the only option that covers all three goals within 13 weeks. Scores run from 1 to 5, where 5 is best; for risk, lower is better.

| Rank | Option | Feasibility | Goals covered | Novelty | Risk | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **A.** Frozen encoder bank → cached tokens → fusion search → fusion pretraining → evidential heads | 4 | 3 of 3 | 4 | Medium | Chosen |
| 2 | **B.** One expert per modality, combined by Dempster-Shafer and inverse-variance weighting | 5 | 2 of 3 (no cross-modal attention to explain) | 2 | Low | Built in week 5 as baseline and fallback |
| 3 | **E.** OmniFuser C2F with pretrained encoders, extended to N modalities | 4 | 2 of 3 (two-stream design, milling only) | 2 | Low | One fusion candidate inside A |
| 4 | **D.** Signals rendered as images into one vision-language model (VLT, Time-VLM, VisionTS style) | 3 | 2 of 3 | 2 | Medium | Loses phase and fine timing; at most one ablation |
| 5 | **C.** Qwen as the fusion backbone with LoRA (BearLLM, ChatTS style) | 2 | 3 of 3 | 3 | High | Needs generated instruction data, is weak at numeric targets, and its made-up statements conflict with trust |

Why A wins:

- Frozen encoders make features cacheable, so each fusion search or pretraining run takes minutes to hours, not days.
- Masks handle missing modalities, which the corpus requires.
- Cross-modal attention gives Goal 3 something real to explain.
- B's evidential heads add per-modality trust at little extra cost.

## Audit of the earlier plan: gaps and fixes

The audit found 24 gaps; the first five would have biased the results. Each fix is part of the design in the sections below.

| # | Area | Gap or lapse | Fix |
| --- | --- | --- | --- |
| 1 | Selection | The encoder probe and fusion search chose on "validation units" without saying which datasets. Choosing on KAIST or PHM 2010 would leak the targets | Both gates use only inner-validation units of the source datasets. The targets are untouched until the final runs, and gate rules are fixed in advance |
| 2 | Resampling | 12.8 kHz upsamples CWRU's 12 kHz. Nonastreda's 1 kHz force cannot be brought to kHz rates. No anti-aliasing was specified | Two signal families: 12 kHz for vibration, current and AE; the native rate for low-rate force. Polyphase resampling with an anti-alias filter |
| 3 | Signal level | MOMENT and Mantis normalize each input, which removes absolute energy, the main sign of wear and faults (OmniFuser F6 had the same flaw) | Level tokens per channel: log RMS, log peak, kurtosis and crest factor, each relative to the machine's healthy baseline |
| 4 | Slow channels | Temperature was to go through MOMENT, but a 1 s window of temperature is constant | Slow-channel token from per-window mean and slope, relative to the baseline |
| 5 | Cross-validation | Pretraining must never see in-domain test units, but no nested structure was defined | Five unit-grouped outer folds shared by all source datasets, one pretraining run per fold |
| 6 | Labels | Nonastreda labels a whole pass (about 12 windows) and XJTU-SY labels a snapshot; how window outputs become pass predictions was undefined | Bag aggregation: average evidence (classes) and average NIG parameters (wear, RUL) over a pass's windows |
| 7 | RUL | No RUL target was defined | Normalized RUL from a causal first-prediction time, with RMSE, MAE and the PHM 2012 score |
| 8 | Masked modeling | Perceiver outputs latents, not tokens, so masked tokens had nowhere to be predicted | A one-layer cross-attention decoder (Perceiver IO style) queried by each masked token's position and modality |
| 9 | Contrastive | Windows from the same recording served as negatives (false negatives) | Negatives come only from other units |
| 10 | Contrastive | Only two pair types were listed | Four pair types: Paderborn vibration↔current, CWRU drive-end↔fan-end, XJTU-SY horizontal↔vertical, Nonastreda force↔photo |
| 11 | Wording | Pretraining was called self-supervised, but text anchoring uses labels | Called multi-task pretraining (self-supervised plus supervised), with an ablation that drops the supervised term |
| 12 | Zero-shot | A learnable text projection over about 60 classes overfits and breaks zero-shot | Text tower and prototypes frozen; only the sensor-side projection is learned; 3-5 paraphrases per class are averaged |
| 13 | Uncertainty | Dempster-Shafer assumes independent sources, but the fused head shares inputs with the per-modality heads, so the combined result is overconfident | ETMC-style combination plus temperature scaling. ECE is reported before and after, with an ablation without the fused opinion |
| 14 | PEFT | LoRA on a 3 M-parameter fusion stage gains nothing over full fine-tuning | LoRA (r = 8) on the sensor encoder's last 4 blocks, with online encoding of the small support sets |
| 15 | Few-shot | k-shot support sets leave no validation data for early stopping | Fixed adaptation recipes, chosen on simulated few-shot tasks from source datasets |
| 16 | Normalization | Training statistics do not exist for an unseen machine | Each encoder's own per-input normalization, plus level tokens relative to the target's healthy support windows or the unit's first windows |
| 17 | Shortcuts | Sensor-slot and rate embeddings could encode which dataset a sample comes from | A generic slot vocabulary (axis, placement) and no dataset ID. A probe reports how well the dataset can be predicted from z |
| 18 | Shortcuts | Context text could correlate with labels | Automatic check: a classifier on the context token alone must stay at chance for the label within each dataset |
| 19 | KAIST | Acoustic data exists only for some faults at 0 Nm, so it is absent from the 4 Nm test | Acoustic dropped for KAIST |
| 20 | Images | Nonastreda photos are 1550×500, while DINOv2 needs sides that are multiples of 14 | Resize to 224×672 (16×48 patches) |
| 21 | Scope | Anomaly detection and false alarms appear in the Project 14 context but not in the plan | Healthy-only anomaly detection with false-alarm rate at 95 % detection (see the anomaly section) |
| 22 | XAI | The physics score needed spectrum tokens | Band-occlusion attribution, which works with any encoder |
| 23 | Honesty | The fusion corpus is about 20k source windows (about 30k with the targets), small for a "foundation" claim | State that foundation capacity comes from the pretrained encoders; MBHM is an optional extra corpus for encoder pretraining |
| 24 | Licences | Paderborn is CC BY-NC, so weights trained on it inherit the non-commercial terms | Release weights under NC terms, or a variant trained without Paderborn |

## Datasets and preprocessing

All high-rate signals are anti-aliased and resampled to 12 kHz and cut into 1.024 s windows (12,288 samples). Nonastreda force stays at its native 1 kHz, in 8.192 s windows. Slow channels become one summary token per window.

| Dataset (role) | Signals used, native rate → model rate | Unit; labels | Window; label rule | Specific steps |
| --- | --- | --- | --- | --- |
| Nonastreda (source; CC BY 4.0) | Fx, Fy, Fz at 1 kHz (Kistler 9257B), kept at 1 kHz; photos of the previous pass: tool, chip, workpiece | 10 tools, 378 passes after dedup; wear class (sharp < 71 µm ≤ used < 110 µm ≤ dulled) and flank wear in µm | 8.192 s, about 12 per pass; one pass = one bag | Dedup and trimming via `omnifuser.data.load_nonastreda(pool=1)`. No Fz level tokens (sensor drift); no run-number input. Baseline = the tool's first pass. A tool's first pass has no photo, so the photo modality is masked |
| CWRU (source) | Drive-end and fan-end accelerometers at 12 kHz; normal data at 48 kHz → 12 kHz | One specimen per condition; 10 classes (normal; inner race, outer race and ball at 0.007, 0.014 and 0.021 in) | 1.024 s, 50 % overlap in training | Speed comes from each file's rpm field. Baseline = the normal recording at the same load. Used for pretraining only, with no in-domain claim |
| Paderborn, artificial damage + healthy (source; CC BY-NC 4.0) | 1 vibration channel and 2 motor currents at 64 kHz → 12 kHz; temperature → slow token; speed, torque and radial load → context text | 6 healthy and 12 artificially damaged bearings; healthy, outer race, inner race | 1.024 s; 3 windows per 4 s recording | Four operating settings named in the file code (e.g. N15\_M07\_F10; the meaning of each part to be checked in the KAt docs). Baseline = healthy training bearings at the same setting. Auxiliary-channel rates to check in the measurement log |
| XJTU-SY (source) | Horizontal and vertical accelerometers at 25.6 kHz → 12 kHz; one 1.28 s snapshot per minute | 15 LDK UER204 bearings under 3 conditions (2100 rpm / 12 kN, 2250 rpm / 11 kN, 2400 rpm / 10 kN); normalized RUL | First 1.024 s of each snapshot; one snapshot = one sample | RUL target defined below. Baseline = the bearing's first 10 snapshots |
| KAIST (target; CC BY 4.0) | 4 housing accelerometers and 3 motor currents at 25.6 kHz → 12 kHz; 2 thermocouples → slow token; acoustic dropped | One specimen per condition; 15 classes (normal; inner race and outer race at 0.3, 1.0 and 3.0 mm; misalignment at 0.1, 0.3 and 0.5 mm; unbalance at 5 masses) | 1.024 s; 60-120 s per file | NSK 6205 bearing at 3010 rpm; loads 0, 2 and 4 Nm. Mixed MAT and TDMS formats (`nptdms`). Baseline = normal windows in the support set |
| Paderborn, real damage (target) | As Paderborn above | 14 bearings with real damage plus 3 held-out healthy bearings | As above | Healthy bearings K001-K006 split 3 source / 3 target |
| PHM 2010 (target) | Fx, Fy, Fz (N) and Vx, Vy, Vz (g) at 50 kHz → 12 kHz; AE-RMS → slow token | 6 cutters, about 315 cuts each; flank wear for c1, c4 and c6 (maximum over 3 flutes, in µm) | 1.024 s; one cut = one bag | Unlabelled cutters c2, c3 and c5 are used neither in pretraining nor in testing. Baseline = the cutter's first 5 cuts |
| Najafi thermal (source, optional; CC BY 4.0) | Thermal images → 224×224 | One motor, 11 conditions | One image | Random split; image pathway and label space only, never a transfer claim |

Steps applied to every high-rate channel:

1. **Load and convert units:** acceleration in g, force in N, current in A, temperature in °C. Keep the metadata: speed, load and bearing geometry.
2. **Anti-alias and resample** with `scipy.signal.resample_poly` (Kaiser FIR). The up/down ratio is exact: 64 kHz → 12 kHz is 3/16; 25.6 kHz → 12 kHz is 15/32; 50 kHz → 12 kHz is 6/25.
3. **Window** without overlap at evaluation. Training uses 50 % overlap only for small sources (CWRU, Nonastreda).
4. **Per-window inputs for the encoders:**
   - **T (time):** 24 segments of 512 samples at 12 kHz (16 segments for the 8.192 s force windows).
   - **F (spectrum):** log magnitude of the FFT of the window, averaged into 512 bins, with Hann tapering.
   - **E (envelope):** for bearings only, the envelope spectrum on an axis of shaft orders. Steps: band-pass 1.0-5.5 kHz, Hilbert envelope, FFT, then resampling of 0-25 shaft orders into 512 bins. The fixed band uses no labels.
5. **Level descriptors:** log RMS, log peak, kurtosis and crest factor, each minus the baseline's median. The baseline is the machine's or unit's healthy reference, computed causally or from the support set (BearLLM's reference-signal idea).
6. **Slow channels:** per-window mean and slope, minus the baseline.
7. **Storage:** resampled windows in float16 shards per dataset (about 24 KB per channel-window; about 4 GB in total), with a manifest of unit, condition, label, time and source file.

RUL target for XJTU-SY:

- The first prediction time (FPT) is the first snapshot whose RMS exceeds the baseline mean + 3σ for 3 consecutive snapshots. It uses only past snapshots, so it is causal.
- Before the FPT, RUL = 1. After it, RUL falls linearly to 0 at failure.
- Metrics: RMSE, MAE and the PHM 2012 score.

## Unified sample format and time alignment

Every dataset is converted into one `Sample` record. Channels from the same acquisition device share a window. Slow items (photos, slow channels, text) are attached by an as-of join with their age. An absent modality is an empty token set with a false mask, never zeros.

```python
@dataclass(frozen=True)
class ChannelWindow:
    family: str        # vibration | force | current | ae | slow
    axis: str          # x | y | z | radial | none
    placement: str     # drive_end | fan_end | housing | spindle | table | motor_phase
    rate_hz: float     # model rate after resampling (12_000 or 1_000)
    signal: np.ndarray # (L,) float16, resampled
    level: np.ndarray  # (4,) log RMS, log peak, kurtosis, crest; minus the baseline

@dataclass(frozen=True)
class ImageItem:
    kind: str          # tool | chip | workpiece | thermal
    path: str
    captured_at: float # seconds on the unit's clock

@dataclass(frozen=True)
class Sample:
    dataset: str; unit: str; bag: str     # unit = tool, bearing, cutter; bag = pass, cut, recording
    t_start: float; t_end: float
    channels: tuple[ChannelWindow, ...]
    images: tuple[ImageItem, ...]        # only items with captured_at < t_start
    context_text: str                    # machine, sensors, speed, load, material; never the state
    condition: dict                      # rpm, load, geometry (for the order axis)
    labels: dict                         # class, wear_um, rul (None when absent)
    label_space: str                     # which class prototypes apply, e.g. "kaist15"
```

Alignment rules:

| Case | Rule |
| --- | --- |
| Channels on one acquisition device (Paderborn vibration + currents; PHM 2010's 7 channels; XJTU-SY H + V; CWRU DE + FE) | Same window boundaries, sample-synchronous |
| Channels on two devices (KAIST: vibration on a Siemens DAQ, current and temperature on an NI DAQ) | Synchronization not guaranteed: windows align by recording and elapsed time only. KAIST is a target, so no contrastive pair depends on it |
| Photos taken after each pass (Nonastreda) | The as-of join attaches the newest photo captured before the window starts, i.e. the previous pass's photo. Age = time since capture plus a pass counter |
| Slow channels (temperature, AE-RMS) | Summary over the same window (mean, slope) |
| Context text | One per unit and operating condition; age = 0 |

Why the joins are causal: Nonastreda photos taken after the pass show the label (OmniFuser finding F1). The as-of rule makes that leak impossible by construction, and a unit test checks that `captured_at < t_start` for every attached item.

Text written once per dataset (stored in `pdmfm/texts/*.yaml`):

- **Context**, e.g. "Rotating machine test rig. Deep-groove ball bearing NSK 6205 at 3010 rpm under a 4 Nm load. Sensors: four housing accelerometers, three motor-phase current transformers, two thermocouples."
- **Fault descriptions** (3-5 paraphrases per class), e.g. "Outer-race defect of 1.0 mm on a deep-groove ball bearing." They never mention speed or load, so no class can be matched through the context text.

## Model architecture

The model is about 3.5 M trainable parameters on top of frozen encoders. Each sample becomes at most 96 tokens of width 256, which a fusion stage reduces to one vector z. Text-anchored evidential heads read z and a summary vector per modality. Shapes below are per sample; B is the batch size and d = 256.

&#91;embedded content: Model architecture · 4 input types, 1 fusion stage, evidential heads\]

The per-modality heads read summaries built before fusion, so their opinions stay independent. The combination step turns those opinions into a trust value per modality and a conflict score.

| Block | Input → output | Trainable | Details |
| --- | --- | --- | --- |
| Sensor encoder (Gate 1 picks one: Mantis-8M, MOMENT-base, own PatchTST) | per channel: T (24, 512) → (24, dₑ), mean over groups of 4 → (6, dₑ); F or E (512) → (1, dₑ) | No; LoRA when adapting | dₑ = 256 (Mantis), 768 (MOMENT-base), 128 (PatchTST). Nonastreda force: T (16, 512) → (4, dₑ) |
| Level token | (4) → (1, d) | Yes | MLP 4 → 64 → d. Restores the absolute level that the encoders normalize away |
| Slow token | (2) per slow channel → (1, d) | Yes | MLP of mean and slope minus the baseline |
| Image encoder: DINOv2-S/14 | photo 224×672 → class token + patch grid pooled to 2×4 = (9, 384); thermal 224×224 → (5, 384) | No | At most 3 photos per sample (tool, chip, workpiece) |
| Text encoder: Qwen3-Embedding-0.6B | context sentence → (1, 1024) | No | Fault-description prototypes P (C × 1024) come from the same model, as the mean of 3-5 paraphrases, L2-normalized |
| Adapters, one per encoder family | (n, dₑ) → (n, d) | Yes | LayerNorm then Linear. Added embeddings: modality (10 kinds), axis (5), placement (8), token role (7), time of the segment centre (sinusoidal, in seconds), age (log seconds plus a pass counter), rate bucket. No dataset ID |
| Token set | ≤ 96 tokens → padded (96, d) + key mask | - | ≤ 8 channels × 8 tokens (6 time, 1 spectral, 1 level), ≤ 4 slow, ≤ 27 image, 1 text |
| Fusion (default: Perceiver resampler; Gate 2 decides) | (96, d) + mask → 32 latents (32, d) → z (d) by mean | Yes, about 3.1 M | 2 blocks of \[cross-attention latents←tokens, then 2 self-attention layers\]; 8 heads, feed-forward 2d, pre-LN, dropout 0.1 |
| Modality summaries | each modality's adapter tokens → zₘ (d) | Yes | Attention pooling with one learned query per modality, built from adapter tokens only, so each modality's opinion is independent of the others |
| Class head (evidential, text-anchored) | z → q = normalize(W\_q z), W\_q: d → 1024; logits = s·qPᵀ + b; evidence e = softplus(logits); α = e + 1 | Yes (W\_q, s, b); P frozen | One head for z and one per modality on zₘ. Classes are restricted to the sample's label space |
| Regression heads (NIG), one per target (wear µm, normalized RUL) | z → (γ, ν, α, β) | Yes | MLP d → 128 → 4 with softplus constraints; per-modality heads fused as in MoNIG |
| Bag aggregation | window outputs → pass, cut or recording output | - | Mean evidence (classes), mean NIG parameters (regression) |
| Pretraining-only heads | masked decoder: query embeddings → 1 cross-attention on latents → dₑ; contrastive projector: d → 256 → 128 | Yes, discarded after pretraining | The decoder predicts the frozen encoder feature of each masked token |

Evidence becomes a belief bₖ = eₖ/S and an uncertainty u = K/S, with S = Σₖ αₖ over the K classes. Two opinions combine by Dempster's rule, with conflict C = Σ\_{i≠j} b¹\_i b²\_j:

```latex
b_k=\frac{b^1_k b^2_k+b^1_k u^2+b^2_k u^1}{1-C},\qquad u=\frac{u^1u^2}{1-C}
```

The combined opinion is the fused head's opinion combined with each modality's opinion in turn (ETMC style). Per-modality trust is 1 − uₘ. Conflict C is reported because it flags modalities that disagree.

Losses on labelled windows. The classification loss is TMC's: an expected cross-entropy under the Dirichlet, plus an annealed KL term with λ\_t = min(1, epoch/10). The regression loss is Amini et al.'s NIG loss with λ\_R = 0.01.

```latex
\mathcal{L}_{cls}=\sum_k y_k\,(\psi(S)-\psi(\alpha_k))+\lambda_t\,\mathrm{KL}\!\left(\mathrm{Dir}(\tilde\alpha)\,\|\,\mathrm{Dir}(\mathbf{1})\right)
```

```latex
\mathcal{L}_{reg}=\mathrm{NLL}_{NIG}(y;\gamma,\nu,\alpha,\beta)+\lambda_R\,|y-\gamma|\,(2\nu+\alpha)
```

The supervised loss L\_sup applies L\_cls to the fused head, to each modality head and to the combined opinion, and adds L\_reg where a regression target exists.

## Self-supervised pretraining and modality alignment

Pretraining happens at three levels. Only the last one is new work, and it runs on cached features in about 1 A100-hour per run. Every level excludes all target datasets and each outer fold's test units. The fold's inner-validation units are also excluded, because they choose the checkpoint.

| Level | What is trained | Data | Objective | Cost |
| --- | --- | --- | --- | --- |
| 0. Inherited | Nothing; the public checkpoints are used as they are | MOMENT: Time-series Pile; Mantis: about 7 M series; DINOv2: LVD-142M images; Qwen3-Embedding: web text | As published | 0 |
| 1. Sensor encoder (optional; used if Gate 1 picks it) | Own PatchTST: d = 128, 3 layers, 16 heads, patch 16 without overlap, RevIN | Vibration channel-windows of source training units, cut into 512-sample inputs; MBHM only if its metadata lets held-out bearings be removed | Masked-patch reconstruction (PatchTST §3.2): mask 40 % of patches, MSE on the masked ones | About 2-3 A100-hours |
| 2. Fusion stage (new) | Adapters, level and slow MLPs, fusion stage, heads | All source training units, about 20k windows | L = L\_mask + 0.5 L\_nce + L\_sup | About 1 A100-hour × 15 runs (5 folds × 3 seeds) |

The three losses of level 2:

- **Masked cross-modal modeling (L\_mask, self-supervised).**
  - Per sample, drop each present modality with p = 0.3, always keeping one.
  - Mask 40 % of the remaining tokens: their content becomes a learned \[MASK\] vector, while their position and type embeddings stay.
  - The pretraining decoder predicts each hidden token's frozen, LayerNorm-ed encoder feature (loss 1 − cosine), and the raw descriptors of level and slow tokens (MSE).
  - A dropped modality must be predicted from the others, which forces cross-modal information into the latents. The targets are frozen, so the model cannot collapse to a trivial solution.
- **Cross-modal contrastive alignment (L\_nce, self-supervised).**
  - The fusion stage runs twice: once on modality group A only, once on group B only.
  - The two pooled vectors go through the projector, into a symmetric InfoNCE with temperature τ learned from 0.07.
  - Negatives are other samples in the batch from *other units*, so recordings of the same unit are never false negatives.
  - Pairs: Paderborn vibration ↔ motor currents; CWRU drive-end ↔ fan-end; XJTU-SY horizontal ↔ vertical; Nonastreda force ↔ previous-pass photos.
- **Supervised text anchoring (L\_sup).** L\_cls and L\_reg from the architecture section, on source labels only. Because labels are used, the pretraining is called multi-task, and one ablation removes L\_sup.

Training recipe for level 2:

- AdamW, learning rate 3e-4, weight decay 0.05, cosine schedule with 5 % warmup, 100 epochs, batch 256, bf16.
- Datasets are sampled in proportion to n\_d^0.5 (n\_d = number of samples in dataset d), and units are balanced within a dataset.
- Checkpoint = best inner-validation macro-F1 averaged with normalized RMSE.
- A collapse monitor tracks the per-dimension standard deviation of z.

Modality alignment works on five levels:

| Level | Mechanism | Check |
| --- | --- | --- |
| Time | As-of join and shared windows (previous section) | Unit test: no item captured after the window start |
| Scale and units | Resampling into families; level tokens relative to the machine's own healthy baseline | Level distributions of healthy windows overlap across datasets |
| Representation | Adapters into one 256-d space with type embeddings; L\_nce; cross-modal prediction in L\_mask | Cross-modal retrieval Recall@1 on validation units; alignment and uniformity (Wang & Isola 2020) |
| Semantics | Every dataset's classes as frozen Qwen prototypes; z projected into the same text space | Zero-shot accuracy on a held-out fault type |
| Shared vs private | C2F-N keeps OmniFuser's shared/private split; in the other candidates, the modality summaries zₘ are the private paths | Drop in accuracy when each modality is removed |

Ablations (on the outer folds): no L\_mask; no L\_nce; no L\_sup (pure self-supervised, then a linear probe); no level tokens; VLT-style statistics prompts instead of context text.

## End-to-end pipeline and training strategy

The pipeline has 12 stages. All stages after stage 2 read cached features, except LoRA, full fine-tuning and band-occlusion explanations, which re-encode small sets. The whole plan needs about 110 A100-hours, about 28 hours of wall time on Ruche's 4 GPUs.

&#91;embedded content: Pipeline · source lane per outer fold, target lane locked until stage 8\]

Only the pretrained backbones cross from the source lane to the target lane. Both gates and all checkpoints are decided before any target result exists.

| # | Stage | Input → output | Recipe | A100-h |
| --- | --- | --- | --- | --- |
| 0 | Ingest | Downloads → `raw/` with SHA-256 checksums and licence notes | Week-1 checklist | 0 |
| 1 | Preprocess | Raw files → window shards and a manifest | Datasets section | CPU |
| 2 | Encode and cache | Windows, photos, texts → feature store (float16 shards) | Each frozen encoder runs once; the cache key is encoder name + checkpoint hash + preprocessing signature; a test compares cached and online features | 2 |
| 3 | Gate 1: encoder probe | Cached features → chosen sensor encoder and input arms | Logistic or ridge regression on 4 source tasks; see the evaluation section | CPU (+3 if own PatchTST is pretrained) |
| 4 | Plan B baseline | Per-modality features → per-modality evidential heads, combined by Dempster-Shafer and MoNIG | L\_sup; AdamW 1e-3; 50 epochs; 5 folds × 3 seeds | 5 |
| 5 | Gate 2: fusion search | 5 candidates → chosen fusion stage | L\_sup with modality dropout 0.3; AdamW 3e-4; 50 epochs; 5 folds × 3 seeds per candidate | 37 |
| 6 | Fusion pretraining | Chosen design → 5 fold backbones (× 3 seeds) | Level 2 of the pretraining section | 15 |
| 7 | In-domain evaluation | Backbones → scores on each fold's test units (Nonastreda, Paderborn artificial, XJTU-SY) | Plus Nonastreda 10-fold leave-one-tool-out, fine-tuned from a backbone pretrained without Nonastreda | 3 |
| 8 | Adaptation to targets | Backbones → KAIST, Paderborn real and PHM 2010 results | Zero-shot, linear probe, LoRA, full fine-tuning; k ∈ {1, 5, 10}; 5 support draws (table below) | 35 |
| 9 | Calibration | Logits → calibrated probabilities | Temperature scaling on inner-validation units in-domain, and on the support set for targets when k ≥ 5 | <1 |
| 10 | Anomaly and RUL baselines | Healthy support sets, health indicators → AUROC, false-alarm rates, TimesFM RUL | Anomaly section | 5 |
| 11 | Explanations | Test windows → attributions, physics score, cards | Trust section | 5 |
| 12 | Report | All results → tables, significance tests, figures | Wilcoxon tests and bootstrap intervals with Holm correction | CPU |

Adaptation recipes. These are fixed before any target is touched, chosen once on simulated few-shot tasks: Paderborn artificial-damage validation bearings act as a pseudo-target.

| Method | What trains | Learning rate | Steps (batch 32) | Encoding |
| --- | --- | --- | --- | --- |
| Zero-shot | Nothing; target class prototypes only | - | 0 | Cached |
| Linear probe | W\_q, s, b and the NIG head | 1e-3 | 300 | Cached |
| LoRA | r = 8, α = 16 on the q and v projections of the sensor encoder's last 4 blocks; adapters and fusion fully trained | 1e-4 (heads 1e-3) | 300 | Online (support sets of at most 150 windows) |
| Full fine-tuning | Everything except the text tower and prototypes | 3e-5 encoder, 1e-4 rest | 300 | Online |

The image encoder stays frozen during adaptation: every target is sensor-only, and Nonastreda is in-domain. There is no early stopping, because k-shot support sets leave no validation data. The fixed step count was chosen on the pseudo-target.

## Cross-validation and evaluation protocol

Evaluation is nested in three layers:

- **Outer folds:** five unit-grouped folds shared by all source datasets measure in-domain results.
- **Inner validation:** units inside each fold's training part make every gate and checkpoint decision.
- **Held-out datasets:** three targets, touched only by the final adaptation runs, measure transfer.

The gate rules below are fixed before any result is seen.

| Source dataset | Fold k test units | Fold k inner validation | Notes |
| --- | --- | --- | --- |
| Nonastreda | Tools T(2k−1), T(2k) | Tool T(2k+1), wrapping round | The 10-fold leave-one-tool-out comparison with the dataset authors' \~75 % runs separately (pipeline stage 7) |
| Paderborn artificial + healthy K001-K003 | 3 of 15 bearings, stratified: at least 1 outer-race and 1 inner-race bearing; healthy bearings in folds 1-3 | 3 bearings of the next fold | Macro-F1 over the classes present in each fold. Check the bearing lists in the KAt fact sheets |
| XJTU-SY | Bearing1\_k, Bearing2\_k, Bearing3\_k | Bearings 1, 2, 3 of fold k+1 | One bearing per operating condition in every split |
| CWRU | None (pretraining only) | 3 hp recordings | Only serves Gate 1's cross-load probe |

**Gate 1 (encoder probe).**

- Inputs: frozen features, mean-pooled per input arm and standardized.
- Models: logistic regression (L2, C from 1e-3 to 10) or ridge regression.
- Four tasks on inner-validation units:
  1. CWRU 10 classes, trained on 0-2 hp and tested on 3 hp;
  2. Paderborn 3 classes, split by bearing;
  3. XJTU-SY normalized RUL, split by bearing;
  4. Nonastreda wear class from force only, split by tool.
- Rule: lowest mean rank across the four tasks wins; on a tie, the cheaper encoder.
- Candidates: Mantis-8M, MOMENT-base, own PatchTST, UniFault (only if its weights load), each crossed with input arms T, F, E and S.

**Gate 2 (fusion search).**

- Score: mean over 5 folds × 3 seeds of macro-F1 (Nonastreda, Paderborn) and 1 − normalized RMSE (Nonastreda wear, XJTU-SY RUL), on inner-validation units.
- A candidate qualifies only if it beats Plan B by at least 1.0 point, with a one-sided Wilcoxon p < 0.05 over the 15 paired runs.
- If none qualifies, Plan B becomes the model.
- Ties within 0.5 points are broken by the smaller accuracy drop when a modality is missing, then lower ECE, then fewer parameters.

Held-out targets (no part of them is used before stage 8):

| Target | Shift tested | Support set (k ∈ {1, 5, 10}, 5 draws) | Test set | Metrics |
| --- | --- | --- | --- | --- |
| KAIST, 15 classes | New machine, five sensor types, new fault types (misalignment, unbalance) | k windows per class from the 0 and 2 Nm recordings; zero-shot uses no fault examples, only 10 healthy reference windows for the level baseline | All 4 Nm windows | Macro-F1, balanced accuracy, ECE; zero-shot accuracy on unseen fault types |
| Paderborn real damage, healthy / outer race / inner race | Artificial → real damage | One bearing per class (healthy from K004-K006), k windows each; bearings redrawn per draw | All other real-damage bearings and the remaining healthy ones | Macro-F1, balanced accuracy, ECE |
| PHM 2010 flank wear | New milling machine, material and cutter; 50 kHz signals | k ∈ {10, 50, all} labelled cuts from two cutters | All cuts of the third cutter (3 folds: c1, c4, c6) | RMSE and MAE in µm; 90 % interval coverage |

Baselines in every table:

- majority class, and persistence (Nonastreda: the previous pass's label);
- handcrafted features (RMS, kurtosis, band energies, energy at the defect frequencies) with gradient boosting;
- PatchTST trained from scratch;
- MOMENT and Mantis linear probes;
- UniFault fine-tuned (if its weights load);
- Plan B;
- C2F-N;
- TimesFM health-indicator forecasting (RUL only).

Statistics:

- Wilcoxon signed-rank tests on paired runs (fold × seed × draw);
- 95 % bootstrap intervals that resample units;
- Holm correction within each table;
- mean ± std over 3 seeds.

Automatic leakage checks, run every time:

- unit-disjoint splits;
- no attached item captured after its window starts;
- a classifier on the context token alone stays at chance for the label;
- a probe reports how well z predicts the source dataset;
- Nonastreda accuracy is reported next to persistence and as skill over persistence.

## Anomaly detection, TimesFM and AnomalyGPT

Anomaly detection becomes a fourth task, scored without fault labels, because Project 14 promises fewer false alarms. TimesFM 3.0 joins as the forecasting stage of a two-stage RUL arm. AnomalyGPT is not adopted; its idea (a few normal images in, an anomaly map plus text out) is covered by anomalib.

| Component | Verdict | Role in this plan | Why |
| --- | --- | --- | --- |
| [TimesFM 3.0](https://github.com/google-research/timesfm) (Google, August 2026; about 330M parameters; 9 quantiles; covariates; context up to 15,360 points) | Adopted as a baseline and a prognosis stage | Forecasts a health indicator to its failure threshold. The indicator is a per-snapshot or per-cut degradation score from our model, or RMS as a plain baseline. RUL = steps until the forecast crosses the threshold; the 0.1 and 0.9 quantiles give an interval | It is a forecaster, so it is the wrong tool for kHz vibration classification: 15k points cover 1.28 s at 12 kHz, and embeddings are not exposed. It is the right tool for slow health trends. Its weights are non-commercial; TimesFM 2.5 (Apache-2.0, 200M) is the licence-clean fallback |
| TimesFM for slow channels | Optional | Forecast-residual alarms on temperature and current RMS: an alarm when a value leaves the 0.1-0.9 band 3 times in a row | Cheap, and gives a second false-alarm baseline |
| [AnomalyGPT](https://github.com/CASIA-IVA-Lab/AnomalyGPT) (AAAI 2024; ImageBind + Vicuna-7B via PandaGPT; trained on MVTec-AD and VisA with simulated anomalies) | Not adopted; at most a qualitative demo on 20 photos | - | Images only. Built for local defects on normal objects, not gradual flank wear. Nonastreda has no segmentation masks, so localization cannot be scored. 7B LLM; CC BY-NC-SA 4.0; its text is not grounded in sensor evidence |
| [anomalib](https://github.com/openvinotoolkit/anomalib) PatchCore and WinCLIP (Apache-2.0) | Adopted for photos | Memory bank = photos of sharp tools from training tools. A heatmap per photo becomes the visual part of the explanation card; the image anomaly score becomes an extra feature | Few normal images are enough. Without masks, the check is the Spearman correlation between image anomaly score and flank wear in µm |
| LLM time-series anomaly methods (SigLLM, AnomalyLLM style) | Not adopted | - | Same reasons as option C: heavy, weak on numbers, hard to make faithful |

Sensor anomaly protocol:

1. **Setting.** Only healthy windows of the target machine are available: k\_h ∈ {10, 50}, 5 draws. The test set holds the remaining healthy windows and all faulty ones.
2. **Scores (no fault labels):**
   - (a) cosine k-nearest-neighbour distance (k = 5) from z to the healthy support;
   - (b) masked-modeling error: mean 1 − cosine over 10 random masks;
   - (c) zero-shot text score: probability of the "healthy machine" prototype against "machine with a fault" paraphrases.
3. **Baselines:**
   - an RMS 3σ alarm;
   - an isolation forest on handcrafted features;
   - k-nearest-neighbour distance on MOMENT or Mantis features;
   - MOMENT's reconstruction anomaly score.
4. **Datasets.** KAIST (4 Nm; normal against 14 fault classes); Paderborn real damage (healthy against damaged); XJTU-SY in-domain folds (onset detection).
5. **Metrics.** AUROC, AUPRC and false-alarm rate at 95 % detection. For XJTU-SY: detection lead time relative to the first prediction time, and false alarms before it.

Two-stage RUL arm on XJTU-SY:

1. The model scores each snapshot.
2. Smooth the scores causally (exponential moving average, 5 snapshots).
3. TimesFM 3.0 forecasts the score from the snapshots seen so far.
4. RUL = minutes until the median forecast crosses the training-fold failure level.

This arm is compared against the direct NIG RUL head and against TimesFM on raw RMS, with the same metrics and interval coverage.

## Trust: uncertainty, calibration and explanations

Every prediction carries a calibrated probability, a trust value per modality and explanations at three levels. An explanation is shown only if it passes two faithfulness tests.

Uncertainty and calibration:

- **Classes:** uncertainty u = K/S, per-modality trust 1 − uₘ, and the Dempster conflict C between modalities.
- **Regression:** the NIG head gives aleatoric uncertainty β/(α−1) and epistemic uncertainty β/(ν(α−1)), and from them a 90 % interval.
- **Calibration metrics:** ECE (15 bins), Brier score, NLL and reliability diagrams. Temperature scaling is fitted on inner-validation units, or on the support set for targets when k ≥ 5. Regression reports 90 % interval coverage and mean width.

Explanations at three levels:

| Level | Method | Faithfulness test |
| --- | --- | --- |
| Modality | Trust 1 − uₘ; conflict C; latent cross-attention mass per modality; drop in probability when the modality is removed | The removal effect is the ground truth; attention mass is shown only if it ranks modalities the same way (Spearman ≥ 0.6) |
| Token (time segment, spectral token, photo region, text) | Gradient-weighted attention relevance through the fusion layers (Chefer et al., ICCV 2021, `hila-chefer/Transformer-MM-Explainability`); raw attention rollout as a comparison | Deletion curve: remove the top-x % tokens and track the probability; area under the curve compared with random order |
| Frequency | Band occlusion: notch each band of the raw window, re-encode, record the drop in probability (64 bands on the Hz axis, or on the order axis for bearings). Captum Integrated Gradients on F/E inputs when those arms are used | Physics alignment score (below); weight-randomization test (Adebayo et al. 2018) |
| Image (Nonastreda) | anomalib PatchCore heatmap and attention over the 2×4 photo regions | Spearman correlation of image anomaly score with flank wear (µm) |

Physics alignment score (PAS). Take A\_b, the attribution of frequency band b. Let F be the set of bands within ±2 % of a defect frequency f\_d or its harmonics (k = 1..5), with ± shaft-rate sidebands for inner-race faults:

```latex
\mathrm{PAS}=\frac{\sum_{b\in\mathcal{F}}A_b\,/\,\sum_b A_b}{|\mathcal{F}|\,/\,N_{bands}}
```

PAS > 1 means the model attends to the physical defect lines more than chance. It is reported on correctly classified faulty windows and must exceed the score of a weight-randomized model. The defect frequencies, for n balls of diameter d on pitch diameter D, contact angle φ and shaft rate f\_r:

```latex
f_{BPFO}=\tfrac{n}{2}f_r\left(1-\tfrac{d}{D}\cos\phi\right),\quad f_{BPFI}=\tfrac{n}{2}f_r\left(1+\tfrac{d}{D}\cos\phi\right),\quad f_{BSF}=\tfrac{D}{2d}f_r\left(1-\left(\tfrac{d}{D}\cos\phi\right)^2\right)
```

| Bearing | Geometry | Speed | BPFO | BPFI |
| --- | --- | --- | --- | --- |
| KAIST, NSK 6205 | n 9, d 7.90 mm, D 38.5 mm, φ 0° | 3010 rpm | ≈ 179 Hz | ≈ 272 Hz |
| CWRU, SKF 6205 | n 9, d 7.94 mm, D 39.04 mm | 1797 rpm | ≈ 107 Hz | ≈ 162 Hz |
| Paderborn, 6203 (check against the fact sheets) | n 8, d 6.75 mm, D ≈ 28.5 mm | 1500 rpm | ≈ 76 Hz | ≈ 124 Hz |
| XJTU-SY, LDK UER204 (check against the dataset paper) | to confirm | 2100-2400 rpm | - | - |

Milling has no bearing lines. The PHM 2010 equivalent is the tooth-passing frequency (teeth × rpm / 60), once the spindle speed is confirmed. Nonastreda publishes no spindle speed and its 1 kHz rate cuts off at 500 Hz, so it gets no physics score.

Explanation card (template; a small Qwen instruct model may reword it, and any output with a number or name not in the card falls back to the template):

```markdown
State: outer-race defect, 1.0 mm (probability 0.91, calibrated; uncertainty 0.06)
Modalities: vibration trust 0.93 | current trust 0.71 | temperature trust 0.40 | conflict 0.08
Evidence: 62 % of frequency attribution within 2 % of BPFO 179 Hz and its harmonics (PAS 7.4)
Nearest fault description: "Outer-race defect of 1.0 mm on a deep-groove ball bearing."
Inputs: vibration and current windows 1.0 s; temperature summary; photo age n/a
Caution: none (faithfulness tests passed)
```

## Reuse versus new work

About 80 % of the code is assembled from verified work: encoders, fusion layers, uncertainty losses, PEFT, attribution tools and the OmniFuser evaluation stack. Six parts are new; each has a concrete route and a test.

| Part | Source | Licence | How it is used |
| --- | --- | --- | --- |
| Sensor encoders | [MOMENT](https://github.com/moment-timeseries-foundation-model/moment) (`momentfm`), [Mantis](https://github.com/vfeofanov/mantis) (`mantis-tsfm`, `paris-noah/Mantis-8M`), PatchTST (HF `transformers`), [UniFault](https://github.com/emadeldeen24/UniFault) | MIT; Apache-2.0; Apache-2.0; MIT | As is, frozen; Gate 1 picks one. UniFault only if its "Tiny" checkpoint loads |
| Spectrogram encoder | [AST](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | BSD-3-Clause | As is, as probe arm S (vibration at 16 kHz looks like audio) |
| Image and text encoders | DINOv2-S/14 (`facebook/dinov2-small`); Qwen3-Embedding-0.6B | Apache-2.0 (check the model cards) | As is, frozen |
| Signal level and healthy reference | [BearLLM](https://github.com/SIA-IDE/BearLLM) (AAAI 2025): frequency-domain input with a fault-free reference signal | No licence file seen | Idea only (baseline-relative level tokens); no code copied |
| Extra vibration corpus | [MBHM](https://huggingface.co/datasets/SIA-IDE/MBHM) (13 GB, 9 public bearing sets, 24,000-point windows) | MIT | Only if its metadata maps every window to a source dataset and bearing |
| Dataset loaders | `paderborn-bearing`, `vibdata` (PyPI); [PHM-Vibench](https://github.com/PHMbench/PHM-Vibench) as a reference | Various; Apache-2.0 | Readers reused where they fit; PHM-Vibench says it is not release-ready |
| Fusion layers | Perceiver (`lucidrains/perceiver-pytorch`); MBT (rewrite of Google's scenic code, about 60 lines); C2F from `omnifuser/models.py` | MIT; Apache-2.0; own | Candidates in the fusion search |
| Uncertainty | [TMC / ETMC](https://github.com/hanmenghan/TMC) (Dempster-Shafer); MoNIG (`MaHuanAAA/MoNIG`); deep evidential regression (Amini et al. 2020) | No licence seen; unchecked; paper | Rewritten from the papers (about 60 lines) and cited |
| Modality balance | OGM-GE (`GeWu-Lab/OGM-GE_CVPR2022`) | Check | Only if one modality dominates training |
| PEFT, attribution, explanations | HF `peft` (LoRA); Captum; Chefer et al. relevance (`hila-chefer/Transformer-MM-Explainability`) | Apache-2.0; BSD-3-Clause; MIT (check) | As is |
| Photo anomaly maps | [anomalib](https://github.com/openvinotoolkit/anomalib) PatchCore, WinCLIP | Apache-2.0 | As is |
| Forecasting stage | [TimesFM](https://github.com/google-research/timesfm) 3.0 (fallback 2.5) | Code Apache-2.0; 3.0 weights non-commercial; 2.5 weights Apache-2.0 | As is, zero-shot; LoRA fine-tuning example available |
| Evaluation stack | OmniFuser: `classification_report`, `ExperimentTracker`, the resume and fold-summary patterns of `cross_validation.py`, the `run_experiment_suite.py` runner pattern, `slurm/` scripts, the `make_nonastreda` test fixture, `load_nonastreda` | Own | Imported or copied; `regression_report` is not used (it hard-codes Fx/Fy/Fz) |

New work, with its route and its test:

1. **Modality-incomplete corpus with causal alignment.** Route: `Sample` schema, eight adapters, as-of join. Test: unit-disjoint splits; no item captured after its window starts.
2. **Baseline-relative level and slow tokens.** Route: four descriptors per channel minus the machine's healthy median. Test: the no-level-token ablation; healthy distributions overlap across datasets.
3. **Cross-modal masked latent modeling on cached features.** Route: whole-modality and token masking, a Perceiver IO style decoder, frozen targets. Test: the no-L\_mask ablation; cross-modal retrieval.
4. **Text-anchored evidential label space.** Route: frozen Qwen prototypes with a learned sensor-side projection; evidence from cosine logits. Test: zero-shot on an unseen fault type (KAIST misalignment and unbalance); calibration.
5. **Evidential token fusion.** Route: per-modality opinions from independent summaries, combined ETMC-style with the fused opinion. Test: missing-modality curves; conflict against corruption level.
6. **Physics alignment score with band occlusion.** Route: notch, re-encode, attribute, compare with defect lines. Test: a synthetic tone at BPFO scores > 1; a randomized model scores ≈ 1.

## Repository, environment and compute

The new repository lives in `Desktop\Multimodal AI for Predictive Maintenance` (not yet under git). `Data/` and `Research papers/` there are git-ignored, and `omnifuser` is installed from the sibling `OmniFuser` folder as an editable dependency.

```text
pyproject.toml          python >=3.11,<3.13
configs/                one YAML per stage (preprocess, cache, probe, fusion, pretrain, adapt, explain)
pdmfm/
  schema.py             ChannelWindow, ImageItem, Sample; as-of join
  texts/                context and fault-description YAML per dataset
  datasets/             nonastreda, cwru, paderborn, xjtu_sy, kaist, phm2010, thermal_motor
  preprocess.py         resampling, windowing, T/F/E inputs, level and slow descriptors, baselines
  encoders/             registry; mantis, moment, patchtst, ast, dinov2, qwen wrappers
  cache.py              float16 feature shards, manifest with encoder signature
  model/                adapters, tokens, fusion/{late,concat,perceiver,mbt,c2f_n}, heads
  losses.py             evidential, NIG, masked latent, InfoNCE
  protocols.py          outer folds, inner validation, held-out targets, few-shot draws
  train.py              supervised and pretraining loops (bf16, resume, tracker)
  adapt.py              zero-shot, linear probe, LoRA, full fine-tuning
  anomaly.py            kNN, masked error, text score; TimesFM RUL stage
  explain/              relevance, band occlusion, PAS, anomalib maps, cards
  report.py             tables, Wilcoxon tests, bootstrap intervals, Holm
scripts/                one entry point per pipeline stage
slurm/                  copied from OmniFuser (job.sbatch, chain.sh, status.sh)
tests/                  synthetic-data tests only
```

Environment and compute:

- **Main environment:** Python 3.11; torch; `transformers` ≥ 4.51 (needed for Qwen3); `peft`; `captum`; `scikit-learn`; `scipy`; `h5py`; `nptdms`; `mantis-tsfm`; `timesfm[torch]`; `anomalib`.
- **MOMENT** gets its own environment if `momentfm` pins old libraries. Features are cached, so it is needed only for extraction.
- **Ruche:** the gpua100 partition (4×A100 40 GB per node, 24 h jobs, at most 4 GPUs per user).
- **Storage:** `$WORKDIR` holds 500 GB. Raw data is about 60-90 GB without the optional MBHM (13 GB); window shards about 4 GB; feature caches under 20 GB.
- **Offline nodes:** all Hugging Face weights are downloaded on the login node into `$WORKDIR/hf_cache`, and `HF_HUB_OFFLINE=1` is set in jobs, in case compute nodes have no internet (to check in week 1).
- **Reproducibility:** seeds; deterministic flags for the runs reported in the paper; every config and data checksum saved with each run by `ExperimentTracker`.

## Timeline, gates and priorities

The design is fixed by Gate 2 on Dec 13. Every later step runs on cached features, and the adaptation jobs run on the queue over the holidays.

&#91;embedded content: Project 14 timeline · 13 weeks + 1 buffer, 2 gates\]

Gate 1 picks the sensor encoder and input arms; Gate 2 picks the fusion stage, or falls back to Plan B. Both rules are in the evaluation section.

If time runs short, cut from the bottom of this table:

| Must | Should | Could |
| --- | --- | --- |
| Nonastreda, Paderborn, XJTU-SY, CWRU and KAIST readers | PHM 2010 reader and transfer | Najafi thermal set; MBHM corpus |
| Gate 1 with Mantis and MOMENT, arms T and E | Arms F and S; own PatchTST | UniFault arm |
| Plan B; fusion search with late, concat, Perceiver and C2F-N | MBT candidate; fusion pretraining | Zero-shot unseen fault types |
| KAIST and Paderborn real damage, k ∈ {1, 5, 10}, linear probe and LoRA | Full fine-tuning; anomaly detection | TimesFM RUL arm; anomalib photo maps |
| Evidential heads, calibration, PAS, deletion test | Explanation cards | Qwen rephrasing; 5 seeds instead of 3 |

## Risks, week-1 checklist and tests

The two largest risks are weak frozen encoders on vibration and fusion that fails to beat late aggregation. Gates 1 and 2 turn each into a planned fallback.

| Risk | Likelihood | Effect | Mitigation |
| --- | --- | --- | --- |
| Frozen general-purpose encoders are weak on kHz vibration | Medium | Low accuracy everywhere | Gate 1 includes own PatchTST, envelope inputs and a handcrafted-feature baseline |
| Fusion fails to beat late aggregation | Medium | Goal 1's claim shrinks | Gate 2 falls back to Plan B; the paper then argues trust, text-anchored transfer and physics-checked explanations |
| Small fusion corpus (about 20k source windows) | High | A "foundation" claim is easy to challenge | State that capacity comes from the encoders; the held-out tests carry the claim; MBHM if usable |
| KAIST and CWRU have one specimen per condition | Certain | They test condition shift, not new units | Say so in every table; Paderborn real damage and PHM 2010 test new units |
| High variance of few-shot results | High | Unclear rankings | 5 draws, bootstrap intervals, paired tests |
| Dempster-Shafer overconfidence | Medium | Poor calibration | Temperature scaling; ablation without the fused opinion |
| Unfaithful explanations | Medium | Goal 3 fails | Deletion and randomization tests; failing explanations are hidden |
| Non-commercial terms (Paderborn data, TimesFM 3.0 weights) | Certain | Release restrictions | NC weights, or a variant without Paderborn; TimesFM 2.5 fallback |
| Ruche queues and the holiday period | Medium | Delays | Cached features keep runs short; Lightning AI as backup; jobs queued over the holidays; one buffer week |

Week-1 checklist:

| Item | How to settle it | Status |
| --- | --- | --- |
| UniFault "Tiny" checkpoint loads and its input format | Load it in a scratch environment and run one CWRU window | Open |
| `momentfm` version pins | Read its `requirements.txt`; use a separate environment if needed | Open |
| MBHM maps each window to a source dataset and bearing | Inspect `metadata.sqlite` | Open |
| XJTU-SY licence and download; LDK UER204 geometry | Official page and the dataset paper | Open |
| PHM 2010: spindle speed, sampling rate, cutter, data access | Apparatus paper on the PHM Society challenge page; HF mirror `AI4Manufacturing/PHM2010` | Open |
| Paderborn: bearing lists (artificial, real), 6203 geometry, auxiliary channel rates, condition codes | KAt fact sheets and measurement log | Open |
| KAIST: TDMS reading, DAQ synchronization | `nptdms` on one file; the article's methods | Open |
| Nonastreda sampling rate | Data in Brief article: 1 kHz, Kistler 9257B | Settled |
| Nonastreda full download: chip and workpiece photo sizes | Mendeley archive | Open |
| Internet access on Ruche compute nodes | One test job; otherwise a pre-filled HF cache | Open |
| Code licences: TMC, MoNIG, Chefer relevance, OGM-GE | Repo pages; rewrite from the paper when there is none | Open |
| Najafi thermal set: image count and resolution | Readme PDF in the archive | Open |

Tests (pytest on synthetic data, so continuous integration needs no datasets or weights):

| Module | Test |
| --- | --- |
| `schema.py` | The as-of join never attaches an item captured after the window starts; a missing modality gives an empty token set with a false mask |
| `datasets/*` | Each reader on a tiny synthetic file in its real format (MAT, TDMS, CSV): shapes, rates, labels; unit-disjoint splits |
| `preprocess.py` | Resampling keeps a 1 kHz tone and removes energy above the new Nyquist; level descriptors on known signals |
| `cache.py` | Cached features equal online encoding within 1e-3; a new encoder signature invalidates the cache |
| `model/fusion/*` | Masking a modality equals removing its tokens; finite outputs with one modality; parameter budget ≤ 3.2 M |
| `losses.py` | Dempster's rule on agreeing, conflicting and vacuous opinions matches hand calculations; NIG loss and variances on known values |
| `protocols.py` | No target dataset or outer-fold test unit reaches pretraining, gates or checkpoint selection |
| `explain/` | A synthetic tone at BPFO scores PAS > 1; a randomized model scores ≈ 1 |
| End to end | CPU smoke run on two synthetic datasets, from cache to explanation card, in under 2 minutes |

## Sources

Papers provided with the request: [PatchTST (arXiv 2211.14730)](https://arxiv.org/abs/2211.14730), [OmniFuser (arXiv 2511.01320)](https://arxiv.org/abs/2511.01320), [VLT (arXiv 2607.14510)](https://arxiv.org/abs/2607.14510), and `FUTURE_WORK_AND_ANALYSIS.md` in the OmniFuser repository.

Pages opened while planning:

- Encoders and models: [MOMENT](https://github.com/moment-timeseries-foundation-model/moment) · [Mantis](https://github.com/vfeofanov/mantis) · [UniFault](https://github.com/emadeldeen24/UniFault) · [AST model card](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) · [TimesFM](https://github.com/google-research/timesfm)
- Related systems: [BearLLM](https://github.com/SIA-IDE/BearLLM) · [MBHM dataset](https://huggingface.co/datasets/SIA-IDE/MBHM) · [PHM-Vibench](https://github.com/PHMbench/PHM-Vibench) · [TMC / ETMC](https://github.com/hanmenghan/TMC) · [AnomalyGPT](https://github.com/CASIA-IVA-Lab/AnomalyGPT)
- Datasets: [Nonastreda on Mendeley](https://data.mendeley.com/datasets/m892d2wtzh/1) · [Nonastreda article (Data in Brief)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12341703/) · [KAIST rotating-machine dataset article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10036499/) · [Najafi thermal induction-motor images](https://data.mendeley.com/datasets/m4sbt8hbvk/3)

Found through search only (to open in week 1):

- Paderborn KAt data centre
- XJTU-SY dataset pages
- PHM 2010 challenge page
- MoNIG (arXiv 2111.08456)
- OGM-GE (CVPR 2022)
- Qwen3-Embedding model card
- anomalib
- Chefer et al. relevance repository
- MaintNet, AssetOpsBench and FailureSensorIQ (text resources)
