# EDA artifacts and their interpretation

The JSON summaries and PNG figures are useful audit evidence, not independent test folds
or a definition of the current protocol. Original artifacts are preserved. See the
[current dataset audit](../DATASET_AUDIT.md) and [plan](../PLAN.md) for current decisions.

| Artifact | What it establishes | Consequence |
| --- | --- | --- |
| `summary/nonastreda.json`, `figures/nonastreda_wear_force.png` | Historical 378-pass deduplicated view; Fy AC RMS/wear correlation 0.70; Fz mean drifts by recording block | Preserve force-level descriptors and strong schedule/persistence baselines; exclude Fz level shortcuts; do not reconstruct RUL from this reduced view |
| `summary/nonastreda_lifecycle.json` | New all-512-pass lifecycle, 378 distinct force recordings, both operating clocks | Use confirmed-wear endpoints and disclose missing-operation estimates |
| `summary/matwi.json` | New 1,700-recording recorded-life proxy; duration variability and pure-flank label availability | Keep RUL separate from flank/adhesion labels; no universal wear failure threshold |
| `summary/cwru.json`, CWRU figures | Recorder roll-off alone separates healthy/fault recordings; band limiting removes the top-band energy | Keep common 5 kHz band; diagnosis accuracy alone is not proof of transferable health features |
| `summary/kaist.json`, temperature/current/order figures | Fault groups recorded on different days; current-channel absence is class dependent; current fundamental differs from shaft speed | Exclude the two class-dependent current channels, relative temperature only, state-paired rather than sample-synchronous modalities, vibration-derived shaft speed |
| `summary/xjtu_sy.json`, RMS trajectories | FPT is highly sensitive to reference length/consecutive rule; the short rule alarms early on long lives | Use the selected 10%-reference / 10-consecutive offline target and kurtosis fallback, never the old headline FPT field as ground truth |
| `summary/najafi_thermal.json`, example images | One motor; 95.7% nearest neighbors share an acquisition folder; even simple pixel baselines saturate | Keep optional; no expensive fusion search or claims of new-machine transfer/RUL |
| `summary/nasa_milling.json`, wear-by-case figure | Only 146/167 runs have wear labels; case trajectories are sparse; level/envelope inputs | Case-disjoint sparse wear regression, not high-frequency bearing tasks or invented full life |
| `summary/mbhm.json` | Large aggregate with overlapping CWRU/PU/XJTU source data | Excluding PU/XJTU alone is not proof of per-unit isolation; require original unit joins before admission |

## Important provenance distinctions

- `nonastreda.json` still describes 378 passes and 4,241 windows. The RUL lifecycle keeps
  512 logical passes. These are different views, not conflicting inventory counts.
- XJTU's historical `fpt_index`, `healthy_fraction`, and headline median correspond to an
  earlier short alarm rule. Prefer `healthy_fraction_by_fpt_rule.b10pct_3sd_c10` when
  reading that summary; it is RMS-only and can be null where the implemented kurtosis
  fallback supplies the label. For example, Bearing3_1 moves from 0.006 of life under the
  short rule to 0.924 under the selected rule.
- There was no MATWI summary among the original EDA artifacts. The two new lifecycle
  summaries distinguish current targets from the earlier wear/window evidence.
- Historical KAIST EDA includes labeled baseline results. KAIST can be described as held
  out from model training/search, but not as a dataset whose labels were never inspected.
  The current audit does not fit new KAIST models or use those baseline scores to tune.
- The figures are observations about acquisition and data structure. Correlations do not
  verify a physical endpoint, causal mechanism, or a model's generalization claim.
