# Project progress

## 15 November 2026: repository and scientific audit

The current specification is [docs/PLAN.md](docs/PLAN.md). The audit corrects bag-level
selection, alarm confirmation timing, run identity, CUDA RNG resumption, and modality
availability. Native Nonastreda ingestion replaces the private dependency. Versioned
lifecycle conversion adds confirmed-wear RUL for Nonastreda and recorded-life proxy RUL
for MATWI; both milling datasets have healthy-only anomaly protocols.

Tests are organized by category and external-resource requirements. The public CPU demo
runs from synthetic raw MAT/CSV files with stand-in encoders. GPU speedups and final
scientific performance remain subject to cluster validation.

Evidence and limitations: [audit](docs/AUDIT.md), [dataset audit](docs/DATASET_AUDIT.md),
and [testing guide](docs/TESTING.md).

Earlier experiment decisions, pilot scores, and research notes remain preserved in the
[historical log](docs/archive/PROJECT_LOG-through-2026-11-05.md). They describe older
protocols and must not be silently pooled with revised runs.
