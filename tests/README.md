# Test suites

See [the testing guide and requirement matrix](../docs/TESTING.md).

- `unit/`: numerical, schema and decision contracts.
- `integration/`: pipeline boundaries, readers, models, caching and resumption.
- `e2e/`: synthetic complete workflows.
- `acceptance/`: explicitly requested real data and hardware.
- `support/`: shared fixtures.

Slow research-quality checks retain their learning assertions and run separately.
