# Contributing to call-asr-bench

Contributions should improve reproducible ASR benchmarking for real telephone
audio. Prefer bounded changes with an explicit benchmark or maintainer need over
activity-only cleanup.

## Start with an issue

For non-trivial work, use or open an issue before implementation. The issue
should define the problem, scope, compatibility constraints, and acceptance
criteria. Public result-schema changes and new normalization/scoring semantics
must be designed explicitly before code is written.

Keep one primary issue per pull request when practical. Use a focused branch such
as `feat/<topic>`, `fix/<topic>`, or `docs/<topic>` and include `Closes #<issue>`
in the PR body when the work fully resolves it.

## Development setup

Core development uses only the default package plus the dev extra:

```bash
uv sync --extra dev
```

Before marking a PR ready for review, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
```

CI runs the same checks on Python 3.10 through 3.13 with a locked environment.
Do not rely on a dependency that is absent from `uv.lock`.

## Test-first behavior changes

For code changes, add or change a focused test that demonstrates the missing or
incorrect behavior before implementing the fix when practical. A useful RED
state must fail for the intended behavioral reason, not because the test itself
is malformed or unformatted.

Tests in the default suite must be deterministic, CPU-safe, and self-contained.
They must not:

- download model weights;
- require a GPU;
- call external ASR services or other public network endpoints;
- depend on user credentials or secrets;
- depend on mutable external datasets.

Use fake/injected model or HTTP clients for adapter tests. Tiny repository-owned
fixtures are acceptable when their provenance and reproduction path are clear.

## Adding an ASR adapter

Adapters implement the `ASRAdapter` contract from
`src/callasr/adapters/base.py`. Keep model construction and provider-specific
behavior inside the adapter; the benchmark runner must remain adapter-agnostic.

An adapter should expose stable, non-secret metadata:

- `name`;
- `model`;
- `device`;
- `compute_type`;
- `decoding_options` containing only information safe and useful to serialize in
  benchmark artifacts.

Heavy or provider-specific dependencies belong in an optional extra. Import them
lazily so the core package and default CI remain lightweight. Missing optional
dependencies should produce an actionable `AdapterError`; do not misreport a
missing transitive dependency as a missing top-level package.

Never place API keys, tokens, passwords, authorization headers, or secret-bearing
URLs in adapter metadata, logs, result artifacts, or error messages.

Adapter unit tests should use injected fake models/clients and must not download
checkpoints or contact real services.

## Adding an audio impairment

Keep low-level deterministic transforms separate from runner/CLI/schema wiring
when that separation makes the behavior easier to verify. Existing examples
include additive noise, packet loss, jitter/late-frame loss, and gain/clipping.

Impairments should:

- validate configuration explicitly;
- avoid mutating the caller's input buffers/payloads;
- make randomness deterministic from an explicit seed;
- preserve existing random streams when a new independent impairment is added;
- document exact ordering when multiple impairments compose.

If a new impairment configuration is needed to reproduce a saved run, design the
result-schema change explicitly rather than adding undocumented serialized
fields.

## Adding or changing a metric

Metric normalization is part of the public benchmark contract. Keep a new metric
in its own pipeline unless it intentionally changes an existing metric.

For example, numeric critical-entity scoring does not alter WER/CER
normalization. Tests should cover representative Russian and English cases when
the metric claims to support both, plus edge cases that make the normalization
or matching semantics explicit.

If metric results or diagnostics are added to saved artifacts, follow the schema
versioning policy in [`docs/releasing.md`](docs/releasing.md). Never change the
meaning of an existing serialized metric silently.

## Optional dependencies

Add provider/model dependencies under `[project.optional-dependencies]` instead
of core `dependencies` unless every user needs them. Refresh `uv.lock` after
changing dependency metadata.

Core CI should still pass with only:

```bash
uv sync --locked --extra dev
```

A new optional integration is not complete merely because its upstream source
code exists. Prefer a stable installable dependency path over ad-hoc Git pins for
normal project support.

## Pull requests

Keep PRs reviewable and scoped. The description should state:

- what changed and why;
- the issue it resolves;
- relevant compatibility or schema implications;
- how the behavior was verified;
- any intentional limitation or deferred follow-up.

Do not mix unrelated refactors, documentation cleanup, and feature work into one
PR. Draft PRs are useful while tests intentionally fail during a test-first
cycle. Mark a PR ready only when the intended implementation is complete and the
full CI matrix is green.

Repository maintainers normally squash feature PRs so intermediate RED/fix/style
commits do not become permanent `main` history.

## Releases

Release/version/schema rules and the maintainer release gate are documented in
[`docs/releasing.md`](docs/releasing.md). Published tags and release artifacts
are immutable.
