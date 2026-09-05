# Release and versioning policy

This project keeps the Python package version and benchmark artifact schema as
separate compatibility contracts.

## Package version

`call-asr-bench` uses semantic versions for published package releases and Git
tags. While the project is pre-1.0, a minor release such as `0.3.0` may include
intentional public API or artifact-contract changes, but they must be documented
in the changelog and never introduced silently.

For every release:

- `pyproject.toml` contains the version being released;
- `uv.lock` is regenerated after the version or dependency metadata changes;
- the Git tag is exactly `v<package-version>`;
- the GitHub Release title matches the tag;
- the tag points to the release-prep commit that passed the release gate;
- published tags and release artifacts are immutable. Never move or recreate a
  released tag to change old behavior.

## Benchmark artifact schema

`schema_version` describes the JSON result contract, not the package version.
Increment it whenever a change alters the serialized result shape or the meaning
required to interpret a serialized field. Examples include adding impairment
configuration that is needed for reproducibility, adding a new scored metric to
the result contract, removing or renaming fields, or changing a field's units or
semantics.

Do not increment the artifact schema for implementation-only changes that leave
the serialized contract and its meaning unchanged, such as adding a new adapter
that fits the existing adapter metadata shape or adding a report command that
only reads existing artifacts.

A schema bump is monotonic. Old published artifacts are never rewritten. Readers
that support multiple schemas must name the supported versions explicitly and
must not guess the meaning of an unknown schema.

The package version and schema version do not need to match. A single package
release publishes one current writer schema while readers such as
`callasr compare` may support older schemas as well.

## Release preparation

Use a dedicated release-prep issue and branch. Keep functional feature work out
of the release-prep PR.

1. Confirm the intended milestone/TODO set is complete or explicitly deferred.
2. Set the package version in `pyproject.toml`.
3. Regenerate `uv.lock` with `uv lock` and verify `uv sync --locked` succeeds.
4. Add the release section to `CHANGELOG.md`, including compatibility notes and
   known deferred work.
5. Check README examples and documented schema version against current code.
6. Run the release gate below on the exact release-prep head.
7. Merge the release-prep PR only after the gate is green.
8. Create annotated tag `v<version>` on that merged release commit.
9. Build wheel and sdist from that tagged commit and attach them to a non-draft,
   non-prerelease GitHub Release.
10. Verify independently that the tag points to the intended commit and that both
    distribution artifacts are attached.

If a temporary GitHub Actions workflow is required for an operation that the
connected maintainer tooling cannot perform directly, keep it branch-only and
remove it before the final release-prep diff or release tag.

## Release gate

Run the same supported-Python matrix used by CI. On the release-prep head, every
Python 3.10 through 3.13 job must pass:

```bash
uv sync --locked --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
```

Also build the distributions once from the exact release commit:

```bash
uv build
```

A green unit-test matrix without a successful locked install is not a release
gate. Likewise, a successful build does not replace the test matrix.

## Release notes

Release notes should describe observable user-facing changes rather than commit
activity. Include, when relevant:

- new adapters, impairments, metrics, CLI commands, and result-schema changes;
- compatibility behavior for old result artifacts;
- optional dependency/install commands;
- security-sensitive configuration such as API-key handling;
- important limitations or deliberately deferred work.

Do not present synthetic smoke fixtures as model-quality evidence, and do not
claim support for an integration that remains blocked on an unstable upstream
packaging path.
