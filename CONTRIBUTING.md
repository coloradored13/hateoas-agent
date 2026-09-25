# Contributing to hateoas-agent

Thanks for your interest in contributing!

## Development setup

```bash
git clone https://github.com/coloradored13/hateoas-agent.git
cd hateoas-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,anthropic,mcp]"
```

The core library has **zero runtime dependencies** — `anthropic` and `mcp`
are optional extras. Keep it that way: new features must not add required
runtime dependencies.

## Running tests

```bash
pytest tests/ -q              # full suite (adversarial tests excluded)
pytest tests/ -k orch -q      # orchestration tests only
pytest tests/ -k async -q     # async tests only
```

The adversarial red-team suite hits the live Claude API and runs as a
standalone script, not via pytest:

```bash
ANTHROPIC_API_KEY=sk-... python3 tests/test_adversarial.py
```

It also runs nightly in CI when the `ANTHROPIC_API_KEY` repository secret is
configured.

## Linting and formatting

```bash
ruff check src/ tests/
ruff format src/ tests/
```

CI enforces both `ruff check` and `ruff format --check`. A pre-commit hook is
available in the repo history if you want to wire it up locally.

## Pull requests

- Branch from `main`; keep PRs focused on one change.
- Add or update tests for any behavior change — the suite currently sits at
  ~92% coverage and CI runs it on Python 3.10–3.13.
- Update `CHANGELOG.md` under `[Unreleased]` for user-visible changes.
- New APIs should work with all three definition styles (action-centric,
  state-centric, Resource) where applicable, and must not break the
  `HasHateoas` protocol contract.

## Release process (maintainers)

Releases publish to PyPI via [trusted publishing](https://docs.pypi.org/trusted-publishers/)
— no API tokens are stored in the repo.

1. Update the `version` in `pyproject.toml` and move `[Unreleased]` entries
   to a new version section in `CHANGELOG.md`.
2. Merge to `main`, then tag and push:

   ```bash
   git tag v0.3.0
   git push origin v0.3.0
   ```

3. The `Release` workflow builds the sdist/wheel, publishes to PyPI, and
   creates a GitHub release.

**One-time setup** (already done if releases exist): on PyPI, add a trusted
publisher for the `hateoas-agent` project pointing at this repo, workflow
`release.yml`, environment `pypi`. On GitHub, create the `pypi` environment
under Settings → Environments.
