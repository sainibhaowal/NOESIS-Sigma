# Contributing to NOESIS-Σ

Thank you for helping improve NOESIS-Σ / SigmaStorm. This repository is a mechanism-level research
prototype, so contributions should be reproducible, narrowly scoped, and clear about what the
evidence does and does not establish.

## Before you start

Please read the [README](README.md), [DESIGN.md](DESIGN.md), and the relevant material under
[`docs/`](docs/). For security issues, do not open a public issue; follow [SECURITY.md](SECURITY.md).

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
pre-commit install
```

The project requires Python 3.10 or newer. The local pre-commit hooks use the active environment's
executables, so activate `.venv` before running them.

## Required checks

Run the checks relevant to your change before opening a pull request:

```bash
pre-commit run --all-files
pytest -q
python scripts/gate.py
```

The gate includes numerical equivalence, fixed-state, and synthetic recall checks and may take
longer on CPU. Documentation-only changes should still pass pre-commit and link checks.

## What to change

- Keep claims in documentation and results tied to a script, configuration, or recorded artifact.
- Add or update tests when behavior changes.
- Keep public source, small configurations, and explanatory documentation in Git.
- Do not commit secrets, local environment files, generated corpora, checkpoints, model weights,
  Kaggle workspace files, or other ignored artifacts.
- If adding a dataset, pretrained artifact, or external asset, document its provenance and terms
  before adding it to the repository.
- Do not silently replace an existing result. Add a dated or versioned record and explain the
  measurement conditions.

## Pull requests

Describe the motivation, the exact change, and the evidence supporting it. Include:

- tests or gates run and their results;
- changes to claims, metrics, or limitations;
- dependency or license changes; and
- any compatibility or migration notes.

Small focused pull requests are easier to review. Please do not include unrelated formatting or
generated-file changes.

## Commits and licensing

Use a concise imperative commit subject, for example `docs: clarify state-size claim`. By submitting
a contribution, you confirm that you have the right to submit it and agree that it may be distributed
under the [Apache License 2.0](LICENSE), consistent with the repository's licensing terms.

Contributors are expected to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
