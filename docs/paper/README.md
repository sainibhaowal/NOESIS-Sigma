# SigmaStorm technical report

LaTeX source for *SigmaStorm: Bounded-State Sequence Modeling without Dense Token-to-Token
Attention*.

**Author:** Ravinder Singh

## Build

From the repository root, a standard TeX Live installation can build the report with:

```bash
cd docs/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Tectonic is an optional alternative when installed; it resolves packages according to its own
configuration and may need network access on the first build:

```bash
cd docs/paper
tectonic -X compile main.tex
```

## Layout

- `main.tex` — preamble, title, abstract, section inputs, bibliography.
- `sections/01_intro.tex` … `sections/09_conclusion.tex` — body.
- `sections/appendix.tex` — configurations, reproduction commands, honesty ledger.
- `references.bib` — bibliography.

## Evidence policy

Every number in the report is traceable to a script or artifact in the repository:
`scripts/gate.py`, `pytest tests`, `scripts/ladder.py`, `scripts/gapd_proof.py`, and the
training/evaluation scripts used for the recorded 21M run. Raw outputs live in `results/`. Unmeasured claims are marked as
such in the claims map (`tab:claims`) and the honesty ledger (`app:ledger`).
