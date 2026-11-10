# SigmaStorm technical report

LaTeX source for *SigmaStorm: Bounded-State Sequence Modeling without Dense Token-to-Token
Attention*.

**Author:** Ravinder Singh

## Build

```bash
cd docs/paper
tectonic -X compile main.tex
```

`tectonic` fetches the required TeX packages on first use and runs the bibliography passes
automatically. A standard TeX Live installation also works:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Layout

- `main.tex` — preamble, title, abstract, section inputs, bibliography.
- `sections/01_intro.tex` … `sections/09_conclusion.tex` — body.
- `sections/appendix.tex` — configurations, reproduction commands, honesty ledger.
- `references.bib` — bibliography.

## Evidence policy

Every number in the report is traceable to a script or artifact in the repository:
`scripts/gate.py`, `pytest tests`, `scripts/ladder.py`, `scripts/gapd_proof.py`, and the 21M
training/evaluation scripts. Raw outputs live in `results/`. Unmeasured claims are marked as
such in the claims map (`tab:claims`) and the honesty ledger (`app:ledger`).
