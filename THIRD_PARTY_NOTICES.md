# Third-party notices

This file identifies external software, data, and research references used by the project. It is a
notice and provenance guide, not a relicensing document. The [Apache License 2.0](LICENSE) applies
only to original NOESIS-Σ / SigmaStorm source and project materials for which Ravinder Singh has
the right to grant that license. Third-party materials retain their own terms.

## Direct software dependencies

The runtime dependencies declared in `pyproject.toml` and `requirements.txt` are:

- [PyTorch](https://github.com/pytorch/pytorch) (`torch`)
- [NumPy](https://github.com/numpy/numpy) (`numpy`)
- [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers) (`tokenizers`)

Corpus construction optionally uses:

- [Hugging Face Datasets](https://github.com/huggingface/datasets) (`datasets`)

Development and verification use Black, Ruff, mypy, Bandit, pre-commit, and pytest. These tools and
their transitive dependencies are installed from their package distributions and are not copied
into this repository. Users must follow the license notices shipped by the relevant distributions.

## External data

When `scripts/build_corpus.py --source tinystories` is used, the script streams
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) from the Hugging Face Hub.
TinyStories is not included in the Git repository. Review the dataset card and its current terms
before downloading, redistributing, or publishing derived data.

The local `data/` directory, Kaggle workspace, generated corpora, checkpoints, and run outputs are
ignored or otherwise kept outside the public source distribution. They are not relicensed by this
repository's Apache-2.0 license.

## Research references

The papers listed in [`docs/paper/references.bib`](docs/paper/references.bib) are citations used to
explain related work. They are not bundled or relicensed by this project; their authors and
publishers retain their own rights and terms.

## Project assets

The files under [`assets/`](assets/) are repository-specific visuals used by the README and project
documentation. If an asset is replaced with material derived from an external source, its source,
license, and required attribution must be added here before distribution.

## Updating this notice

When adding a dependency, dataset, pretrained artifact, copied code, or externally derived asset,
record its source and applicable terms here and verify that the intended use is compatible with the
project's license and distribution plan.
