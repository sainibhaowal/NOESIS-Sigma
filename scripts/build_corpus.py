#!/usr/bin/env python3
"""Tokenise a text corpus into the uint32 shards ``training/data.py`` expects.

Two input modes:

* ``--source tinystories`` streams ``roneneldan/TinyStories`` from the Hugging Face hub
  (needs ``datasets`` and network access).
* ``--source text --input <path>`` reads a plain text file or a directory of ``.txt`` files.

An 8K byte-level BPE tokenizer is trained once and written to ``<out_dir>/tokenizer.json``;
documents are separated by a single EOS token, so every token in the shards is a training token.

    python scripts/build_corpus.py --source tinystories --out_dir data/tinystories \
        --train_tokens 300_000_000 --val_tokens 5_000_000

Writes ``metadata.json`` for the train split and ``val/metadata.json`` for the validation split.
The builder is resumable per shard: existing full shards are kept and reported.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
EOS_ID = SPECIAL_TOKENS.index("[EOS]")


# ------------------------------------------------------------------- sources


def _tinystories_docs(split: str, max_docs: int) -> Iterator[str]:
    from datasets import load_dataset

    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    for index, row in enumerate(dataset):
        if max_docs and index >= max_docs:
            return
        text = row.get("text") or ""
        if text.strip():
            yield text


def _text_docs(path: Path, max_docs: int) -> Iterator[str]:
    files = sorted(path.rglob("*.txt")) if path.is_dir() else [path]
    seen = 0
    for file in files:
        with file.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield line
                seen += 1
                if max_docs and seen >= max_docs:
                    return


def document_stream(source: str, split: str, input_path: Path | None, max_docs: int) -> Iterable[str]:
    if source == "tinystories":
        return _tinystories_docs(split, max_docs)
    if input_path is None:
        raise SystemExit("--input is required with --source text")
    return _text_docs(input_path, max_docs)


# ----------------------------------------------------------------- tokenizer


def train_tokenizer(docs: Iterable[str], vocab_size: int, out_path: Path, sample_docs: int = 200_000) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    def batched() -> Iterator[list[str]]:
        batch: list[str] = []
        for index, doc in enumerate(docs):
            if index >= sample_docs:
                break
            batch.append(doc)
            if len(batch) >= 1000:
                yield batch
                batch = []
        if batch:
            yield batch

    started = time.time()
    tokenizer.train_from_iterator(batched(), trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A",
        special_tokens=[("[BOS]", tokenizer.token_to_id("[BOS]"))],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    print(f"trained tokenizer with {tokenizer.get_vocab_size()} tokens in {time.time() - started:.1f}s")


# -------------------------------------------------------------------- shards


class ShardWriter:
    """Append-only uint32 shard writer; closes a shard every ``shard_tokens`` tokens."""

    def __init__(self, root: Path, shard_tokens: int, eos_id: int) -> None:
        self.root = root
        self.shard_tokens = shard_tokens
        self.eos_id = eos_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []
        self.buffer: list[np.ndarray] = []
        self.buffered = 0
        self.total = 0

    def add(self, ids: np.ndarray) -> None:
        if ids.size:
            self.buffer.append(ids)
            self.buffered += int(ids.size)
        while self.buffered >= self.shard_tokens:
            self._flush(self.shard_tokens)

    def _flush(self, count: int) -> None:
        joined = np.concatenate(self.buffer) if len(self.buffer) > 1 else self.buffer[0]
        chunk, rest = joined[:count], joined[count:]
        index = len(self.entries)
        name = f"shard-{index:05d}.bin"
        with (self.root / name).open("wb") as handle:
            chunk.astype(np.uint32).tofile(handle)
        self.entries.append({"path": name, "tokens": int(chunk.size)})
        self.buffer = [rest] if rest.size else []
        self.buffered = int(rest.size)
        self.total += int(chunk.size)
        print(f"  wrote {name}: {chunk.size:,} tokens (total {self.total:,})", flush=True)

    def finish(self) -> int:
        if self.buffered:
            self._flush(self.buffered)
        (self.root / "metadata.json").write_text(
            json.dumps(
                {"dtype": "uint32", "eos_token_id": self.eos_id, "shards": self.entries},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return self.total


def tokenize_split(
    tokenizer,
    docs: Iterable[str],
    target_tokens: int,
    writer: ShardWriter,
    batch_docs: int = 512,
) -> int:
    started = time.time()
    batch: list[str] = []
    done = 0
    for doc in docs:
        batch.append(doc)
        if len(batch) < batch_docs:
            continue
        done += _encode_batch(tokenizer, batch, writer, target_tokens - done)
        batch = []
        if done >= target_tokens:
            break
        if done % 20_000_000 < batch_docs * 100:
            elapsed = time.time() - started
            rate = done / max(elapsed, 1e-9)
            print(f"  {done:,} / {target_tokens:,} tokens ({rate:,.0f} tok/s)", flush=True)
    if batch and done < target_tokens:
        done += _encode_batch(tokenizer, batch, writer, target_tokens - done)
    return done


def _encode_batch(tokenizer, docs: list[str], writer: ShardWriter, remaining: int) -> int:
    encodings = tokenizer.encode_batch(docs, add_special_tokens=True)
    eos = writer.eos_id
    pieces = []
    written = 0
    for encoding in encodings:
        ids = np.asarray(encoding.ids, dtype=np.uint32)
        take = min(int(ids.size) + 1, remaining - written)
        if take <= 0:
            break
        pieces.append(ids[: take - 1])
        pieces.append(np.asarray([eos], dtype=np.uint32))
        written += take
    if pieces:
        writer.add(np.concatenate(pieces))
    return written


# ---------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="tinystories", choices=["tinystories", "text"])
    parser.add_argument("--input", default=None, help="file or directory for --source text")
    parser.add_argument("--out_dir", default="data/tinystories")
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--train_tokens", type=int, default=300_000_000)
    parser.add_argument("--val_tokens", type=int, default=5_000_000)
    parser.add_argument("--shard_tokens", type=int, default=50_000_000)
    parser.add_argument("--sample_docs", type=int, default=200_000, help="docs used to train the BPE")
    parser.add_argument("--max_docs", type=int, default=0, help="0 streams the whole split")
    parser.add_argument("--tokenizer", default=None, help="reuse an existing tokenizer.json")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(args.tokenizer) if args.tokenizer else out / "tokenizer.json"
    input_path = Path(args.input) if args.input else None

    if tokenizer_path.exists():
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"reusing tokenizer {tokenizer_path}")
    else:
        if args.source != "tinystories":
            docs = document_stream(args.source, "train", input_path, args.sample_docs)
        else:
            docs = document_stream(args.source, "train", input_path, args.sample_docs)
        train_tokenizer(docs, args.vocab_size, tokenizer_path, args.sample_docs)
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    vocab = tokenizer.get_vocab_size()
    if vocab > args.vocab_size:
        raise SystemExit(f"tokenizer has {vocab} tokens, config expects {args.vocab_size}")

    print(f"building train split -> {out}", flush=True)
    train_writer = ShardWriter(out, args.shard_tokens, EOS_ID)
    tokenize_split(
        tokenizer,
        document_stream(args.source, "train", input_path, args.max_docs),
        args.train_tokens,
        train_writer,
    )
    train_tokens = train_writer.finish()

    val_tokens = 0
    if args.val_tokens > 0:
        print(f"building validation split -> {out / 'val'}", flush=True)
        val_writer = ShardWriter(out / "val", args.shard_tokens, EOS_ID)
        tokenize_split(
            tokenizer,
            document_stream(args.source, "validation", input_path, args.max_docs),
            args.val_tokens,
            val_writer,
        )
        val_tokens = val_writer.finish()

    summary = {
        "source": args.source,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "vocab_size": vocab,
        "eos_token_id": EOS_ID,
    }
    (out / "build.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
