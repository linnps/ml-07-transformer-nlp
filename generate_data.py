"""
Synthetic sentiment-classification text generator.

Template-based: each example is constructed from a sentence template
plus randomly-chosen lexical slot fillers (item, adjective, intensifier,
optional negation / adversative). The label is determined by the
combination of adjective polarity *and* whether negation is present —
which means models that ignore the "not" word will get those examples
wrong.

Why synthetic?
- Zero copyright concerns (no IMDb / Yelp / Amazon).
- We control the difficulty: the share of negated examples and the
  length of the vocabulary are knobs.
- Each test example carries a structural label (template id, has-negation
  flag) so we can do per-slice error analysis post-hoc.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

ITEMS = ["product", "movie", "book", "restaurant", "phone", "trip",
         "show", "service", "course", "song"]
POS_ADJS = ["amazing", "wonderful", "delightful", "excellent", "fantastic",
            "great", "lovely", "outstanding", "brilliant", "superb"]
NEG_ADJS = ["awful", "terrible", "horrible", "disappointing", "boring",
            "frustrating", "miserable", "dreadful", "weak", "poor"]
INTENSIFIERS = ["really", "very", "absolutely", "honestly", "quite", "remarkably"]

POSITIVE_TEMPLATES = [
    "The {item} was {intensifier} {pos_adj} .",
    "I {intensifier} loved the {item} , it was {pos_adj} .",
    "What a {pos_adj} {item} that was .",
]
NEGATIVE_TEMPLATES = [
    "The {item} was {intensifier} {neg_adj} .",
    "I {intensifier} hated the {item} , it was {neg_adj} .",
    "What a {neg_adj} {item} that was .",
]
NEGATED_POSITIVE = [
    "The {item} was not {intensifier} {neg_adj} , I actually enjoyed it .",
    "It was not {neg_adj} at all — the {item} was {pos_adj} .",
]
NEGATED_NEGATIVE = [
    "The {item} was not {intensifier} {pos_adj} , I really disliked it .",
    "It was not {pos_adj} at all — the {item} was {neg_adj} .",
]


@dataclass
class DataConfig:
    n_train: int = 1500
    n_test: int = 400
    negation_ratio: float = 0.25
    seed: int = 42


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        item=rng.choice(ITEMS),
        pos_adj=rng.choice(POS_ADJS),
        neg_adj=rng.choice(NEG_ADJS),
        intensifier=rng.choice(INTENSIFIERS),
    )


def _make_one(rng: random.Random, negation_ratio: float):
    use_neg = rng.random() < negation_ratio
    label = rng.randint(0, 1)              # 0 = negative, 1 = positive
    if use_neg:
        templates = NEGATED_POSITIVE if label == 1 else NEGATED_NEGATIVE
        struct = "negated"
    else:
        templates = POSITIVE_TEMPLATES if label == 1 else NEGATIVE_TEMPLATES
        struct = "plain"
    text = _fill(rng.choice(templates), rng)
    return text, label, struct


def generate(cfg: DataConfig):
    rng = random.Random(cfg.seed)
    train, test = [], []
    for _ in range(cfg.n_train):
        train.append(_make_one(rng, cfg.negation_ratio))
    for _ in range(cfg.n_test):
        test.append(_make_one(rng, cfg.negation_ratio))
    return train, test


def vocab_from(corpus) -> dict[str, int]:
    """Whitespace tokenizer with <pad> and <unk>."""
    vocab = {"<pad>": 0, "<unk>": 1}
    for text, _, _ in corpus:
        for tok in text.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic sentiment dataset.")
    p.add_argument("--n-train", type=int, default=1500)
    p.add_argument("--n-test", type=int, default=400)
    p.add_argument("--negation-ratio", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    args = p.parse_args()

    cfg = DataConfig(n_train=args.n_train, n_test=args.n_test,
                     negation_ratio=args.negation_ratio, seed=args.seed)
    train, test = generate(cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def dump(name, rows):
        with open(args.out_dir / name, "w") as f:
            for text, lab, struct in rows:
                f.write(json.dumps({"text": text, "label": lab, "struct": struct}) + "\n")
    dump("train.jsonl", train)
    dump("test.jsonl", test)
    print(f"Train: {len(train)}, Test: {len(test)}")
    print(f"Saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
