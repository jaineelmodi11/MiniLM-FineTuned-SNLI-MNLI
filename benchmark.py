"""Compare this fine-tune against the model it started from.

Two measurements:

  triplet accuracy  On held-out SNLI/MNLI triplets, how often is the entailed
                    sentence closer to the anchor than the contradicting one.
  retrieval         24 sentences in 12 paraphrase pairs. For each sentence, is
                    its nearest neighbour its own paraphrase.

Run: python benchmark.py
"""
import itertools

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import TripletEvaluator

PAIRS = [
    ["A man is playing a guitar on stage.", "A musician performs for a crowd."],
    ["The dog ran through the park.", "A canine sprinted across the field."],
    ["She is reading a book.", "A woman reads a novel."],
    ["The chef is chopping vegetables.", "A cook slices produce in a kitchen."],
    ["A child is riding a bicycle.", "A kid pedals a bike down the street."],
    ["The plane landed safely.", "An aircraft touched down without incident."],
    ["He is repairing a car engine.", "A mechanic fixes a motor."],
    ["Snow is falling on the mountain.", "Flakes drift down over the peaks."],
    ["The team won the championship.", "They took home the title."],
    ["She sent an email to her manager.", "A woman messaged her boss."],
    ["The cat slept on the windowsill.", "A feline napped by the window."],
    ["Prices at the store increased.", "The shop raised its costs."],
]

MODELS = [
    ("this repo (SoftmaxLoss)", "./fine_tuned_model"),
    ("base all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
]


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    test = load_dataset("sentence-transformers/all-nli", "triplet")["test"]
    test = test.select(range(3000))
    evaluator = TripletEvaluator(
        anchors=test["anchor"], positives=test["positive"],
        negatives=test["negative"], name="nli",
    )
    flat = [s for pair in PAIRS for s in pair]

    print(f"{'model':28} {'triplet acc':>12} {'retrieval':>12}")
    print("-" * 54)
    for label, path in MODELS:
        model = SentenceTransformer(path)
        triplet = evaluator(model)["nli_cosine_accuracy"]
        emb = model.encode(flat, show_progress_bar=False)
        hits = sum(
            1 for i in range(len(flat))
            if max((j for j in range(len(flat)) if j != i),
                   key=lambda j: cosine(emb[i], emb[j])) // 2 == i // 2
        )
        print(f"{label:28} {triplet:>12.4f} {hits:>8}/{len(flat)}")


if __name__ == "__main__":
    main()
