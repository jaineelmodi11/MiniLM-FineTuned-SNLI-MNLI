<div align="center">

# 🔬 MiniLM Fine-Tuned (SNLI/MNLI)

**Fine-tuning `all-MiniLM-L6-v2` on NLI data, and measuring whether it actually helped.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![sentence-transformers](https://img.shields.io/badge/sentence--transformers-4.1-orange.svg)](https://sbert.net)

</div>

<p align="center">
  <img src="docs/benchmark.svg" alt="Benchmark comparing this fine-tune against base all-MiniLM-L6-v2" width="740">
</p>

---

## 📖 What this is

`fine_tuned_model/` holds `all-MiniLM-L6-v2` fine-tuned on 549k SNLI and MNLI
pairs for 4 epochs with `SoftmaxLoss`.

It is worse than the model it started from, and the point of this repo is now
the measurement rather than the model. Run [`benchmark.py`](benchmark.py) and
you get the table below.

---

## 📊 Results

Held-out SNLI/MNLI test set, 3,000 triplets. **Triplet accuracy** asks how often
the entailed sentence sits closer to the anchor than the contradicting one.
**Retrieval** takes 24 sentences arranged in 12 paraphrase pairs and asks whether
each sentence's nearest neighbour is its own paraphrase.

| Model | Triplet accuracy | Retrieval |
| :--- | ---: | ---: |
| This fine-tune (`SoftmaxLoss`) | 0.7007 | 7/24 |
| Base `all-MiniLM-L6-v2` | **0.9430** | **24/24** |
| Retrained with `MultipleNegativesRankingLoss` | **0.9453** | **24/24** |

The fine-tune sits **24 accuracy points below** the off-the-shelf model. On the
retrieval check it gets 7 of 24 while the base model gets all of them.

---

## 🔍 Why SoftmaxLoss hurts here

`SoftmaxLoss` puts a classification head on top of `[u, v, |u-v|]` and trains
that head to predict the NLI label. Nothing in the objective asks the two
embeddings to be close to each other when the sentences mean the same thing.
The head learns the task, and the embedding space drifts away from the one
`all-MiniLM-L6-v2` shipped with.

It is the original 2019 SBERT recipe, and it was superseded for exactly this
reason.

---

## 🔁 Retraining with the right loss

`MultipleNegativesRankingLoss` optimises the thing that was actually wanted:
pull an anchor toward its entailed sentence, push it away from everything else
in the batch. Retrained on 100k triplets for one epoch, it reaches **0.9453**.

That beats the base model by 0.2 points, which is close to nothing, and the
reason is worth stating plainly: `all-MiniLM-L6-v2` was already trained on over
a billion sentence pairs, SNLI and MNLI among them. There was never much
headroom. If you want better embeddings than `all-MiniLM-L6-v2`, NLI fine-tuning
is not the lever.

---

## 🧪 Reproduce

```bash
pip install sentence-transformers datasets
python benchmark.py
```

Downloads the `sentence-transformers/all-nli` test split and evaluates both
models. Takes a couple of minutes on CPU.

---

## 📌 On the 91.2% figure

An earlier version of this README reported 91.2% MNLI matched dev accuracy.
That number came from a logistic regression trained **on top of** these
embeddings to classify NLI, which is a different measurement from whether the
embeddings themselves are any good. Both can be true at once. The classifier is
not in this repo, so that figure is not reproducible here.

---

## 📄 License

MIT. See [LICENSE](LICENSE).
