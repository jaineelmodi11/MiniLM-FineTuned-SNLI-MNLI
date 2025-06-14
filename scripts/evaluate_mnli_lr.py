# scripts/evaluate_mnli_lr.py

from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def get_features(model, premises, hypos, batch_size=128):
    """Encode premise/hypothesis pairs into concatenated embeddings."""
    feats = []
    for i in range(0, len(premises), batch_size):
        emb_p = model.encode(premises[i:i+batch_size], show_progress_bar=False)
        emb_h = model.encode(hypos   [i:i+batch_size], show_progress_bar=False)
        feats.append(np.hstack([emb_p, emb_h]))
    return np.vstack(feats)

def main():
    # Load your fine-tuned model
    model = SentenceTransformer('fine_tuned_model')

    # Load & filter MNLI train
    ds_train = load_dataset('multi_nli', split='train')
    ds_train = ds_train.filter(lambda x: x['label'] != -1)
    premises = ds_train['premise']
    hypos    = ds_train['hypothesis']
    labels   = np.array(ds_train['label'])

    print("Encoding train set...")
    X_train = get_features(model, premises, hypos)
    print("Fitting LogisticRegression on train set...")
    clf = LogisticRegression(
        multi_class='multinomial',
        solver='saga',
        max_iter=1000,
        n_jobs=-1
    )
    clf.fit(X_train, labels)

    # Load & filter MNLI matched dev
    ds_dev = load_dataset('multi_nli', split='validation_matched')
    ds_dev = ds_dev.filter(lambda x: x['label'] != -1)
    premises_dev = ds_dev['premise']
    hypos_dev    = ds_dev['hypothesis']
    labels_dev   = np.array(ds_dev['label'])

    print("Encoding dev (matched) set...")
    X_dev = get_features(model, premises_dev, hypos_dev)
    print("Predicting on dev set...")
    preds = clf.predict(X_dev)

    acc = accuracy_score(labels_dev, preds)
    print(f"\nMNLI matched dev accuracy (logreg on embeddings): {acc*100:.2f}%\n")

if __name__ == "__main__":
    main()
