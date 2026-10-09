"""Train the letter classifier from data/landmarks.csv.

Run:  python train.py
Creates letter_model.joblib, which main.py loads automatically.
"""
import csv
import os
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "data", "landmarks.csv")
MODEL_FILE = os.path.join(HERE, "letter_model.joblib")


def main():
    if not os.path.exists(DATA_FILE):
        raise SystemExit("No data yet. Run collect_data.py first.")

    labels, rows = [], []
    with open(DATA_FILE, newline="") as f:
        for row in csv.reader(f):
            if len(row) == 64:
                labels.append(row[0])
                rows.append([float(v) for v in row[1:]])
    X, y = np.array(rows, dtype=np.float32), np.array(labels)

    counts = Counter(str(v) for v in y)
    print("Samples per letter:", dict(sorted(counts.items())))
    if len(counts) < 2:
        raise SystemExit("Record at least 2 different letters first.")
    thin = [k for k, v in counts.items() if v < 20]
    if thin:
        print("Warning: fewer than 20 samples for", sorted(thin))

    stratify = y if min(counts.values()) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=stratify)
    model = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    model.fit(Xtr, ytr)
    print(f"\nHeld-out accuracy: {model.score(Xte, yte):.1%}\n")
    print(classification_report(yte, model.predict(Xte), zero_division=0))

    model.fit(X, y)  # final model uses all the data
    joblib.dump(model, MODEL_FILE)
    print("Saved", MODEL_FILE)


if __name__ == "__main__":
    main()
