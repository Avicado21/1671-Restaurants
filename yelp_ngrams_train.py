from pathlib import Path
import numpy as np
from scipy import sparse
import json

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


BASE_DIR = Path("yelp")

NGRAM_DIR = BASE_DIR / "features/capped"

with open("yelp/features/capped/ngram_1_1/label_classes.json", "r") as file:
    data = json.load(file)

cuisine_names = list(data.values())


def train_ngram_model(ngram_range, tfidf=False):
    # Load the datasets for ngrams
    
    TARGET_DIR = NGRAM_DIR / f"ngram_{ngram_range}"

    X_train = sparse.load_npz(TARGET_DIR / "X_train.npz")
    X_dev = sparse.load_npz(TARGET_DIR / "X_dev.npz")
    
    if tfidf:
        
        tfidf_transformer = TfidfTransformer()
        
        X_train = tfidf_transformer.fit_transform(X_train)
        X_dev = tfidf_transformer.transform(X_dev)

    y_train = np.load(TARGET_DIR / "y_train.npy")
    y_dev = np.load(TARGET_DIR / "y_dev.npy")
    
    # Training a basic Logistic regression model
    model = LogisticRegression(multi_class="multinomial", max_iter=1000, random_state=1671)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_dev)

    # Evaluation
    accuracy = accuracy_score(y_dev, y_pred)
    macro_f1 = f1_score(y_dev, y_pred, average="macro")
    precision = precision_score(y_dev, y_pred, average="macro")
    recall = recall_score(y_dev, y_pred, average="macro")
    
    if ngram_range == "1_3":
        print(classification_report(
            y_dev,
            y_pred,
            labels=range(18),
            target_names=cuisine_names,
        ))

    return {
        "tf-idf": tfidf,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1, 
    }

    
# Baseline testing for all 5 configurations
# for (ngram, tfidf_flag) in [("1_1", True), ("1_2", False), ("1_3", True), ("2_2", False), ("3_3", False)]:
#     result = train_ngram_model(ngram, tfidf_flag)
#     print(ngram, result)

for ngram in ["1_1", "1_3"]:
    result = train_ngram_model(ngram, tfidf=True)
    print(ngram, result, flush=True)
    
# print("Training features:", X_train.shape)
# print("Development features:", X_dev.shape)

# print("Training labels:", y_train.shape)
# print("Development labels:", y_dev.shape)