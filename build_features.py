"""Preprocess review text and build n-gram feature sets.

Reads the train/dev/test CSVs produced by build_labeled_datasets.py,
cleans and tokenizes the review text, then builds count-based
unigram, unigram+bigram, and unigram+bigram+trigram feature
matrices for a multinomial logistic regression baseline.

Pipeline:
    1. Clean text: lowercase, collapse newlines/tabs/whitespace,
       strip URLs, drop punctuation and digit-only tokens.
    2. Tokenize with NLTK's word_tokenize when available, falling
       back to a regex word tokenizer otherwise.
    3. Optionally strip the restaurant's own name from its reviews
       (see STRIP_BUSINESS_NAME below).
    4. Remove English stopwords.
    5. Vectorize with CountVectorizer at several n-gram ranges.

Two decisions worth stating in the writeup:

  * The vectorizer is FIT ON TRAIN ONLY and merely applied to dev
    and test. Fitting on all splits would let test vocabulary leak
    into the feature space and inflate scores.

  * Stopwords are removed BEFORE n-grams are formed, so a bigram
    like "not good" becomes impossible while "pad thai" survives
    intact. For cuisine classification that trade is acceptable,
    since dish names rarely hinge on function words, but it would
    be the wrong default for sentiment work.

Outputs, per variant and n-gram range, into FEATURES_DIR:
    X_{split}.npz        sparse count matrix (scipy CSR)
    y_{split}.npy        integer-encoded labels
    vocabulary.json      feature name -> column index
    label_classes.json   integer label -> label name
    feature_stats.txt    shapes, vocab sizes, sparsity
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

# Which variant to featurize: "capped" (balanced) or "natural".
# The balanced set is the one intended for the n-gram baseline; the
# natural set is far larger and trigram vocabularies on it get
# expensive fast.
VARIANT = "capped"

DATASETS_DIR = "datasets"
FEATURES_DIR = "features"
SPLITS = ["train", "dev", "test"]

# N-gram ranges to build, as (min_n, max_n).
#
# Cumulative ranges (1,1) (1,2) (1,3) match the project plan: start
# with unigrams, then add bigrams, then trigrams, so each set is a
# superset of the previous one.
#
# Isolated ranges (2,2) and (3,3) are included so the contribution of
# bigrams and trigrams can be measured on their own. Without these,
# comparing (1,2) against (1,3) under a feature cap mostly measures
# which terms won a fixed budget rather than whether higher-order
# n-grams carry signal.
NGRAM_RANGES = [(1, 1), (1, 2), (1, 3), (2, 2), (3, 3)]

# Ignore terms appearing in fewer than this many documents. Without
# a floor, trigram vocabularies run to millions of columns that each
# appear once and contribute nothing but noise and memory pressure.
MIN_DF = 5

# Hard cap on vocabulary size, keeping the most frequent terms.
# None means no cap.
#
# An earlier run at 200,000 saw both (1,2) and (1,3) land on exactly
# 200,000 features, meaning the cap, not the data, decided the
# vocabulary. Because unigrams and bigrams are far more frequent than
# trigrams, a tight budget fills with lower-order terms and the
# "trigram" set ends up being a bigram set with a thin trigram layer:
# nnz rose only 4% from (1,2) to (1,3). The matrices are extremely
# sparse (~0.04% density), and nnz rather than column count drives
# memory, so a larger cap costs little.
MAX_FEATURES = 1_000_000

# If True, tokens matching words in the review's own business_name
# are removed. A review of "Taco Bell" containing "taco bell" would
# otherwise hand the model the label for free, since business names
# frequently name the cuisine. Leaving this True makes the learned
# features reflect food vocabulary rather than brand memorization.
STRIP_BUSINESS_NAME = True

# Rows per chunk when reading the CSVs.
CHUNK_SIZE = 50_000

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NON_LETTER_PATTERN = re.compile(r"[^a-z\s']")
WHITESPACE_PATTERN = re.compile(r"\s+")
WORD_PATTERN = re.compile(r"[a-z][a-z']+")

# Fallback stopword list, used only if NLTK is unavailable. Kept
# deliberately close to NLTK's English list.
FALLBACK_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him",
    "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did",
    "doing", "a", "an", "the", "and", "but", "if", "or", "because",
    "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now",
}


def load_nltk_tools() -> tuple:
    """Load NLTK's tokenizer and stopwords, with regex fallbacks.

    NLTK needs its 'punkt' and 'stopwords' corpora downloaded once:
        python -m nltk.downloader punkt punkt_tab stopwords

    Returns:
        A tuple of (tokenize function, stopword set, backend name).
    """
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        # Touch both corpora so a missing download fails here rather
        # than part-way through a long run.
        stop = set(stopwords.words("english"))
        word_tokenize("test sentence")
        return word_tokenize, stop, "nltk"

    except Exception as exc:
        print(f"NLTK unavailable ({type(exc).__name__}), using regex "
              f"tokenizer and built-in stopword list.", file=sys.stderr)
        print("  For NLTK: pip install nltk && python -m nltk.downloader "
              "punkt punkt_tab stopwords", file=sys.stderr)
        return (lambda text: WORD_PATTERN.findall(text),
                FALLBACK_STOPWORDS, "regex")


def clean_text(text: str) -> str:
    """Lowercase and strip URLs, punctuation, and extra whitespace.

    Args:
        text: Raw review text.

    Returns:
        Cleaned lowercase text with newlines and tabs collapsed to
        single spaces.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = NON_LETTER_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def name_tokens(business_name: str) -> set:
    """Extract lowercase tokens from a business name.

    Args:
        business_name: The restaurant's name.

    Returns:
        A set of lowercase word tokens from the name.
    """
    if not isinstance(business_name, str):
        return set()
    return set(WORD_PATTERN.findall(business_name.lower()))


def preprocess_row(text: str,
                   business_name: str,
                   tokenize,
                   stop_words: set) -> str:
    """Clean, tokenize and filter one review into a token string.

    Args:
        text: Raw review text.
        business_name: The reviewed restaurant's name.
        tokenize: Tokenizer function.
        stop_words: Stopwords to remove.

    Returns:
        A space-joined string of surviving tokens, ready for a
        whitespace-splitting vectorizer.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)

    drop = stop_words
    if STRIP_BUSINESS_NAME:
        drop = drop | name_tokens(business_name)

    kept = [
        tok for tok in tokens
        if len(tok) > 1 and tok not in drop and not tok.isdigit()
    ]
    return " ".join(kept)


def preprocess_split(csv_path: Path,
                     tokenize,
                     stop_words: set,
                     chunk_size: int) -> tuple:
    """Preprocess one split's CSV into token strings and labels.

    Args:
        csv_path: Path to the split CSV.
        tokenize: Tokenizer function.
        stop_words: Stopwords to remove.
        chunk_size: Rows to read per chunk.

    Returns:
        A tuple of (list of token strings, list of label strings).
    """
    docs = []
    labels = []

    reader = pd.read_csv(
        csv_path, usecols=["business_name", "label", "text"],
        chunksize=chunk_size,
    )
    for chunk in reader:
        for row in chunk.itertuples(index=False):
            docs.append(preprocess_row(
                row.text, row.business_name, tokenize, stop_words
            ))
            labels.append(row.label)

    print(f"  {csv_path.name}: {len(docs):,} documents")
    return docs, labels


def encode_labels(train_labels: list) -> dict:
    """Map label strings to integer ids, sorted for stability.

    Args:
        train_labels: Label strings from the training split.

    Returns:
        A dict of label string -> integer id.
    """
    classes = sorted(set(train_labels))
    return {label: i for i, label in enumerate(classes)}


def build_ngram_features(docs: dict,
                         labels: dict,
                         label_ids: dict,
                         ngram_range: tuple,
                         output_dir: Path) -> dict:
    """Vectorize all splits at one n-gram range and save to disk.

    The vectorizer is fit on train only, then applied to dev and
    test, so no dev/test vocabulary enters the feature space.

    Args:
        docs: split name -> list of token strings.
        labels: split name -> list of label strings.
        label_ids: label string -> integer id.
        ngram_range: (min_n, max_n) passed to CountVectorizer.
        output_dir: Directory for this n-gram range's outputs.

    Returns:
        A dict of summary statistics for reporting.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        min_df=MIN_DF,
        max_features=MAX_FEATURES,
        # Text is already cleaned, tokenized and stopword-filtered,
        # so the vectorizer only needs to split on whitespace.
        analyzer="word",
        token_pattern=r"\S+",
        lowercase=False,
    )

    x_train = vectorizer.fit_transform(docs["train"])
    matrices = {"train": x_train}
    for split in SPLITS:
        if split != "train":
            matrices[split] = vectorizer.transform(docs[split])

    # Count how many features are unigrams, bigrams, trigrams. If a
    # range's vocabulary sits exactly at MAX_FEATURES, the cap rather
    # than the data chose the vocabulary and the composition below
    # shows which orders won that budget.
    order_counts = Counter(
        term.count(" ") + 1 for term in vectorizer.vocabulary_
    )

    stats = {
        "ngram_range": ngram_range,
        "vocab_size": len(vectorizer.vocabulary_),
        "order_counts": order_counts,
        "capped": len(vectorizer.vocabulary_) == MAX_FEATURES,
        "shapes": {},
        "nnz": {},
    }

    for split in SPLITS:
        matrix = matrices[split]
        sparse.save_npz(output_dir / f"X_{split}.npz", matrix)

        y = np.array([label_ids[l] for l in labels[split]], dtype=np.int32)
        np.save(output_dir / f"y_{split}.npy", y)

        stats["shapes"][split] = matrix.shape
        stats["nnz"][split] = matrix.nnz

    with open(output_dir / "vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(
            {term: int(idx) for term, idx in vectorizer.vocabulary_.items()},
            f,
        )

    classes = {idx: label for label, idx in label_ids.items()}
    with open(output_dir / "label_classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    return stats


def format_stats(all_stats: list, backend: str) -> str:
    """Format feature statistics into a readable report.

    Args:
        all_stats: List of per-n-gram-range stats dicts.
        backend: Which tokenizer backend was used.

    Returns:
        A formatted multi-line report string.
    """
    lines = [
        "N-GRAM FEATURE SETS",
        "=" * 64,
        f"Variant: {VARIANT}",
        f"Tokenizer: {backend}",
        f"min_df: {MIN_DF} | max_features: {MAX_FEATURES}",
        f"Business name stripped from text: {STRIP_BUSINESS_NAME}",
    ]

    order_name = {1: "unigram", 2: "bigram", 3: "trigram"}

    for stats in all_stats:
        lo, hi = stats["ngram_range"]
        if lo == hi:
            title = f"({lo},{hi}) {order_name[hi]} only"
        else:
            title = f"({lo},{hi}) up to {order_name[hi]}"
        lines.append(f"\n{title}")
        lines.append("-" * 64)

        vocab_line = f"  vocabulary: {stats['vocab_size']:,} features"
        if stats["capped"]:
            vocab_line += "   *** AT MAX_FEATURES CAP ***"
        lines.append(vocab_line)

        composition = ", ".join(
            f"{order_name[n]} {stats['order_counts'][n]:,}"
            for n in sorted(stats["order_counts"])
        )
        lines.append(f"  composition: {composition}")
        for split in SPLITS:
            rows, cols = stats["shapes"][split]
            nnz = stats["nnz"][split]
            density = nnz / (rows * cols) * 100 if rows and cols else 0
            lines.append(
                f"  {split:<6} {rows:>9,} x {cols:>9,}  "
                f"nnz {nnz:>12,}  density {density:.4f}%"
            )

    return "\n".join(lines)


def main() -> None:
    """Preprocess the chosen variant and build all n-gram sets."""
    variant_dir = Path(DATASETS_DIR) / VARIANT
    if not variant_dir.exists():
        raise FileNotFoundError(
            f"Variant directory not found: '{variant_dir}'. "
            "Run build_labeled_datasets.py first."
        )

    tokenize, stop_words, backend = load_nltk_tools()

    print(f"Preprocessing '{VARIANT}' variant...")
    docs = {}
    labels = {}
    for split in SPLITS:
        docs[split], labels[split] = preprocess_split(
            variant_dir / f"{split}.csv", tokenize, stop_words, CHUNK_SIZE
        )

    label_ids = encode_labels(labels["train"])
    print(f"Classes: {len(label_ids)}")

    features_root = Path(FEATURES_DIR) / VARIANT
    all_stats = []

    for ngram_range in NGRAM_RANGES:
        lo, hi = ngram_range
        print(f"Vectorizing n-gram range ({lo},{hi})...")
        out_dir = features_root / f"ngram_{lo}_{hi}"
        all_stats.append(build_ngram_features(
            docs, labels, label_ids, ngram_range, out_dir
        ))

    report = format_stats(all_stats, backend)
    print("\n" + report)

    features_root.mkdir(parents=True, exist_ok=True)
    stats_path = features_root / "feature_stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved feature stats to '{stats_path}'")


if __name__ == "__main__":
    main()