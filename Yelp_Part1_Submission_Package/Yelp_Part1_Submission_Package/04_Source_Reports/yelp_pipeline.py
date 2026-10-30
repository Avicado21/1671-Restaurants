"""End-to-end pipeline: Yelp Open Dataset -> n-gram feature matrices.

Runs every stage of the restaurant-cuisine classification data
pipeline in order. Each stage skips itself if its output already
exists, so reruns are cheap; pass --force to rebuild anyway.

Stages:
    1 download   Fetch the dataset tar from a URL (optional).
    2 extract    Unpack the tar into JSON files.
    3 reviews    Filter review.json to businesses tagged
                 "Restaurants", joined with business fields.
                 -> yelp_restaurant_reviews.csv
    4 stats      Summary statistics on the full restaurant CSV.
                 -> dataset_stats_summary.txt
    5 audit      Yelp category frequencies and label-collision
                 analysis.
                 -> category_counts.csv, category_audit_report.txt
    6 datasets   Assign cuisine labels, resolve collisions, split
                 train/dev/test grouped by business, and write both
                 the natural (unbalanced) and capped (balanced)
                 variants.
                 -> datasets/natural/, datasets/capped/,
                    datasets/dataset_distribution.txt,
                    datasets/label_map.txt
    7 features   Preprocess text and build n-gram count matrices.
                 -> features/capped/ngram_*/
    8 split      Optional. Split the main CSV into parts under a size
                 limit, for hosting in a size-capped repo. Runs only
                 when --split-parts is passed.
                 -> yelp_restaurant_reviews_parts/

Usage:
    python3 yelp_pipeline.py --url "<signed Yelp URL>"
    python3 yelp_pipeline.py --tar "Yelp JSON/yelp_dataset.tar"
    python3 yelp_pipeline.py --extracted yelp_dataset_extracted
    python3 yelp_pipeline.py --from-stage 6
    python3 yelp_pipeline.py --only 7 --force
    python3 yelp_pipeline.py --tar . --split-parts

Note on the download stage: Yelp's dataset link is not a stable
public URL. It is issued per user after accepting the Terms of Use
and it expires, so --url only works with a fresh link you copied
yourself. If you already downloaded the tar, use --tar instead.

Note on licensing: the Yelp Open Dataset Terms of Use restrict this
data to non-commercial academic use, prohibit redistributing it to
third parties, and require Yelp's review before public release of
findings. Keep every artifact this script produces private to your
course project.
"""

import argparse
import csv
import json
import random
import re
import sys
import tarfile
import urllib.request
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
TAR_PATH = "yelp_dataset.tar"
EXTRACT_DIR = "yelp_dataset_extracted"
REVIEWS_CSV = "yelp_restaurant_reviews.csv"
DATASETS_DIR = "datasets"
FEATURES_DIR = "features"

STATS_SUMMARY = "dataset_stats_summary.txt"
CATEGORY_COUNTS = "category_counts.csv"
CATEGORY_REPORT = "category_audit_report.txt"
PARTS_DIR = "yelp_restaurant_reviews_parts"

# How many top cities/states to report in the stats summary.
TOP_N = 10

# How many categories to print in the audit report. The full list
# always goes to CATEGORY_COUNTS.
AUDIT_TOP_N = 60

# Per-part size ceiling for the optional split stage. Set under the
# target limit to leave margin, since the last row written before the
# size check can push slightly over.
MAX_PART_SIZE_BYTES = 24 * 1024 * 1024  # 24 MB

# ----------------------------------------------------------------
# Stage 3: restaurant filtering
# ----------------------------------------------------------------
RESTAURANT_CATEGORY = "Restaurants"
REVIEW_FIELDS = [
    "review_id", "business_id", "business_name", "city", "state",
    "stars", "date", "text",
]

# ----------------------------------------------------------------
# Stage 4: labeling and splitting
# ----------------------------------------------------------------
SPLIT_FRACTIONS = {"train": 0.8, "dev": 0.1, "test": 0.1}
RANDOM_SEED = 1671
CHUNK_SIZE = 100_000

# Labels totalling fewer than this many reviews are dropped. At well
# under 1% of the data a class is effectively never predicted and
# only depresses macro-F1.
MIN_REVIEWS_PER_LABEL = 15_000

# Per-class review ceiling for the capped variant. None uses the
# smallest surviving class, which maximizes balance but discards most
# of the largest classes.
CAPPED_CEILING = None

# A business matching "American" plus exactly one specific cuisine is
# assigned that cuisine rather than dropped. American is Yelp's broad
# default tag and carries less information than a specific cuisine.
# Without this, American collisions cost Southern 59% of its reviews,
# French 42% and British & Irish 41% -- a systematic bias against the
# cuisines closest to American dining.
CUISINE_BEATS_AMERICAN = True
AMERICAN_LABEL = "American"

# Businesses matching no cuisine are dropped rather than bucketed, so
# no residual "Other" class exists.
KEEP_UNLABELED_AS_OTHER = False
OTHER_LABEL = "Other"

DATASET_FIELDS = [
    "review_id", "business_id", "business_name", "city", "state",
    "stars", "date", "label", "text",
]

# Inclusion rule: a Yelp category is eligible as a label only if it
# names a national or regional cuisine. Dish tags (Salad, Pizza,
# Tacos), format tags (Fast Food, Diners), dietary tags (Vegan) and
# meal tags (Breakfast & Brunch) are excluded. Subtypes are folded
# into their parent cuisine to raise class sizes and remove
# within-cuisine collisions.
LABEL_MAP = {
    # Merged: Yelp's New/Traditional split tracks style and price
    # register rather than cuisine, and is inconsistently applied.
    "American": {"American (New)", "American (Traditional)"},
    "Mexican": {"Mexican", "Tex-Mex", "New Mexican Cuisine"},
    "Italian": {"Italian"},
    "Chinese": {
        "Chinese", "Szechuan", "Cantonese", "Shanghainese",
        "Taiwanese", "Hong Kong Style Cafe", "Hainan",
    },
    "Japanese": {"Japanese"},
    "Korean": {"Korean"},
    "Thai": {"Thai"},
    "Vietnamese": {"Vietnamese"},
    "South Asian": {
        "Indian", "Pakistani", "Bangladeshi", "Sri Lankan",
        "Himalayan/Nepalese",
    },
    # Collapsed: Greek/Mediterranean/Middle Eastern produced the two
    # largest collision pairs in the audit and Yelp applies them
    # near-interchangeably to the same restaurants.
    "Eastern Mediterranean": {
        "Greek", "Mediterranean", "Middle Eastern", "Lebanese",
        "Turkish", "Persian/Iranian", "Israeli", "Egyptian",
        "Afghan", "Moroccan",
    },
    "French": {"French"},
    "Spanish": {"Spanish"},
    "German": {"German", "Austrian"},
    "British & Irish": {"Irish", "British", "Scottish"},
    "Eastern European": {
        "Polish", "Russian", "Ukrainian", "Hungarian", "Czech",
        "Georgian", "Uzbek", "Eastern European",
    },
    "Latin American": {
        "Latin American", "Colombian", "Venezuelan", "Salvadoran",
        "Honduran", "Argentine", "Peruvian", "Brazilian",
    },
    "Caribbean": {
        "Caribbean", "Cuban", "Puerto Rican", "Dominican", "Haitian",
    },
    "Cajun/Creole": {"Cajun/Creole"},
    "Southern": {"Southern", "Soul Food"},
    "Hawaiian": {"Hawaiian"},
    "African": {"African", "Ethiopian", "Senegalese", "Somali"},
}

# ----------------------------------------------------------------
# Stage 5: features
# ----------------------------------------------------------------
FEATURE_VARIANT = "capped"
SPLITS = ["train", "dev", "test"]

# Cumulative ranges match the project plan; isolated (2,2) and (3,3)
# let the contribution of each order be measured on its own.
NGRAM_RANGES = [(1, 1), (1, 2), (1, 3), (2, 2), (3, 3)]

MIN_DF = 5

# A 200,000 cap was previously binding on both (1,2) and (1,3),
# letting the budget rather than the data pick the vocabulary. The
# matrices are ~0.02% dense and nnz drives memory, so a large cap is
# cheap.
MAX_FEATURES = 1_000_000

# Tokens from a review's own business_name are removed. Restaurant
# names frequently contain the cuisine ("Tokyo Sushi Bar"), which
# would hand the model the label directly. Note the cost: genuine
# cuisine vocabulary is also removed when the name contains it.
STRIP_BUSINESS_NAME = True

FEATURE_CHUNK_SIZE = 50_000

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NON_LETTER_PATTERN = re.compile(r"[^a-z\s']")
WHITESPACE_PATTERN = re.compile(r"\s+")
WORD_PATTERN = re.compile(r"[a-z][a-z']+")

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


# ================================================================
# Stage 1: download
# ================================================================

def stage_download(url: str, tar_path: Path, force: bool) -> None:
    """Download the dataset tar from a URL.

    Args:
        url: A signed Yelp download URL, copied from their site.
        tar_path: Where to save the tar.
        force: Re-download even if the file exists.
    """
    if tar_path.exists() and not force:
        size_gb = tar_path.stat().st_size / 1e9
        print(f"[1/8] tar already present ({size_gb:.2f} GB), skipping")
        return

    print(f"[1/8] Downloading to '{tar_path}' (this is several GB)...")

    def report(block_num, block_size, total_size):
        if total_size <= 0:
            return
        done = block_num * block_size
        pct = min(100, done / total_size * 100)
        sys.stdout.write(
            f"\r  {pct:5.1f}%  {done/1e9:.2f} / {total_size/1e9:.2f} GB"
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, tar_path, reporthook=report)
    print("\n  done")


# ================================================================
# Stage 2: extract
# ================================================================

def stage_extract(tar_path: Path, extract_dir: Path, force: bool) -> Path:
    """Extract the dataset tar, or locate already-extracted JSON.

    Accepts a tar file, a directory containing one, or a directory of
    already-extracted JSON files, so it works whatever state the
    download is in.

    Args:
        tar_path: Path to a tar file or a directory holding one.
        extract_dir: Where to extract to.
        force: Re-extract even if JSON files are already present.

    Returns:
        The directory containing the extracted JSON files.
    """
    if extract_dir.exists() and list(extract_dir.rglob("*.json")) \
            and not force:
        print(f"[2/8] Already extracted in '{extract_dir}', skipping")
        return extract_dir

    path = Path(tar_path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: '{path}'")

    if path.is_dir():
        if list(path.rglob("*.json")):
            print(f"[2/8] Using extracted JSON in '{path}'")
            return path
        tars = list(path.rglob("*.tar"))
        if not tars:
            raise FileNotFoundError(
                f"No .json or .tar found under '{path}'"
            )
        path = tars[0]

    if not tarfile.is_tarfile(path):
        raise ValueError(f"Not a valid tar file: '{path}'")

    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[2/8] Extracting '{path}' -> '{extract_dir}'...")
    with tarfile.open(path) as tar:
        tar.extractall(path=extract_dir, filter="data")
    print("  done")
    return extract_dir


def find_json(extract_dir: Path, fragment: str) -> Path:
    """Locate a dataset JSON file by a distinguishing name fragment.

    Args:
        extract_dir: Directory the tar was extracted into.
        fragment: Substring to match, e.g. "business" or "review".

    Returns:
        Path to the matching JSON file.
    """
    matches = list(extract_dir.rglob(f"*{fragment}*.json"))
    if not matches:
        raise FileNotFoundError(
            f"No '*{fragment}*.json' under '{extract_dir}'"
        )
    return matches[0]


# ================================================================
# Stage 3: restaurant reviews CSV
# ================================================================

def parse_categories(raw) -> list:
    """Split Yelp's comma-separated category string into a list.

    Args:
        raw: The raw "categories" field, a comma-separated string
            or None.

    Returns:
        A list of cleaned category strings.
    """
    if not raw:
        return []
    return [c.strip() for c in raw.split(",") if c.strip()]


def stage_reviews(extract_dir: Path,
                  reviews_csv: Path,
                  force: bool) -> None:
    """Filter review.json down to restaurant reviews only.

    review.json holds millions of lines, so it is streamed one line
    at a time and written straight out rather than loaded into a
    DataFrame.

    Args:
        extract_dir: Directory holding the extracted JSON files.
        reviews_csv: Output CSV path.
        force: Rebuild even if the CSV exists.
    """
    if reviews_csv.exists() and not force:
        print(f"[3/8] '{reviews_csv}' exists, skipping")
        return

    business_path = find_json(extract_dir, "business")
    review_path = find_json(extract_dir, "review")

    restaurants = {}
    with open(business_path, "r", encoding="utf-8") as f:
        for line in f:
            business = json.loads(line)
            if RESTAURANT_CATEGORY in parse_categories(
                    business.get("categories")):
                restaurants[business["business_id"]] = {
                    "name": business.get("name", ""),
                    "city": business.get("city", ""),
                    "state": business.get("state", ""),
                }
    print(f"[3/8] {len(restaurants):,} restaurant businesses")

    n_total = n_kept = 0
    with open(review_path, "r", encoding="utf-8") as infile, \
            open(reviews_csv, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(
            out, fieldnames=REVIEW_FIELDS, lineterminator="\n"
        )
        writer.writeheader()

        for line in infile:
            n_total += 1
            review = json.loads(line)
            business = restaurants.get(review["business_id"])
            if business is None:
                continue
            n_kept += 1
            writer.writerow({
                "review_id": review["review_id"],
                "business_id": review["business_id"],
                "business_name": business["name"],
                "city": business["city"],
                "state": business["state"],
                "stars": review["stars"],
                "date": review["date"],
                "text": review["text"],
            })

    print(f"  scanned {n_total:,} reviews, kept {n_kept:,}")


# ================================================================
# Stage 4: summary statistics on the full restaurant CSV
# ================================================================

def stage_stats(reviews_csv: Path, out_path: Path, force: bool) -> None:
    """Compute summary statistics over the full restaurant CSV.

    Streams the file in chunks and keeps only running aggregates, so
    memory stays flat regardless of file size.

    Args:
        reviews_csv: Path to the restaurant reviews CSV.
        out_path: Where to write the summary report.
        force: Rebuild even if the report exists.
    """
    if out_path.exists() and not force:
        print(f"[4/8] '{out_path}' exists, skipping")
        return

    print("[4/8] Computing dataset statistics...")

    total = 0
    missing = None
    ratings = Counter()
    businesses = set()
    cities = Counter()
    states = Counter()
    min_date = max_date = None
    char_sum = word_sum = 0
    char_min = char_max = None

    for chunk in pd.read_csv(reviews_csv, chunksize=CHUNK_SIZE):
        total += len(chunk)
        chunk_missing = chunk.isna().sum()
        missing = chunk_missing if missing is None else missing + chunk_missing

        ratings.update(chunk["stars"].dropna().astype(int))
        businesses.update(chunk["business_id"].dropna().unique())
        cities.update(chunk["city"].dropna())
        states.update(chunk["state"].dropna())

        dates = pd.to_datetime(chunk["date"], errors="coerce")
        lo, hi = dates.min(), dates.max()
        if pd.notna(lo) and (min_date is None or lo < min_date):
            min_date = lo
        if pd.notna(hi) and (max_date is None or hi > max_date):
            max_date = hi

        text = chunk["text"].dropna()
        lengths = text.str.len()
        char_sum += lengths.sum()
        word_sum += text.str.split().str.len().sum()
        if char_min is None or lengths.min() < char_min:
            char_min = lengths.min()
        if char_max is None or lengths.max() > char_max:
            char_max = lengths.max()

    lines = [
        "YELP RESTAURANT REVIEWS - DATASET SUMMARY",
        "=" * 50,
        f"\nTotal rows (reviews): {total:,}",
        f"Unique businesses: {len(businesses):,}",
        "\nMissing values per column:",
    ]
    for column in REVIEW_FIELDS:
        lines.append(f"  {column}: {int(missing.get(column, 0)):,}")

    lines.append("\nRating distribution:")
    for rating in sorted(ratings):
        n = ratings[rating]
        lines.append(f"  {rating} star: {n:,} ({n/total*100:.1f}%)")

    lines.append(
        f"\nDate range: {min_date.date()} to {max_date.date()}"
    )
    lines.append(f"\nTop {TOP_N} cities by review count:")
    for city, n in cities.most_common(TOP_N):
        lines.append(f"  {city}: {n:,}")
    lines.append(f"\nTop {TOP_N} states by review count:")
    for state, n in states.most_common(TOP_N):
        lines.append(f"  {state}: {n:,}")

    lines.append("\nReview text length:")
    lines.append(f"  Average: {char_sum/total:.1f} characters, "
                 f"{word_sum/total:.1f} words")
    lines.append(f"  Shortest: {char_min:.0f} characters")
    lines.append(f"  Longest: {char_max:.0f} characters")

    report = "\n".join(lines)
    print(report)
    out_path.write_text(report, encoding="utf-8")
    print(f"  saved '{out_path}'")


# ================================================================
# Stage 5: category frequency and collision audit
# ================================================================

def stage_audit(extract_dir: Path,
                reviews_csv: Path,
                counts_path: Path,
                report_path: Path,
                force: bool) -> None:
    """Audit Yelp categories and analyze label collisions.

    Produces the evidence behind the taxonomy decisions: which Yelp
    categories exist and how common they are, how many businesses
    match no label / one label / several, and which label pairs
    collide most often.

    Args:
        extract_dir: Directory holding the extracted JSON files.
        reviews_csv: Path to the restaurant reviews CSV.
        counts_path: Where to write the category frequency CSV.
        report_path: Where to write the audit report.
        force: Rebuild even if outputs exist.
    """
    if report_path.exists() and not force:
        print(f"[5/8] '{report_path}' exists, skipping")
        return

    print("[5/8] Auditing categories and collisions...")

    restaurants = {}
    with open(find_json(extract_dir, "business"), "r",
              encoding="utf-8") as f:
        for line in f:
            business = json.loads(line)
            cats = parse_categories(business.get("categories"))
            if RESTAURANT_CATEGORY in cats:
                restaurants[business["business_id"]] = {
                    "name": business.get("name", ""),
                    "categories": cats,
                }

    review_counts = count_reviews_per_business(reviews_csv)

    # Per-category totals across restaurant businesses.
    biz_counts = Counter()
    cat_reviews = Counter()
    for business_id, info in restaurants.items():
        n = review_counts.get(business_id, 0)
        for category in info["categories"]:
            biz_counts[category] += 1
            cat_reviews[category] += n

    counts_df = pd.DataFrame(
        [
            {"category": c, "n_businesses": n,
             "n_reviews": cat_reviews[c]}
            for c, n in biz_counts.items()
        ]
    ).sort_values("n_businesses", ascending=False).reset_index(drop=True)
    counts_df.to_csv(counts_path, index=False)

    # Label coverage and collisions, before any tiebreak is applied,
    # so the raw collision picture is visible.
    n_unlabeled = n_single = n_collision = 0
    rev_unlabeled = rev_single = rev_collision = 0
    per_label_biz = Counter()
    per_label_rev = Counter()
    pairs = Counter()
    examples = defaultdict(list)

    for business_id, info in restaurants.items():
        matched = match_labels(info["categories"])
        n = review_counts.get(business_id, 0)

        if not matched:
            n_unlabeled += 1
            rev_unlabeled += n
        elif len(matched) == 1:
            n_single += 1
            rev_single += n
            label = next(iter(matched))
            per_label_biz[label] += 1
            per_label_rev[label] += n
        else:
            n_collision += 1
            rev_collision += n
            for pair in combinations(sorted(matched), 2):
                pairs[pair] += 1
                if len(examples[pair]) < 3:
                    examples[pair].append(info["name"])

    total_biz = n_unlabeled + n_single + n_collision
    total_rev = rev_unlabeled + rev_single + rev_collision

    def pct(part, whole):
        return (part / whole * 100) if whole else 0

    lines = [
        "CATEGORY FREQUENCY AMONG RESTAURANT BUSINESSES",
        "=" * 60,
        f"Distinct categories found: {len(counts_df):,}",
        f"\nTop {AUDIT_TOP_N} by business count:",
        f"{'category':<35}{'businesses':>12}{'reviews':>14}",
    ]
    for _, row in counts_df.head(AUDIT_TOP_N).iterrows():
        lines.append(
            f"{row['category'][:34]:<35}"
            f"{row['n_businesses']:>12,}{row['n_reviews']:>14,}"
        )

    lines += [
        "\n\nLABEL ASSIGNMENT (before any collision tiebreak)",
        "=" * 60,
        f"\nBusinesses matching no label:   {n_unlabeled:>8,} "
        f"({pct(n_unlabeled, total_biz):.1f}%)",
        f"Businesses matching one label:  {n_single:>8,} "
        f"({pct(n_single, total_biz):.1f}%)",
        f"Businesses matching 2+ labels:  {n_collision:>8,} "
        f"({pct(n_collision, total_biz):.1f}%)",
        f"\nReviews matching no label:      {rev_unlabeled:>10,} "
        f"({pct(rev_unlabeled, total_rev):.1f}%)",
        f"Reviews matching one label:     {rev_single:>10,} "
        f"({pct(rev_single, total_rev):.1f}%)",
        f"Reviews matching 2+ labels:     {rev_collision:>10,} "
        f"({pct(rev_collision, total_rev):.1f}%)",
        "\n\nPER-LABEL TOTALS (single-label businesses only)",
        "-" * 60,
        f"{'label':<25}{'businesses':>12}{'reviews':>14}",
    ]
    for label in sorted(LABEL_MAP):
        lines.append(
            f"{label[:24]:<25}{per_label_biz[label]:>12,}"
            f"{per_label_rev[label]:>14,}"
        )

    lines += ["\n\nCOLLIDING LABEL PAIRS", "-" * 60]
    if not pairs:
        lines.append("No collisions found.")
    else:
        for pair, n in pairs.most_common():
            lines.append(f"{pair[0]} + {pair[1]}: {n:,} businesses")
            lines.append(f"    e.g. {', '.join(examples[pair])}")

    report = "\n".join(lines)
    report_path.write_text(report, encoding="utf-8")
    print(f"  distinct categories: {len(counts_df):,} | "
          f"collisions: {n_collision:,} businesses")
    print(f"  saved '{counts_path}' and '{report_path}'")


# ================================================================
# Stage 6: labeled datasets
# ================================================================

def match_labels(categories: list) -> set:
    """Find which labels a business's categories match.

    Several categories mapping to the same label is not a collision,
    which is why this returns a set of labels rather than a count.

    Args:
        categories: The business's Yelp category strings.

    Returns:
        The set of matching label names.
    """
    cats = set(categories)
    return {
        label for label, label_cats in LABEL_MAP.items()
        if cats & label_cats
    }


def build_business_labels(business_path: Path) -> dict:
    """Assign one label per restaurant, resolving or dropping ties.

    Args:
        business_path: Path to business.json.

    Returns:
        A dict of business_id -> label.
    """
    labels = {}
    n_rest = n_unlabeled = n_collision = n_resolved = 0

    with open(business_path, "r", encoding="utf-8") as f:
        for line in f:
            business = json.loads(line)
            cats = parse_categories(business.get("categories"))
            if RESTAURANT_CATEGORY not in cats:
                continue

            n_rest += 1
            matched = match_labels(cats)

            if not matched:
                n_unlabeled += 1
                if KEEP_UNLABELED_AS_OTHER:
                    labels[business["business_id"]] = OTHER_LABEL
            elif len(matched) == 1:
                labels[business["business_id"]] = next(iter(matched))
            elif (CUISINE_BEATS_AMERICAN and len(matched) == 2
                  and AMERICAN_LABEL in matched):
                labels[business["business_id"]] = next(
                    iter(matched - {AMERICAN_LABEL})
                )
                n_resolved += 1
            else:
                n_collision += 1

    print(f"  restaurants {n_rest:,} | labeled {len(labels):,} | "
          f"unlabeled {n_unlabeled:,} | collisions {n_collision:,}")
    if n_resolved:
        print(f"  ({n_resolved:,} American+cuisine ties kept as cuisine)")
    return labels


def count_reviews_per_business(reviews_csv: Path) -> Counter:
    """Count reviews per business_id.

    Args:
        reviews_csv: Path to the restaurant reviews CSV.

    Returns:
        A Counter of business_id -> review count.
    """
    counts = Counter()
    for chunk in pd.read_csv(
            reviews_csv, usecols=["business_id"], chunksize=CHUNK_SIZE):
        counts.update(chunk["business_id"])
    return counts


def drop_small_labels(business_labels: dict,
                      review_counts: Counter) -> dict:
    """Remove labels whose total review count is under the minimum.

    Args:
        business_labels: business_id -> label.
        review_counts: business_id -> review count.

    Returns:
        The filtered business_labels dict.
    """
    if MIN_REVIEWS_PER_LABEL <= 0:
        return business_labels

    per_label = Counter()
    for business_id, label in business_labels.items():
        per_label[label] += review_counts.get(business_id, 0)

    small = {
        label: n for label, n in per_label.items()
        if n < MIN_REVIEWS_PER_LABEL
    }
    if small:
        listed = ", ".join(
            f"{k} ({v:,})"
            for k, v in sorted(small.items(), key=lambda x: x[1])
        )
        print(f"  dropping {len(small)} label(s) under "
              f"{MIN_REVIEWS_PER_LABEL:,} reviews: {listed}")

    return {
        b: l for b, l in business_labels.items() if l not in small
    }


def assign_splits(business_labels: dict,
                  review_counts: Counter) -> dict:
    """Assign businesses to splits, stratified by label.

    Allocation is greedy by review volume rather than business count,
    so splits land near their target proportions even though
    businesses vary hugely in review count. A business is never
    divided across splits, which prevents a model from memorizing
    restaurant-specific vocabulary seen in training.

    Args:
        business_labels: business_id -> label.
        review_counts: business_id -> review count.

    Returns:
        A dict of business_id -> split name.
    """
    rng = random.Random(RANDOM_SEED)
    by_label = defaultdict(list)
    for business_id, label in business_labels.items():
        by_label[label].append(business_id)

    splits = {}
    for label, business_ids in by_label.items():
        rng.shuffle(business_ids)
        total = sum(review_counts.get(b, 0) for b in business_ids)
        targets = {s: total * f for s, f in SPLIT_FRACTIONS.items()}
        filled = {s: 0 for s in SPLIT_FRACTIONS}

        for business_id in business_ids:
            chosen = max(
                SPLIT_FRACTIONS,
                key=lambda s: targets[s] - filled[s],
            )
            splits[business_id] = chosen
            filled[chosen] += review_counts.get(business_id, 0)

    return splits


def write_natural(reviews_csv: Path,
                  business_labels: dict,
                  business_splits: dict,
                  out_dir: Path) -> dict:
    """Write the natural-distribution variant.

    Args:
        reviews_csv: Path to the restaurant reviews CSV.
        business_labels: business_id -> label.
        business_splits: business_id -> split name.
        out_dir: Base datasets directory.

    Returns:
        A dict of split -> Counter of label -> count.
    """
    natural = out_dir / "natural"
    natural.mkdir(parents=True, exist_ok=True)

    files, writers = {}, {}
    for split in SPLIT_FRACTIONS:
        files[split] = open(
            natural / f"{split}.csv", "w", newline="", encoding="utf-8"
        )
        writers[split] = csv.DictWriter(
            files[split], fieldnames=DATASET_FIELDS, lineterminator="\n"
        )
        writers[split].writeheader()

    dist = {split: Counter() for split in SPLIT_FRACTIONS}

    for chunk in pd.read_csv(reviews_csv, chunksize=CHUNK_SIZE):
        for row in chunk.itertuples(index=False):
            label = business_labels.get(row.business_id)
            if label is None:
                continue
            split = business_splits[row.business_id]
            writers[split].writerow({
                "review_id": row.review_id,
                "business_id": row.business_id,
                "business_name": row.business_name,
                "city": row.city,
                "state": row.state,
                "stars": row.stars,
                "date": row.date,
                "label": label,
                "text": row.text,
            })
            dist[split][label] += 1

    for f in files.values():
        f.close()
    return dist


def write_capped(out_dir: Path, dist: dict, ceiling: int) -> dict:
    """Downsample the natural variant toward a per-class ceiling.

    Each row is kept with probability ceiling/class_size, so every
    review in an over-represented class has an equal chance of being
    kept and no class is favoured by position in the file.

    Args:
        out_dir: Base datasets directory.
        dist: split -> Counter of label -> count, from write_natural.
        ceiling: Target max reviews per class in train.

    Returns:
        A dict of split -> Counter of label -> kept count.
    """
    natural = out_dir / "natural"
    capped = out_dir / "capped"
    capped.mkdir(parents=True, exist_ok=True)

    kept = {split: Counter() for split in SPLIT_FRACTIONS}

    for split, fraction in SPLIT_FRACTIONS.items():
        rng = random.Random(RANDOM_SEED)
        split_ceiling = max(
            1, int(ceiling * fraction / SPLIT_FRACTIONS["train"])
        )
        keep_prob = {
            label: min(1.0, split_ceiling / n) if n else 0.0
            for label, n in dist[split].items()
        }

        with open(capped / f"{split}.csv", "w", newline="",
                  encoding="utf-8") as out:
            writer = csv.DictWriter(
                out, fieldnames=DATASET_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            for chunk in pd.read_csv(
                    natural / f"{split}.csv", chunksize=CHUNK_SIZE):
                for row in chunk.itertuples(index=False):
                    if rng.random() <= keep_prob[row.label]:
                        writer.writerow({
                            field: getattr(row, field)
                            for field in DATASET_FIELDS
                        })
                        kept[split][row.label] += 1

    return kept


def format_distribution(title: str, dist: dict) -> list:
    """Format a split/label distribution as report lines.

    Args:
        title: Section heading.
        dist: split -> Counter of label -> count.

    Returns:
        A list of report lines.
    """
    lines = [f"\n{title}", "=" * 72]
    labels = sorted(
        {l for c in dist.values() for l in c},
        key=lambda x: -dist["train"][x],
    )
    totals = {s: sum(dist[s].values()) for s in SPLIT_FRACTIONS}
    grand = sum(totals.values())

    lines.append(
        f"{'label':<24}" + "".join(f"{s:>12}" for s in SPLIT_FRACTIONS)
        + f"{'total':>12}{'%':>8}"
    )
    lines.append("-" * 72)
    for label in labels:
        counts = [dist[s][label] for s in SPLIT_FRACTIONS]
        total = sum(counts)
        pct = total / grand * 100 if grand else 0
        lines.append(
            f"{label[:23]:<24}"
            + "".join(f"{c:>12,}" for c in counts)
            + f"{total:>12,}{pct:>7.1f}%"
        )
    lines.append("-" * 72)
    lines.append(
        f"{'TOTAL':<24}"
        + "".join(f"{totals[s]:>12,}" for s in SPLIT_FRACTIONS)
        + f"{grand:>12,}"
    )
    if grand:
        biggest = max(dist["train"].values())
        smallest = min(dist["train"].values())
        lines.append(
            f"classes {len(labels)} | majority "
            f"{biggest/totals['train']*100:.1f}% of train | "
            f"imbalance {biggest/smallest:.1f}x"
        )
    return lines


def write_label_map_doc(out_path: Path, kept_labels: set) -> None:
    """Write the taxonomy to a readable file for documentation.

    Args:
        out_path: Where to write the label map documentation.
        kept_labels: Labels that survived the minimum-size filter.
    """
    lines = [
        "LABEL MAP",
        "=" * 64,
        "",
        "Inclusion rule: a Yelp category is eligible as a label only",
        "if it names a national or regional cuisine. Dish tags",
        "(Salad, Pizza, Tacos), format tags (Fast Food, Diners),",
        "dietary tags (Vegan) and meal tags (Breakfast & Brunch) are",
        "excluded. The rule leaves no residual bucket, so there is no",
        "'Other' class: a business matching no cuisine is dropped.",
        "",
        "Collision policy:",
        "  1. A business matching 'American' plus exactly one cuisine",
        "     is assigned that cuisine."
        + ("" if CUISINE_BEATS_AMERICAN else "  [DISABLED in this run]"),
        "  2. A business matching two or more cuisines is dropped.",
        "",
        f"Minimum reviews per label: {MIN_REVIEWS_PER_LABEL:,}",
        "",
        f"KEPT ({len(kept_labels)} labels)",
        "-" * 64,
    ]
    for label in sorted(kept_labels):
        cats = ", ".join(sorted(LABEL_MAP[label]))
        lines.append(f"{label}: {cats}")

    dropped = set(LABEL_MAP) - kept_labels
    if dropped:
        lines += [
            "",
            f"MAPPED BUT DROPPED ({len(dropped)} labels, under the "
            f"minimum size)",
            "-" * 64,
        ]
        for label in sorted(dropped):
            cats = ", ".join(sorted(LABEL_MAP[label]))
            lines.append(f"{label}: {cats}")
        lines += [
            "",
            "Note: these labels are still matched during assignment,",
            "then removed afterward. Deleting them from LABEL_MAP",
            "instead would change the results, since a business",
            "tagged with a dropped label plus a kept one is currently",
            "a collision rather than a single-label match.",
        ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  saved '{out_path}'")


def stage_datasets(extract_dir: Path,
                   reviews_csv: Path,
                   out_dir: Path,
                   force: bool) -> None:
    """Build labeled, split, natural and capped dataset variants.

    Args:
        extract_dir: Directory holding the extracted JSON files.
        reviews_csv: Path to the restaurant reviews CSV.
        out_dir: Base datasets directory.
        force: Rebuild even if outputs exist.
    """
    marker = out_dir / "dataset_distribution.txt"
    if marker.exists() and not force:
        print(f"[6/8] '{marker}' exists, skipping")
        return

    print("[6/8] Building labeled datasets...")
    out_dir.mkdir(parents=True, exist_ok=True)

    business_labels = build_business_labels(
        find_json(extract_dir, "business")
    )
    review_counts = count_reviews_per_business(reviews_csv)
    business_labels = drop_small_labels(business_labels, review_counts)
    business_splits = assign_splits(business_labels, review_counts)

    natural_dist = write_natural(
        reviews_csv, business_labels, business_splits, out_dir
    )
    ceiling = CAPPED_CEILING or min(natural_dist["train"].values())
    print(f"  capping at ~{ceiling:,} train reviews per class")
    capped_dist = write_capped(out_dir, natural_dist, ceiling)

    report = "\n".join(
        format_distribution("NATURAL VARIANT (real distribution)",
                            natural_dist)
        + format_distribution(
            f"CAPPED VARIANT (~{ceiling:,} per class in train)",
            capped_dist)
    )
    print(report)
    marker.write_text(report, encoding="utf-8")
    print(f"  saved '{marker}'")

    write_label_map_doc(
        out_dir / "label_map.txt", set(business_labels.values())
    )


# ================================================================
# Stage 7: features
# ================================================================

def load_nltk_tools() -> tuple:
    """Load NLTK's tokenizer and stopwords, with regex fallbacks.

    NLTK needs its corpora downloaded once:
        python -m nltk.downloader punkt punkt_tab stopwords

    Returns:
        A tuple of (tokenize function, stopword set, backend name).
    """
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        stop = set(stopwords.words("english"))
        word_tokenize("test sentence")
        return word_tokenize, stop, "nltk"
    except Exception as exc:
        print(f"  NLTK unavailable ({type(exc).__name__}), using regex "
              f"tokenizer", file=sys.stderr)
        return (lambda t: WORD_PATTERN.findall(t),
                FALLBACK_STOPWORDS, "regex")


def clean_text(text) -> str:
    """Lowercase and strip URLs, punctuation and extra whitespace.

    Args:
        text: Raw review text.

    Returns:
        Cleaned lowercase text, newlines and tabs collapsed.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = NON_LETTER_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def preprocess_row(text, business_name, tokenize, stop_words) -> str:
    """Clean, tokenize and filter one review into a token string.

    Args:
        text: Raw review text.
        business_name: The reviewed restaurant's name.
        tokenize: Tokenizer function.
        stop_words: Stopwords to remove.

    Returns:
        A space-joined string of surviving tokens.
    """
    tokens = tokenize(clean_text(text))
    drop = stop_words
    if STRIP_BUSINESS_NAME and isinstance(business_name, str):
        drop = drop | set(WORD_PATTERN.findall(business_name.lower()))
    return " ".join(
        t for t in tokens
        if len(t) > 1 and t not in drop and not t.isdigit()
    )


def preprocess_split(csv_path: Path, tokenize, stop_words) -> tuple:
    """Preprocess one split's CSV into token strings and labels.

    Args:
        csv_path: Path to the split CSV.
        tokenize: Tokenizer function.
        stop_words: Stopwords to remove.

    Returns:
        A tuple of (token strings list, label strings list).
    """
    docs, labels = [], []
    for chunk in pd.read_csv(
            csv_path,
            usecols=["business_name", "label", "text"],
            chunksize=FEATURE_CHUNK_SIZE):
        for row in chunk.itertuples(index=False):
            docs.append(preprocess_row(
                row.text, row.business_name, tokenize, stop_words
            ))
            labels.append(row.label)
    print(f"  {csv_path.name}: {len(docs):,} documents")
    return docs, labels


def build_ngram_features(docs: dict,
                         labels: dict,
                         label_ids: dict,
                         ngram_range: tuple,
                         out_dir: Path) -> dict:
    """Vectorize all splits at one n-gram range and save to disk.

    The vectorizer is fit on train only and merely applied to dev and
    test, so no dev/test vocabulary enters the feature space.

    Args:
        docs: split -> list of token strings.
        labels: split -> list of label strings.
        label_ids: label string -> integer id.
        ngram_range: (min_n, max_n) for CountVectorizer.
        out_dir: Output directory for this range.

    Returns:
        A dict of summary statistics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        min_df=MIN_DF,
        max_features=MAX_FEATURES,
        analyzer="word",
        token_pattern=r"\S+",
        lowercase=False,
    )
    matrices = {"train": vectorizer.fit_transform(docs["train"])}
    for split in SPLITS:
        if split != "train":
            matrices[split] = vectorizer.transform(docs[split])

    stats = {
        "ngram_range": ngram_range,
        "vocab_size": len(vectorizer.vocabulary_),
        "order_counts": Counter(
            term.count(" ") + 1 for term in vectorizer.vocabulary_
        ),
        "capped": len(vectorizer.vocabulary_) == MAX_FEATURES,
        "shapes": {},
        "nnz": {},
    }

    for split in SPLITS:
        matrix = matrices[split]
        sparse.save_npz(out_dir / f"X_{split}.npz", matrix)
        np.save(
            out_dir / f"y_{split}.npy",
            np.array([label_ids[l] for l in labels[split]],
                     dtype=np.int32),
        )
        stats["shapes"][split] = matrix.shape
        stats["nnz"][split] = matrix.nnz

    with open(out_dir / "vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(
            {t: int(i) for t, i in vectorizer.vocabulary_.items()}, f
        )
    with open(out_dir / "label_classes.json", "w",
              encoding="utf-8") as f:
        json.dump({i: l for l, i in label_ids.items()}, f, indent=2)

    return stats


def format_feature_stats(all_stats: list, backend: str) -> str:
    """Format feature statistics into a readable report.

    Args:
        all_stats: Per-range stats dicts.
        backend: Tokenizer backend used.

    Returns:
        A formatted report string.
    """
    order_name = {1: "unigram", 2: "bigram", 3: "trigram"}
    lines = [
        "N-GRAM FEATURE SETS",
        "=" * 64,
        f"Variant: {FEATURE_VARIANT}",
        f"Tokenizer: {backend}",
        f"min_df: {MIN_DF} | max_features: {MAX_FEATURES:,}",
        f"Business name stripped from text: {STRIP_BUSINESS_NAME}",
    ]

    for stats in all_stats:
        lo, hi = stats["ngram_range"]
        title = (f"({lo},{hi}) {order_name[hi]} only" if lo == hi
                 else f"({lo},{hi}) up to {order_name[hi]}")
        lines.append(f"\n{title}")
        lines.append("-" * 64)

        vocab_line = f"  vocabulary: {stats['vocab_size']:,} features"
        if stats["capped"]:
            vocab_line += "   *** AT MAX_FEATURES CAP ***"
        lines.append(vocab_line)
        lines.append("  composition: " + ", ".join(
            f"{order_name[n]} {stats['order_counts'][n]:,}"
            for n in sorted(stats["order_counts"])
        ))

        for split in SPLITS:
            rows, cols = stats["shapes"][split]
            nnz = stats["nnz"][split]
            density = nnz / (rows * cols) * 100 if rows and cols else 0
            lines.append(
                f"  {split:<6} {rows:>9,} x {cols:>9,}  "
                f"nnz {nnz:>12,}  density {density:.4f}%"
            )

    return "\n".join(lines)


def stage_features(datasets_dir: Path,
                   features_dir: Path,
                   force: bool) -> None:
    """Preprocess text and build all n-gram feature sets.

    Args:
        datasets_dir: Base datasets directory.
        features_dir: Base features directory.
        force: Rebuild even if outputs exist.
    """
    root = features_dir / FEATURE_VARIANT
    marker = root / "feature_stats.txt"
    if marker.exists() and not force:
        print(f"[7/8] '{marker}' exists, skipping")
        return

    variant_dir = datasets_dir / FEATURE_VARIANT
    if not variant_dir.exists():
        raise FileNotFoundError(f"Not found: '{variant_dir}'")

    print(f"[7/8] Building features from '{FEATURE_VARIANT}' variant...")
    tokenize, stop_words, backend = load_nltk_tools()

    docs, labels = {}, {}
    for split in SPLITS:
        docs[split], labels[split] = preprocess_split(
            variant_dir / f"{split}.csv", tokenize, stop_words
        )

    classes = sorted(set(labels["train"]))
    label_ids = {label: i for i, label in enumerate(classes)}
    print(f"  classes: {len(label_ids)}")

    all_stats = []
    for ngram_range in NGRAM_RANGES:
        lo, hi = ngram_range
        print(f"  vectorizing ({lo},{hi})...")
        all_stats.append(build_ngram_features(
            docs, labels, label_ids, ngram_range,
            root / f"ngram_{lo}_{hi}",
        ))

    report = format_feature_stats(all_stats, backend)
    print("\n" + report)
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(report, encoding="utf-8")
    print(f"\n  saved '{marker}'")


# ================================================================
# Stage 8: split the main CSV into size-capped parts (optional)
# ================================================================

def stage_split(reviews_csv: Path,
                parts_dir: Path,
                force: bool) -> None:
    """Split the main CSV into parts under a size ceiling.

    Useful when the CSV must live in a repo with a per-file size
    limit. Each part repeats the header row, so every part is a valid
    standalone CSV. Reassemble by concatenating the parts in numeric
    order and keeping only the first header.

    Args:
        reviews_csv: Path to the restaurant reviews CSV.
        parts_dir: Directory to write the part files into.
        force: Rebuild even if parts already exist.
    """
    if parts_dir.exists() and list(parts_dir.glob("*.csv")) \
            and not force:
        print(f"[8/8] Parts already in '{parts_dir}', skipping")
        return

    parts_dir.mkdir(parents=True, exist_ok=True)
    stem = reviews_csv.stem
    print(f"[8/8] Splitting '{reviews_csv}' into "
          f"{MAX_PART_SIZE_BYTES/1024/1024:.0f} MB parts...")

    def open_part(number, header):
        path = parts_dir / f"{stem}_part{number}.csv"
        handle = open(path, "w", newline="", encoding="utf-8")
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        return handle, writer

    with open(reviews_csv, "r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        header = next(reader)
        number = 1
        handle, writer = open_part(number, header)

        for row in reader:
            row_size = len(",".join(row).encode("utf-8")) + 1
            if handle.tell() + row_size > MAX_PART_SIZE_BYTES:
                handle.close()
                number += 1
                handle, writer = open_part(number, header)
            writer.writerow(row)

        handle.close()

    print(f"  wrote {number} part(s) to '{parts_dir}'")


# ================================================================
# Driver
# ================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Yelp restaurant-cuisine data pipeline.",
    )
    parser.add_argument(
        "--url",
        help="Signed Yelp download URL (stage 1). Expires, so copy a "
             "fresh one from Yelp's site.",
    )
    parser.add_argument(
        "--tar", default=TAR_PATH,
        help="Path to the tar file, or a folder containing it.",
    )
    parser.add_argument(
        "--extracted", default=EXTRACT_DIR,
        help="Directory for (or containing) extracted JSON files.",
    )
    parser.add_argument(
        "--from-stage", type=int, default=1, choices=range(1, 9),
        help="Start at this stage.",
    )
    parser.add_argument(
        "--only", type=int, choices=range(1, 9),
        help="Run only this stage.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild stages even if their outputs exist.",
    )
    parser.add_argument(
        "--split-parts", action="store_true",
        help="Also run stage 8, splitting the main CSV into "
             "size-capped parts for repo hosting.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the pipeline stages selected on the command line."""
    args = parse_args()

    first = args.only or args.from_stage
    last = args.only or 8

    tar_path = Path(args.tar)
    extract_dir = Path(args.extracted)
    reviews_csv = Path(REVIEWS_CSV)
    datasets_dir = Path(DATASETS_DIR)
    features_dir = Path(FEATURES_DIR)

    if first <= 1 <= last:
        if args.url:
            stage_download(args.url, tar_path, args.force)
        else:
            print("[1/8] No --url given, skipping download")

    if first <= 2 <= last:
        extract_dir = stage_extract(tar_path, extract_dir, args.force)

    if first <= 3 <= last:
        stage_reviews(extract_dir, reviews_csv, args.force)

    if first <= 4 <= last:
        stage_stats(reviews_csv, Path(STATS_SUMMARY), args.force)

    if first <= 5 <= last:
        stage_audit(extract_dir, reviews_csv, Path(CATEGORY_COUNTS),
                    Path(CATEGORY_REPORT), args.force)

    if first <= 6 <= last:
        stage_datasets(extract_dir, reviews_csv, datasets_dir,
                       args.force)

    if first <= 7 <= last:
        stage_features(datasets_dir, features_dir, args.force)

    # Stage 8 only runs when explicitly asked for, since splitting is
    # for repo hosting rather than analysis.
    if first <= 8 <= last:
        if args.split_parts or args.only == 8:
            stage_split(reviews_csv, Path(PARTS_DIR), args.force)
        else:
            print("[8/8] --split-parts not given, skipping")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
