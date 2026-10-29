"""Build labeled train/dev/test datasets of restaurant reviews.

Takes business.json (for category tags) plus the filtered reviews CSV
and produces two dataset variants, each split into train/dev/test:

    natural/  - the real class distribution, imbalanced. American
                dominates by a wide margin.
    capped/   - every class downsampled toward a common ceiling, so
                no class can swamp the others during training.

Key decisions encoded here, all of which belong in the writeup:

  * Labels come from Yelp's business-level "categories" field and are
    propagated to every review of that business. Reviews are not
    individually annotated, so the label describes the restaurant,
    not necessarily the dish discussed in any single review.

  * Only Yelp categories naming a national or regional cuisine are
    eligible as labels. Dish, format, dietary, meal and venue tags
    are excluded. Greek, Mediterranean and Middle Eastern are
    collapsed into one "Eastern Mediterranean" label, since the audit
    showed they were the largest source of collisions (926 colliding
    businesses across those three pairs) and Yelp applies them
    near-interchangeably.

  * Businesses matching two or more labels are DROPPED, not
    reassigned. The audit measured this at 6.2% of businesses and
    9.3% of reviews, small enough to discard without a tiebreak rule.

  * Businesses matching no label are dropped. Set
    KEEP_UNLABELED_AS_OTHER to True to retain them as a residual
    "Other" class instead.

  * Splits are grouped by business_id so that all reviews of one
    restaurant land in exactly one split. Without this, a model can
    memorize restaurant-specific vocabulary (including the
    restaurant's own name appearing in its reviews) and score far
    higher on dev than it deserves.
"""

import csv
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

BUSINESS_JSON_PATH = "yelp_dataset_extracted/yelp_academic_dataset_business.json"
REVIEWS_CSV_PATH = "yelp_restaurant_reviews.csv"
OUTPUT_DIR = "datasets"

RESTAURANT_CATEGORY = "Restaurants"

# Split proportions, applied at the business level.
SPLIT_FRACTIONS = {"train": 0.8, "dev": 0.1, "test": 0.1}

# Seed for reproducible splits and downsampling.
RANDOM_SEED = 1671

# Rows per chunk when streaming the reviews CSV.
CHUNK_SIZE = 100_000

# Classes with fewer than this many reviews are dropped entirely.
# At 0.2-0.5% of the data a class is effectively never predicted and
# only drags macro-F1 down. Set to 0 to keep every class.
MIN_REVIEWS_PER_LABEL = 15_000

# Per-class review ceiling for the capped variant, applied within
# each split. Set to None to use the size of the smallest surviving
# class instead of a fixed number.
CAPPED_CEILING = None

# If True, businesses matching no cuisine label are kept as "Other"
# rather than dropped. The professor advised against a residual
# class, so this defaults to False.
KEEP_UNLABELED_AS_OTHER = False
OTHER_LABEL = "Other"

# If True, a business matching "American" plus exactly one specific
# cuisine is assigned that cuisine rather than dropped as a
# collision. Rationale: "American (Traditional)/(New)" is Yelp's
# broad default tag and carries far less information than a specific
# cuisine tag, so "Southern, American (Traditional)" is better read
# as a Southern restaurant than as an unresolvable conflict.
#
# Without this rule, American collisions cost Southern 59% of its
# reviews, French 42%, British & Irish 41%, and pushed German and
# Hawaiian below the minimum class size entirely. The bias is
# systematic rather than random: the cuisines closest to American
# dining lose the most, which is exactly the wrong thing to let
# happen silently.
#
# Genuine cuisine-vs-cuisine collisions (e.g. Chinese + Thai) are
# still dropped regardless of this setting.
CUISINE_BEATS_AMERICAN = True
AMERICAN_LABEL = "American"

# Columns written to the output CSVs.
OUTPUT_FIELDS = [
    "review_id",
    "business_id",
    "business_name",
    "city",
    "state",
    "stars",
    "date",
    "label",
    "text",
]

LABEL_MAP = {
    "American": {
        "American (New)", "American (Traditional)",
    },
    "Mexican": {
        "Mexican", "Tex-Mex", "New Mexican Cuisine",
    },
    "Italian": {
        "Italian",
    },
    "Chinese": {
        "Chinese", "Szechuan", "Cantonese", "Shanghainese",
        "Taiwanese", "Hong Kong Style Cafe", "Hainan",
    },
    "Japanese": {
        "Japanese",
    },
    "Korean": {
        "Korean",
    },
    "Thai": {
        "Thai",
    },
    "Vietnamese": {
        "Vietnamese",
    },
    "South Asian": {
        "Indian", "Pakistani", "Bangladeshi", "Sri Lankan",
        "Himalayan/Nepalese",
    },
    # Greek + Mediterranean + Middle Eastern collapsed: these three
    # produced the two largest collision pairs in the audit and Yelp
    # uses them near-interchangeably for the same gyro/kebab/falafel
    # restaurants.
    "Eastern Mediterranean": {
        "Greek", "Mediterranean", "Middle Eastern", "Lebanese",
        "Turkish", "Persian/Iranian", "Israeli", "Egyptian",
        "Afghan", "Moroccan",
    },
    "French": {
        "French",
    },
    "Spanish": {
        "Spanish",
    },
    "German": {
        "German", "Austrian",
    },
    "British & Irish": {
        "Irish", "British", "Scottish",
    },
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
    "Cajun/Creole": {
        "Cajun/Creole",
    },
    "Southern": {
        "Southern", "Soul Food",
    },
    "Hawaiian": {
        "Hawaiian",
    },
    "African": {
        "African", "Ethiopian", "Senegalese", "Somali",
    },
}


def parse_categories(raw_categories) -> list:
    """Split Yelp's comma-separated category string into a list.

    Args:
        raw_categories: The raw "categories" field, a single
            comma-separated string or None.

    Returns:
        A list of cleaned category strings, empty if missing.
    """
    if not raw_categories:
        return []
    return [c.strip() for c in raw_categories.split(",") if c.strip()]


def assign_label(categories: list, label_map: dict) -> set:
    """Find which labels a business's categories match.

    Args:
        categories: The business's Yelp category strings.
        label_map: Label -> set of eligible Yelp categories.

    Returns:
        The set of matching labels. Several categories mapping to the
        same label is not a collision, which is why this returns a
        set of labels rather than a count of matched categories.
    """
    category_set = set(categories)
    return {
        label
        for label, label_categories in label_map.items()
        if category_set & label_categories
    }


def build_business_labels(business_json_path: str,
                          label_map: dict) -> tuple:
    """Assign one label per restaurant business, dropping collisions.

    Args:
        business_json_path: Path to business.json.
        label_map: Label -> set of eligible Yelp categories.

    Returns:
        A tuple of (business_id -> label dict, stats dict).
    """
    labels = {}
    n_restaurants = 0
    n_unlabeled = 0
    n_collision = 0
    n_american_resolved = 0
    collision_pairs = Counter()

    with open(business_json_path, "r", encoding="utf-8") as f:
        for line in f:
            business = json.loads(line)
            categories = parse_categories(business.get("categories"))

            if RESTAURANT_CATEGORY not in categories:
                continue

            n_restaurants += 1
            matched = assign_label(categories, label_map)

            if not matched:
                n_unlabeled += 1
                if KEEP_UNLABELED_AS_OTHER:
                    labels[business["business_id"]] = OTHER_LABEL
            elif len(matched) == 1:
                labels[business["business_id"]] = next(iter(matched))
            elif (CUISINE_BEATS_AMERICAN
                  and len(matched) == 2
                  and AMERICAN_LABEL in matched):
                # American plus exactly one cuisine: keep the cuisine.
                cuisine = next(iter(matched - {AMERICAN_LABEL}))
                labels[business["business_id"]] = cuisine
                n_american_resolved += 1
            else:
                n_collision += 1
                for pair in combinations(sorted(matched), 2):
                    collision_pairs[pair] += 1

    stats = {
        "n_restaurants": n_restaurants,
        "n_unlabeled": n_unlabeled,
        "n_collision": n_collision,
        "n_american_resolved": n_american_resolved,
        "n_labeled": len(labels),
        "collision_pairs": collision_pairs,
    }

    print(f"Restaurants: {n_restaurants:,} | labeled: {len(labels):,} | "
          f"unlabeled: {n_unlabeled:,} | collisions dropped: "
          f"{n_collision:,}")
    if n_american_resolved:
        print(f"  (of which {n_american_resolved:,} American+cuisine "
              f"conflicts resolved to the cuisine)")
    return labels, stats


def count_reviews_per_business(reviews_csv_path: str,
                               chunk_size: int) -> Counter:
    """Count reviews per business_id in the reviews CSV.

    Args:
        reviews_csv_path: Path to the filtered reviews CSV.
        chunk_size: Rows to read per chunk.

    Returns:
        A Counter of business_id -> review count.
    """
    counts = Counter()
    reader = pd.read_csv(
        reviews_csv_path, usecols=["business_id"], chunksize=chunk_size
    )
    for chunk in reader:
        counts.update(chunk["business_id"])
    return counts


def drop_small_labels(business_labels: dict,
                      review_counts: Counter,
                      min_reviews: int) -> tuple:
    """Remove labels whose total review count is below a threshold.

    Args:
        business_labels: business_id -> label.
        review_counts: business_id -> review count.
        min_reviews: Minimum reviews for a label to be kept.

    Returns:
        A tuple of (filtered business_labels, dropped label dict).
    """
    if min_reviews <= 0:
        return business_labels, {}

    per_label = Counter()
    for business_id, label in business_labels.items():
        per_label[label] += review_counts.get(business_id, 0)

    dropped = {
        label: count
        for label, count in per_label.items()
        if count < min_reviews
    }

    if dropped:
        print(f"Dropping {len(dropped)} label(s) under "
              f"{min_reviews:,} reviews: "
              + ", ".join(f"{k} ({v:,})" for k, v in
                          sorted(dropped.items(), key=lambda x: x[1])))

    filtered = {
        business_id: label
        for business_id, label in business_labels.items()
        if label not in dropped
    }
    return filtered, dropped


def assign_splits(business_labels: dict,
                  review_counts: Counter,
                  split_fractions: dict,
                  seed: int) -> dict:
    """Assign each business to a split, stratified by label.

    Businesses are shuffled within each label and allocated to splits
    by review volume rather than business count, so the splits come
    out near the intended proportions even though businesses have
    very different review counts.

    Args:
        business_labels: business_id -> label.
        review_counts: business_id -> review count.
        split_fractions: split name -> target fraction.
        seed: Random seed for the shuffle.

    Returns:
        A dict of business_id -> split name.
    """
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for business_id, label in business_labels.items():
        by_label[label].append(business_id)

    splits = {}
    for label, business_ids in by_label.items():
        rng.shuffle(business_ids)

        label_reviews = sum(review_counts.get(b, 0) for b in business_ids)
        targets = {
            name: label_reviews * frac
            for name, frac in split_fractions.items()
        }
        filled = {name: 0 for name in split_fractions}

        # Greedily place each business in whichever split is furthest
        # below its target, which keeps proportions close without
        # ever splitting a business across two sets.
        for business_id in business_ids:
            n = review_counts.get(business_id, 0)
            deficit = {
                name: targets[name] - filled[name]
                for name in split_fractions
            }
            chosen = max(deficit, key=deficit.get)
            splits[business_id] = chosen
            filled[chosen] += n

    return splits


def write_natural_variant(reviews_csv_path: str,
                          business_labels: dict,
                          business_splits: dict,
                          output_dir: Path,
                          chunk_size: int) -> dict:
    """Stream reviews and write the natural-distribution variant.

    Args:
        reviews_csv_path: Path to the filtered reviews CSV.
        business_labels: business_id -> label.
        business_splits: business_id -> split name.
        output_dir: Base output directory.
        chunk_size: Rows to read per chunk.

    Returns:
        A dict of split name -> Counter of label -> review count.
    """
    natural_dir = output_dir / "natural"
    natural_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    writers = {}
    for split in SPLIT_FRACTIONS:
        path = natural_dir / f"{split}.csv"
        files[split] = open(path, "w", newline="", encoding="utf-8")
        writers[split] = csv.DictWriter(
            files[split], fieldnames=OUTPUT_FIELDS, lineterminator="\n"
        )
        writers[split].writeheader()

    distribution = {split: Counter() for split in SPLIT_FRACTIONS}

    reader = pd.read_csv(reviews_csv_path, chunksize=chunk_size)
    for chunk in reader:
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
            distribution[split][label] += 1

    for f in files.values():
        f.close()

    print(f"Wrote natural variant to '{natural_dir}'")
    return distribution


def write_capped_variant(output_dir: Path,
                         distribution: dict,
                         ceiling: int,
                         seed: int,
                         chunk_size: int) -> dict:
    """Downsample the natural variant toward a per-class ceiling.

    Each row is kept with probability ceiling/class_size, so final
    counts land near the ceiling rather than exactly on it. This
    avoids holding any class fully in memory.

    Args:
        output_dir: Base output directory.
        distribution: split -> Counter of label -> count, from the
            natural variant.
        ceiling: Target max reviews per class per split.
        seed: Random seed for sampling.
        chunk_size: Rows to read per chunk.

    Returns:
        A dict of split name -> Counter of label -> kept count.
    """
    natural_dir = output_dir / "natural"
    capped_dir = output_dir / "capped"
    capped_dir.mkdir(parents=True, exist_ok=True)

    kept = {split: Counter() for split in SPLIT_FRACTIONS}

    for split, fraction in SPLIT_FRACTIONS.items():
        rng = random.Random(seed)
        split_ceiling = max(1, int(ceiling * fraction / SPLIT_FRACTIONS["train"]))

        keep_prob = {}
        for label, count in distribution[split].items():
            keep_prob[label] = min(1.0, split_ceiling / count) if count else 0

        in_path = natural_dir / f"{split}.csv"
        out_path = capped_dir / f"{split}.csv"

        with open(out_path, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(
                outfile, fieldnames=OUTPUT_FIELDS, lineterminator="\n"
            )
            writer.writeheader()

            reader = pd.read_csv(in_path, chunksize=chunk_size)
            for chunk in reader:
                for row in chunk.itertuples(index=False):
                    if rng.random() <= keep_prob[row.label]:
                        writer.writerow({
                            field: getattr(row, field)
                            for field in OUTPUT_FIELDS
                        })
                        kept[split][row.label] += 1

    print(f"Wrote capped variant to '{capped_dir}'")
    return kept


def format_distribution(title: str, distribution: dict) -> list:
    """Format a split/label distribution as report lines.

    Args:
        title: Heading for this section.
        distribution: split -> Counter of label -> count.

    Returns:
        A list of report lines.
    """
    lines = [f"\n{title}", "=" * 64]
    labels = sorted(
        {label for counter in distribution.values() for label in counter},
        key=lambda x: -distribution["train"][x],
    )

    header = f"{'label':<24}" + "".join(
        f"{split:>12}" for split in SPLIT_FRACTIONS
    ) + f"{'total':>12}"
    lines.append(header)

    for label in labels:
        counts = [distribution[split][label] for split in SPLIT_FRACTIONS]
        lines.append(
            f"{label[:23]:<24}"
            + "".join(f"{c:>12,}" for c in counts)
            + f"{sum(counts):>12,}"
        )

    totals = [sum(distribution[split].values()) for split in SPLIT_FRACTIONS]
    lines.append(
        f"{'TOTAL':<24}"
        + "".join(f"{t:>12,}" for t in totals)
        + f"{sum(totals):>12,}"
    )
    return lines


def main() -> None:
    """Build labeled, split, natural and capped dataset variants."""
    if not Path(BUSINESS_JSON_PATH).exists():
        raise FileNotFoundError(
            f"business.json not found at '{BUSINESS_JSON_PATH}'"
        )
    if not Path(REVIEWS_CSV_PATH).exists():
        raise FileNotFoundError(
            f"Reviews CSV not found at '{REVIEWS_CSV_PATH}'"
        )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    business_labels, _ = build_business_labels(
        BUSINESS_JSON_PATH, LABEL_MAP
    )

    print("Counting reviews per business...")
    review_counts = count_reviews_per_business(REVIEWS_CSV_PATH, CHUNK_SIZE)

    business_labels, _ = drop_small_labels(
        business_labels, review_counts, MIN_REVIEWS_PER_LABEL
    )

    business_splits = assign_splits(
        business_labels, review_counts, SPLIT_FRACTIONS, RANDOM_SEED
    )

    print("Writing natural variant...")
    natural_dist = write_natural_variant(
        REVIEWS_CSV_PATH, business_labels, business_splits,
        output_dir, CHUNK_SIZE,
    )

    train_counts = natural_dist["train"]
    ceiling = CAPPED_CEILING or min(train_counts.values())
    print(f"Capping classes at ~{ceiling:,} train reviews each")

    capped_dist = write_capped_variant(
        output_dir, natural_dist, ceiling, RANDOM_SEED, CHUNK_SIZE
    )

    report_lines = format_distribution(
        "NATURAL VARIANT (real distribution)", natural_dist
    )
    report_lines += format_distribution(
        f"CAPPED VARIANT (~{ceiling:,} per class in train)", capped_dist
    )
    report = "\n".join(report_lines)

    print("\n" + report)
    report_path = output_dir / "dataset_distribution.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved distribution report to '{report_path}'")


if __name__ == "__main__":
    main()
