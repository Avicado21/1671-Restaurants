"""Audit Yelp business categories and analyze label collisions.

Groundwork for assigning restaurant-type labels to reviews. The
reviews CSV produced by extract_restaurant_reviews.py does NOT carry
category information (categories were used only as a yes/no filter
for "Restaurants"), so this script goes back to business.json.

Two modes, controlled by whether LABEL_MAP below is filled in:

    LABEL_MAP empty  -> AUDIT ONLY
        Reports every distinct Yelp category string found among
        restaurant businesses, with how many businesses carry it and
        (optionally) how many reviews those businesses account for.
        Use this output to decide which Yelp categories belong under
        each of your labels.

    LABEL_MAP filled -> AUDIT + LABEL ANALYSIS
        Additionally assigns a label to each business, reports how
        many businesses match zero / one / multiple labels, breaks
        down which label pairs collide, and reports final row counts
        after dropping collisions and unlabeled businesses.

Outputs a category frequency CSV and a text report.
"""

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

BUSINESS_JSON_PATH = "yelp_dataset_extracted/yelp_academic_dataset_business.json"
REVIEWS_CSV_PATH = "yelp_restaurant_reviews.csv"

CATEGORY_COUNTS_PATH = "category_counts.csv"
REPORT_PATH = "category_audit_report.txt"

# The category string Yelp uses to tag a business as a restaurant.
# Businesses without this tag are skipped, matching the filter used
# in extract_restaurant_reviews.py.
RESTAURANT_CATEGORY = "Restaurants"

# Set False to skip the reviews CSV pass. Business counts alone are
# much faster; review counts matter for judging class balance, since
# a label with few businesses can still have many reviews.
COUNT_REVIEWS = True

# Rows per chunk when counting reviews per business.
CHUNK_SIZE = 100_000

# How many categories to print to the console (full list goes to CSV).
TOP_N = 60

# ---------------------------------------------------------------
# FILL THIS IN with your group's taxonomy, then re-run the script.
# Maps your label -> the set of Yelp category strings that count as
# that label. Category strings must match Yelp's exactly; use the
# audit output to get them right.
#
# Example shape (replace entirely with your own):
#     LABEL_MAP = {
#         "Mexican": {"Mexican", "Tacos", "Tex-Mex"},
#         "Chinese": {"Chinese", "Dim Sum", "Szechuan"},
#     }
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# Candidate taxonomy: restaurant type by national/regional cuisine.
#
# Inclusion rule: a Yelp category is eligible as a label only if it
# names a national or regional cuisine. Dish tags (Salad, Pizza,
# Tacos, Sushi Bars, Chicken Wings), format tags (Fast Food, Diners,
# Food Trucks, Steakhouses, Cafes), dietary tags (Vegan, Vegetarian,
# Gluten-Free), meal tags (Breakfast & Brunch) and venue tags (Bars,
# Nightlife, Caterers) are excluded by this rule. This keeps the
# label set on a single axis and removes the need for an "Other"
# bucket, since anything not matching a cuisine is simply unlabeled
# and dropped.
#
# Subtypes are folded into their parent cuisine (e.g. Szechuan ->
# Chinese) to raise class sizes and eliminate within-cuisine
# collisions. Split them back out if your group prefers finer
# classes, e.g. Tex-Mex as its own label separate from Mexican.
#
# Deliberately EXCLUDED and worth discussing in the writeup:
#   American (New), American (Traditional) - ~2.0M reviews combined,
#       the two largest categories, but not a national cuisine under
#       the rule above and expected to collide with nearly every
#       dish and format tag. Test a variant that includes them.
#   Asian Fusion - ~195k reviews, explicitly a blend label, so it
#       collides with every Asian cuisine by construction.
#   Salad, Desserts, Bakeries, Sandwiches, Pizza, Burgers, Seafood,
#       Barbeque, Steakhouses - dish or format, not cuisine.
# ---------------------------------------------------------------
LABEL_MAP = {
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
    "Middle Eastern": {
        "Middle Eastern", "Lebanese", "Turkish", "Persian/Iranian",
        "Israeli", "Egyptian", "Afghan", "Moroccan",
    },
    "Mediterranean": {
        "Mediterranean",
    },
    "Greek": {
        "Greek",
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
        raw_categories: The raw "categories" field, which is a single
            comma-separated string or None.

    Returns:
        A list of cleaned category strings, empty if the field was
        missing or blank.
    """
    if not raw_categories:
        return []
    return [c.strip() for c in raw_categories.split(",") if c.strip()]


def load_restaurant_businesses(business_json_path: str) -> dict:
    """Read business.json and keep categories for restaurants only.

    Args:
        business_json_path: Path to the business.json file.

    Returns:
        A dict mapping business_id -> {"name": str,
        "categories": list of category strings}.
    """
    restaurants = {}

    with open(business_json_path, "r", encoding="utf-8") as f:
        for line in f:
            business = json.loads(line)
            categories = parse_categories(business.get("categories"))

            if RESTAURANT_CATEGORY in categories:
                restaurants[business["business_id"]] = {
                    "name": business.get("name", ""),
                    "categories": categories,
                }

    print(f"Found {len(restaurants):,} restaurant businesses")
    return restaurants


def count_reviews_per_business(reviews_csv_path: str,
                               chunk_size: int) -> Counter:
    """Count how many reviews each business_id has in the CSV.

    Args:
        reviews_csv_path: Path to the filtered reviews CSV.
        chunk_size: Rows to read per chunk.

    Returns:
        A Counter mapping business_id -> review count.
    """
    counts = Counter()

    reader = pd.read_csv(
        reviews_csv_path,
        usecols=["business_id"],
        chunksize=chunk_size,
    )
    for chunk in reader:
        counts.update(chunk["business_id"])

    print(f"Counted reviews for {len(counts):,} businesses")
    return counts


def audit_categories(restaurants: dict,
                     review_counts: Counter) -> pd.DataFrame:
    """Tally how often each Yelp category appears among restaurants.

    Args:
        restaurants: Dict from load_restaurant_businesses().
        review_counts: business_id -> review count, or an empty
            Counter if review counting was skipped.

    Returns:
        A DataFrame with one row per category, columns
        "category", "n_businesses", "n_reviews", sorted by
        n_businesses descending.
    """
    business_counts = Counter()
    category_review_counts = Counter()

    for business_id, info in restaurants.items():
        n_reviews = review_counts.get(business_id, 0)
        for category in info["categories"]:
            business_counts[category] += 1
            category_review_counts[category] += n_reviews

    rows = [
        {
            "category": category,
            "n_businesses": count,
            "n_reviews": category_review_counts[category],
        }
        for category, count in business_counts.items()
    ]

    df = pd.DataFrame(rows)
    df = df.sort_values("n_businesses", ascending=False)
    df = df.reset_index(drop=True)
    return df


def assign_labels(categories: list, label_map: dict) -> set:
    """Find which of your labels a business's categories match.

    Args:
        categories: The business's Yelp category strings.
        label_map: Your label -> set of Yelp categories mapping.

    Returns:
        The set of matching labels. Empty if nothing matched; more
        than one element means a cross-label collision. Note that
        several categories mapping to the SAME label is not a
        collision, which is why this returns a set of labels rather
        than a count of matched categories.
    """
    category_set = set(categories)
    return {
        label
        for label, label_categories in label_map.items()
        if category_set & label_categories
    }


def analyze_labels(restaurants: dict,
                   label_map: dict,
                   review_counts: Counter) -> dict:
    """Assign labels and measure coverage and collisions.

    Args:
        restaurants: Dict from load_restaurant_businesses().
        label_map: Your label -> set of Yelp categories mapping.
        review_counts: business_id -> review count, or empty Counter.

    Returns:
        A dict of analysis results for reporting.
    """
    n_unlabeled = 0
    n_single = 0
    n_collision = 0

    reviews_unlabeled = 0
    reviews_single = 0
    reviews_collision = 0

    per_label_businesses = Counter()
    per_label_reviews = Counter()
    collision_pairs = Counter()
    collision_examples = defaultdict(list)

    for business_id, info in restaurants.items():
        labels = assign_labels(info["categories"], label_map)
        n_reviews = review_counts.get(business_id, 0)

        if not labels:
            n_unlabeled += 1
            reviews_unlabeled += n_reviews
        elif len(labels) == 1:
            n_single += 1
            reviews_single += n_reviews
            label = next(iter(labels))
            per_label_businesses[label] += 1
            per_label_reviews[label] += n_reviews
        else:
            n_collision += 1
            reviews_collision += n_reviews
            for pair in combinations(sorted(labels), 2):
                collision_pairs[pair] += 1
                if len(collision_examples[pair]) < 3:
                    collision_examples[pair].append(info["name"])

    return {
        "n_unlabeled": n_unlabeled,
        "n_single": n_single,
        "n_collision": n_collision,
        "reviews_unlabeled": reviews_unlabeled,
        "reviews_single": reviews_single,
        "reviews_collision": reviews_collision,
        "per_label_businesses": per_label_businesses,
        "per_label_reviews": per_label_reviews,
        "collision_pairs": collision_pairs,
        "collision_examples": collision_examples,
    }


def format_audit_section(category_df: pd.DataFrame, top_n: int) -> list:
    """Format the category frequency table as report lines.

    Args:
        category_df: DataFrame from audit_categories().
        top_n: How many rows to include.

    Returns:
        A list of report lines.
    """
    lines = []
    lines.append("CATEGORY FREQUENCY AMONG RESTAURANT BUSINESSES")
    lines.append("=" * 60)
    lines.append(f"Distinct categories found: {len(category_df):,}")
    lines.append(f"\nTop {top_n} by business count:")
    lines.append(f"{'category':<35}{'businesses':>12}{'reviews':>14}")

    for _, row in category_df.head(top_n).iterrows():
        lines.append(
            f"{row['category'][:34]:<35}"
            f"{row['n_businesses']:>12,}"
            f"{row['n_reviews']:>14,}"
        )

    return lines


def format_label_section(analysis: dict, label_map: dict) -> list:
    """Format the label coverage and collision results as lines.

    Args:
        analysis: Dict from analyze_labels().
        label_map: Your label -> set of Yelp categories mapping.

    Returns:
        A list of report lines.
    """
    lines = []
    lines.append("\n\nLABEL ASSIGNMENT")
    lines.append("=" * 60)

    total_businesses = (
        analysis["n_unlabeled"]
        + analysis["n_single"]
        + analysis["n_collision"]
    )
    total_reviews = (
        analysis["reviews_unlabeled"]
        + analysis["reviews_single"]
        + analysis["reviews_collision"]
    )

    def pct(part, whole):
        return (part / whole * 100) if whole else 0

    lines.append(f"\nBusinesses matching no label:   "
                 f"{analysis['n_unlabeled']:>8,} "
                 f"({pct(analysis['n_unlabeled'], total_businesses):.1f}%)")
    lines.append(f"Businesses matching one label:  "
                 f"{analysis['n_single']:>8,} "
                 f"({pct(analysis['n_single'], total_businesses):.1f}%)")
    lines.append(f"Businesses matching 2+ labels:  "
                 f"{analysis['n_collision']:>8,} "
                 f"({pct(analysis['n_collision'], total_businesses):.1f}%)")

    lines.append(f"\nReviews matching no label:      "
                 f"{analysis['reviews_unlabeled']:>10,} "
                 f"({pct(analysis['reviews_unlabeled'], total_reviews):.1f}%)")
    lines.append(f"Reviews matching one label:     "
                 f"{analysis['reviews_single']:>10,} "
                 f"({pct(analysis['reviews_single'], total_reviews):.1f}%)")
    lines.append(f"Reviews matching 2+ labels:     "
                 f"{analysis['reviews_collision']:>10,} "
                 f"({pct(analysis['reviews_collision'], total_reviews):.1f}%)")

    lines.append("\n\nPER-LABEL TOTALS (single-label businesses only)")
    lines.append("-" * 60)
    lines.append(f"{'label':<25}{'businesses':>12}{'reviews':>14}")
    for label in sorted(label_map):
        lines.append(
            f"{label[:24]:<25}"
            f"{analysis['per_label_businesses'][label]:>12,}"
            f"{analysis['per_label_reviews'][label]:>14,}"
        )

    lines.append("\n\nCOLLIDING LABEL PAIRS")
    lines.append("-" * 60)
    if not analysis["collision_pairs"]:
        lines.append("No collisions found.")
    else:
        for pair, count in analysis["collision_pairs"].most_common():
            examples = ", ".join(analysis["collision_examples"][pair])
            lines.append(f"{pair[0]} + {pair[1]}: {count:,} businesses")
            lines.append(f"    e.g. {examples}")

    lines.append("\n\nIF DROPPING COLLISIONS AND UNLABELED BUSINESSES")
    lines.append("-" * 60)
    lines.append(f"Businesses kept: {analysis['n_single']:,} "
                 f"of {total_businesses:,}")
    lines.append(f"Reviews kept:    {analysis['reviews_single']:,} "
                 f"of {total_reviews:,}")

    return lines


def main() -> None:
    """Run the category audit and, if configured, label analysis."""
    if not Path(BUSINESS_JSON_PATH).exists():
        raise FileNotFoundError(
            f"business.json not found at '{BUSINESS_JSON_PATH}'. "
            "Update BUSINESS_JSON_PATH to point at the extracted file."
        )

    restaurants = load_restaurant_businesses(BUSINESS_JSON_PATH)

    review_counts = Counter()
    if COUNT_REVIEWS:
        if Path(REVIEWS_CSV_PATH).exists():
            review_counts = count_reviews_per_business(
                REVIEWS_CSV_PATH, CHUNK_SIZE
            )
        else:
            print(f"Reviews CSV not found at '{REVIEWS_CSV_PATH}', "
                  "reporting business counts only")

    category_df = audit_categories(restaurants, review_counts)
    category_df.to_csv(CATEGORY_COUNTS_PATH, index=False)
    print(f"Saved full category counts to '{CATEGORY_COUNTS_PATH}'")

    report_lines = format_audit_section(category_df, TOP_N)

    if LABEL_MAP:
        analysis = analyze_labels(restaurants, LABEL_MAP, review_counts)
        report_lines += format_label_section(analysis, LABEL_MAP)
    else:
        report_lines.append(
            "\n\nLABEL_MAP is empty, so label/collision analysis was "
            "skipped. Fill in LABEL_MAP near the top of this script "
            "with your taxonomy and re-run."
        )

    report = "\n".join(report_lines)
    print("\n" + report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved report to '{REPORT_PATH}'")


if __name__ == "__main__":
    main()
