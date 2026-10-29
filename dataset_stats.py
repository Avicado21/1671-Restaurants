"""Compute summary statistics on yelp_restaurant_reviews.csv.

Built for the filtered restaurant-reviews CSV produced by
extract_restaurant_reviews.py, which can be large (potentially
several GB, millions of rows). Reads the file in chunks rather than
all at once, so memory use stays bounded regardless of file size.

Reports:
    - Total row count and column names
    - Missing values per column
    - Rating distribution (count and percent per star rating)
    - Date range covered (earliest / latest review)
    - Number of unique businesses, cities, and states
    - Top cities and states by review count
    - Review text length statistics (characters and words)

Writes a human-readable summary to both the console and a text file.
"""

from collections import Counter

import pandas as pd

INPUT_PATH = "yelp_restaurant_reviews.csv"
STATS_OUTPUT_PATH = "dataset_stats_summary.txt"

# Rows processed per chunk. Lower this if memory is tight, raise it
# if the file is small enough that larger chunks run faster.
CHUNK_SIZE = 100_000

# How many top cities/states to report.
TOP_N = 10


def compute_stats(input_path: str, chunk_size: int) -> dict:
    """Stream the CSV in chunks and accumulate summary statistics.

    Args:
        input_path: Path to the restaurant reviews CSV.
        chunk_size: Number of rows to read per chunk.

    Returns:
        A dict of accumulated statistics, ready for formatting.
    """
    total_rows = 0
    missing_counts = None
    rating_counts = Counter()
    business_ids = set()
    cities = Counter()
    states = Counter()
    min_date = None
    max_date = None

    # Running totals for review length, avoiding storing every
    # individual length in memory.
    char_len_sum = 0
    char_len_min = None
    char_len_max = None
    word_len_sum = 0

    reader = pd.read_csv(input_path, chunksize=chunk_size)

    for chunk in reader:
        total_rows += len(chunk)

        chunk_missing = chunk.isna().sum()
        if missing_counts is None:
            missing_counts = chunk_missing
        else:
            missing_counts += chunk_missing

        rating_counts.update(chunk["stars"].dropna().astype(int))
        business_ids.update(chunk["business_id"].dropna().unique())
        cities.update(chunk["city"].dropna())
        states.update(chunk["state"].dropna())

        chunk_dates = pd.to_datetime(chunk["date"], errors="coerce")
        chunk_min = chunk_dates.min()
        chunk_max = chunk_dates.max()
        if pd.notna(chunk_min) and (min_date is None or chunk_min < min_date):
            min_date = chunk_min
        if pd.notna(chunk_max) and (max_date is None or chunk_max > max_date):
            max_date = chunk_max

        text_lengths = chunk["text"].dropna().str.len()
        word_counts = chunk["text"].dropna().str.split().str.len()

        char_len_sum += text_lengths.sum()
        word_len_sum += word_counts.sum()

        chunk_char_min = text_lengths.min()
        chunk_char_max = text_lengths.max()
        if char_len_min is None or chunk_char_min < char_len_min:
            char_len_min = chunk_char_min
        if char_len_max is None or chunk_char_max > char_len_max:
            char_len_max = chunk_char_max

    return {
        "total_rows": total_rows,
        "missing_counts": missing_counts,
        "rating_counts": rating_counts,
        "n_unique_businesses": len(business_ids),
        "top_cities": cities.most_common(TOP_N),
        "top_states": states.most_common(TOP_N),
        "min_date": min_date,
        "max_date": max_date,
        "avg_char_len": char_len_sum / total_rows if total_rows else 0,
        "min_char_len": char_len_min,
        "max_char_len": char_len_max,
        "avg_word_len": word_len_sum / total_rows if total_rows else 0,
    }


def format_report(stats: dict, columns: list) -> str:
    """Turn the accumulated stats dict into a readable text report.

    Args:
        stats: The dict returned by compute_stats().
        columns: List of column names in the source CSV, used to
            report missing values per column in order.

    Returns:
        A formatted multi-line string summarizing the dataset.
    """
    lines = []
    lines.append("YELP RESTAURANT REVIEWS - DATASET SUMMARY")
    lines.append("=" * 50)

    lines.append(f"\nTotal rows (reviews): {stats['total_rows']:,}")
    lines.append(f"Unique businesses: {stats['n_unique_businesses']:,}")

    lines.append("\nMissing values per column:")
    for column in columns:
        missing = stats["missing_counts"].get(column, 0)
        lines.append(f"  {column}: {missing:,}")

    lines.append("\nRating distribution:")
    total = stats["total_rows"]
    for rating in sorted(stats["rating_counts"]):
        count = stats["rating_counts"][rating]
        pct = (count / total * 100) if total else 0
        lines.append(f"  {rating} star: {count:,} ({pct:.1f}%)")

    lines.append(
        f"\nDate range: {stats['min_date'].date()} to "
        f"{stats['max_date'].date()}"
    )

    lines.append(f"\nTop {TOP_N} cities by review count:")
    for city, count in stats["top_cities"]:
        lines.append(f"  {city}: {count:,}")

    lines.append(f"\nTop {TOP_N} states by review count:")
    for state, count in stats["top_states"]:
        lines.append(f"  {state}: {count:,}")

    lines.append("\nReview text length:")
    lines.append(f"  Average: {stats['avg_char_len']:.1f} characters, "
                  f"{stats['avg_word_len']:.1f} words")
    lines.append(f"  Shortest: {stats['min_char_len']:.0f} characters")
    lines.append(f"  Longest: {stats['max_char_len']:.0f} characters")

    return "\n".join(lines)


def main() -> None:
    """Compute and report statistics for the restaurant reviews CSV."""
    columns = pd.read_csv(INPUT_PATH, nrows=0).columns.tolist()

    print(f"Computing statistics for '{INPUT_PATH}' in chunks of "
          f"{CHUNK_SIZE:,} rows...")
    stats = compute_stats(INPUT_PATH, CHUNK_SIZE)

    report = format_report(stats, columns)
    print("\n" + report)

    with open(STATS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved summary to '{STATS_OUTPUT_PATH}'")


if __name__ == "__main__":
    main()
