"""Cleaning pipeline for Yelp_Restaurant_Reviews.csv.

Despite the filename, this dataset consists entirely of reviews for
dessert and sweets businesses (ice cream shops, bakeries, chocolate
shops, etc.) rather than general restaurants.

Issues addressed by this script:
    1. Rows where Review Text is an Excel formula error ("#ERROR!"),
       which is spreadsheet junk rather than real data.
    2. Mojibake / double-encoded text (e.g. "purAed" -> "pured"),
       caused by UTF-8 bytes having been read as Latin-1 at some
       point in the file's history.
    3. Leading/trailing whitespace in review text.
    4. Date strings that should be real datetime values.
    5. Business name, which only exists buried inside the Yelp URL,
       extracted into its own readable column.
"""

import re
from urllib.parse import unquote

import pandas as pd

INPUT_PATH = "Yelp_Restaurant_Reviews.csv"
OUTPUT_PATH = "Yelp_Restaurant_Reviews_clean.csv"

# Marker Excel writes into a cell when a formula fails to evaluate.
ERROR_PLACEHOLDER = "#ERROR!"

# Matches a trailing " -<digits>" disambiguator that Yelp appends to
# a business slug when multiple locations share the same base name,
# e.g. "...-french-bakery-11" -> drop the "-11".
TRAILING_ID_PATTERN = re.compile(r"\s\d+$")


def fix_mojibake(text: str) -> str:
    """Repair text that was UTF-8 but got decoded as Latin-1.

    Args:
        text: A single review's raw text.

    Returns:
        The repaired text if the round-trip succeeds, otherwise the
        original text unchanged (meaning it was never mis-decoded).
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def extract_business_name(url: str) -> str:
    """Pull a human-readable business name out of a Yelp URL.

    Example:
        "https://www.yelp.com/biz/am%C3%A9lies-french-bakery-11"
        -> "amelies french bakery"

    Args:
        url: The full Yelp business URL for a review.

    Returns:
        A readable business name with URL-encoding decoded, hyphens
        replaced by spaces, and the trailing disambiguator removed.
    """
    slug = url.split("/biz/")[-1]
    slug = unquote(slug)
    slug = slug.replace("-", " ")
    slug = TRAILING_ID_PATTERN.sub("", slug)
    return slug.strip()


def drop_error_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows whose review text is just a spreadsheet error.

    Args:
        df: The raw reviews DataFrame.

    Returns:
        A copy of df with "#ERROR!" rows removed.
    """
    is_error = df["Review Text"].str.strip() == ERROR_PLACEHOLDER
    n_dropped = int(is_error.sum())
    print(f"Dropped {n_dropped} rows with '{ERROR_PLACEHOLDER}' text")
    return df[~is_error].copy()


def clean_review_text(df: pd.DataFrame) -> pd.DataFrame:
    """Fix encoding issues and strip stray whitespace in review text.

    Args:
        df: The reviews DataFrame (after error rows are dropped).

    Returns:
        A copy of df with a cleaned "Review Text" column.
    """
    df = df.copy()
    df["Review Text"] = df["Review Text"].apply(fix_mojibake)
    df["Review Text"] = df["Review Text"].str.strip()
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the "Date" column from strings to real datetimes.

    Args:
        df: The reviews DataFrame.

    Returns:
        A copy of df with "Date" parsed as datetime64.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    return df


def add_business_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a readable "Business" column derived from "Yelp URL".

    Args:
        df: The reviews DataFrame.

    Returns:
        A copy of df with a new "Business" column.
    """
    df = df.copy()
    df["Business"] = df["Yelp URL"].apply(extract_business_name)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline on the raw reviews DataFrame.

    Args:
        df: The raw reviews DataFrame, as loaded from CSV.

    Returns:
        A cleaned DataFrame, ready for analysis or model training.
    """
    df = drop_error_rows(df)
    df = clean_review_text(df)
    df = parse_dates(df)
    df = add_business_column(df)
    df = df.reset_index(drop=True)
    return df


def main() -> None:
    """Load the raw CSV, clean it, and save the result."""
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows")

    clean_df = clean(df)
    print(f"Final shape: {clean_df.shape}")
    print(clean_df.head())

    clean_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved cleaned data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
