"""Extract restaurant-only reviews from the official Yelp Open Dataset.

The Yelp Open Dataset ships as a .tar archive containing several
newline-delimited JSON files (one JSON object per line):
    yelp_academic_dataset_business.json
    yelp_academic_dataset_review.json
    yelp_academic_dataset_user.json
    yelp_academic_dataset_checkin.json
    yelp_academic_dataset_tip.json

This script:
    1. Locates the dataset (extracting a tar archive first if the
       given DATASET_PATH is a .tar file; used directly if it is
       already a folder of extracted JSON files).
    2. Reads business.json and keeps only businesses whose
       "categories" field contains "Restaurants".
    3. Streams review.json line-by-line (it is too large to load
       into memory all at once) and keeps only reviews written for
       one of those restaurant business_ids.
    4. Writes the matching reviews to a single CSV, joined with a
       few useful business fields (name, city, state).

Note on licensing: the Yelp Open Dataset Terms of Use restrict this
data to non-commercial academic use, prohibit redistributing it to
third parties, and require Yelp's review before any public release
of findings. Keep the extracted files and output CSV private to
your project / course, and don't publish the raw CSV anywhere.
"""

import csv
import json
import tarfile
from pathlib import Path

# Path to either the downloaded .tar file OR a folder that already
# contains the extracted yelp_academic_dataset_*.json files.
DATASET_PATH = "."

EXTRACT_DIR = "yelp_dataset_extracted"
OUTPUT_PATH = "yelp_restaurant_reviews.csv"

# The category string Yelp uses to tag a business as a restaurant.
RESTAURANT_CATEGORY = "Restaurants"

# Columns written to the output CSV, in order.
OUTPUT_FIELDS = [
    "review_id",
    "business_id",
    "business_name",
    "city",
    "state",
    "stars",
    "date",
    "text",
]


def resolve_dataset_path(dataset_path: str, extract_dir: str) -> Path:
    """Get a folder of extracted JSON files, handling several layouts.

    Accepts any of these for dataset_path, so the script works
    regardless of where things are unpacked:
        - A directory that already contains the JSON files directly.
        - A directory that contains a .tar file somewhere inside it
          (e.g. a "downloads" folder you dropped the tar into).
        - A .tar file itself.
    In the tar cases, the archive is extracted first.

    Args:
        dataset_path: Path to a folder or a .tar archive, as
            described above.
        extract_dir: Directory to extract into, if a tar archive is
            found (unused if the JSON files are already present).

    Returns:
        The Path to a directory containing the JSON files.

    Raises:
        FileNotFoundError: If dataset_path does not exist, or if a
            directory is given but no JSON or tar file is found
            anywhere inside it.
        ValueError: If dataset_path is a file but not a valid tar
            archive.
    """
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: '{path}'")

    if path.is_file():
        if not tarfile.is_tarfile(path):
            raise ValueError(f"'{path}' is not a valid tar file")
        return _extract_tar(path, extract_dir)

    # path is a directory: look for JSON files first, then for a
    # tar file nested inside (searched recursively, since it may be
    # a level or two deep, e.g. "Yelp JSON/yelp_dataset.tar").
    if list(path.rglob("*.json")):
        print(f"Using already-extracted folder: '{path}'")
        return path

    tar_matches = list(path.rglob("*.tar"))
    if tar_matches:
        return _extract_tar(tar_matches[0], extract_dir)

    raise FileNotFoundError(
        f"No .json or .tar files found anywhere under '{path}'"
    )


def _extract_tar(tar_path: Path, extract_dir: str) -> Path:
    """Extract a tar archive to extract_dir.

    Args:
        tar_path: Path to the .tar file to extract.
        extract_dir: Directory to extract the contents into.

    Returns:
        The Path to the extraction directory.
    """
    extract_path = Path(extract_dir)
    extract_path.mkdir(exist_ok=True)

    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_path)

    print(f"Extracted '{tar_path}' to '{extract_path}'")
    return extract_path


def find_json_file(extract_path: Path, name_fragment: str) -> Path:
    """Locate a dataset JSON file by a distinguishing name fragment.

    The tar sometimes nests files in a subdirectory, so this searches
    recursively rather than assuming a fixed layout.

    Args:
        extract_path: Directory the tar was extracted into.
        name_fragment: Substring to match in the filename, e.g.
            "business" or "review".

    Returns:
        The Path to the matching JSON file.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
    matches = list(extract_path.rglob(f"*{name_fragment}*.json"))
    if not matches:
        raise FileNotFoundError(
            f"No file matching '*{name_fragment}*.json' found under "
            f"{extract_path}"
        )
    return matches[0]


def load_restaurant_businesses(business_json_path: Path) -> dict:
    """Read business.json and keep only restaurant businesses.

    business.json is small enough (~150k lines) to load fully into
    memory, unlike review.json.

    Args:
        business_json_path: Path to the business.json file.

    Returns:
        A dict mapping business_id -> {"name", "city", "state"} for
        every business tagged with the "Restaurants" category.
    """
    restaurants = {}

    with open(business_json_path, "r", encoding="utf-8") as f:
        for line in f:
            business = json.loads(line)

            # "categories" is a single comma-separated string or None,
            # not a list, per the dataset schema.
            categories = business.get("categories") or ""
            category_list = [c.strip() for c in categories.split(",")]

            if RESTAURANT_CATEGORY in category_list:
                restaurants[business["business_id"]] = {
                    "name": business.get("name", ""),
                    "city": business.get("city", ""),
                    "state": business.get("state", ""),
                }

    print(f"Found {len(restaurants)} restaurant businesses")
    return restaurants


def write_restaurant_reviews(
    review_json_path: Path,
    restaurants: dict,
    output_path: str,
) -> None:
    """Stream review.json and write only restaurant reviews to CSV.

    Reviews are processed one line at a time so memory usage stays
    constant regardless of how large review.json is.

    Args:
        review_json_path: Path to the review.json file.
        restaurants: Lookup dict from load_restaurant_businesses().
        output_path: Path to write the filtered CSV to.
    """
    n_total = 0
    n_matched = 0

    with open(review_json_path, "r", encoding="utf-8") as infile, \
            open(output_path, "w", newline="", encoding="utf-8") as outfile:

        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for line in infile:
            n_total += 1
            review = json.loads(line)
            business_id = review["business_id"]

            business = restaurants.get(business_id)
            if business is None:
                continue  # Not a restaurant review, skip it.

            n_matched += 1
            writer.writerow({
                "review_id": review["review_id"],
                "business_id": business_id,
                "business_name": business["name"],
                "city": business["city"],
                "state": business["state"],
                "stars": review["stars"],
                "date": review["date"],
                "text": review["text"],
            })

    print(f"Scanned {n_total} reviews, kept {n_matched} restaurant reviews")
    print(f"Saved filtered reviews to '{output_path}'")


def main() -> None:
    """Run the full extract -> filter -> write pipeline."""
    extract_path = resolve_dataset_path(DATASET_PATH, EXTRACT_DIR)

    business_json_path = find_json_file(extract_path, "business")
    review_json_path = find_json_file(extract_path, "review")

    restaurants = load_restaurant_businesses(business_json_path)
    write_restaurant_reviews(review_json_path, restaurants, OUTPUT_PATH)


if __name__ == "__main__":
    main()
