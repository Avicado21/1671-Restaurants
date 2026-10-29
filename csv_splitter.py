"""Split or reassemble the restaurant reviews CSV, via command line.

Built for yelp_restaurant_reviews.csv, which can exceed GitHub's
100 MB per-file limit. This script has two modes, chosen by the
first command-line argument:

    python3 csv_splitter.py split
        Splits INPUT_PATH into multiple part files, each under
        MAX_PART_SIZE_BYTES, written into OUTPUT_DIR. Every part
        gets the header row repeated at the top so it is a valid,
        independently-readable CSV on its own.

    python3 csv_splitter.py reassemble
        Reads every part file in OUTPUT_DIR (in order) and stitches
        them back into a single file at INPUT_PATH, writing the
        header only once at the very top.

Rows are streamed in and out one at a time in both modes, so memory
usage stays constant regardless of how large the file is.
"""

import csv
import re
import sys
from pathlib import Path

INPUT_PATH = "yelp_restaurant_reviews.csv"
OUTPUT_DIR = "yelp_restaurant_reviews_parts"

# Target max size per output file. Set a bit below the 25 MB target
# to leave a safety margin, since the exact byte count of the last
# row written before the check can push slightly over.
MAX_PART_SIZE_BYTES = 24 * 1024 * 1024  # 24 MB

# Matches the numeric suffix in a part filename, e.g. "..._part12"
# -> "12", used to sort parts in the correct numeric order during
# reassembly (plain alphabetical sort would put part10 before part2).
PART_NUMBER_PATTERN = re.compile(r"_part(\d+)\.csv$")


def split_csv(
    input_path: str,
    output_dir: str,
    max_part_size_bytes: int,
) -> None:
    """Split input_path into size-capped part files in output_dir.

    Args:
        input_path: Path to the large source CSV.
        output_dir: Directory to write the part files into. Created
            if it does not already exist.
        max_part_size_bytes: Approximate max size, in bytes, for
            each output part file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    input_stem = Path(input_path).stem

    with open(input_path, "r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        header = next(reader)

        part_number = 1
        part_file, writer = _open_new_part(
            output_path, input_stem, part_number, header
        )

        for row in reader:
            # +1 roughly accounts for the newline character.
            row_size = len(",".join(row).encode("utf-8")) + 1

            if part_file.tell() + row_size > max_part_size_bytes:
                part_file.close()
                part_number += 1
                part_file, writer = _open_new_part(
                    output_path, input_stem, part_number, header
                )

            writer.writerow(row)

        part_file.close()

    print(f"Split '{input_path}' into {part_number} part file(s) "
          f"under '{output_path}'")


def _open_new_part(
    output_path: Path,
    input_stem: str,
    part_number: int,
    header: list,
):
    """Open a new part file and write the header row into it.

    Args:
        output_path: Directory the part file should be created in.
        input_stem: Base filename (without extension) of the
            original input CSV, used to name the part file.
        part_number: 1-indexed number of this part, used in the
            filename.
        header: The header row to write at the top of this part.

    Returns:
        A tuple of (open file object, csv.writer for that file).
    """
    part_path = output_path / f"{input_stem}_part{part_number}.csv"
    part_file = open(part_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(part_file, lineterminator="\n")
    writer.writerow(header)
    print(f"Writing '{part_path}'")
    return part_file, writer


def find_part_files(output_dir: str) -> list:
    """Find and numerically sort all part files in output_dir.

    Args:
        output_dir: Directory containing the "*_partN.csv" files.

    Returns:
        A list of Paths sorted by their numeric part suffix, e.g.
        part1, part2, ..., part10 (not alphabetical order, which
        would incorrectly put part10 before part2).

    Raises:
        FileNotFoundError: If no part files are found.
    """
    output_path = Path(output_dir)
    all_parts = list(output_path.glob("*_part*.csv"))

    matched = []
    for part in all_parts:
        match = PART_NUMBER_PATTERN.search(part.name)
        if match:
            matched.append((int(match.group(1)), part))

    if not matched:
        raise FileNotFoundError(
            f"No '*_partN.csv' files found under '{output_path}'"
        )

    matched.sort(key=lambda pair: pair[0])
    return [part for _, part in matched]


def reassemble_csv(output_dir: str, input_path: str) -> None:
    """Stitch all part files back into a single CSV.

    Args:
        output_dir: Directory containing the part files to combine.
        input_path: Path to write the reassembled CSV to.
    """
    part_paths = find_part_files(output_dir)
    print(f"Found {len(part_paths)} part file(s) to reassemble")

    with open(input_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile, lineterminator="\n")
        header_written = False

        for part_path in part_paths:
            with open(part_path, "r", newline="", encoding="utf-8") as pf:
                reader = csv.reader(pf)
                header = next(reader)

                if not header_written:
                    writer.writerow(header)
                    header_written = True

                for row in reader:
                    writer.writerow(row)

            print(f"Merged '{part_path}'")

    print(f"Reassembled {len(part_paths)} part file(s) into "
          f"'{input_path}'")


def main() -> None:
    """Parse the command-line mode and run split or reassemble."""
    valid_modes = {"split", "reassemble"}

    if len(sys.argv) != 2 or sys.argv[1] not in valid_modes:
        print(f"Usage: python3 {sys.argv[0]} <split|reassemble>")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "split":
        split_csv(INPUT_PATH, OUTPUT_DIR, MAX_PART_SIZE_BYTES)
    elif mode == "reassemble":
        reassemble_csv(OUTPUT_DIR, INPUT_PATH)


if __name__ == "__main__":
    main()
