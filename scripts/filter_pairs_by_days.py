#!/usr/bin/env python3
"""Filter interferometric pairs by temporal baseline from run files.

This script extracts interferometric pairs from ISCE run files and filters
them based on the specified temporal baseline (in days).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import operator
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Pattern to match date pairs like 20220103_20220115
DATE_PAIR_PATTERN = re.compile(r"(\d{8})_(\d{8})")

# Map operator strings to functions
OPERATORS = {
    "lt": operator.lt,
    "le": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
    "ge": operator.ge,
    "gt": operator.gt,
}


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYYMMDD format.

    Parameters
    ----------
    date_str : str
        Date string in YYYYMMDD format.

    Returns
    -------
    datetime
        Parsed datetime object.
    """
    return datetime.strptime(date_str, "%Y%m%d")


def calculate_temporal_baseline(date1_str: str, date2_str: str) -> int:
    """Calculate temporal baseline in days between two dates.

    Parameters
    ----------
    date1_str : str
        First date string in YYYYMMDD format.
    date2_str : str
        Second date string in YYYYMMDD format.

    Returns
    -------
    int
        Absolute temporal baseline in days.
    """
    date1 = parse_date(date1_str)
    date2 = parse_date(date2_str)
    return abs((date2 - date1).days)


def extract_date_pair(line: str) -> tuple[str, str] | None:
    """Extract date pair from a command line.

    Parameters
    ----------
    line : str
        A line from the run file.

    Returns
    -------
    tuple[str, str] | None
        Tuple of (date1, date2) if found, None otherwise.
    """
    match = DATE_PAIR_PATTERN.search(line)
    if match:
        return match.group(1), match.group(2)
    return None


def filter_run_file(
    input_file: Path,
    threshold_days: int,
    operator_str: str = "le",
    output_file: Path | None = None,
) -> list[str]:
    """Filter run file by temporal baseline.

    Parameters
    ----------
    input_file : Path
        Path to the input run file.
    threshold_days : int
        Temporal baseline threshold in days.
    operator_str : str, optional
        Comparison operator ('lt', 'le', 'eq', 'ne', 'ge', 'gt'), default 'le'.
    output_file : Path | None, optional
        Path to the output file. If None, auto-generated.

    Returns
    -------
    list[str]
        List of filtered command lines.
    """
    if not input_file.exists():
        logger.error("Input file does not exist: %s", input_file)
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    if operator_str not in OPERATORS:
        raise ValueError(f"Invalid operator: {operator_str}")

    op_func = OPERATORS[operator_str]
    filtered_lines: list[str] = []

    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and wait commands
            if not line or line == "wait":
                continue

            # Extract date pair
            date_pair = extract_date_pair(line)
            if date_pair is None:
                logger.warning("No date pair found in line: %s", line)
                continue

            date1, date2 = date_pair
            temporal_baseline = calculate_temporal_baseline(date1, date2)

            if op_func(temporal_baseline, threshold_days):
                # Remove trailing ' &' if present
                line = line.rstrip()
                if line.endswith(" &"):
                    line = line[:-2].rstrip()
                elif line.endswith("&"):
                    line = line[:-1].rstrip()
                filtered_lines.append(line)
                logger.debug(
                    "Included: %s_%s (baseline: %d days, op: %s %d)",
                    date1,
                    date2,
                    temporal_baseline,
                    operator_str,
                    threshold_days,
                )
            else:
                logger.debug(
                    "Excluded: %s_%s (baseline: %d days, op: %s %d)",
                    date1,
                    date2,
                    temporal_baseline,
                    operator_str,
                    threshold_days,
                )

    # Determine output file path
    if output_file is None:
        # Create output directory: <operator>_<days>
        output_dir = input_file.parent / f"{operator_str}{threshold_days}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output file inside the directory with same name as input
        output_file = output_dir / input_file.name

    # Write filtered lines
    with output_file.open("w", encoding="utf-8") as f:
        for line in filtered_lines:
            f.write(line + "\n")

    logger.info("Total pairs found: %d", len(filtered_lines))
    logger.info("Output written to: %s", output_file)

    return filtered_lines


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Filter interferometric pairs by temporal baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Filter pairs with temporal baseline <= 24 days (default)
    python filter_pairs_by_days.py run_file -d 24

    # Filter pairs with temporal baseline > 24 days
    python filter_pairs_by_days.py run_file -d 24 --operator gt

    # Filter pairs with temporal baseline == 12 days
    python filter_pairs_by_days.py run_file -d 12 --op eq
        """,
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input run file",
    )
    parser.add_argument(
        "-d",
        "--days",
        type=int,
        required=True,
        help="Temporal baseline threshold in days",
    )
    parser.add_argument(
        "--op",
        "--operator",
        dest="operator",
        choices=["lt", "le", "eq", "ne", "ge", "gt"],
        default="le",
        help="Comparison operator (default: le)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: <input_file>_<op>_<days>days)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        filter_run_file(args.input_file, args.days, args.operator, args.output)
    except FileNotFoundError:
        sys.exit(1)
    except Exception as e:
        logger.error("Error processing file: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
