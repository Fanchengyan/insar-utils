#!/usr/bin/env python3
"""
Pipeline runner script for InSAR processing workflow.

Executes run files in sequential order (run_01 through run_11).
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(log_file: str | None = None, work_dir: Path | None = None) -> Path:
    """
    Set up logging configuration.

    Parameters
    ----------
    log_file : str | None
        Path to the log file. If None, a log file will be automatically
        generated with timestamp. If it's a filename (not an absolute path),
        the log file will be placed in work_dir.
    work_dir : Path | None
        Working directory. Used to determine where to place the log file
        if log_file is not an absolute path. Defaults to current directory.

    Returns
    -------
    Path
        Path to the log file being used.
    """
    if work_dir is None:
        work_dir = Path.cwd()

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"pipeline_run_{timestamp}.log"

    log_path = Path(log_file)

    # If log_file is not an absolute path, place it in work_dir
    if not log_path.is_absolute():
        log_path = work_dir / log_path

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger.info("Log file: %s", log_path.resolve())

    return log_path


def run_command_from_file(filepath: Path) -> tuple[bool, int]:
    """
    Read and execute the command from a run file.

    Parameters
    ----------
    filepath : Path
        Path to the run file.

    Returns
    -------
    tuple[bool, int]
        A tuple of (success, return_code).
    """
    logger.info("=" * 80)
    logger.info("Running: %s", filepath.name)
    logger.info("=" * 80)

    # Read the command from the file
    try:
        with open(filepath, "r") as f:
            command = f.read().strip()
    except Exception as e:
        logger.exception("Failed to read %s: %s", filepath, e)
        return False, -1

    if not command:
        logger.warning("%s is empty, skipping...", filepath)
        return True, 0

    logger.info("Command: %s", command)
    logger.info("Started at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Execute the command using bash
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=False,
            stdout=sys.stdout,
            stderr=sys.stderr,
            executable="/bin/bash",
        )

        logger.info("Finished at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        if result.returncode == 0:
            logger.info("SUCCESS: %s completed successfully", filepath.name)
            return True, result.returncode
        else:
            logger.error(
                "%s failed with return code %d", filepath.name, result.returncode
            )
            return False, result.returncode

    except Exception as e:
        logger.exception("Exception while running %s: %s", filepath, e)
        return False, -1


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline runner script for InSAR processing workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-l",
        "--log-file",
        type=str,
        default=None,
        help="Path to log file. If not specified, a log file will be "
        "automatically generated with timestamp (pipeline_run_YYYYMMDD_HHMMSS.log). "
        "If a filename is provided (not an absolute path), the log file will be "
        "placed in the working directory.",
    )
    parser.add_argument(
        "-w",
        "--work-dir",
        type=str,
        default=None,
        help="Working directory for the pipeline. If not specified, "
        "the current directory will be used.",
    )
    return parser.parse_args()


def main() -> int:
    """
    Main pipeline execution function.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()

    # Determine working directory
    if args.work_dir is not None:
        work_dir = Path(args.work_dir).resolve()
        if not work_dir.exists():
            print(f"ERROR: Working directory does not exist: {work_dir}")
            return 1
        if not work_dir.is_dir():
            print(f"ERROR: Specified path is not a directory: {work_dir}")
            return 1
    else:
        work_dir = Path.cwd()

    # Setup logging
    log_path = setup_logging(args.log_file, work_dir)

    logger.info("Working directory: %s", work_dir)

    dem_dir = "/home/fancy/workspace/Data/UCM/pairs_frame/Sentinel1TileScenes_DESCENDING_106_461/dem"
    cmd = f"ln -sf {dem_dir}/* ./"
    logger.info("Creating symlinks: %s", cmd)
    os.popen(cmd)
    logger.info("Symlink done")

    dem_files = [f for f in os.listdir() if f.startswith("dem")]
    logger.info("DEM files: %s", dem_files)

    run_files = [i.name for i in sorted(work_dir.glob("run_*"))]
    logger.info("Run files: %s", run_files)

    logger.info("#" * 80)
    logger.info("# InSAR Processing Pipeline")
    logger.info("# Started at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("# Working directory: %s", work_dir)
    logger.info("# Log file: %s", log_path.resolve())
    logger.info("#" * 80)

    # Track execution results
    results: list[tuple[str, bool, int]] = []

    # Execute each run file in order
    for run_file in run_files[10:]:  # Exclude the last unwrap step
        filepath = work_dir / run_file

        if not filepath.exists():
            logger.error("%s not found!", run_file)
            results.append((run_file, False, -1))
            logger.error("Stopping pipeline due to missing file.")
            break

        success, return_code = run_command_from_file(filepath)
        results.append((run_file, success, return_code))

        if not success:
            logger.error("Pipeline stopped at %s", run_file)
            logger.error("Please fix the error before continuing.")
            break

    # Print summary
    logger.info("#" * 80)
    logger.info("# Pipeline Summary")
    logger.info("# Finished at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("#" * 80)

    for run_file, success, return_code in results:
        status = "SUCCESS" if success else "FAILED"
        logger.info("%s - %s (exit code: %d)", status, run_file, return_code)

    # Determine overall success
    all_success = all(success for _, success, _ in results)
    total_run = len(results)
    total_expected = len(run_files)

    logger.info("Completed %d/%d steps", total_run, total_expected)

    if all_success and total_run == total_expected:
        logger.info("Pipeline completed successfully!")
        return 0
    else:
        logger.error("Pipeline failed or incomplete!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
