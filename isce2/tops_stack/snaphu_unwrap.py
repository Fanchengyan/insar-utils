#!/usr/bin/env python3
"""SNAPHU Unwrapping Module using snaphu library.

This module provides functions to unwrap interferograms using the snaphu library
with support for parallel processing via nproc parameter.
"""

import configparser
import logging
from pathlib import Path
from typing import Literal

import isceobj
import numpy as np
import snaphu


def create_meta_file(raster_path: str, width: int, height: int, dtype: str) -> None:
    """Create metadata file for raster using ISCE image object."""
    unwImage = isceobj.Image.createImage()
    unwImage.setFilename(raster_path)
    unwImage.setWidth(width)
    unwImage.setLength(height)
    unwImage.dataType = dtype
    unwImage.setAccessMode("read")
    unwImage.renderHdr()


def parse_config(config_path: str) -> dict[str, str | float] | None:
    """Parse configuration file and extract unwrapping parameters.

    Parameters
    ----------
    config_path : str
        Path to configuration file.

    Returns
    -------
    dict[str, str | float] | None
        Dictionary with config parameters containing:
        - ifg: Path to interferogram file
        - unw: Path to output unwrapped file
        - coh: Path to coherence file
        - rlks: Range looks
        - alks: Azimuth looks
        - method: Unwrapping method
        Returns None if parsing fails.

    Notes
    -----
    Expects INI format with a 'Function-1' section containing unwrapping parameters.
    """
    try:
        config = configparser.ConfigParser()
        config.read(config_path)

        if "Function-1" not in config:
            logging.error(f"No 'Function-1' section found in {config_path}")
            return None

        func_section = config["Function-1"]

        # Extract required parameters
        result = {
            "ifg": func_section.get("ifg", "").strip(),
            "unw": func_section.get("unw", "").strip(),
            "coh": func_section.get("coh", "").strip(),
            "method": func_section.get("method", "snaphu").strip(),
        }

        # Extract optional numeric parameters
        try:
            result["rlks"] = float(func_section.get("rlks", "1"))
            result["alks"] = float(func_section.get("alks", "1"))
        except ValueError as e:
            logging.warning(f"Failed to parse rlks/alks in {config_path}: {e}")
            result["rlks"] = 1.0
            result["alks"] = 1.0

        # Validate required paths
        if not all([result["ifg"], result["unw"], result["coh"]]):
            logging.error(f"Missing required paths in {config_path}")
            return None

        return result

    except Exception as e:
        logging.error(f"Failed to parse config {config_path}: {e}")
        return None


def calculate_nlooks(
    rlks: float, alks: float, range_spacing: float = 0.8, azimuth_spacing: float = 0.8
) -> float:
    """Calculate equivalent number of looks.

    Parameters
    ----------
    rlks : float
        Range looks.
    alks : float
        Azimuth looks.
    range_spacing : float, default=0.8
        Range pixel spacing (for Sentinel-1).
    azimuth_spacing : float, default=0.8
        Azimuth pixel spacing (for Sentinel-1).

    Returns
    -------
    float
        Equivalent number of looks.

    Notes
    -----
    Formula: nlooks = (rlks * alks) / (range_spacing * azimuth_spacing)
    Default spacing values are for Sentinel-1 data.
    """
    return (rlks * alks) / (range_spacing * azimuth_spacing)


def unwrap_from_config(
    config_path: str,
    nproc: int = 1,
    ntiles: tuple[int, int] | None = None,
    tile_overlap: int = 0,
    nlooks: float | None = None,
    cost: Literal["smooth", "defo"] = "defo",
    init: Literal["mst", "mcf"] = "mcf",
    min_region_size: int = 100,
    tile_cost_thresh: int = 500,
    phase_grad_window: tuple[int, int] = (7, 7),
    min_conncomp_frac: float = 0.01,
    single_tile_reoptimize: bool = True,
    regrow_conncomps: bool = True,
    scratchdir: str | None = None,
    delete_scratch: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, str | bool]:
    """Execute SNAPHU unwrapping from configuration file.

    Parameters
    ----------
    config_path : str
        Path to configuration file.
    nproc : int, default=1
        Number of parallel processing cores.
    ntiles : tuple[int, int] | None, optional
        Tile grid dimensions (rows, cols). If None, no tiling is used.
    tile_overlap : int, default=0
        Tile overlap in pixels.
    nlooks : float | None, optional
        Equivalent number of looks. If None, calculated from config rlks/alks.
    cost : Literal["smooth", "defo", "p-norm"], default="defo"
        Cost function to use for unwrapping.
    init : Literal["mst", "mcf"], default="mcf"
        Initialization method for unwrapping algorithm.
        'mst' = Minimum Spanning Tree, 'mcf' = Minimum Cost Flow.
    min_region_size : int, default=100
        Minimum size (in pixels) of a region to be unwrapped separately.
    tile_cost_thresh : int, default=500
        Cost threshold for determining region boundaries in tiled unwrapping.
    phase_grad_window : tuple[int, int], default=(7, 7)
        Sliding window size (rows, cols) for computing phase gradients.
    min_conncomp_frac : float, default=0.01
        Minimum fraction of pixels that a connected component must contain
        to be retained in the output (range: 0.0 to 1.0).
    single_tile_reoptimize : bool, default=True
        If True, re-optimize the solution using a single tile after
        completing tiled unwrapping. Improves quality but increases runtime.
    regrow_conncomps : bool, default=True
        If True, regrow connected components after tiled unwrapping
        to merge components split by tile boundaries.
    scratchdir : str | None, optional
        Directory for storing intermediate scratch files. If None,
        a temporary directory is automatically created.
    delete_scratch : bool, default=True
        If True, delete the scratch directory and its contents after unwrapping.
        Set to False for debugging or to preserve intermediate results.
    logger : logging.Logger | None, optional
        Logger instance. If None, uses default logger.

    Returns
    -------
    dict[str, str | bool]
        Dictionary with execution results containing:
        - config: Path to config file
        - success: True if unwrapping completed successfully
        - message: Status message
        - output: Output file path
        - error: Error message if failed

    Raises
    ------
    FileNotFoundError
        If input files (ifg, coh) do not exist.
    ValueError
        If configuration is invalid.

    Notes
    -----
    This function:
    1. Parses the configuration file
    2. Opens input rasters (interferogram, coherence)
    3. Creates output rasters (unwrapped phase, connected components)
    4. Calls snaphu.unwrap() with specified parameters
    5. Handles errors and logs progress
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    result = {
        "config": config_path,
        "success": False,
        "message": "",
        "output": "",
        "error": "",
    }

    try:
        # Parse configuration
        config_info = parse_config(config_path)
        if config_info is None:
            result["message"] = "Failed to parse configuration"
            result["error"] = "Invalid or missing configuration parameters"
            logger.error(f"Failed to parse config: {config_path}")
            return result

        ifg_path = config_info["ifg"]
        unw_path = config_info["unw"]
        coh_path = config_info["coh"]
        conncomp_path = unw_path + ".conncomp"

        # Validate input files exist
        if not Path(ifg_path).exists():
            result["message"] = f"Interferogram file not found: {ifg_path}"
            result["error"] = result["message"]
            logger.error(result["message"])
            return result

        if not Path(coh_path).exists():
            result["message"] = f"Coherence file not found: {coh_path}"
            result["error"] = result["message"]
            logger.error(result["message"])
            return result

        # Calculate nlooks if not provided
        if nlooks is None:
            rlks = config_info.get("rlks", 1.0)
            alks = config_info.get("alks", 1.0)
            nlooks = calculate_nlooks(rlks, alks)
            logger.debug(
                f"Calculated nlooks={nlooks:.2f} from rlks={rlks}, alks={alks}"
            )

        # Open input rasters
        logger.info(f"Opening input files for {Path(config_path).name}")
        igram = snaphu.io.Raster(ifg_path)
        corr = snaphu.io.Raster(coh_path)
        height, width = igram.shape

        # Create output directory if it doesn't exist
        unw_dir = Path(unw_path).parent
        unw_dir.mkdir(parents=True, exist_ok=True)

        # Create output rasters

        unw = np.memmap(unw_path, dtype=np.float32, mode="w+", shape=(height, width))
        conncomp = np.memmap(
            conncomp_path, dtype=np.uint8, mode="w+", shape=(height, width)
        )

        # Prepare unwrap parameters
        unwrap_params = {
            "igram": igram,
            "corr": corr,
            "nlooks": nlooks,
            "unw": unw,
            "conncomp": conncomp,
            "nproc": nproc,
            "cost": cost,
            "init": init,
            "min_region_size": min_region_size,
            "tile_cost_thresh": tile_cost_thresh,
            "phase_grad_window": phase_grad_window,
            "min_conncomp_frac": min_conncomp_frac,
            "single_tile_reoptimize": single_tile_reoptimize,
            "regrow_conncomps": regrow_conncomps,
            "delete_scratch": delete_scratch,
        }

        # Add scratchdir if specified
        if scratchdir is not None:
            unwrap_params["scratchdir"] = scratchdir

        # Add tiling parameters if specified
        if ntiles is not None:
            unwrap_params["ntiles"] = ntiles
            unwrap_params["tile_overlap"] = tile_overlap
            logger.info(f"Using tiling: ntiles={ntiles}, overlap={tile_overlap} pixels")

        # Log unwrapping parameters
        logger.info(f"Unwrapping Parameters: \n {unwrap_params}")

        # Execute unwrapping
        snaphu.unwrap(**unwrap_params)

        create_meta_file(unw_path, width, height, "FLOAT")
        create_meta_file(conncomp_path, width, height, "BYTE")

        # Verify output was created
        if not Path(unw_path).exists():
            result["message"] = "Unwrapping completed but output file not found"
            result["error"] = result["message"]
            logger.error(result["message"])
            return result

        # # Check output file size
        # output_size = Path(unw_path).stat().st_size
        # output_size_mb = output_size / (1024 * 1024)  # Convert to MB
        # if output_size_mb < 1.0:  # Less than 1MB
        #     result['message'] = f'Output file too small ({output_size_mb:.3f} MB < 1.0 MB)'
        #     result['error'] = result['message']
        #     logger.warning(result['message'])
        #     return result

        # Success
        result["success"] = True
        result["message"] = "Unwrapping completed successfully"
        result["output"] = unw_path
        logger.info(f"Successfully unwrapped: {unw_path} ({output_size_mb:.2f} MB)")

    except Exception as e:
        result["message"] = f"Exception during unwrapping: {str(e)}"
        result["error"] = str(e)
        logger.error(f"Unwrapping failed for {config_path}: {e}", exc_info=True)

    return result


def batch_unwrap(
    config_paths: list[str],
    nproc: int = 1,
    ntiles: tuple[int, int] | None = None,
    tile_overlap: int = 0,
    skip_existing: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, int]:
    """Execute batch unwrapping for multiple configuration files.

    Parameters
    ----------
    config_paths : list[str]
        List of configuration file paths.
    nproc : int, default=1
        Number of parallel processing cores per task.
    ntiles : tuple[int, int] | None, optional
        Tile grid dimensions (rows, cols).
    tile_overlap : int, default=0
        Tile overlap in pixels.
    skip_existing : bool, default=True
        Whether to skip tasks with existing output files.
    logger : logging.Logger | None, optional
        Logger instance.

    Returns
    -------
    dict[str, int]
        Dictionary with execution statistics containing:
        - total: Total number of tasks
        - skipped: Number of skipped tasks
        - completed: Number of successfully completed tasks
        - failed: Number of failed tasks

    Notes
    -----
    This function processes tasks sequentially (no parallel execution at task level).
    Parallelism is achieved within each task via the nproc parameter.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    stats = {"total": len(config_paths), "skipped": 0, "completed": 0, "failed": 0}

    for i, config_path in enumerate(config_paths, 1):
        config_name = Path(config_path).name

        # Check if output should be skipped
        if skip_existing:
            config_info = parse_config(config_path)
            if config_info:
                unw_path = config_info.get("unw", "")
                if unw_path and isinstance(unw_path, str) and Path(unw_path).exists():
                    file_size = Path(unw_path).stat().st_size
                    file_size_mb = file_size / (1024 * 1024)  # Convert to MB

                    if file_size_mb >= 1.0:  # At least 1MB
                        stats["skipped"] += 1
                        logger.info(
                            f"[{i}/{stats['total']}] SKIP: {config_name} "
                            f"(output exists, {file_size_mb:.2f} MB)"
                        )
                        continue
                    else:
                        # File is too small, delete and regenerate
                        logger.warning(
                            f"[{i}/{stats['total']}] DELETE: {config_name} - file too small "
                            f"({file_size_mb:.3f} MB < 1.0 MB), will regenerate"
                        )
                        try:
                            Path(unw_path).unlink()
                            # Also delete corresponding conncomp file if exists
                            conncomp_path = str(unw_path) + ".conncomp"
                            if Path(conncomp_path).exists():
                                Path(conncomp_path).unlink()
                                logger.info(
                                    f"Deleted associated conncomp file: {conncomp_path}"
                                )
                        except Exception as e:
                            logger.error(f"Failed to delete {unw_path}: {e}")

        # Execute unwrapping
        logger.info(f"[{i}/{stats['total']}] Processing: {config_name}")
        result = unwrap_from_config(
            config_path=config_path,
            nproc=nproc,
            ntiles=ntiles,
            tile_overlap=tile_overlap,
            logger=logger,
        )

        if result["success"]:
            stats["completed"] += 1
            logger.info(f"[{i}/{stats['total']}] SUCCESS: {config_name}")
        else:
            stats["failed"] += 1
            logger.error(
                f"[{i}/{stats['total']}] FAILED: {config_name} - {result['message']}"
            )

    return stats
