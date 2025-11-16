#!/usr/bin/env python3
"""InSAR Stack Looks Variants Generator.

This module creates minimal InSAR stack variants for multilooking experiments
with different look values. It copies only the configuration files and run scripts
that use looks parameters, keeping all other files referenced from the source stack.

Examples
--------
Basic usage to create stack variants:

    $ python create_stack_variants.py

The script will:
1. Copy config files that use looks parameters (merge, filter, unwrap)
2. Copy corresponding run scripts (run_14, run_15, run_16)
3. Update looks parameters (alks, rlks, azimuth_looks, range_looks)
4. Update output paths to point to new stack
5. Keep input data references pointing to stack_1_4

Notes
-----
Files copied per variant:
- config_merge_igram_* (185 files) - multilooking干涉图
- config_igram_filt_coh_* (185 files) - 滤波和相干性
- config_igram_unw_* (185 files) - 解缠
- run_14_merge_burst_igram - 合并脚本
- run_15_filter_coherence - 滤波脚本
- run_16_unwrap - 解缠脚本
- run_pipeline.py - 流水线运行脚本

Total: ~555 small config files + 4 run scripts ≈ 2-5 MB per variant

Author: Generated for InSAR processing workflow
Date: 2025-11-12
"""

import logging
import re
import shutil
import sys
from pathlib import Path

from typing_extensions import TypeAlias

# Type aliases
StackName: TypeAlias = str
LooksValue: TypeAlias = int
VariantTuple: TypeAlias = tuple[LooksValue, LooksValue]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class StackVariantGenerator:
    """Generates minimal InSAR stack variants for multilooking experiments.

    This class creates ultra-lightweight stack copies containing only
    files that use looks parameters.

    Parameters
    ----------
    source_stack : str
        Source stack directory name (e.g., 'stack_1_4')
    base_dir : Path | None, optional
        Base directory containing stacks, by default None (uses current directory)

    Attributes
    ----------
    source_stack : str
        Name of the source stack directory
    base_dir : Path
        Base directory path containing stacks
    source_path : Path
        Full path to the source stack directory
    source_alks : int
        Source azimuth looks value
    source_rlks : int
        Source range looks value

    Raises
    ------
    ValueError
        If source_stack name format is invalid

    Examples
    --------
    Create a generator for stack_1_4:

        >>> generator = StackVariantGenerator('stack_1_4')
        >>> generator.create_variant(2, 8)
        PosixPath('/path/to/stack_2_8')

    Notes
    -----
    The source stack name must follow the pattern: stack_{alks}_{rlks}
    where alks and rlks are integer values.
    """

    # Config file patterns that use looks parameters
    CONFIG_PATTERNS: list[str] = [
        "config_merge_igram_*",  # Multilooking
        "config_igram_filt_coh_*",  # Filtering and coherence
        "config_igram_unw_*",  # Unwrapping
    ]

    # Run scripts that process with looks
    RUN_SCRIPTS: list[str] = [
        "run_14_merge_burst_igram",  # Merge and multilook
        "run_15_filter_coherence",  # Filter and coherence
        "run_16_unwrap",  # Unwrap
        "run_pipeline.py",  # Pipeline runner script
    ]

    def __init__(self, source_stack: StackName, base_dir: Path | None = None) -> None:
        """Initialize the StackVariantGenerator.

        Parameters
        ----------
        source_stack : str
            Source stack directory name (e.g., 'stack_1_4')
        base_dir : Path | None, optional
            Base directory containing stacks, by default None

        Raises
        ------
        ValueError
            If source_stack name format is invalid
        """
        self.source_stack: StackName = source_stack
        self.base_dir: Path = Path(base_dir) if base_dir else Path.cwd()
        self.source_path: Path = self.base_dir / source_stack

        # Extract source looks from stack name
        parts: list[str] = source_stack.split("_")
        if len(parts) >= 3:
            try:
                self.source_alks: LooksValue = int(parts[1])
                self.source_rlks: LooksValue = int(parts[2])
                logger.info(
                    f"Initialized generator for {source_stack} "
                    f"(alks={self.source_alks}, rlks={self.source_rlks})"
                )
            except ValueError as e:
                logger.error(f"Invalid looks values in stack name: {source_stack}")
                raise ValueError(
                    f"Invalid looks values in stack name format: {source_stack}"
                ) from e
        else:
            logger.error(f"Invalid stack name format: {source_stack}")
            raise ValueError(f"Invalid source stack name format: {source_stack}")

    def create_variant(
        self,
        target_alks: LooksValue,
        target_rlks: LooksValue,
        output_dir: Path | str | None = None,
        dry_run: bool = False,
    ) -> Path | None:
        """Create a minimal stack variant for multilooking experiments.

        Parameters
        ----------
        target_alks : int
            Target azimuth looks value
        target_rlks : int
            Target range looks value
        output_dir : Path | None, optional
            Output directory for the variant, by default None (uses self.base_dir)
        dry_run : bool, optional
            If True, only show what would be done without executing, by default False

        Returns
        -------
        Path | None
            Path to the created stack variant, or None if skipped

        Examples
        --------
        Create a new stack variant:

            >>> generator = StackVariantGenerator('stack_1_4')
            >>> new_stack = generator.create_variant(2, 8)
            >>> print(new_stack)
            PosixPath('/path/to/stack_2_8')

        Create variant in custom directory:

            >>> new_stack = generator.create_variant(2, 8, output_dir=Path('/custom/dir'))

        Notes
        -----
        Only files using looks parameters are copied. All input data remains
        in the source stack.
        """
        target_stack: StackName = f"stack_{target_alks}_{target_rlks}"
        base_output_dir: Path = Path(output_dir) if output_dir else self.base_dir
        target_path: Path = base_output_dir / target_stack

        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}Creating variant: {target_stack}"
        )
        logger.info(f"Source: {self.source_path}")
        logger.info(f"Target: {target_path}")
        logger.info(f"Looks: alks={target_alks}, rlks={target_rlks}")
        logger.info("Strategy: Copy only files using looks parameters")

        if target_path.exists():
            logger.warning(f"Target directory already exists: {target_path}")
            response: str = input(f"Overwrite {target_stack}? (y/n): ")
            if response.lower() not in ("y", "yes"):
                logger.info(f"User cancelled overwrite for {target_stack}")
                return None
            if not dry_run:
                logger.info(f"Removing existing directory: {target_path}")
                shutil.rmtree(target_path)

        if dry_run:
            logger.info("Would create directory structure")
            logger.info("Would copy looks-related config files")
            logger.info("Would copy run scripts (14, 15, 16)")
            logger.info("Would update looks parameters")
            return target_path

        # Create directory structure
        logger.info(f"Creating directory structure: {target_path}")
        configs_dir: Path = target_path / "configs"
        run_files_dir: Path = target_path / "run_files"
        merged_igrams: Path = target_path / "merged" / "interferograms"

        configs_dir.mkdir(parents=True, exist_ok=True)
        run_files_dir.mkdir(parents=True, exist_ok=True)
        merged_igrams.mkdir(parents=True, exist_ok=True)

        # Copy looks-related config files
        logger.info("Copying looks-related configuration files...")
        total_configs: int = self._copy_looks_configs(
            target_path, target_stack, target_alks, target_rlks
        )
        logger.info(f"Copied {total_configs} config files")

        # Copy run scripts
        logger.info("Copying run scripts...")
        scripts_copied: int = self._copy_run_scripts(target_path, target_stack)
        logger.info(f"Copied {scripts_copied} run scripts")

        logger.info(f"Successfully created {target_stack}")
        logger.info(f"Input data will be read from {self.source_stack}")
        logger.info(f"Processed outputs will be saved to {target_stack}/merged/")
        return target_path

    def _copy_looks_configs(
        self,
        target_path: Path,
        target_stack: StackName,
        target_alks: LooksValue,
        target_rlks: LooksValue,
    ) -> int:
        """Copy and update configuration files that use looks parameters.

        Parameters
        ----------
        target_path : Path
            Path to the target stack directory
        target_stack : str
            Target stack name
        target_alks : int
            Target azimuth looks value
        target_rlks : int
            Target range looks value

        Returns
        -------
        int
            Total number of configuration files copied

        Notes
        -----
        Copies and updates:
        - config_merge_igram_* : Updates azimuth_looks, range_looks, output paths
        - config_igram_filt_coh_* : Updates azimuth_looks, range_looks, output paths
        - config_igram_unw_* : Updates alks, rlks, output unw path only

        Path replacement strategy:
        - Input paths (ifg, input, reference, etc.) are resolved to real paths (symlinks resolved)
        - Output paths (unw, outfile, filt, coh) are updated to point to target directory
        - Two-step process: 1) Resolve symlinks, 2) Replace output paths
        - This ensures all paths work correctly even with symlinks and different output_dir
        """
        source_configs_dir: Path = self.source_path / "configs"
        target_configs_dir: Path = target_path / "configs"

        if not source_configs_dir.exists():
            logger.error(f"Source configs directory not found: {source_configs_dir}")
            return 0

        # Resolve paths to handle symlinks
        # Get real path for source (resolves symlinks)
        source_real_path: str = str(self.source_path.resolve())
        # Get target base path for output file replacement
        target_base: str = str(target_path.absolute())

        logger.debug(f"Source real path (resolved): {source_real_path}")
        logger.debug(f"Target base path: {target_base}")
        logger.debug(f"Source stack name: {self.source_stack}")

        total_copied: int = 0

        for pattern in self.CONFIG_PATTERNS:
            config_files: list[Path] = list(source_configs_dir.glob(pattern))
            if not config_files:
                logger.warning(f"No files found for pattern: {pattern}")
                continue

            logger.info(f"Processing {len(config_files)} files matching {pattern}")

            for i, source_file in enumerate(config_files, 1):
                if i % 50 == 0:
                    logger.debug(f"Progress: {i}/{len(config_files)} files")

                try:
                    # Read source file
                    with open(source_file, "r", encoding="utf-8") as f:
                        content: str = f.read()

                    # Update looks parameters
                    content = re.sub(
                        r"alks\s*:\s*\d+",
                        f"alks : {target_alks}",
                        content,
                    )
                    content = re.sub(
                        r"rlks\s*:\s*\d+",
                        f"rlks : {target_rlks}",
                        content,
                    )
                    content = re.sub(
                        r"azimuth_looks\s*:\s*\d+",
                        f"azimuth_looks : {target_alks}",
                        content,
                    )
                    content = re.sub(
                        r"range_looks\s*:\s*\d+",
                        f"range_looks : {target_rlks}",
                        content,
                    )

                    # Step 1: Resolve all paths containing source_stack to real paths
                    # This handles symlinks in source paths (both input and output)
                    # Match: /any/path/stack_1_4/
                    # Replace with: /real/source/path/
                    content = re.sub(
                        rf"[^\s]*/{re.escape(self.source_stack)}/",
                        f"{source_real_path}/",
                        content,
                    )

                    # Step 2: Update output paths for different file types
                    # Replace source_real_path with target_base for output fields only
                    if "config_igram_unw_" in source_file.name:
                        # For unwrap: only update unw: output path
                        content = re.sub(
                            rf"(unw\s*:\s*){re.escape(source_real_path)}",
                            rf"\1{target_base}",
                            content,
                        )
                    elif "config_merge_igram_" in source_file.name:
                        # For merge: update outfile path
                        content = re.sub(
                            rf"(outfile\s*:\s*){re.escape(source_real_path)}",
                            rf"\1{target_base}",
                            content,
                        )
                    elif "config_igram_filt_coh_" in source_file.name:
                        # For filter: update filt and coh output paths
                        content = re.sub(
                            rf"(filt\s*:\s*){re.escape(source_real_path)}",
                            rf"\1{target_base}",
                            content,
                        )
                        content = re.sub(
                            rf"(coh\s*:\s*){re.escape(source_real_path)}",
                            rf"\1{target_base}",
                            content,
                        )

                    # Write to target
                    target_file: Path = target_configs_dir / source_file.name
                    with open(target_file, "w", encoding="utf-8") as f:
                        f.write(content)

                    total_copied += 1

                except Exception as e:
                    logger.error(f"Error processing {source_file.name}: {e}")

        logger.info(f"Copied and updated {total_copied} config files")
        return total_copied

    def _copy_run_scripts(
        self,
        target_path: Path,
        target_stack: StackName,
    ) -> int:
        """Copy and update run scripts.

        Parameters
        ----------
        target_path : Path
            Path to the target stack directory
        target_stack : str
            Target stack name

        Returns
        -------
        int
            Number of run scripts copied

        Notes
        -----
        Copies run_14_merge_burst_igram, run_15_filter_coherence, run_16_unwrap, run_pipeline.py
        and updates config paths to point to new stack.

        Path replacement strategy:
        - Step 1: Resolve symlinks - all paths are converted to real paths
        - Step 2: Replace config paths to point to target directory
        - This ensures correct paths even when source has symlinks and output_dir differs
        """
        source_run_dir: Path = self.source_path / "run_files"
        target_run_dir: Path = target_path / "run_files"

        if not source_run_dir.exists():
            logger.error(f"Source run_files directory not found: {source_run_dir}")
            return 0

        # Resolve paths to handle symlinks
        source_real_path: str = str(self.source_path.resolve())
        target_base: str = str(target_path.absolute())

        copied_count: int = 0

        for script_name in self.RUN_SCRIPTS:
            source_script: Path = source_run_dir / script_name
            target_script: Path = target_run_dir / script_name

            if not source_script.exists():
                logger.warning(f"Run script not found: {script_name}")
                continue

            try:
                # Read source script
                with open(source_script, "r", encoding="utf-8") as f:
                    content: str = f.read()

                # Step 1: Resolve all paths containing source_stack to real paths
                # This handles symlinks
                content = re.sub(
                    rf"[^\s]*/{re.escape(self.source_stack)}/",
                    f"{source_real_path}/",
                    content,
                )

                # Step 2: Replace config paths to point to target
                # Match: /real/source/path/configs/...
                # Replace with: /target/base/path/configs/...
                content = re.sub(
                    rf"{re.escape(source_real_path)}/configs/",
                    f"{target_base}/configs/",
                    content,
                )

                # Write to target
                with open(target_script, "w", encoding="utf-8") as f:
                    f.write(content)

                copied_count += 1
                logger.info(f"Copied {script_name}")

            except Exception as e:
                logger.error(f"Error copying {script_name}: {e}")

        return copied_count

    def create_multiple_variants(
        self,
        variants: list[VariantTuple],
        output_dir: Path | str | None = None,
        dry_run: bool = False,
    ) -> list[Path]:
        """Create multiple minimal stack variants.

        Parameters
        ----------
        variants : list[tuple[int, int]]
            List of tuples (alks, rlks) for each variant to create
        output_dir : Path | None, optional
            Output directory for variants, by default None (uses self.base_dir)
        dry_run : bool, optional
            If True, only show what would be done without executing, by default False

        Returns
        -------
        list[Path]
            List of paths to created stack variants

        Examples
        --------
        Create multiple variants at once:

            >>> generator = StackVariantGenerator('stack_1_4')
            >>> variants = [(2, 8), (3, 12), (4, 16)]
            >>> created = generator.create_multiple_variants(variants)
            >>> print(len(created))
            3
        """
        created_stacks: list[Path] = []

        logger.info("=" * 70)
        logger.info("InSAR Stack Multilooking Variants Generator")
        logger.info("=" * 70)
        logger.info(f"Source stack: {self.source_stack}")
        logger.info(f"Source looks: alks={self.source_alks}, rlks={self.source_rlks}")
        logger.info(f"Variants to create: {len(variants)}")
        logger.info("Strategy: Copy only files using looks parameters")
        logger.info("=" * 70)

        for alks, rlks in variants:
            stack_path: Path | None = self.create_variant(
                alks, rlks, output_dir=output_dir, dry_run=dry_run
            )
            if stack_path:
                created_stacks.append(stack_path)

        logger.info("=" * 70)
        logger.info("Summary")
        logger.info("=" * 70)
        logger.info(f"Successfully created {len(created_stacks)} stack variant(s)")
        for stack_path in created_stacks:
            logger.info(f"  - {stack_path.name}")
        logger.info("Files copied per variant:")
        logger.info("  config_merge_igram_* (merge with multilooking)")
        logger.info("  config_igram_filt_coh_* (filtering and coherence)")
        logger.info("  config_igram_unw_* (unwrapping)")
        logger.info("  run_14, run_15, run_16, run_pipeline.py (run scripts)")
        logger.info("All other files are referenced from source stack")
        logger.info("=" * 70)

        return created_stacks


def main() -> int:
    """Execute the main script workflow.

    Returns
    -------
    int
        Exit code (0 for success, 1 for error)

    Notes
    -----
    The workflow consists of:
    1. Initialize generator with source stack
    2. Display variants to be created
    3. Perform dry run preview
    4. Ask user for confirmation
    5. Create actual variants if confirmed

    Examples
    --------
    Run from command line:

        $ python create_stack_variants.py
    """
    # Configuration
    source_stack: StackName = "stack_1_4"

    # Define variants to create: (azimuth_looks, range_looks)
    variants: list[VariantTuple] = [
        (2, 8),  # stack_2_8
        (3, 12),  # stack_3_12
        (4, 16),
        (5, 20),
        (6, 24),
        (7, 28),
        (8, 32),
        (9, 36),
        (10, 40),
    ]

    # Initialize generator
    try:
        generator: StackVariantGenerator = StackVariantGenerator(source_stack)
    except ValueError as e:
        logger.error(f"Failed to initialize generator: {e}")
        return 1

    # Display information
    logger.info(
        "This script will create minimal stack variants for multilooking experiments:"
    )
    for alks, rlks in variants:
        logger.info(f"  - stack_{alks}_{rlks} (alks={alks}, rlks={rlks})")
    logger.info(f"Source: {source_stack}")
    logger.info(f"Base directory: {generator.base_dir}")
    logger.info("What will be copied:")
    logger.info("  config_merge_igram_* files")
    logger.info("  config_igram_filt_coh_* files")
    logger.info("  config_igram_unw_* files")
    logger.info("  run_14_merge_burst_igram")
    logger.info("  run_15_filter_coherence")
    logger.info("  run_16_unwrap")
    logger.info("  run_pipeline.py")
    logger.info("What will NOT be copied (references source):")
    logger.info("  - All other config files")
    logger.info("  - All other run files")
    logger.info("  - All data directories (interferograms, SLC, etc.)")
    logger.info("Disk usage: ~2-5 MB per variant (vs ~100 GB for full copy)")

    # Create variants
    generator.create_multiple_variants(
        variants, output_dir="/home/fancy/workspace/cryo_data/UCM", dry_run=False
    )
    logger.info("All operations completed successfully!")
    logger.info("To run processing for a variant:")
    logger.info("  cd stack_2_8")
    logger.info("  bash run_files/run_14_merge_burst_igram    # Merge with new looks")
    logger.info("  bash run_files/run_15_filter_coherence     # Filter with new looks")
    logger.info("  bash run_files/run_16_unwrap               # Unwrap with new looks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
