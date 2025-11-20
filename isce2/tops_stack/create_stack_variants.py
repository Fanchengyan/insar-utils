#!/usr/bin/env python3
"""InSAR Stack Looks Variants Generator.

This module creates minimal InSAR stack variants for multilooking experiments
with different look values. It copies only the configuration files and run scripts
that use looks parameters, keeping all other files referenced from the source stack.

Examples
--------
Basic usage to create stack variants:

    $ python create_stack_variants.py -s stack_1_4 -v 2,8 3,12 4,16

With custom base and output directories:

    $ python create_stack_variants.py -s stack_1_4 -b /path/to/stacks \\
        -o /output/path -v 2,8 3,12 4,16

Preview changes without executing (dry-run):

    $ python create_stack_variants.py -s stack_1_4 -v 2,8 3,12 --dry-run

Get help:

    $ python create_stack_variants.py -h

The script will:
1. Copy config files that use looks parameters (merge, filter, unwrap)
2. Copy corresponding run scripts (run_14, run_15, run_16)
3. Update looks parameters (alks, rlks, azimuth_looks, range_looks)
4. Update output paths to point to new stack
5. Keep input data references pointing to source stack
"""

import argparse
import json
import logging
import re
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

    def __init__(
        self,
        source_stack: StackName,
        base_dir: Path | None = None,
        resolve_symlinks: bool = False,
    ) -> None:
        """Initialize the StackVariantGenerator.

        Parameters
        ----------
        source_stack : str
            Source stack directory name (e.g., 'stack_1_4')
        base_dir : Path | None, optional
            Base directory containing stacks, by default None
        resolve_symlinks : bool, optional
            If True, resolve symbolic links to real paths; if False, preserve symlink paths, by default False

        Raises
        ------
        ValueError
            If source_stack name format is invalid
        """
        self.source_stack: StackName = source_stack
        self.base_dir: Path = Path(base_dir) if base_dir else Path.cwd()
        self.source_path: Path = self.base_dir / source_stack
        self.resolve_symlinks: bool = resolve_symlinks

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
            logger.info(f"Target directory already exists: {target_path} - will overwrite files")

        if dry_run:
            logger.info("Would create directory structure")
            logger.info("Would copy looks-related config files")
            logger.info("Would copy run scripts (14, 15, 16)")
            logger.info("Would update looks parameters")
            return target_path

        # Create directory structure (will not fail if exists due to exist_ok=True)
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

        # Handle path resolution based on resolve_symlinks flag
        # Get path for source (resolve symlinks or preserve them)
        if self.resolve_symlinks:
            source_abs_path: str = str(self.source_path.resolve())
        else:
            source_abs_path: str = str(self.source_path.absolute())

        # Get target base path for output file replacement
        target_base: str = str(target_path.absolute())

        logger.debug(f"Source path ({'resolved' if self.resolve_symlinks else 'preserved'}): {source_abs_path}")
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

                    # Step 1: Update all paths containing source_stack
                    # This handles path resolution based on resolve_symlinks setting
                    # Match: /any/path/stack_1_4/
                    # Replace with: /source/path/ (resolved if flag=True, absolute if flag=False)
                    content = re.sub(
                        rf"[^\s]*/{re.escape(self.source_stack)}/",
                        f"{source_abs_path}/",
                        content,
                    )

                    # Step 2: Update output paths for different file types
                    # Replace source_abs_path with target_base for output fields only
                    if "config_igram_unw_" in source_file.name:
                        # For unwrap: only update unw: output path
                        content = re.sub(
                            rf"(unw\s*:\s*){re.escape(source_abs_path)}",
                            rf"\1{target_base}",
                            content,
                        )
                    elif "config_merge_igram_" in source_file.name:
                        # For merge: update outfile path
                        content = re.sub(
                            rf"(outfile\s*:\s*){re.escape(source_abs_path)}",
                            rf"\1{target_base}",
                            content,
                        )
                    elif "config_igram_filt_coh_" in source_file.name:
                        # For filter: update filt and coh output paths
                        content = re.sub(
                            rf"(filt\s*:\s*){re.escape(source_abs_path)}",
                            rf"\1{target_base}",
                            content,
                        )
                        content = re.sub(
                            rf"(coh\s*:\s*){re.escape(source_abs_path)}",
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

        # Handle path resolution based on resolve_symlinks flag
        if self.resolve_symlinks:
            source_abs_path: str = str(self.source_path.resolve())
        else:
            source_abs_path: str = str(self.source_path.absolute())

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

                # Step 1: Update all paths containing source_stack
                # This handles path resolution based on resolve_symlinks setting
                content = re.sub(
                    rf"[^\s]*/{re.escape(self.source_stack)}/",
                    f"{source_abs_path}/",
                    content,
                )

                # Step 2: Replace config paths to point to target
                # Match: /source/path/configs/...
                # Replace with: /target/base/path/configs/...
                content = re.sub(
                    rf"{re.escape(source_abs_path)}/configs/",
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


def parse_variants(variant_strings: list[str]) -> list[VariantTuple]:
    """Parse variant strings into tuples.

    Parameters
    ----------
    variant_strings : list[str]
        List of variant strings like ['2,8', '3,12', '4,16']

    Returns
    -------
    list[tuple[int, int]]
        List of (alks, rlks) tuples

    Raises
    ------
    ValueError
        If variant format is invalid

    Examples
    --------
    >>> parse_variants(['2,8', '3,12'])
    [(2, 8), (3, 12)]
    """
    variants: list[VariantTuple] = []
    for v in variant_strings:
        try:
            parts = v.split(',')
            if len(parts) != 2:
                raise ValueError(f"Expected format 'alks,rlks', got: {v}")
            alks, rlks = int(parts[0]), int(parts[1])
            variants.append((alks, rlks))
        except (ValueError, AttributeError) as e:
            raise ValueError(
                f"Invalid variant format: {v}. Expected: alks,rlks (e.g., 2,8)"
            ) from e
    return variants


def load_config_file(config_path: Path) -> dict:
    """Load variants from a JSON configuration file.

    Parameters
    ----------
    config_path : Path
        Path to JSON configuration file

    Returns
    -------
    dict
        Configuration dictionary with 'variants', 'source_stack', etc.

    Examples
    --------
    Config file format (variants.json):
    {
        "source_stack": "stack_1_4",
        "base_dir": "/path/to/stacks",
        "output_dir": "/output/path",
        "variants": [
            [2, 8],
            [3, 12],
            [4, 16]
        ]
    }
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from: {config_path}")
        return config
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Failed to load config file: {e}")
        raise


def main() -> int:
    """Command line interface for creating InSAR stack variants.

    Returns
    -------
    int
        Exit code (0 for success, 1 for error)

    Examples
    --------
    Run from command line:

        $ python create_stack_variants.py -s stack_1_4 -v 2,8 3,12 4,16
        $ python create_stack_variants.py -s stack_1_4 -v 2,8 3,12 --dry-run
        $ python create_stack_variants.py -c config.json
    """
    parser = argparse.ArgumentParser(
        description='Create minimal InSAR stack variants for multilooking experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create variants with default settings
  %(prog)s -s stack_1_4 -v 2,8 3,12 4,16

  # Specify custom directories
  %(prog)s -s stack_1_4 -b /path/to/stacks -o /output/path -v 2,8 3,12

  # Preview without executing
  %(prog)s -s stack_1_4 -v 2,8 3,12 --dry-run

  # Use configuration file
  %(prog)s -c variants.json

  # With custom log level
  %(prog)s -s stack_1_4 -v 2,8 3,12 --log-level DEBUG

Variant Format:
  Each variant is specified as 'alks,rlks' where:
    alks = azimuth looks (integer)
    rlks = range looks (integer)
  Example: 2,8 creates stack_2_8 with alks=2, rlks=8

Config File Format (JSON):
  {
    "source_stack": "stack_1_4",
    "base_dir": "/path/to/stacks",
    "output_dir": "/output/path",
    "variants": [[2, 8], [3, 12], [4, 16]]
  }

What Gets Copied:
  - config_merge_igram_* files (merge with multilooking)
  - config_igram_filt_coh_* files (filtering and coherence)
  - config_igram_unw_* files (unwrapping)
  - run_14, run_15, run_16, run_pipeline.py (run scripts)

  All other files remain in the source stack (~2-5 MB vs ~100 GB)
        """
    )

    # Source stack configuration
    parser.add_argument(
        '-s', '--source-stack',
        type=str,
        help='Source stack directory name (e.g., stack_1_4). Required unless using -c/--config-file.'
    )

    parser.add_argument(
        '-b', '--base-dir',
        type=Path,
        help='Base directory containing stacks (default: current directory)'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        help='Output directory for variants (default: same as base-dir)'
    )

    # Variant specification
    parser.add_argument(
        '-v', '--variants',
        nargs='+',
        metavar='ALKS,RLKS',
        help='Variant specifications as alks,rlks pairs (e.g., 2,8 3,12 4,16). Required unless using -c/--config-file.'
    )

    # Configuration file
    parser.add_argument(
        '-c', '--config-file',
        type=Path,
        help='JSON config file with variant specifications (alternative to -s and -v)'
    )

    # Execution options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without executing'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--resolve-symlinks',
        action='store_true',
        help='Resolve symbolic links to real paths instead of preserving symlink paths'
    )

    args = parser.parse_args()

    # Configure logging level
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)

    # Parse configuration
    source_stack: str | None = None
    base_dir: Path | None = None
    output_dir: Path | None = None
    variants: list[VariantTuple] = []

    if args.config_file:
        # Load from config file
        try:
            config = load_config_file(args.config_file)
            source_stack = config.get('source_stack')
            base_dir = Path(config['base_dir']) if 'base_dir' in config else None
            output_dir = Path(config['output_dir']) if 'output_dir' in config else None

            # Parse variants from config
            variant_list = config.get('variants', [])
            for v in variant_list:
                if isinstance(v, list) and len(v) == 2:
                    variants.append((int(v[0]), int(v[1])))
                else:
                    logger.error(f"Invalid variant format in config: {v}")
                    return 1

        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
            return 1
    else:
        # Use command-line arguments
        if not args.source_stack:
            parser.error("--source-stack is required when not using --config-file")
        if not args.variants:
            parser.error("--variants is required when not using --config-file")

        source_stack = args.source_stack
        base_dir = args.base_dir
        output_dir = args.output_dir

        try:
            variants = parse_variants(args.variants)
        except ValueError as e:
            logger.error(f"Invalid variant specification: {e}")
            return 1

    # Validate required parameters
    if not source_stack:
        logger.error("source_stack not specified")
        return 1
    if not variants:
        logger.error("No variants specified")
        return 1

    # Initialize generator
    try:
        generator = StackVariantGenerator(
            source_stack,
            base_dir=base_dir,
            resolve_symlinks=args.resolve_symlinks
        )
    except ValueError as e:
        logger.error(f"Failed to initialize generator: {e}")
        return 1

    # Display information
    logger.info("=" * 70)
    logger.info("InSAR Stack Multilooking Variants Generator")
    logger.info("=" * 70)
    logger.info(f"Source stack: {source_stack}")
    logger.info(f"Base directory: {generator.base_dir}")
    logger.info(f"Output directory: {output_dir if output_dir else generator.base_dir}")
    logger.info(f"Resolve symlinks: {args.resolve_symlinks}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 70)
    logger.info("Variants to create:")
    for alks, rlks in variants:
        logger.info(f"  - stack_{alks}_{rlks} (alks={alks}, rlks={rlks})")
    logger.info("=" * 70)
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
    logger.info("=" * 70)

    # Create variants
    try:
        generator.create_multiple_variants(
            variants,
            output_dir=output_dir,
            dry_run=args.dry_run
        )
    except Exception as e:
        logger.error(f"Failed to create variants: {e}")
        return 1

    if not args.dry_run:
        logger.info("All operations completed successfully!")
        logger.info("To run processing for a variant:")
        logger.info(f"  cd stack_{variants[0][0]}_{variants[0][1]}")
        logger.info("  bash run_files/run_14_merge_burst_igram    # Merge with new looks")
        logger.info("  bash run_files/run_15_filter_coherence     # Filter with new looks")
        logger.info("  bash run_files/run_16_unwrap               # Unwrap with new looks")

    return 0


if __name__ == "__main__":
    sys.exit(main())
