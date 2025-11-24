#!/usr/bin/env python3
"""Hybrid Parallel SNAPHU Unwrapping Script.

This script combines two levels of parallelism:
1. Outer level: ProcessPoolExecutor for processing multiple interferograms simultaneously
2. Inner level: snaphu's nproc parameter for tile-based parallel processing within each task

This provides optimal performance for large batches of interferograms.

Usage
-----
    # Manual resource allocation
    python parallel_unwrap_hybrid.py -f unw_cmd -n 2 --nproc 8 --ntiles 4 4 --tile_overlap 100
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


from tqdm import tqdm


import snaphu_unwrap


def setup_logging(log_file: str | None = None) -> logging.Logger:
    """Configure logging to output to both console and file.
    
    Parameters
    ----------
    log_file : str | None, optional
        Path to log file. If None, only console output is used.
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger('snaphu_unwrap_hybrid')
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_tasks_from_cmd_file(cmd_file: str, logger: logging.Logger) -> list[str]:
    """Load task list from command file.
    
    Parameters
    ----------
    cmd_file : str
        Path to command file (e.g., unw_cmd).
    logger : logging.Logger
        Logger instance.
        
    Returns
    -------
    list[str]
        List of config file paths from the command file.
        
    Notes
    -----
    Expected command format: SentinelWrapper.py -c /path/to/config_file
    Lines starting with '#' are treated as comments and ignored.
    """
    tasks = []
    
    try:
        with open(cmd_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Extract config file path from command
                # Format: SentinelWrapper.py -c /path/to/config_file
                if '-c' in line:
                    parts = line.split('-c')
                    if len(parts) > 1:
                        config_path = parts[1].strip()
                        
                        if Path(config_path).exists():
                            tasks.append(config_path)
                        else:
                            logger.warning(f"Config file not found: {config_path}")
    
    except Exception as e:
        logger.error(f"Failed to load tasks from {cmd_file}: {e}")
        return []
    
    return tasks


def execute_task_wrapper(
    config_path: str,
    nproc: int,
    ntiles: tuple[int, int] | None,
    tile_overlap: int
) -> dict[str, str | bool]:
    """Wrapper function for executing unwrap task in parallel.
    
    This function is designed to be called by ProcessPoolExecutor.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file.
    nproc : int
        Number of cores for parallel processing within this task.
    ntiles : tuple[int, int] | None
        Tile grid dimensions (rows, cols).
    tile_overlap : int
        Tile overlap in pixels.
        
    Returns
    -------
    dict[str, str | bool]
        Dictionary with execution results.
        
    Notes
    -----
    This function creates its own logger to avoid issues with
    multiprocessing and shared logger instances.
    """
    # Create a simple logger for this worker
    logger = logging.getLogger(f'worker_{os.getpid()}')
    logger.setLevel(logging.INFO)
    
    try:
        result = snaphu_unwrap.unwrap_from_config(
            config_path=config_path,
            nproc=nproc,
            ntiles=ntiles,
            tile_overlap=tile_overlap,
            logger=logger
        )
        return result
    except Exception as e:
        return {
            'config': config_path,
            'success': False,
            'message': f'Exception: {str(e)}',
            'output': '',
            'error': str(e)
        }


def calculate_resource_allocation(
    total_cores: int,
    num_tasks: int,
    outer_workers: int | None = None,
    nproc: int | None = None
) -> tuple[int, int]:
    """Calculate optimal resource allocation for hybrid parallelism.
    
    Parameters
    ----------
    total_cores : int
        Total available CPU cores.
    num_tasks : int
        Total number of tasks to process.
    outer_workers : int | None, optional
        Desired number of outer workers. If None, auto-calculated.
    nproc : int | None, optional
        Desired nproc per task. If None, auto-calculated.
        
    Returns
    -------
    tuple[int, int]
        (outer_workers, nproc) - Optimized resource allocation.
        
    Notes
    -----
    Strategy:
    - If both specified: validate and warn if over-allocated
    - If only outer_workers specified: calculate nproc
    - If only nproc specified: calculate outer_workers
    - If neither specified: auto-calculate based on num_tasks
    
    Auto-calculation heuristics:
    - Few tasks (< 4): Maximize nproc, minimize outer_workers
    - Many tasks (> 8): Balance outer_workers and nproc
    - Medium tasks (4-8): Slight preference for outer parallelism
    """
    # Both specified - validate only
    if outer_workers is not None and nproc is not None:
        return outer_workers, nproc
    
    # Only outer_workers specified
    if outer_workers is not None:
        nproc = max(1, total_cores // outer_workers)
        return outer_workers, nproc
    
    # Only nproc specified
    if nproc is not None:
        outer_workers = max(1, min(num_tasks, total_cores // nproc))
        return outer_workers, nproc
    
    # Auto-calculate both
    if num_tasks <= 3:
        # Few tasks: maximize nproc
        outer_workers = min(2, num_tasks)
        nproc = max(1, total_cores // outer_workers)
    elif num_tasks <= 8:
        # Medium tasks: balance
        outer_workers = min(4, num_tasks, total_cores // 2)
        nproc = max(1, total_cores // outer_workers)
    else:
        # Many tasks: prefer outer parallelism
        # Try to use 4-8 cores per task
        target_nproc = min(8, max(4, total_cores // 4))
        outer_workers = max(1, min(num_tasks, total_cores // target_nproc))
        nproc = max(1, total_cores // outer_workers)
    
    return outer_workers, nproc


def run_hybrid_parallel(
    tasks: list[str],
    outer_workers: int,
    nproc: int,
    ntiles: tuple[int, int] | None = None,
    tile_overlap: int = 50,
    skip_existing: bool = True,
    logger: logging.Logger | None = None
) -> dict[str, int]:
    """Execute tasks using hybrid parallelism.
    
    Parameters
    ----------
    tasks : list[str]
        List of config file paths.
    outer_workers : int
        Number of parallel workers (outer parallelism).
    nproc : int
        Number of cores per task (inner parallelism).
    ntiles : tuple[int, int] | None, optional
        Tile grid dimensions (rows, cols).
    tile_overlap : int, default=50
        Tile overlap in pixels.
    skip_existing : bool, default=True
        Whether to skip tasks with existing output.
    logger : logging.Logger | None, optional
        Logger instance. If None, uses default logger.
        
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
    This function implements two-level parallelism:
    - Outer: ProcessPoolExecutor with 'outer_workers' workers
    - Inner: Each worker uses 'nproc' cores via snaphu.unwrap()
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    stats = {
        'total': len(tasks),
        'skipped': 0,
        'completed': 0,
        'failed': 0
    }
    
    # Filter tasks based on existing output
    tasks_to_run = []
    for task in tasks:
        if skip_existing:
            config_info = snaphu_unwrap.parse_config(task)
            if config_info:
                unw_path = config_info.get('unw', '')
                if unw_path and Path(unw_path).exists():
                    file_size = Path(unw_path).stat().st_size
                    if file_size >= 1024:  # At least 1KB
                        stats['skipped'] += 1
                        logger.info(
                            f"SKIP: {Path(task).name} "
                            f"(output exists, {file_size} bytes)"
                        )
                        continue
        
        tasks_to_run.append(task)
    
    # Log execution plan
    total_cores_used = outer_workers * nproc
    available_cores = os.cpu_count() or 1
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Total tasks: {stats['total']}")
    logger.info(f"Skipped (output exists): {stats['skipped']}")
    logger.info(f"To execute: {len(tasks_to_run)}")
    logger.info(f"{'='*60}")
    logger.info(f"RESOURCE ALLOCATION:")
    logger.info(f"  Outer workers:      {outer_workers}")
    logger.info(f"  Cores per task:     {nproc}")
    logger.info(f"  Total cores used:   {total_cores_used}")
    logger.info(f"  Available cores:    {available_cores}")
    if total_cores_used > available_cores:
        logger.warning(
            f"  ⚠️  Over-allocated by {total_cores_used - available_cores} cores!"
        )
    if ntiles:
        logger.info(f"  Tiling:             {ntiles[0]} × {ntiles[1]} "
                   f"(overlap: {tile_overlap} px)")
    logger.info(f"{'='*60}\n")
    
    if not tasks_to_run:
        logger.info("No tasks to execute. All outputs already exist.")
        return stats
    
    # Execute tasks in parallel
    with ProcessPoolExecutor(max_workers=outer_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                execute_task_wrapper,
                task,
                nproc,
                ntiles,
                tile_overlap
            ): task
            for task in tasks_to_run
        }
        
        # Setup progress bar if tqdm is available
        pbar = tqdm(
                total=len(tasks_to_run),
                desc="Processing",
                unit="task",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        
        # Process completed tasks
        for i, future in enumerate(as_completed(future_to_task), 1):
            task = future_to_task[future]
            task_name = Path(task).name
            
            try:
                result = future.result()
                
                if result['success']:
                    stats['completed'] += 1
                    logger.info(f"[{i}/{len(tasks_to_run)}] SUCCESS: {task_name}")
                    pbar.set_postfix_str(f"✓ {task_name[:40]}")
                else:
                    stats['failed'] += 1
                    logger.error(
                        f"[{i}/{len(tasks_to_run)}] FAILED: {task_name} - "
                        f"{result['message']}"
                    )
                    if result['error']:
                        logger.error(f"  Error: {result['error'][:200]}")
                    pbar.set_postfix_str(f"✗ {task_name[:40]}")
            
            except Exception as e:
                stats['failed'] += 1
                logger.error(
                    f"[{i}/{len(tasks_to_run)}] EXCEPTION: {task_name} - {str(e)}"
                )
                pbar.set_postfix_str(f"✗ {task_name[:40]}")
            
            # Update progress bar
            pbar.update(1)
        
        # Close progress bar
        pbar.close()
    
    return stats


def main() -> None:
    """Main function for hybrid parallel unwrapping script.
    
    Raises
    ------
    SystemExit
        Exits with code 1 if any tasks failed, otherwise exits with code 0.
    """
    parser = argparse.ArgumentParser(
        description='Hybrid parallel SNAPHU unwrapping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto resource allocation (recommended)
  python parallel_unwrap_hybrid.py -f unw_cmd -n 4 --nproc 4 --ntiles 4 4
  
  # Maximize outer parallelism (many small tasks)
  python parallel_unwrap_hybrid.py -f unw_cmd -n 8 --nproc 2
  
  # Maximize inner parallelism (few large tasks)
  python parallel_unwrap_hybrid.py -f unw_cmd -n 2 --nproc 8 --ntiles 4 4
  
  # Full auto mode (auto-calculate both -n and --nproc)
  python parallel_unwrap_hybrid.py -f unw_cmd --auto
  
  # With custom log file
  python parallel_unwrap_hybrid.py -f unw_cmd -n 4 --nproc 4 -o custom.log

Resource Allocation Strategy:
  Total cores used = outer_workers (-n) × nproc (--nproc)
  
  Recommendations:
  - 8 cores total:  -n 2 --nproc 4  or  -n 4 --nproc 2 --ntiles 4 4
  - 16 cores total: -n 4 --nproc 4  or  -n 8 --nproc 2
  - 32 cores total: -n 8 --nproc 4  or  -n 4 --nproc 8
  
  Use --auto to let the script decide based on number of tasks.
        """
    )
    
    # Required arguments
    parser.add_argument(
        '-f', '--cmd-file',
        type=str,
        required=True,
        help='Path to command file (e.g., unw_cmd)'
    )
    
    # Parallel processing arguments
    parser.add_argument(
        '-n', '--num-workers',
        type=int,
        default=None,
        help='Number of parallel workers (outer parallelism). default: auto'
    )
    parser.add_argument(
        '--nproc',
        type=int,
        default=None,
        help='Number of cores per task (inner parallelism). default: auto'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Auto-calculate both --num-workers and --nproc based on available cores and task count'
    )
    
    # Tiling arguments
    parser.add_argument(
        '--ntiles',
        type=int,
        nargs=2,
        default=(4, 4),
        metavar=('ROWS', 'COLS'),
        help='Tile grid dimensions (rows cols), e.g., --ntiles 2 3'
    )
    parser.add_argument(
        '--tile-overlap',
        type=int,
        default=50,
        help='Tile overlap in pixels (default: 50)'
    )
    
    # Other arguments
    parser.add_argument(
        '-o', '--log-file',
        type=str,
        default=None,
        help='Path to log file (optional, auto-generated if not specified)'
    )
    parser.add_argument(
        '-F', '--no-skip-existing',
        action='store_true',
        help='Do not skip tasks with existing output files'
    )
    
    args = parser.parse_args()
    
    # Auto-generate log file name if not specified
    if args.log_file is None:
        cmd_file_stem = Path(args.cmd_file).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.log_file = f"{cmd_file_stem}_hybrid_{timestamp}.log"
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    logger.info("="*60)
    logger.info("Hybrid Parallel SNAPHU Unwrapping Script")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {args.log_file}")
    logger.info("="*60)
    
    # Validate ntiles if provided
    ntiles = tuple(args.ntiles)
    if ntiles[0] < 1 or ntiles[1] < 1:
        logger.error("Invalid ntiles: both dimensions must be >= 1")
        sys.exit(1)

    # Load tasks
    logger.info(f"Loading tasks from command file: {args.cmd_file}")
    tasks = load_tasks_from_cmd_file(args.cmd_file, logger)
    
    if not tasks:
        logger.error("No tasks loaded. Exiting.")
        sys.exit(1)
    
    logger.info(f"Loaded {len(tasks)} tasks\n")
    
    # Calculate resource allocation
    total_cores = os.cpu_count() or 1
    
    if args.auto:
        # Full auto mode
        outer_workers, nproc = calculate_resource_allocation(
            total_cores=total_cores,
            num_tasks=len(tasks)
        )
        logger.info(f"AUTO mode: Calculated -n {outer_workers} --nproc {nproc}")
    else:
        # Use provided values or auto-calculate missing ones
        outer_workers, nproc = calculate_resource_allocation(
            total_cores=total_cores,
            num_tasks=len(tasks),
            outer_workers=args.num_workers,
            nproc=args.nproc
        )
        
        if args.num_workers is None:
            logger.info(f"Auto-calculated outer workers: {outer_workers}")
        if args.nproc is None:
            logger.info(f"Auto-calculated nproc: {nproc}")
    
    # Validate resource allocation
    total_cores_requested = outer_workers * nproc
    if total_cores_requested > total_cores * 1.5:
        logger.warning(
            f"Resource over-allocation detected: "
            f"{outer_workers} × {nproc} = {total_cores_requested} cores "
            f"(available: {total_cores})"
        )
        logger.warning("This may cause performance degradation due to over-subscription")
    
    # Run tasks with hybrid parallelism
    skip_existing = not args.no_skip_existing
    stats = run_hybrid_parallel(
        tasks=tasks,
        outer_workers=outer_workers,
        nproc=nproc,
        ntiles=ntiles,
        tile_overlap=args.tile_overlap,
        skip_existing=skip_existing,
        logger=logger
    )
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("EXECUTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total tasks:       {stats['total']}")
    logger.info(f"Skipped:           {stats['skipped']}")
    logger.info(f"Completed:         {stats['completed']}")
    logger.info(f"Failed:            {stats['failed']}")
    logger.info(f"End time:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file:          {args.log_file}")
    logger.info(f"{'='*60}")
    
    # Exit with appropriate code
    if stats['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
