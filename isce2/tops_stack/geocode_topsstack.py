#!/usr/bin/env python3
"""TopsStack interferogram geocoding automation class."""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal

import isce
import isceobj
from isceobj.Planet.Planet import Planet
from stdproc.rectify.geocode.Geocodable import Geocodable
from zerodop.geozero import createGeozero

# Import topsStack utilities
sys.path.append('/DATA/DATA5/fancy/github/isce2/contrib/stack/topsStack')
import s1a_isce_utils as ut
from baselineGrid import getMergedOrbit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GeocodeTopsStack:
    """
    TopsStack interferogram geocoding automation class.

    This class automates the geocoding process for topsStack interferograms by:
    1. Automatically parsing stack workspace structure
    2. Identifying master and slave dates from interferogram pair names
    3. Matching files in merged/interferograms directory
    4. Batch geocoding processing

    Parameters
    ----------
    stack_workspace : str
        Path to the topsStack workspace (e.g., 'stack_1_4/')
    dem_file : str
        Path to DEM file (without .xml extension)
    range_looks : int
        Number of range looks
    azimuth_looks : int
        Number of azimuth looks
    bbox : list[float] | None, optional
        Bounding box [South, North, West, East], by default None

    Attributes
    ----------
    stack_dir : Path
        Stack workspace directory
    reference_dir : Path
        Reference acquisition directory
    secondary_base_dir : Path
        Base directory for secondary acquisitions (coreg_secondarys)
    merged_dir : Path
        Merged interferograms directory
    dem_file : str
        DEM file path
    range_looks : int
        Number of range looks
    azimuth_looks : int
        Number of azimuth looks
    bbox : list[float] | None
        Bounding box

    Examples
    --------
    >>> geocoder = GeocodeTopsStack(
    ...     stack_workspace='stack_1_4',
    ...     dem_file='/path/to/dem',
    ...     range_looks=10,
    ...     azimuth_looks=2,
    ...     bbox=[36.0, 38.0, -121.0, -119.0]
    ... )
    >>> geocoder.geocode_pair('20220103_20220115', ['filt_fine.unw'])
    >>> geocoder.geocode_all(file_patterns=['filt_fine.unw', 'filt_fine.cor'])
    """

    def __init__(
        self,
        stack_workspace: str,
        dem_file: str,
        range_looks: int,
        azimuth_looks: int,
        bbox: list[float] | None = None,
    ) -> None:
        """
        Initialize GeocodeTopsStack.

        Parameters
        ----------
        stack_workspace : str
            Path to the topsStack workspace
        dem_file : str
            Path to DEM file (without .xml extension)
        range_looks : int
            Number of range looks
        azimuth_looks : int
            Number of azimuth looks
        bbox : list[float] | None, optional
            Bounding box [South, North, West, East], by default None

        Raises
        ------
        ValueError
            If workspace or DEM validation fails
        """
        self.stack_dir = Path(stack_workspace).resolve()
        self.reference_dir = self.stack_dir / "reference"
        self.secondary_base_dir = self.stack_dir / "coreg_secondarys"
        self.merged_dir = self.stack_dir / "merged" / "interferograms"
        self.dem_file = dem_file
        self.range_looks = range_looks
        self.azimuth_looks = azimuth_looks
        self.bbox = bbox

        # Validate inputs
        self._validate_workspace()
        self._validate_dem()

        logger.info(f"Initialized GeocodeTopsStack for workspace: {self.stack_dir}")

    def _validate_workspace(self) -> None:
        """
        Validate stack workspace structure.

        Raises
        ------
        ValueError
            If required directories are missing
        """
        if not self.stack_dir.exists():
            logger.error(f"Stack workspace not found: {self.stack_dir}")
            raise ValueError(f"Stack workspace not found: {self.stack_dir}")

        if not self.reference_dir.exists():
            logger.error(f"Reference directory not found: {self.reference_dir}")
            raise ValueError(f"Reference directory not found: {self.reference_dir}")

        if not self.secondary_base_dir.exists():
            logger.error(f"Secondary base directory not found: {self.secondary_base_dir}")
            raise ValueError(f"Secondary base directory not found: {self.secondary_base_dir}")

        if not self.merged_dir.exists():
            logger.error(f"Merged interferograms directory not found: {self.merged_dir}")
            raise ValueError(f"Merged interferograms directory not found: {self.merged_dir}")

        logger.info("Workspace structure validated successfully")

    def _validate_dem(self) -> None:
        """
        Validate DEM file existence.

        Raises
        ------
        ValueError
            If DEM file or its XML metadata is missing
        """
        dem_xml = f"{self.dem_file}.xml"
        if not os.path.exists(self.dem_file):
            logger.error(f"DEM file not found: {self.dem_file}")
            raise ValueError(f"DEM file not found: {self.dem_file}")

        if not os.path.exists(dem_xml):
            logger.error(f"DEM XML file not found: {dem_xml}")
            raise ValueError(f"DEM XML file not found: {dem_xml}")

        logger.info(f"DEM validated: {self.dem_file}")

    def parse_master_slave_dates(self, pair_name: str) -> tuple[str, str]:
        """
        Parse master and slave dates from interferogram pair name.

        Parameters
        ----------
        pair_name : str
            Interferogram pair name (e.g., '20220103_20220115')

        Returns
        -------
        tuple[str, str]
            Tuple of (master_date, slave_date)

        Raises
        ------
        ValueError
            If pair name format is invalid

        Examples
        --------
        >>> geocoder.parse_master_slave_dates('20220103_20220115')
        ('20220103', '20220115')
        """
        pattern = r'^(\d{8})_(\d{8})$'
        match = re.match(pattern, pair_name)
        if not match:
            logger.error(f"Invalid pair name format: {pair_name}")
            raise ValueError(f"Invalid pair name format: {pair_name}. Expected: YYYYMMDD_YYYYMMDD")

        master_date, slave_date = match.groups()
        logger.debug(f"Parsed {pair_name} -> master: {master_date}, slave: {slave_date}")
        return master_date, slave_date

    def find_interferogram_pairs(self, date_pattern: str | None = None) -> list[str]:
        """
        Find all interferogram pair directories in merged/interferograms.

        Parameters
        ----------
        date_pattern : str | None, optional
            Pattern to match specific pairs (e.g., '20220103_*' or '*_20220115').
            Supports glob-style wildcards. If None, returns all pairs.

        Returns
        -------
        list[str]
            Sorted list of interferogram pair names

        Examples
        --------
        >>> geocoder.find_interferogram_pairs()
        ['20220103_20220115', '20220103_20220127', ...]
        >>> geocoder.find_interferogram_pairs('20220103_*')
        ['20220103_20220115', '20220103_20220127']
        """
        if date_pattern:
            pairs = sorted([p.name for p in self.merged_dir.glob(date_pattern) if p.is_dir()])
        else:
            pairs = sorted([p.name for p in self.merged_dir.iterdir() if p.is_dir()])

        logger.info(f"Found {len(pairs)} interferogram pairs")
        return pairs

    def get_reference_dir(self, master_date: str) -> str:
        """
        Get reference directory path for master date.

        For topsStack, reference is typically in coreg_secondarys/{master_date}
        or in the reference directory if it's the stack reference.

        Parameters
        ----------
        master_date : str
            Master date (YYYYMMDD format)

        Returns
        -------
        str
            Path to reference directory

        Raises
        ------
        ValueError
            If reference directory not found
        """
        # First try coreg_secondarys
        ref_path = self.secondary_base_dir / master_date
        if ref_path.exists():
            logger.debug(f"Using reference from coreg_secondarys: {ref_path}")
            return str(ref_path)

        # Fallback to reference directory
        if self.reference_dir.exists():
            logger.debug(f"Using stack reference directory: {self.reference_dir}")
            return str(self.reference_dir)

        logger.error(f"Reference directory not found for master: {master_date}")
        raise ValueError(f"Reference directory not found for master: {master_date}")

    def get_secondary_dir(self, slave_date: str) -> str:
        """
        Get secondary directory path for slave date.

        Parameters
        ----------
        slave_date : str
            Slave date (YYYYMMDD format)

        Returns
        -------
        str
            Path to secondary directory

        Raises
        ------
        ValueError
            If secondary directory not found
        """
        sec_path = self.secondary_base_dir / slave_date
        if not sec_path.exists():
            logger.error(f"Secondary directory not found: {sec_path}")
            raise ValueError(f"Secondary directory not found: {sec_path}")

        logger.debug(f"Found secondary directory: {sec_path}")
        return str(sec_path)

    def find_files_to_geocode(
        self,
        pair_list: list[str] | None = None,
        file_patterns: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """
        Find all files to geocode for specified interferogram pairs.

        Parameters
        ----------
        pair_list : list[str] | None, optional
            List of interferogram pairs to process. If None, processes all pairs.
        file_patterns : list[str] | None, optional
            File patterns to match (e.g., ['filt_fine.unw', 'filt_fine.cor']).
            Supports glob patterns like 'filt_*.unw'.
            Default: ['filt_fine.unw']

        Returns
        -------
        dict[str, list[str]]
            Dictionary mapping pair names to list of absolute file paths

        Examples
        --------
        >>> files = geocoder.find_files_to_geocode(
        ...     pair_list=['20220103_20220115'],
        ...     file_patterns=['filt_fine.unw', 'filt_fine.cor']
        ... )
        {'20220103_20220115': ['/path/to/filt_fine.unw', '/path/to/filt_fine.cor']}
        """
        if pair_list is None:
            pair_list = self.find_interferogram_pairs()

        if file_patterns is None:
            file_patterns = ['filt_fine.unw']

        result = {}
        for pair_name in pair_list:
            pair_dir = self.merged_dir / pair_name
            if not pair_dir.exists():
                logger.warning(f"Pair directory not found: {pair_dir}")
                continue

            files = []
            for pattern in file_patterns:
                matched_files = list(pair_dir.glob(pattern))
                files.extend([str(f.resolve()) for f in matched_files if f.is_file()])

            if files:
                result[pair_name] = sorted(files)
                logger.info(f"Found {len(files)} files for pair {pair_name}")
            else:
                logger.warning(f"No matching files found for pair {pair_name}")

        return result

    def geocode_pair(
        self,
        pair_name: str,
        file_list: list[str],
        output_dir: str | None = None,
    ) -> None:
        """
        Geocode files for a single interferogram pair.

        Parameters
        ----------
        pair_name : str
            Interferogram pair name (e.g., '20220103_20220115')
        file_list : list[str]
            List of absolute file paths to geocode
        output_dir : str | None, optional
            Output directory for geocoded files. If None, saves in same directory
            as input files with .geo extension

        Raises
        ------
        ValueError
            If master/slave directories not found or geocoding fails

        Examples
        --------
        >>> geocoder.geocode_pair(
        ...     '20220103_20220115',
        ...     ['/path/to/filt_fine.unw']
        ... )
        """
        logger.info(f"Starting geocoding for pair: {pair_name}")

        # Parse dates
        master_date, slave_date = self.parse_master_slave_dates(pair_name)

        # Get reference and secondary directories
        try:
            reference_dir = self.get_reference_dir(master_date)
            secondary_dir = self.get_secondary_dir(slave_date)
        except ValueError as e:
            logger.error(f"Failed to get directories for {pair_name}: {e}")
            raise

        # Filter existing files
        existing_files = [f for f in file_list if os.path.exists(f)]
        if not existing_files:
            logger.warning(f"No existing files found for {pair_name}")
            return

        logger.info(f"Geocoding {len(existing_files)} files for {pair_name}")
        logger.info(f"Reference: {reference_dir}")
        logger.info(f"Secondary: {secondary_dir}")

        # Create mock inps object
        class MockInps:
            pass

        inps = MockInps()
        inps.reference = reference_dir
        inps.secondary = secondary_dir
        inps.numberRangeLooks = self.range_looks
        inps.numberAzimuthLooks = self.azimuth_looks

        # Run geocoding
        try:
            self._run_geocode(inps, existing_files, self.bbox, self.dem_file)
            logger.info(f"Successfully geocoded {pair_name}")
        except Exception as e:
            logger.error(f"Failed to geocode {pair_name}: {e}")
            raise

    def _run_geocode(
        self,
        inps,
        prodlist: list[str],
        bbox: list[float] | None,
        demfilename: str,
        is_offset_mode: bool = False,
    ) -> None:
        """
        Run geocoding process (adapted from geocodeIsce.py).

        Parameters
        ----------
        inps : object
            Input parameters object with reference, secondary, range/azimuth looks
        prodlist : list[str]
            List of file paths to geocode
        bbox : list[float] | None
            Bounding box [S, N, W, E]
        demfilename : str
            DEM file path
        is_offset_mode : bool, optional
            Whether in offset mode, by default False
        """
        logger.info("Geocoding images...")

        # Get swath lists
        referenceSwathList = ut.getSwathList(inps.reference)
        secondarySwathList = ut.getSwathList(inps.secondary)
        swathList = list(sorted(set(referenceSwathList + secondarySwathList)))

        # Load frames
        frames = []
        for swath in swathList:
            referenceProduct = ut.loadProduct(
                os.path.join(inps.secondary, f'IW{swath}.xml')
            )
            frames.append(referenceProduct)

        orb = getMergedOrbit(frames)

        # Calculate bounding box
        if bbox is None:
            bboxes = []
            for frame in frames:
                bboxes.append(frame.getBbox())

            snwe = [
                min([x[0] for x in bboxes]),
                max([x[1] for x in bboxes]),
                min([x[2] for x in bboxes]),
                max([x[3] for x in bboxes]),
            ]
        else:
            snwe = list(bbox)
            if len(snwe) != 4:
                logger.error("Bounding box must have 4 values [S, N, W, E]")
                raise ValueError('Bounding box should be a list/tuple of length 4')

        # Identify corners and dimensions
        topSwath = min(frames, key=lambda x: x.sensingStart)
        leftSwath = min(frames, key=lambda x: x.startingRange)

        # Get required values from product
        burst = frames[0].bursts[0]
        t0 = topSwath.sensingStart
        dtaz = burst.azimuthTimeInterval
        r0 = leftSwath.startingRange
        dr = burst.rangePixelSize
        wvl = burst.radarWavelength
        planet = Planet(pname='Earth')

        # Setup DEM
        demImage = isceobj.createDemImage()
        demImage.load(demfilename + '.xml')

        # Geocode each file
        ge = Geocodable()
        for prod in prodlist:
            objGeo = createGeozero()
            objGeo.configure()

            # Configure geocoding parameters
            objGeo.snwe = snwe
            objGeo.demImage = demImage
            objGeo.demCropFilename = os.path.join(os.path.dirname(demfilename), "dem.crop")

            if is_offset_mode:
                objGeo.numberRangeLooks = inps.skipwidth
                objGeo.numberAzimuthLooks = inps.skiphgt
            else:
                objGeo.numberRangeLooks = inps.numberRangeLooks
                objGeo.numberAzimuthLooks = inps.numberAzimuthLooks

            objGeo.lookSide = -1  # S1A is right looking only

            # Create input image and geocode method
            inImage, method = ge.create(prod)
            objGeo.method = method

            objGeo.slantRangePixelSpacing = dr
            objGeo.prf = 1.0 / dtaz
            objGeo.orbit = orb
            objGeo.width = inImage.getWidth()
            objGeo.length = inImage.getLength()
            objGeo.dopplerCentroidCoeffs = [0.0]
            objGeo.radarWavelength = wvl

            if is_offset_mode:
                objGeo.rangeFirstSample = r0 + (inps.offset_left - 1) * dr
                objGeo.setSensingStart(
                    t0 + datetime.timedelta(seconds=((inps.offset_top - 1) * dtaz))
                )
            else:
                objGeo.rangeFirstSample = r0 + ((inps.numberRangeLooks - 1) / 2.0) * dr
                objGeo.setSensingStart(
                    t0 + datetime.timedelta(seconds=(((inps.numberAzimuthLooks - 1) / 2.0) * dtaz))
                )

            objGeo.wireInputPort(name='dem', object=demImage)
            objGeo.wireInputPort(name='planet', object=planet)
            objGeo.wireInputPort(name='tobegeocoded', object=inImage)

            objGeo.geocode()

            logger.info(f"Geocoded: {inImage.filename} -> {inImage.filename}.geo")
            logger.info(f"  Width: {inImage.width}, Length: {inImage.length}")
            logger.info(f"  Range looks: {inps.numberRangeLooks}, Azimuth looks: {inps.numberAzimuthLooks}")
            logger.info(f"  Bbox - S: {objGeo.minimumGeoLatitude}, N: {objGeo.maximumGeoLatitude}")
            logger.info(f"        W: {objGeo.minimumGeoLongitude}, E: {objGeo.maximumGeoLongitude}")

    def geocode_all(
        self,
        pair_list: list[str] | None = None,
        file_patterns: list[str] | None = None,
        output_dir: str | None = None,
    ) -> None:
        """
        Batch geocode all specified interferograms.

        Parameters
        ----------
        pair_list : list[str] | None, optional
            List of pairs to process. If None, processes all pairs.
        file_patterns : list[str] | None, optional
            File patterns to geocode. Default: ['filt_fine.unw']
        output_dir : str | None, optional
            Output directory for geocoded files

        Examples
        --------
        >>> geocoder.geocode_all(
        ...     file_patterns=['filt_fine.unw', 'filt_fine.cor']
        ... )
        """
        logger.info("Starting batch geocoding...")

        # Find files to geocode
        files_dict = self.find_files_to_geocode(pair_list, file_patterns)

        if not files_dict:
            logger.warning("No files found to geocode")
            return

        total_pairs = len(files_dict)
        logger.info(f"Processing {total_pairs} interferogram pairs")

        # Process each pair
        success_count = 0
        fail_count = 0

        for idx, (pair_name, file_list) in enumerate(files_dict.items(), 1):
            logger.info(f"[{idx}/{total_pairs}] Processing {pair_name}")
            try:
                self.geocode_pair(pair_name, file_list, output_dir)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to process {pair_name}: {e}")
                fail_count += 1
                continue

        logger.info(f"Batch geocoding completed: {success_count} succeeded, {fail_count} failed")

    def summary(self) -> dict:
        """
        Return summary of available data and configuration.

        Returns
        -------
        dict
            Summary information including workspace paths, configuration,
            and available interferogram pairs

        Examples
        --------
        >>> info = geocoder.summary()
        >>> print(info['total_pairs'])
        186
        """
        pairs = self.find_interferogram_pairs()
        secondary_dates = sorted([d.name for d in self.secondary_base_dir.iterdir() if d.is_dir()])

        summary = {
            'stack_workspace': str(self.stack_dir),
            'reference_dir': str(self.reference_dir),
            'secondary_base_dir': str(self.secondary_base_dir),
            'merged_dir': str(self.merged_dir),
            'dem_file': self.dem_file,
            'range_looks': self.range_looks,
            'azimuth_looks': self.azimuth_looks,
            'bbox': self.bbox,
            'total_pairs': len(pairs),
            'total_secondary_dates': len(secondary_dates),
            'sample_pairs': pairs[:5] if len(pairs) > 5 else pairs,
            'secondary_dates': secondary_dates,
        }

        return summary

    def print_summary(self) -> None:
        """Print formatted summary of configuration and available data."""
        info = self.summary()

        print("\n" + "=" * 60)
        print("GeocodeTopsStack Configuration Summary")
        print("=" * 60)
        print(f"Stack Workspace    : {info['stack_workspace']}")
        print(f"DEM File           : {info['dem_file']}")
        print(f"Range Looks        : {info['range_looks']}")
        print(f"Azimuth Looks      : {info['azimuth_looks']}")
        print(f"Bounding Box       : {info['bbox']}")
        print(f"\nTotal Pairs        : {info['total_pairs']}")
        print(f"Secondary Dates    : {info['total_secondary_dates']}")
        if info['sample_pairs']:
            print(f"\nSample Pairs:")
            for pair in info['sample_pairs']:
                print(f"  - {pair}")
        print("=" * 60 + "\n")


def main():
    """Command line interface for GeocodeTopsStack."""
    parser = argparse.ArgumentParser(
        description='Batch geocode topsStack interferograms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Geocode all interferograms
  python geocode_topsstack.py -w stack_1_4 -d dem -r 10 -a 2

  # Geocode specific pairs
  python geocode_topsstack.py -w stack_1_4 -d dem -r 10 -a 2 \\
      -p "20220103_*"

  # Geocode multiple file types
  python geocode_topsstack.py -w stack_1_4 -d dem -r 10 -a 2 \\
      -f "filt_fine.unw" "filt_fine.cor"

  # With bounding box
  python geocode_topsstack.py -w stack_1_4 -d dem -r 10 -a 2 \\
      -b 36.0 38.0 -121.0 -119.0
        """,
    )

    parser.add_argument(
        '-w', '--workspace',
        required=True,
        help='Path to topsStack workspace directory',
    )
    parser.add_argument(
        '-d', '--dem',
        required=True,
        help='Path to DEM file (without .xml extension)',
    )
    parser.add_argument(
        '-r', '--range-looks',
        type=int,
        required=True,
        help='Number of range looks',
    )
    parser.add_argument(
        '-a', '--azimuth-looks',
        type=int,
        required=True,
        help='Number of azimuth looks',
    )
    parser.add_argument(
        '-b', '--bbox',
        nargs=4,
        type=float,
        metavar=('S', 'N', 'W', 'E'),
        help='Bounding box: South North West East',
    )
    parser.add_argument(
        '-p', '--pair-pattern',
        help='Pattern to match interferogram pairs (e.g., "20220103_*")',
    )
    parser.add_argument(
        '-f', '--file-patterns',
        nargs='+',
        default=['filt_fine.unw'],
        help='File patterns to geocode (default: filt_fine.unw)',
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Print summary and exit without geocoding',
    )

    args = parser.parse_args()

    # Initialize geocoder
    try:
        geocoder = GeocodeTopsStack(
            stack_workspace=args.workspace,
            dem_file=args.dem,
            range_looks=args.range_looks,
            azimuth_looks=args.azimuth_looks,
            bbox=args.bbox,
        )
    except ValueError as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)

    # Print summary
    geocoder.print_summary()

    if args.summary_only:
        sys.exit(0)

    # Find pairs to process
    if args.pair_pattern:
        pair_list = geocoder.find_interferogram_pairs(args.pair_pattern)
    else:
        pair_list = None

    # Run batch geocoding
    try:
        geocoder.geocode_all(
            pair_list=pair_list,
            file_patterns=args.file_patterns,
        )
    except Exception as e:
        logger.error(f"Batch geocoding failed: {e}")
        sys.exit(1)

    logger.info("All done!")


if __name__ == '__main__':
    main()
