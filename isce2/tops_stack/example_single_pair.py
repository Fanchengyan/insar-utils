#!/usr/bin/env python3
"""Example: Basic usage of GeocodeTopsStack for a single interferogram pair."""

import sys
sys.path.append('/DATA/DATA5/fancy/github/isce2/contrib/stack/topsStack')

from geocode_topsstack import GeocodeTopsStack

# Configuration
STACK_WORKSPACE = '/home/fancy/workspace/DATA4/Data/UCM/stack_1_4'
DEM_FILE = '/path/to/your/dem'  # Change this to your actual DEM path
RANGE_LOOKS = 10
AZIMUTH_LOOKS = 2
BBOX = None  # Or specify [South, North, West, East], e.g., [36.0, 38.0, -121.0, -119.0]

# Initialize geocoder
geocoder = GeocodeTopsStack(
    stack_workspace=STACK_WORKSPACE,
    dem_file=DEM_FILE,
    range_looks=RANGE_LOOKS,
    azimuth_looks=AZIMUTH_LOOKS,
    bbox=BBOX,
)

# Print configuration summary
geocoder.print_summary()

# Example 1: Geocode a single interferogram pair with one file
print("\n" + "="*60)
print("Example 1: Geocode single pair with single file")
print("="*60)
pair_name = '20220103_20220115'
file_path = f'{STACK_WORKSPACE}/merged/interferograms/{pair_name}/filt_fine.unw'

try:
    geocoder.geocode_pair(pair_name, [file_path])
    print(f"✓ Successfully geocoded {pair_name}")
except Exception as e:
    print(f"✗ Failed to geocode {pair_name}: {e}")

# Example 2: Geocode multiple files for one pair
print("\n" + "="*60)
print("Example 2: Geocode single pair with multiple files")
print("="*60)
pair_name = '20220103_20220127'
pair_dir = f'{STACK_WORKSPACE}/merged/interferograms/{pair_name}'
file_list = [
    f'{pair_dir}/filt_fine.unw',
    f'{pair_dir}/filt_fine.cor',
]

try:
    geocoder.geocode_pair(pair_name, file_list)
    print(f"✓ Successfully geocoded {pair_name}")
except Exception as e:
    print(f"✗ Failed to geocode {pair_name}: {e}")

print("\n✓ All done!")
