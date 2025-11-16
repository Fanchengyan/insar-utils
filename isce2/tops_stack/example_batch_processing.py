#!/usr/bin/env python3
"""Example: Batch geocoding of multiple interferogram pairs."""

import sys
sys.path.append('/DATA/DATA5/fancy/github/isce2/contrib/stack/topsStack')

from geocode_topsstack import GeocodeTopsStack

# Configuration
STACK_WORKSPACE = '/home/fancy/workspace/DATA4/Data/UCM/stack_1_4'
DEM_FILE = '/path/to/your/dem'  # Change this to your actual DEM path
RANGE_LOOKS = 10
AZIMUTH_LOOKS = 2
BBOX = None  # Or specify [South, North, West, East]

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

# Example 1: Geocode all interferogram pairs with default file pattern
print("\n" + "="*60)
print("Example 1: Geocode all pairs (filt_fine.unw only)")
print("="*60)
# This will process all interferogram pairs and geocode filt_fine.unw
# geocoder.geocode_all()

# Example 2: Geocode all pairs with multiple file patterns
print("\n" + "="*60)
print("Example 2: Geocode all pairs (multiple files)")
print("="*60)
geocoder.geocode_all(
    file_patterns=['filt_fine.unw', 'filt_fine.cor']
)

# Example 3: Geocode specific pairs using date pattern
print("\n" + "="*60)
print("Example 3: Geocode pairs with master date 20220103")
print("="*60)
# Find all pairs with master date 20220103
pairs = geocoder.find_interferogram_pairs(date_pattern='20220103_*')
print(f"Found {len(pairs)} pairs: {pairs[:5]}...")

geocoder.geocode_all(
    pair_list=pairs,
    file_patterns=['filt_fine.unw']
)

# Example 4: Geocode specific subset of pairs
print("\n" + "="*60)
print("Example 4: Geocode manually selected pairs")
print("="*60)
selected_pairs = [
    '20220103_20220115',
    '20220103_20220127',
    '20220103_20220208',
]

geocoder.geocode_all(
    pair_list=selected_pairs,
    file_patterns=['filt_fine.unw', 'filt_fine.cor']
)

print("\n✓ All batch processing completed!")
