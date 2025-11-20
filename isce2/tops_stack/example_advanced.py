#!/usr/bin/env python3
"""Example: Advanced usage with custom filtering and analysis."""

import sys
sys.path.append('/DATA/DATA5/fancy/github/isce2/contrib/stack/topsStack')

from geocode_topsstack import GeocodeTopsStack
from pathlib import Path

# Configuration
STACK_WORKSPACE = '/home/fancy/workspace/DATA4/Data/UCM/stack_1_4'
DEM_FILE = '/path/to/your/dem'  # Change this to your actual DEM path
RANGE_LOOKS = 10
AZIMUTH_LOOKS = 2
BBOX = [36.0, 38.0, -121.0, -119.0]  # Custom bounding box

# Initialize geocoder
geocoder = GeocodeTopsStack(
    stack_workspace=STACK_WORKSPACE,
    dem_file=DEM_FILE,
    range_looks=RANGE_LOOKS,
    azimuth_looks=AZIMUTH_LOOKS,
    bbox=BBOX,
)

# Example 1: Find and analyze available data
print("="*60)
print("Example 1: Data exploration")
print("="*60)

# Get all pairs
all_pairs = geocoder.find_interferogram_pairs()
print(f"Total interferogram pairs: {len(all_pairs)}")

# Find files for all pairs
files_dict = geocoder.find_files_to_geocode(
    file_patterns=['filt_fine.unw', 'filt_fine.cor', 'fine.int']
)

# Analyze availability
for pair, files in list(files_dict.items())[:5]:
    print(f"\n{pair}:")
    for f in files:
        print(f"  - {Path(f).name}")

# Example 2: Filter pairs by temporal baseline
print("\n" + "="*60)
print("Example 2: Filter by temporal baseline")
print("="*60)

from datetime import datetime

def get_temporal_baseline(pair_name):
    """Calculate temporal baseline in days."""
    master, slave = pair_name.split('_')
    master_date = datetime.strptime(master, '%Y%m%d')
    slave_date = datetime.strptime(slave, '%Y%m%d')
    return abs((slave_date - master_date).days)

# Filter pairs with temporal baseline <= 36 days
short_baseline_pairs = [
    pair for pair in all_pairs
    if get_temporal_baseline(pair) <= 36
]

print(f"Pairs with baseline ≤ 36 days: {len(short_baseline_pairs)}")
print(f"Sample pairs: {short_baseline_pairs[:5]}")

# Geocode short baseline pairs
# geocoder.geocode_all(
#     pair_list=short_baseline_pairs,
#     file_patterns=['filt_fine.unw']
# )

# Example 3: Filter pairs by specific slave date
print("\n" + "="*60)
print("Example 3: Filter by slave date")
print("="*60)

target_slave = '20220115'
pairs_with_slave = [
    pair for pair in all_pairs
    if pair.split('_')[1] == target_slave
]

print(f"Pairs with slave date {target_slave}: {len(pairs_with_slave)}")
print(f"Pairs: {pairs_with_slave}")

# Example 4: Find pairs within date range
print("\n" + "="*60)
print("Example 4: Filter by date range")
print("="*60)

start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 3, 31)

def is_in_date_range(pair_name):
    """Check if pair is within date range."""
    master, slave = pair_name.split('_')
    master_date = datetime.strptime(master, '%Y%m%d')
    slave_date = datetime.strptime(slave, '%Y%m%d')
    return start_date <= master_date <= end_date and start_date <= slave_date <= end_date

filtered_pairs = [pair for pair in all_pairs if is_in_date_range(pair)]
print(f"Pairs in Q1 2022: {len(filtered_pairs)}")

# Example 5: Check file availability before geocoding
print("\n" + "="*60)
print("Example 5: Pre-check file availability")
print("="*60)

files_to_check = geocoder.find_files_to_geocode(
    pair_list=filtered_pairs[:5],
    file_patterns=['filt_fine.unw', 'filt_fine.cor']
)

# Report availability
for pair, files in files_to_check.items():
    has_unw = any('unw' in f for f in files)
    has_cor = any('cor' in f for f in files)
    print(f"{pair}: unw={has_unw}, cor={has_cor}")

# Example 6: Custom file patterns with glob
print("\n" + "="*60)
print("Example 6: Advanced file pattern matching")
print("="*60)

# Match all filtered files
files_dict = geocoder.find_files_to_geocode(
    pair_list=['20220103_20220115'],
    file_patterns=['filt_*.unw', 'filt_*.cor', 'fine.int']
)

for pair, files in files_dict.items():
    print(f"\n{pair}:")
    for f in files:
        print(f"  - {Path(f).name}")

print("\n✓ Advanced examples completed!")
