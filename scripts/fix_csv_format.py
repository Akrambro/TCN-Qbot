#!/usr/bin/env python3
"""
Fix CSV format - add headers and convert tabs to commas
"""

import sys

def fix_csv(input_file, output_file=None):
    if output_file is None:
        output_file = input_file
    
    print(f"Reading {input_file}...")
    
    # Read the file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Found {len(lines):,} rows")
    
    # Add header and convert tabs to commas
    with open(output_file, 'w') as f:
        # Write header
        f.write("timestamp,open,high,low,close,volume\n")
        
        # Process each line
        for line in lines:
            # Replace tabs with commas
            fixed_line = line.replace('\t', ',').strip()
            # Replace multiple spaces with comma
            import re
            fixed_line = re.sub(r'\s+', ',', fixed_line)
            f.write(fixed_line + '\n')
    
    print(f"✅ Fixed CSV saved to: {output_file}")
    print(f"✅ Added header: timestamp,open,high,low,close,volume")
    print(f"✅ Total rows: {len(lines):,}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/fix_csv_format.py <input_csv> [output_csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file
    
    fix_csv(input_file, output_file)
