#!/usr/bin/env python3
"""
JSON Array Splitter - Splits a large JSON array into smaller files
Usage: python split_json.py <input_file> [--chunk-size SIZE]
"""

import json
import argparse
import os
import sys
from pathlib import Path


def split_json(input_file, chunk_size=30):
    """
    Split a JSON array file into smaller chunks.
    
    Args:
        input_file: Path to the input JSON file
        chunk_size: Maximum number of elements per output file
    """
    # Convert to Path object for easier manipulation
    input_path = Path(input_file)
    
    # Validate input file exists
    if not input_path.exists():
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    
    # Read the JSON file
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Validate it's a list
    if not isinstance(data, list):
        print("Error: JSON file must contain an array at the root level.")
        sys.exit(1)
    
    # Create output directory
    output_dir = input_path.parent / f"{input_path.stem}_split"
    output_dir.mkdir(exist_ok=True)
    
    # Calculate number of chunks
    total_items = len(data)
    num_chunks = (total_items + chunk_size - 1) // chunk_size  # Ceiling division
    
    print(f"Splitting {total_items} items into {num_chunks} file(s)...")
    print(f"Output directory: {output_dir}")
    
    # Split and write chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_items)
        chunk = data[start_idx:end_idx]
        
        # Create output filename with zero-padded numbers
        output_file = output_dir / f"{input_path.stem}_part_{i+1:03d}.json"
        
        # Write chunk to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)
        
        print(f"  Created: {output_file.name} ({len(chunk)} items)")
    
    print(f"\nDone! Split into {num_chunks} file(s) in '{output_dir.name}' folder.")


def main():
    parser = argparse.ArgumentParser(
        description='Split a JSON array file into smaller chunks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python split_json.py data.json
  python split_json.py data.json --chunk-size 50
  python split_json.py data.json -c 20
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to the input JSON file containing an array'
    )
    
    parser.add_argument(
        '-c', '--chunk-size',
        type=int,
        default=30,
        help='Maximum number of elements per output file (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Validate chunk size
    if args.chunk_size < 1:
        print("Error: Chunk size must be at least 1.")
        sys.exit(1)
    
    split_json(args.input_file, args.chunk_size)


if __name__ == '__main__':
    main()