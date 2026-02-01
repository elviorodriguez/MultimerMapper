#!/usr/bin/env python3
"""
Script to check if a folder or zip file contains the expected AF3 predictions
based on a JSON input file.

Usage:
    python check_af3_predictions.py <folder_or_zip> <json_file> [options]

Example:
    python check_af3_predictions.py predictions.zip input.json
    python check_af3_predictions.py predictions/ input.json --verbose
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re


class AF3PredictionChecker:
    """Check if AF3 predictions exist for given input specifications."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.missing_predictions = []
        self.found_predictions = []
        
    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"  {message}")
    
    def normalize_name(self, name: str) -> str:
        """Normalize prediction name for comparison."""
        # Convert to lowercase and replace underscores with nothing for comparison
        return name.lower().replace('_', '').replace('-', '')
    
    def get_expected_files(self, prediction_name: str) -> Set[str]:
        """Generate expected file patterns for a prediction."""
        base_name = f"fold_{prediction_name}"
        
        expected = {
            # Main files (5 models)
            f"{base_name}_model_0.cif",
            f"{base_name}_model_1.cif",
            f"{base_name}_model_2.cif",
            f"{base_name}_model_3.cif",
            f"{base_name}_model_4.cif",
            
            # Full data files
            f"{base_name}_full_data_0.json",
            f"{base_name}_full_data_1.json",
            f"{base_name}_full_data_2.json",
            f"{base_name}_full_data_3.json",
            f"{base_name}_full_data_4.json",
            
            # Summary confidence files
            f"{base_name}_summary_confidences_0.json",
            f"{base_name}_summary_confidences_1.json",
            f"{base_name}_summary_confidences_2.json",
            f"{base_name}_summary_confidences_3.json",
            f"{base_name}_summary_confidences_4.json",
            
            # Job request
            f"{base_name}_job_request.json",
        }
        
        return expected
    
    def check_folder(self, folder_path: Path, expected_names: List[str]) -> Tuple[List[str], List[str]]:
        """Check if folder contains expected predictions."""
        self.log(f"Checking folder: {folder_path}")
        
        # Get all subdirectories
        subdirs = {d.name.lower(): d for d in folder_path.iterdir() if d.is_dir()}
        
        found = []
        missing = []
        
        for expected_name in expected_names:
            normalized = self.normalize_name(expected_name)
            
            # Try to find matching directory
            match_found = False
            for subdir_name, subdir_path in subdirs.items():
                if self.normalize_name(subdir_name) == normalized:
                    match_found = True
                    # Check if essential files exist
                    essential_files = self.get_expected_files(expected_name)
                    
                    files_in_dir = set()
                    for f in subdir_path.rglob("*"):
                        if f.is_file():
                            files_in_dir.add(f.name.lower())
                    
                    # Check for at least model files and job request
                    required_files = [
                        f"fold_{expected_name}_model_0.cif".lower(),
                        f"fold_{expected_name}_job_request.json".lower()
                    ]
                    
                    if all(any(rf in fn for fn in files_in_dir) for rf in required_files):
                        found.append(expected_name)
                        self.log(f"✓ Found: {expected_name} (in {subdir_path.name})")
                    else:
                        missing.append(expected_name)
                        self.log(f"✗ Incomplete: {expected_name} (in {subdir_path.name})")
                    break
            
            if not match_found:
                missing.append(expected_name)
                self.log(f"✗ Missing: {expected_name}")
        
        return found, missing
    
    def check_zip(self, zip_path: Path, expected_names: List[str]) -> Tuple[List[str], List[str]]:
        """Check if zip contains expected predictions."""
        self.log(f"Checking zip file: {zip_path}")
        
        found = []
        missing = []
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get all file names in zip
            zip_files = {name.lower(): name for name in zf.namelist()}
            
            # Group files by prediction directory
            predictions_in_zip = {}
            for filepath in zip_files.values():
                parts = Path(filepath).parts
                if len(parts) > 1:
                    pred_dir = parts[0] if parts[0] else (parts[1] if len(parts) > 1 else None)
                    if pred_dir:
                        if pred_dir not in predictions_in_zip:
                            predictions_in_zip[pred_dir] = []
                        predictions_in_zip[pred_dir].append(filepath)
            
            for expected_name in expected_names:
                normalized = self.normalize_name(expected_name)
                
                # Try to find matching prediction in zip
                match_found = False
                for pred_dir, files in predictions_in_zip.items():
                    if self.normalize_name(pred_dir) == normalized:
                        match_found = True
                        
                        # Check for essential files
                        file_basenames = [Path(f).name.lower() for f in files]
                        
                        required_files = [
                            f"fold_{expected_name}_model_0.cif".lower(),
                            f"fold_{expected_name}_job_request.json".lower()
                        ]
                        
                        if all(any(rf in fn for fn in file_basenames) for rf in required_files):
                            found.append(expected_name)
                            self.log(f"✓ Found: {expected_name} (in {pred_dir})")
                        else:
                            missing.append(expected_name)
                            self.log(f"✗ Incomplete: {expected_name} (in {pred_dir})")
                        break
                
                if not match_found:
                    missing.append(expected_name)
                    self.log(f"✗ Missing: {expected_name}")
        
        return found, missing
    
    def load_json_input(self, json_path: Path) -> Tuple[List[str], Dict[str, dict]]:
        """Load and parse the JSON input file.
        
        Returns:
            prediction_names: list of lowercased prediction names used for matching.
            raw_entries: dict mapping each lowercased name to its original full JSON entry,
                         preserved so missing predictions can be written back verbatim.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        prediction_names = []
        raw_entries = {}
        for item in data:
            if 'name' in item:
                key = item['name'].lower()
                prediction_names.append(key)
                raw_entries[key] = item          # keep the original entry untouched
        
        return prediction_names, raw_entries
    
    def check_predictions(self, target_path: Path, json_path: Path) -> Dict:
        """Main method to check predictions."""
        print(f"\n{'='*60}")
        print(f"AF3 Prediction Checker")
        print(f"{'='*60}\n")
        
        # Load expected predictions from JSON
        expected_names, raw_entries = self.load_json_input(json_path)
        print(f"Expected predictions from JSON: {len(expected_names)}")
        
        if self.verbose:
            print(f"\nExpected prediction names:")
            for name in expected_names:
                print(f"  - {name}")
        
        print(f"\nTarget: {target_path}")
        print()
        
        # Check if target is folder or zip
        if target_path.is_dir():
            found, missing = self.check_folder(target_path, expected_names)
        elif target_path.suffix.lower() == '.zip':
            found, missing = self.check_zip(target_path, expected_names)
        else:
            print(f"Error: {target_path} is neither a directory nor a zip file")
            sys.exit(1)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}\n")
        print(f"Total expected:  {len(expected_names)}")
        print(f"Found:           {len(found)} ✓")
        print(f"Missing:         {len(missing)} ✗")
        
        if missing:
            print(f"\n{'='*60}")
            print(f"Missing Predictions:")
            print(f"{'='*60}")
            for name in missing:
                print(f"  ✗ {name}")
        
        if found and self.verbose:
            print(f"\n{'='*60}")
            print(f"Found Predictions:")
            print(f"{'='*60}")
            for name in found:
                print(f"  ✓ {name}")
        
        result = {
            'expected': len(expected_names),
            'found': len(found),
            'missing': len(missing),
            'found_list': found,
            'missing_list': missing,
            'raw_entries': raw_entries,
            'success': len(missing) == 0
        }
        
        return result


def main():
    parser = argparse.ArgumentParser(
        description='Check if AF3 predictions exist for given input JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s predictions.zip input.json
  %(prog)s predictions/ input.json --verbose
  %(prog)s predictions.zip input.json -v --json-output results.json
        """
    )
    
    parser.add_argument(
        'target',
        type=str,
        help='Path to folder or zip file containing predictions'
    )
    
    parser.add_argument(
        'json_input',
        type=str,
        help='Path to JSON file with expected prediction specifications'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--json-output',
        type=str,
        help='Save a JSON file containing only the missing predictions, '
             'in the same format as the input JSON so it can be submitted directly to AF3'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    target_path = Path(args.target)
    json_path = Path(args.json_input)
    
    if not target_path.exists():
        print(f"Error: Target path does not exist: {target_path}")
        sys.exit(1)
    
    if not json_path.exists():
        print(f"Error: JSON file does not exist: {json_path}")
        sys.exit(1)
    
    # Run checker
    checker = AF3PredictionChecker(verbose=args.verbose)
    result = checker.check_predictions(target_path, json_path)
    
    # Save JSON output if requested: write only the missing predictions,
    # preserving the original format so the file can be fed back to AF3 directly.
    if args.json_output:
        missing_entries = [
            result['raw_entries'][name]
            for name in result['missing_list']
            if name in result['raw_entries']
        ]
        output_path = Path(args.json_output)
        with open(output_path, 'w') as f:
            json.dump(missing_entries, f, indent=2)
        print(f"\nMissing predictions saved to: {output_path} ({len(missing_entries)} entries)")
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()