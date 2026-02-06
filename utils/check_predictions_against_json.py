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
import shutil


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
    
    def check_folder(self, folder_path: Path, expected_names: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Check if folder contains expected predictions.
        
        Returns:
            found: list of expected predictions that were found
            missing: list of expected predictions that were not found
            unexpected: list of prediction directories that were not in the JSON
        """
        self.log(f"Checking folder: {folder_path}")
        
        # Get all subdirectories
        subdirs = {d.name.lower(): d for d in folder_path.iterdir() if d.is_dir()}
        
        found = []
        missing = []
        matched_subdirs = set()  # track which subdirs we've matched to expected predictions
        
        for expected_name in expected_names:
            normalized = self.normalize_name(expected_name)
            
            # Try to find matching directory
            match_found = False
            for subdir_name, subdir_path in subdirs.items():
                if self.normalize_name(subdir_name) == normalized:
                    match_found = True
                    matched_subdirs.add(subdir_name)
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
        
        # Find unexpected directories (those not matched to any expected prediction)
        unexpected = []
        for subdir_name, subdir_path in subdirs.items():
            if subdir_name not in matched_subdirs:
                # Check if it looks like a prediction folder (has at least a .cif file)
                has_prediction_files = any(
                    f.suffix.lower() == '.cif' 
                    for f in subdir_path.rglob("*") 
                    if f.is_file()
                )
                if has_prediction_files:
                    unexpected.append(subdir_path.name)
                    self.log(f"⚠ Unexpected: {subdir_path.name} (not in JSON)")
        
        return found, missing, unexpected
    
    def check_zip(self, zip_path: Path, expected_names: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Check if zip contains expected predictions.
        
        Returns:
            found: list of expected predictions that were found
            missing: list of expected predictions that were not found
            unexpected: list of prediction directories that were not in the JSON
        """
        self.log(f"Checking zip file: {zip_path}")
        
        found = []
        missing = []
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get all file names in zip
            zip_files = {name.lower(): name for name in zf.namelist()}
            
            # Group files by prediction directory
            # Search for prediction directories at any depth in the zip
            predictions_in_zip = {}
            for filepath in zip_files.values():
                parts = Path(filepath).parts
                filename = Path(filepath).name.lower()
                
                # Check if this is a prediction file
                if filename.endswith(('.cif', '.json')) and 'fold_' in filename:
                    # Find the immediate parent directory of this prediction file
                    if len(parts) >= 2:
                        # The prediction directory is the parent of the file
                        pred_dir = parts[-2]  # -1 is the filename, -2 is its parent directory
                        
                        if pred_dir not in predictions_in_zip:
                            predictions_in_zip[pred_dir] = []
                        predictions_in_zip[pred_dir].append(filepath)
            
            matched_pred_dirs = set()  # track which prediction dirs we've matched
            
            for expected_name in expected_names:
                normalized = self.normalize_name(expected_name)
                
                # Try to find matching prediction in zip
                match_found = False
                for pred_dir, files in predictions_in_zip.items():
                    if self.normalize_name(pred_dir) == normalized:
                        match_found = True
                        matched_pred_dirs.add(pred_dir)
                        
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
            
            # Find unexpected predictions (those not matched to any expected prediction)
            unexpected = []
            for pred_dir, files in predictions_in_zip.items():
                if pred_dir not in matched_pred_dirs:
                    # Check if it looks like a prediction (has .cif files)
                    has_prediction_files = any(
                        Path(f).suffix.lower() == '.cif' 
                        for f in files
                    )
                    if has_prediction_files:
                        unexpected.append(pred_dir)
                        self.log(f"⚠ Unexpected: {pred_dir} (not in JSON)")
        
        return found, missing, unexpected
    
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
    
    def extract_matched_predictions_to_zip(self, source_path: Path, found_predictions: List[str], 
                                        output_zip_path: Path):
        """Extract matched predictions from source and create a new zip at depth 0.
        
        Args:
            source_path: Path to the source folder or zip file
            found_predictions: List of prediction names that were found
            output_zip_path: Path where the output zip should be created
        """
        if source_path.is_dir():
            self._extract_from_folder(source_path, found_predictions, output_zip_path)
        elif source_path.suffix.lower() == '.zip':
            self._extract_from_zip(source_path, found_predictions, output_zip_path)
        
    def _extract_from_folder(self, folder_path: Path, found_predictions: List[str], 
                            output_zip_path: Path):
        """Extract matched predictions from a folder to a zip file."""
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as out_zip:
            subdirs = {d.name.lower(): d for d in folder_path.iterdir() if d.is_dir()}
            
            for pred_name in found_predictions:
                normalized = self.normalize_name(pred_name)
                
                # Find matching directory
                for subdir_name, subdir_path in subdirs.items():
                    if self.normalize_name(subdir_name) == normalized:
                        # Add all files from this directory to the zip at depth 0
                        for file_path in subdir_path.rglob("*"):
                            if file_path.is_file():
                                # Create archive name: pred_dir/filename
                                arcname = f"{subdir_path.name}/{file_path.relative_to(subdir_path)}"
                                out_zip.write(file_path, arcname)
                                self.log(f"  Added: {arcname}")
                        break

    def _extract_from_zip(self, zip_path: Path, found_predictions: List[str], 
                        output_zip_path: Path):
        """Extract matched predictions from a source zip to a new zip file."""
        with zipfile.ZipFile(zip_path, 'r') as in_zip:
            with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as out_zip:
                # Get all files and group by prediction directory
                zip_files = in_zip.namelist()
                
                predictions_in_zip = {}
                for filepath in zip_files:
                    parts = Path(filepath).parts
                    filename = Path(filepath).name.lower()
                    
                    # Check if this is a prediction file
                    if filename.endswith(('.cif', '.json')) and 'fold_' in filename:
                        if len(parts) >= 2:
                            pred_dir = parts[-2]
                            if pred_dir not in predictions_in_zip:
                                predictions_in_zip[pred_dir] = []
                            predictions_in_zip[pred_dir].append(filepath)
                
                # Extract matched predictions
                for pred_name in found_predictions:
                    normalized = self.normalize_name(pred_name)
                    
                    for pred_dir, files in predictions_in_zip.items():
                        if self.normalize_name(pred_dir) == normalized:
                            # Copy all files from this prediction to output zip at depth 0
                            for filepath in files:
                                # Read from source zip
                                file_data = in_zip.read(filepath)
                                
                                # Create new archive name at depth 0: pred_dir/filename
                                parts = Path(filepath).parts
                                new_arcname = f"{pred_dir}/{parts[-1]}"
                                
                                # Write to output zip
                                out_zip.writestr(new_arcname, file_data)
                                self.log(f"  Added: {new_arcname}")
                            break
    
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
            found, missing, unexpected = self.check_folder(target_path, expected_names)
        elif target_path.suffix.lower() == '.zip':
            found, missing, unexpected = self.check_zip(target_path, expected_names)
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
        print(f"Unexpected:      {len(unexpected)} ⚠")
        
        if missing:
            print(f"\n{'='*60}")
            print(f"Missing Predictions:")
            print(f"{'='*60}")
            for name in missing:
                print(f"  ✗ {name}")
        
        if unexpected:
            print(f"\n{'='*60}")
            print(f"Unexpected Predictions (not in JSON):")
            print(f"{'='*60}")
            for name in unexpected:
                print(f"  ⚠ {name}")
        
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
            'unexpected': len(unexpected),
            'found_list': found,
            'missing_list': missing,
            'unexpected_list': unexpected,
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
    
    parser.add_argument(
    '--predictions-output',
    type=str,
    help='Create a new zip file containing only the matched predictions from the input, '
         'all placed at depth 0 (root level of the zip)'
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

    # Save matched predictions to zip if requested
    if args.predictions_output:
        output_zip_path = Path(args.predictions_output)
        print(f"\nCreating zip with matched predictions...")
        checker.extract_matched_predictions_to_zip(target_path, result['found_list'], output_zip_path)
        print(f"Matched predictions saved to: {output_zip_path} ({len(result['found_list'])} predictions)")
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()