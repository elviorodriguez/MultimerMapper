#!/usr/bin/env python3
import os
import sys
import json
import argparse
import glob
import zipfile
import tempfile
import shutil
from collections import defaultdict
from pathlib import Path

from Bio.PDB import MMCIFParser, PDBIO
from Bio import SeqIO
from Bio.SeqUtils import seq1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert AF3 output (ZIP files or directories) to AF2-style JSON and PDB files"
    )
    parser.add_argument(
        "input_path",
        help="Path to AF3 ZIP file, directory containing ZIP files, or AF3 output folder"
    )
    parser.add_argument(
        "--fasta",
        help="Optional FASTA file to generate __vs__ naming",
        default=None
    )
    parser.add_argument(
        "--out_dir",
        help="Output directory (default: input_path + _AF2_converted)",
        default=None
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep temporary extracted files for debugging"
    )
    return parser.parse_args()


def is_zip_file(path):
    """Check if path is a ZIP file"""
    return os.path.isfile(path) and path.lower().endswith('.zip')


def extract_zip_to_temp(zip_path):
    """Extract ZIP file to a temporary directory"""
    temp_dir = tempfile.mkdtemp(prefix="af3_extract_")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        return temp_dir
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e


def find_af3_files_in_dir(directory):
    """Find AF3 output files in a directory"""
    full_files = sorted(glob.glob(os.path.join(directory, "*_full_data_*.json")))
    summ_files = sorted(glob.glob(os.path.join(directory, "*_summary_confidences_*.json")))
    cif_files = sorted(glob.glob(os.path.join(directory, "*_model_*.cif")))
    
    return full_files, summ_files, cif_files


def get_zip_files_from_path(input_path):
    """Get list of ZIP files from input path"""
    if is_zip_file(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        zip_files = glob.glob(os.path.join(input_path, "*.zip"))
        return sorted(zip_files)
    else:
        return []


def average_plddt_by_residue(cif_path, atom_plddts):
    """
    Read CIF via Biopython, map atom_plddts list to residues by averaging.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("m", cif_path)
    residue_scores = defaultdict(list)
    atom_index = 0

    # collect scores per residue
    for model in structure:
        for chain in model:
            for residue in chain:
                # skip hetero
                if residue.id[0] != " ":
                    continue
                for atom in residue:
                    if atom_index >= len(atom_plddts):
                        raise IndexError(f"Atom index {atom_index} out of range for {cif_path}")
                    residue_scores[(chain.id, residue.id[1])].append(atom_plddts[atom_index])
                    atom_index += 1

    # average in sequence order
    plddt_res = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                scores = residue_scores.get((chain.id, residue.id[1]), [])
                avg = sum(scores) / len(scores) if scores else 0.0
                plddt_res.append(avg)

    return plddt_res


def extract_sequence_from_cif(cif_path):
    """
    Extract one-letter sequence(s) per chain from a CIF file.
    Returns list of sequences, in chain order.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("m", cif_path)
    seqs = []
    for model in structure:
        for chain in model:
            # build three-letter code string
            three_letter = "".join(
                residue.resname
                for residue in chain
                if residue.id[0] == " "
            )
            # convert to one-letter
            try:
                one_letter = seq1(three_letter)
            except Exception:
                # fallback: X for unknowns
                one_letter = "".join(
                    seq1(res) if res.isalpha() else "X"
                    for res in [residue.resname for residue in chain if residue.id[0] == " "]
                )
            seqs.append(one_letter)
    return seqs


def generate_name_from_fasta(fasta_path, cif_files):
    """
    Match the sequence(s) extracted from the first CIF against
    FASTA records, returning a "__vs__" joined name list.
    """
    # load fasta
    fasta_records = list(SeqIO.parse(fasta_path, "fasta"))
    fasta_strs = {str(rec.seq): rec.id for rec in fasta_records}

    # get cif sequences (assume all cif share same sequence ordering)
    cif_seqs = extract_sequence_from_cif(cif_files[0])
    names = []
    for seq in cif_seqs:
        match = None
        # exact match
        if seq in fasta_strs:
            match = fasta_strs[seq]
        else:
            # partial match
            for fseq, fid in fasta_strs.items():
                if seq in fseq or fseq in seq:
                    match = fid
                    break
        names.append(match if match else "unknown")
    return "__vs__".join(names)


def convert_cif_to_pdb(cif_path, pdb_path):
    """
    Convert a CIF file to PDB format via Biopython.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("m", cif_path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)


def process_af3_directory(af3_dir, fasta_path, out_dir, dir_name_prefix):
    """
    Process a single AF3 directory (extracted or original)
    """
    # Find AF3 files
    full_files, summ_files, cif_files = find_af3_files_in_dir(af3_dir)
    
    if not (full_files and summ_files and cif_files):
        print(f"Warning: Could not find expected AF3 files in {af3_dir}", file=sys.stderr)
        return False
    
    # Determine prefix
    if fasta_path:
        prefix = generate_name_from_fasta(fasta_path, cif_files)
    else:
        prefix = dir_name_prefix
    
    # Create output subdirectory
    prediction_out_dir = os.path.join(out_dir, prefix)
    os.makedirs(prediction_out_dir, exist_ok=True)
    
    # Build models list
    models = []
    for full, summ, cif in zip(full_files, summ_files, cif_files):
        idx = os.path.basename(full).split("_")[-1].split(".")[0]
        
        try:
            with open(full, 'r') as f:
                data_full = json.load(f)
            with open(summ, 'r') as f:
                data_summ = json.load(f)
        except Exception as e:
            print(f"Error reading JSON files for {cif}: {e}", file=sys.stderr)
            continue
            
        try:
            # residue pLDDT
            plddt = average_plddt_by_residue(cif, data_full["atom_plddts"])
            # max pae
            pae_matrix = data_full["pae"]
            max_pae = max(max(row) for row in pae_matrix)

            out_json = {
                "plddt":    plddt,
                "max_pae":  max_pae,
                "pae":      data_full["pae"],
                "ptm":      data_summ["ptm"],
                "iptm":     data_summ["iptm"],
            }
            models.append((idx, out_json, cif))
        except Exception as e:
            print(f"Error processing {cif}: {e}", file=sys.stderr)
            continue

    if not models:
        print(f"No valid models found in {af3_dir}", file=sys.stderr)
        return False
    
    # Rank by iPTM descending
    ranked = sorted(models, key=lambda x: x[1]["iptm"], reverse=True)
    
    # Write out AF2-style outputs
    for rank, (idx, jdata, cif) in enumerate(ranked, start=1):
        base = os.path.basename(cif)
        model_n = int(base.split("_")[-1].split(".")[0]) + 1

        # JSON filename
        json_name = (
            f"{prefix}_scores_rank_{rank:03d}"
            f"_alphafold2_multimer_v3_model_{model_n}_seed_000.json"
        )
        try:
            with open(os.path.join(prediction_out_dir, json_name), "w") as outj:
                json.dump(jdata, outj, indent=2)
        except Exception as e:
            print(f"Error writing JSON {json_name}: {e}", file=sys.stderr)
            continue

        # PDB filename
        pdb_name = (
            f"{prefix}_unrelaxed_rank_{rank:03d}"
            f"_alphafold2_multimer_v3_model_{model_n}_seed_000.pdb"
        )
        try:
            convert_cif_to_pdb(cif, os.path.join(prediction_out_dir, pdb_name))
        except Exception as e:
            print(f"Error converting CIF to PDB {pdb_name}: {e}", file=sys.stderr)
            continue

    print(f"Processed {len(ranked)} models for {prefix} -> {prediction_out_dir}")
    return True


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input_path)
    
    # Set up output directory
    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
    else:
        out_dir = f"{input_path.rstrip(os.sep)}_AF2_converted"
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Check if input is a regular AF3 directory (not ZIP)
    if os.path.isdir(input_path):
        full_files, summ_files, cif_files = find_af3_files_in_dir(input_path)
        if full_files and summ_files and cif_files:
            # This is a regular AF3 directory
            print(f"Processing AF3 directory: {input_path}")
            dir_name = os.path.basename(input_path.rstrip(os.sep))
            success = process_af3_directory(input_path, args.fasta, out_dir, dir_name)
            if success:
                print(f"Conversion complete. Files written to: {out_dir}")
            else:
                print("Conversion failed.", file=sys.stderr)
                sys.exit(1)
            return
    
    # Handle ZIP files
    zip_files = get_zip_files_from_path(input_path)
    
    if not zip_files:
        print(f"Error: No ZIP files found in {input_path}, and it's not a valid AF3 directory", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(zip_files)} ZIP file(s) to process")
    
    processed_count = 0
    temp_dirs = []
    
    try:
        for zip_file in zip_files:
            print(f"\nProcessing: {zip_file}")
            
            # Extract ZIP file
            try:
                temp_dir = extract_zip_to_temp(zip_file)
                temp_dirs.append(temp_dir)
            except Exception as e:
                print(f"Error extracting {zip_file}: {e}", file=sys.stderr)
                continue
            
            # Find all AF3 directories inside the extracted content
            af3_dirs = []
            for root, dirs, files in os.walk(temp_dir):
                full_files, summ_files, cif_files = find_af3_files_in_dir(root)
                if full_files and summ_files and cif_files:
                    af3_dirs.append(root)

            if not af3_dirs:
                print(f"No AF3 files found in {zip_file}", file=sys.stderr)
                continue

            # Generate base prefix from ZIP filename
            zip_basename = os.path.splitext(os.path.basename(zip_file))[0]

            # Process each AF3 directory found
            for af3_dir in af3_dirs:
                # For multiple predictions in one ZIP, use the subdirectory name as part of prefix
                if len(af3_dirs) > 1:
                    subdir_name = os.path.basename(af3_dir)
                    dir_prefix = f"{zip_basename}_{subdir_name}"
                else:
                    dir_prefix = zip_basename
                
                success = process_af3_directory(af3_dir, args.fasta, out_dir, dir_prefix)
                if success:
                    processed_count += 1
    
    finally:
        # Clean up temporary directories
        if not args.keep_temp:
            for temp_dir in temp_dirs:
                shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"Temporary directories kept: {temp_dirs}")
    
    print(f"\n=== Summary ===")
    print(f"Total ZIP files found: {len(zip_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Output directory: {out_dir}")
    
    if processed_count == 0:
        print("No files were successfully processed!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()