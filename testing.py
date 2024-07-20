
#################################################################################
# For debuging ##################################################################
#################################################################################

import sys
import argparse

parser = argparse.ArgumentParser(
    description='Process sequences from FASTA and PDB files.')
parser.add_argument(
    'fasta_file_path', type=str,
    help='Path to the input FASTA file')
parser.add_argument(
    'AF2_2mers', type=str,
    help='Path to the directory containing AF2 2mers PDB files')
parser.add_argument(
    '--AF2_Nmers', type=str, default=None, 
    help='Path to the directory containing AF2 Nmers PDB files (optional)')
parser.add_argument('--use_names', action='store_true',
                    help='Use protein names instead of IDs')

args = parser.parse_args()

# Use names?
use_names = args.use_names

# FASTA file
fasta_file_path = args.fasta_file_path
prot_IDs, prot_names, prot_seqs, prot_lens, prot_N = seq_input_from_fasta(
    fasta_file_path, use_names)

# Use names?
if args.use_names:
    prot_IDs_backup = prot_IDs
    prot_IDs = prot_names
    prot_names = prot_IDs_backup

# PDB files
#all_pdb_data = extract_seqs_from_AF2_PDBs(args.AF2_2mers, args.AF2_Nmers)
all_pdb_data = extract_seqs_from_AF2_PDBs(args.AF2_2mers)

# Combine the data from both 
merge_fasta_with_PDB_data(all_pdb_data, prot_IDs, prot_seqs, 
                            prot_seqs, prot_lens, prot_N, use_names)

sliced_PAE_and_pLDDTs = extract_AF2_metrics_from_JSON(all_pdb_data, fasta_file_path)

pairwise_Nmers_df = generate_pairwise_Nmers_df(all_pdb_data, is_debug = False)

all_pdb_data["../../AF2_results/BDF6_HAT1_3-4-5mers/AF2\\YNG2L__vs__EAF6__vs__YEA2__vs__Tb927.6.1240"]["A"]["length"]

test_pdb = PDB.PDBParser(QUIET=True).get_structure(id = "structure",
                                                    file = '../../AF2_results/BDF6_HAT1_3-4-5mers/AF2/BDF6__vs__EPL1__vs__EAF6__vs__YNG2L/BDF6__vs__EPL1__vs__EAF6__vs__YNG2L_unrelaxed_rank_001_alphafold2_multimer_v3_model_2_seed_000.pdb')[0]
for chain in test_pdb.get_chains():
    print(chain)
test_pdb.detach_child("D")


