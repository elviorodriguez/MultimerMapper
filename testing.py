

#################################################################################
############### Python file for debuging ########################################
#################################################################################

from src.input_check import seq_input_from_fasta, extract_seqs_from_AF2_PDBs, merge_fasta_with_PDB_data
from src.metrics_extractor import extract_AF2_metrics_from_JSON, generate_pairwise_2mers_df, generate_pairwise_Nmers_df

# Input
fasta_file = "tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta"
AF2_2mers = "tests/EAF6_EPL1_PHD1/2-mers"
AF2_Nmers = "tests/EAF6_EPL1_PHD1/N-mers"
use_names = True

# FASTA file processing
prot_IDs, prot_names, prot_seqs, prot_lens, prot_N = seq_input_from_fasta(
    fasta_file, use_names)

# PDB files processing
all_pdb_data = extract_seqs_from_AF2_PDBs(AF2_2mers, AF2_Nmers)
# all_pdb_data = extract_seqs_from_AF2_PDBs(AF2_2mers)

# Combine data
merge_fasta_with_PDB_data(all_pdb_data, prot_IDs, prot_seqs, 
                            prot_seqs, prot_lens, prot_N, use_names)

# Extract AF2 metrics
sliced_PAE_and_pLDDTs = extract_AF2_metrics_from_JSON(all_pdb_data, fasta_file)

# Get pairwise info
pairwise_2mers_df = generate_pairwise_2mers_df(all_pdb_data)
pairwise_Nmers_df = generate_pairwise_Nmers_df(all_pdb_data)

print(pairwise_2mers_df)
print(pairwise_Nmers_df)

############  IMPORTS from main module ###########
# from Bio import SeqIO, PDB
# from Bio.PDB import Chain, Superimposer
# from Bio.SeqUtils import seq1
# from Bio.PDB.Polypeptide import protein_letters_3to1
# import igraph
# import plotly.graph_objects as go           # For plotly ploting
# from plotly.offline import plot             # To allow displaying plots
# from src.detect_domains import detect_domains, plot_backbone

