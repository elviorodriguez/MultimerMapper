#################################################################################
################ Python file for debuging MultimerMapper ########################
#################################################################################

from utils.logger_setup import configure_logger
from src.input_check import seq_input_from_fasta, extract_seqs_from_AF2_PDBs, merge_fasta_with_PDB_data
from src.metrics_extractor import extract_AF2_metrics_from_JSON, generate_pairwise_2mers_df, generate_pairwise_Nmers_df
from src.detect_domains import detect_domains
from cfg.default_settings import *

# Input/Output
fasta_file = "tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta"
AF2_2mers = "tests/EAF6_EPL1_PHD1/2-mers"
AF2_Nmers = "tests/EAF6_EPL1_PHD1/N-mers"
out_path = "tests/output"
overwrite = True

# Setup the logger:
logger = configure_logger(out_path)

# FASTA file processing
prot_IDs, prot_names, prot_seqs, prot_lens, prot_N = seq_input_from_fasta(
    fasta_file, use_names, logger = logger)

# PDB files processing
all_pdb_data = extract_seqs_from_AF2_PDBs(AF2_2mers, AF2_Nmers, logger = logger)
# all_pdb_data = extract_seqs_from_AF2_PDBs(AF2_2mers, logger = logger)

# Combine data
merge_fasta_with_PDB_data(all_pdb_data, prot_IDs, prot_seqs, 
                            prot_seqs, prot_lens, prot_N, use_names, logger = logger)

# Extract AF2 metrics
sliced_PAE_and_pLDDTs = extract_AF2_metrics_from_JSON(all_pdb_data, fasta_file, out_path, overwrite = overwrite, logger = logger)

# Get pairwise info
pairwise_2mers_df = generate_pairwise_2mers_df(all_pdb_data, out_path = out_path, save_pairwise_data = save_pairwise_data, 
                                               overwrite = overwrite, logger = logger)
pairwise_Nmers_df = generate_pairwise_Nmers_df(all_pdb_data, out_path = out_path, save_pairwise_data = save_pairwise_data, 
                                               overwrite = overwrite, logger = logger)

# Detect protein domains
display_PAE_domains = False
display_PAE_domains_inline = False
domains_df = detect_domains(sliced_PAE_and_pLDDTs, fasta_file, graph_resolution = graph_resolution,
                            auto_domain_detection = auto_domain_detection,
                            graph_resolution_preset = graph_resolution_preset, save_preset = save_preset,
                            save_png_file = save_PAE_png, show_image = display_PAE_domains,
                            show_inline = display_PAE_domains_inline, show_structure = show_monomer_structures,
                            save_html = save_domains_html, save_tsv = save_domains_tsv,
                            out_path = out_path, overwrite = overwrite, logger = logger)

print(domains_df)

############  IMPORTS from main module ###########
# from Bio import SeqIO, PDB
# from Bio.PDB import Chain, Superimposer
# from Bio.SeqUtils import seq1
# from Bio.PDB.Polypeptide import protein_letters_3to1
# import igraph
# import plotly.graph_objects as go           # For plotly ploting
# from plotly.offline import plot             # To allow displaying plots
# from src.detect_domains import detect_domains, plot_backbone

