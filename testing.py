#################################################################################
################ Python file for debuging MultimerMapper ########################
#################################################################################

import os
import pickle
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

# Check if the pickle file exists
pickle_dir = 'tests/pickles'
pickle_file = os.path.join(pickle_dir, 'variables.pkl')
os.makedirs(pickle_dir, exist_ok=True)
try:
    logger.info(f"Opening pickle file: {pickle_file}. This might take a while...")

    with open(pickle_file, 'rb') as f:
        variables = pickle.load(f)
    print(f"Loaded variables from pickle: {pickle_file}")
    
    # Unpack the variables
    prot_IDs = variables['prot_IDs']
    prot_names = variables['prot_names']
    prot_seqs = variables['prot_seqs']
    prot_lens = variables['prot_lens']
    prot_N = variables['prot_N']
    all_pdb_data = variables['all_pdb_data']
    sliced_PAE_and_pLDDTs = variables['sliced_PAE_and_pLDDTs']
    pairwise_2mers_df = variables['pairwise_2mers_df']
    pairwise_Nmers_df = variables['pairwise_Nmers_df']

    print(f"Unpacking variables from pickle complete")

except FileNotFoundError:
    
    print("Pickle file not found. Running the full protocol...")

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
    
    # Save all variables to a pickle file
    variables = {
        'prot_IDs': prot_IDs,
        'prot_names': prot_names,
        'prot_seqs': prot_seqs,
        'prot_lens': prot_lens,
        'prot_N': prot_N,
        'all_pdb_data': all_pdb_data,
        'sliced_PAE_and_pLDDTs': sliced_PAE_and_pLDDTs,
        'pairwise_2mers_df': pairwise_2mers_df,
        'pairwise_Nmers_df': pairwise_Nmers_df
    }

    with open(pickle_file, 'wb') as f:
        pickle.dump(variables, f)

    logger.warning(f"Saved variables to pickle: {pickle_file}")


# Detect protein domains
display_PAE_domains = True
display_PAE_domains_inline = True
auto_domain_detection = True
graph_resolution_preset = None
manual_domains = "tests/EAF6_EPL1_PHD1/manual_domains.tsv"
manual_domains = None
domains_df = detect_domains(sliced_PAE_and_pLDDTs, fasta_file, graph_resolution = graph_resolution,
                            auto_domain_detection = auto_domain_detection,
                            graph_resolution_preset = graph_resolution_preset, save_preset = save_preset,
                            save_png_file = save_PAE_png, show_image = display_PAE_domains,
                            show_inline = display_PAE_domains_inline, show_structure = show_monomer_structures,
                            save_html = save_domains_html, save_tsv = save_domains_tsv,
                            out_path = out_path, overwrite = overwrite, logger = logger, manual_domains = manual_domains)

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

