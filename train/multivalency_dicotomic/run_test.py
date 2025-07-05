
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import logging

from train.multivalency_dicotomic.parse_raw_data import parse_raw_data
from train.multivalency_dicotomic.count_interaction_modes import analyze_protein_interactions, compute_max_valency, run_multivalency_analysis
# from train.multivalency_merging_matrixes.multivalency_testing import run_multivalency_testing

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configs
N_contacts_cutoff = 3

###############################################################################
####################### Generate Benchmark Input Data #########################
###############################################################################

# ----------------------- matrices_dict data structure ------------------------

# matrices_dict = {
#     (protein1, protein2): {
#         model_id: {
#             'is_contact': numpy_matrix,
#             # other matrices...
#         },
#     (protein1, protein3): {
#         model_id: {...
#         },
#     }
# }

use_saved_matrices = False
save_pickle_dict = False
pickle_file_path = "train/multivalency_dicotomic/raw_matrices_dict.pkl"

if use_saved_matrices:
    with open(pickle_file_path, 'rb') as pickle_file:
        pairwise_contact_matrices = pickle.load(pickle_file)
else:
    fasta_file  = "/home/elvio/Desktop/multivalency_benchmark/multivalency_test.fasta"
    AF2_2mers   = "/home/elvio/Desktop/multivalency_benchmark/multivalency_test_AF_2mers"
    AF2_Nmers   = "/home/elvio/Desktop/multivalency_benchmark/multivalency_test_AF_Nmers/3mers/"
    out_path    = "/home/elvio/Desktop/multivalency_benchmark/multivalency_dicotomic_raw_output"
    
    mm_output = parse_raw_data(fasta_file = fasta_file,
                               AF2_2mers  = AF2_2mers,
                               AF2_Nmers  = AF2_Nmers,
                               out_path   = out_path)
    
    pairwise_contact_matrices = mm_output["pairwise_contact_matrices"]

    if save_pickle_dict:
        with open(pickle_file_path, 'wb') as pickle_file:
            pickle.dump(pairwise_contact_matrices, pickle_file)
            
            
# ----------------------- true_labels_df data structure -----------------------

# true_labels_df = pd.DataFrame({
#     'id1': [(protein_ID1a, protein_ID1b), ...],
#     'id2': [(protein_ID2a, protein_ID2b), ...],
#     'is_multivalent': [n_clusters, ...],
#     other columns with metadata about the protein pairs (not important)...
# })

true_labels_file = "train/multivalency_dicotomic/true_multivalency_labels.tsv"
true_labels_df = pd.read_csv(true_labels_file, sep= "\t")
benchmark_results_path = "/home/elvio/Desktop/multivalency_benchmark/multivalency_dicotomic_raw_output"

###############################################################################
############################### Matrix Analysis ###############################
###############################################################################

# # Get the pairwise contact matrices
# pairwise_contact_matrices = mm_output['pairwise_contact_matrices']

# Compute interaction counts
interaction_counts_df = analyze_protein_interactions(
    pairwise_contact_matrices = pairwise_contact_matrices, 
    N_contacts_cutoff = N_contacts_cutoff,
    logger = logger)

# Compute maximum valency for each protein pair
max_valency_dict = compute_max_valency(interaction_counts_df)

# Run benchmark
results = run_multivalency_analysis(
    interaction_counts_df = interaction_counts_df,
    true_labels_df = true_labels_df,
    output_dir = benchmark_results_path,
    logger=logger
)

# results['roc_results']
