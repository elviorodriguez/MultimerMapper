
import numpy as np
import pandas as pd
import json
import pickle

from train.multivalency.parse_raw_data import parse_raw_data
from train.multivalency.multivalency_testing import run_multivalency_testing


# matrices_dict data structure ---------------------------------

# matrices_dict = {
#     (protein1, protein2): {
#         model_id: {
#             'is_contact': numpy_matrix,
#             # other matrices...
#         }
#     }
# }

use_saved_matrices = True
save_pickle_dict = False
pickle_file_path = "train/multivalency/raw_matrices_dict.pkl"

if use_saved_matrices:
    with open(pickle_file_path, 'rb') as pickle_file:
        matrices_dict = pickle.load(pickle_file)
else:
    fasta_file  = "/home/elvio/Desktop/multivalency_benchmark/multivalency_test.fasta"
    AF2_2mers   = "/home/elvio/Desktop/multivalency_benchmark/multivalency_test_AF_2mers"
    AF2_Nmers   = "/home/elvio/Desktop/multivalency_benchmark/multivalency_test_AF_Nmers"
    out_path    = "/home/elvio/Desktop/multivalency_benchmark/multivalency_test_raw_output"
    mm_raw_data = parse_raw_data(fasta_file = fasta_file,
                                AF2_2mers  = AF2_2mers,
                                AF2_Nmers  = AF2_Nmers,
                                out_path   = out_path)
    matrices_dict = mm_raw_data["pairwise_contact_matrices"]

    if save_pickle_dict:
        with open(pickle_file_path, 'wb') as pickle_file:
            pickle.dump(matrices_dict, pickle_file)

# true_labels_df data structure ---------------------------------

# true_labels_df = pd.DataFrame({
#     'protein1': [(protein_ID1a, protein_ID1b), ...],
#     'protein2': [(protein_ID2a, protein_ID2b), ...],
#     'true_n_clusters': [n_clusters, ...]
# })

true_labels_file = "train/multivalency//true_multivalency_labels.tsv"
true_labels_df = pd.read_csv(true_labels_file, sep= "\t")

results = run_multivalency_testing(matrices_dict, true_labels_df, save_path="results")
