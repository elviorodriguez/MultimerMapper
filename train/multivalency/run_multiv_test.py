
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from train.multivalency.parse_raw_data import parse_raw_data
from train.multivalency.multivalency_testing import run_multivalency_testing

from train.multivalency.visualize_clustering import visualize_clustering_results, ClusteringVisualizer


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

thresholds = {
    'iou': np.linspace(0.01, 0.99, 10),
    'cf': np.linspace(0.01, 0.99, 10),
    'mc': np.linspace(0.01, 20.0, 10),
    'medc': np.linspace(0.01, 20.0, 10)
}

results = run_multivalency_testing(matrices_dict, true_labels_df,
                                   thresholds = thresholds,
                                   save_path="results")

results.keys()



###############################################################################
########################### Results visualization #############################
###############################################################################

# Simple usage - just show the plots
visualize_clustering_results('results')

# Save all visualizations to a directory
visualize_clustering_results('results', output_dir='visualization_output')

# Or use the ClusteringVisualizer class directly for more control
visualizer = ClusteringVisualizer('results')

# Plot ROC and PR curves with custom size
fig, axes = visualizer.plot_curves(figsize=(15, 6))
plt.show()

# Create performance heatmap
visualizer.plot_performance_heatmap(metric='mse')
plt.show()
