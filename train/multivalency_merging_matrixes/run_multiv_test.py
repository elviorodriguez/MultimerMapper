
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from train.multivalency_merging_matrixes.parse_raw_data import parse_raw_data
from train.multivalency_merging_matrixes.multivalency_testing import run_multivalency_testing

###############################################################################
############################## Data Generation ################################
###############################################################################

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

true_labels_file = "train/multivalency/true_multivalency_labels.tsv"
true_labels_df = pd.read_csv(true_labels_file, sep= "\t")
benchmark_results_path = "/home/elvio/Desktop/multivalency_benchmark/benchmark_results"

thresholds = {
    'iou': np.linspace(0.00, 1, 100),
    'cf': np.linspace(0.00, 1, 100),
    'mc': np.linspace(0.00, 50, 100),
    'medc': np.linspace(0.00, 50, 100)
}

results = run_multivalency_testing(matrices_dict, true_labels_df,
                                   thresholds = thresholds,
                                   save_path = benchmark_results_path)

results.keys()


###############################################################################
########################### Results visualization #############################
###############################################################################

# File paths
filepaths = ['/home/elvio/Desktop/multivalency_benchmark/benchmark_results/cf_results.csv',
             '/home/elvio/Desktop/multivalency_benchmark/benchmark_results/iou_results.csv',
             '/home/elvio/Desktop/multivalency_benchmark/benchmark_results/mc_results.csv',
             '/home/elvio/Desktop/multivalency_benchmark/benchmark_results/medc_results.csv']


# ---------------------------------- Script 1 ---------------------------------

from train.multivalency_merging_matrixes.visualize_results import process_files, plot_precision_recall,plot_accuracy_vs_threshold, plot_precision_recall_accuracy_static, plot_precision_recall_accuracy_interactive

# Process the files and get the results
results_df = process_files(filepaths)

# Display the resulting DataFrame
print(results_df)

# Plots
plot_precision_recall(results_df)
plot_accuracy_vs_threshold(results_df)
plot_precision_recall_accuracy_static(results_df, out_dir = benchmark_results_path)
plot_precision_recall_accuracy_interactive(results_df, out_dir = benchmark_results_path)



# ---------------------------------- Script 2 ---------------------------------

from train.multivalency_merging_matrixes.visualize_results import read_csvs, evaluate_clustering_metrics, plot_accuracy_vs_threshold

# Read the clustering results for each threshold
raw_df2 = read_csvs(filepaths)
print(raw_df2)
results_df2 = evaluate_clustering_metrics(raw_df2)

benchmark_results_path2 = "/home/elvio/Desktop/multivalency_benchmark/benchmark_results2"

# Plots
plot_precision_recall(results_df2)
plot_accuracy_vs_threshold(results_df2)
plot_precision_recall_accuracy_static(results_df2)
plot_precision_recall_accuracy_interactive(results_df2, benchmark_results_path2)



###############################################################################
########################### Results visualization #############################
###############################################################################



# # Simple usage - just show the plots
# visualize_clustering_results(benchmark_results_path)

# # Save all visualizations to a directory
# visualize_clustering_results(benchmark_results_path, output_dir=benchmark_results_path)

# # Or use the ClusteringVisualizer class directly for more control
# visualizer = ClusteringVisualizer(benchmark_results_path)

# # Plot ROC and PR curves with custom size
# fig, axes = visualizer.plot_curves(figsize=(15, 6))
# plt.show()

# # Create performance heatmap
# visualizer.plot_performance_heatmap(metric='mse')
# plt.show()
