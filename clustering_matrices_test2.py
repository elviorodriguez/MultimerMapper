#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 23:32:20 2024

@author: elvio
"""

def visualize_clusters(all_pair_matrices, pair, model_keys, labels, reduced_features):
    if labels is None:
        return
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of clusters: {n_clusters}")
    
    plt.figure(figsize=(15, 5))
    
    # Plot cluster assignments
    plt.subplot(131)
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
    plt.title('Cluster Assignments')
    plt.colorbar(scatter)
    
    # Find the maximum dimensions
    max_shape = max(all_pair_matrices[pair][model]['is_contact'].shape 
                    for model in model_keys)
    
    # Function to pad matrices to the same size
    def pad_matrix(matrix, max_shape):
        pad_width = [(0, max_shape[i] - matrix.shape[i]) for i in range(len(max_shape))]
        return np.pad(matrix, pad_width, mode='constant', constant_values=0)
    
    # Plot average contact matrix for all models
    plt.subplot(132)
    padded_matrices = [pad_matrix(all_pair_matrices[pair][model]['is_contact'], max_shape) 
                       for model in model_keys]
    avg_contact_matrix = np.mean(padded_matrices, axis=0)
    plt.imshow(avg_contact_matrix, cmap='viridis')
    plt.title(f"Average Contact Matrix (n={len(model_keys)})")
    plt.colorbar()
    
    # Plot contact frequency
    plt.subplot(133)
    contact_freq = np.mean([pad_matrix(all_pair_matrices[pair][model]['is_contact'], max_shape) > 0 
                            for model in model_keys], axis=0)
    plt.imshow(contact_freq, cmap='viridis')
    plt.title("Contact Frequency")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    # If multiple clusters were found, show average contact matrices for each cluster
    if n_clusters > 1:
        plt.figure(figsize=(5*n_clusters, 5))
        for cluster in range(n_clusters):
            cluster_models = [model for model, label in zip(model_keys, labels) if label == cluster]
            cluster_matrices = [pad_matrix(all_pair_matrices[pair][model]['is_contact'], max_shape) 
                                for model in cluster_models]
            avg_contact_matrix = np.mean(cluster_matrices, axis=0)
            
            plt.subplot(1, n_clusters, cluster + 1)
            plt.imshow(avg_contact_matrix, cmap='viridis')
            plt.title(f"Cluster {cluster} (n={len(cluster_models)})")
            plt.colorbar()
        
        plt.tight_layout()
        plt.show()