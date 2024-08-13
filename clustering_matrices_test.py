
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def preprocess_matrices(all_pair_matrices, pair):
    valid_models = {}
    for model, matrices in all_pair_matrices[pair].items():
        if np.sum(matrices['is_contact']) > 0:  # Check if there are any contacts
            valid_models[model] = matrices
    return valid_models

def create_feature_vector(matrices):
    features = []
    for matrix_type in ['is_contact', 'PAE', 'min_pLDDT', 'distance']:
        features.extend(matrices[matrix_type].flatten())
    return np.array(features)

def cluster_models(all_pair_matrices, pair):
    # Preprocess to remove models with no contacts
    valid_models = preprocess_matrices(all_pair_matrices, pair)
    
    if len(valid_models) == 0:
        print(f"No valid models found for pair {pair}")
        return None, None, None
    
    print(f"Number of valid models: {len(valid_models)}")
    
    # Create feature vectors
    feature_vectors = np.array([create_feature_vector(matrices) for matrices in valid_models.values()])
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_vectors)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    reduced_features = pca.fit_transform(scaled_features)
    
    print(f"Number of PCA components: {reduced_features.shape[1]}")
    
    # Cluster using DBSCAN
    best_silhouette = -1
    best_eps = None
    best_labels = None
    
    for eps in np.arange(0.1, 2.0, 0.1):
        dbscan = DBSCAN(eps=eps, min_samples=2)
        labels = dbscan.fit_predict(reduced_features)
        
        if len(set(labels)) > 1:  # More than one cluster
            score = silhouette_score(reduced_features, labels)
            if score > best_silhouette:
                best_silhouette = score
                best_eps = eps
                best_labels = labels
    
    if best_labels is None:
        print(f"DBSCAN could not find optimal clustering for pair {pair}")
        print("Attempting K-means clustering as fallback")
        
        # Try K-means with 2 clusters as a fallback
        kmeans = KMeans(n_clusters=2, random_state=42)
        best_labels = kmeans.fit_predict(reduced_features)
    
    return list(valid_models.keys()), best_labels, reduced_features

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
    
    # Plot average contact matrix for all models
    plt.subplot(132)
    avg_contact_matrix = np.mean([all_pair_matrices[pair][model]['is_contact'] for model in model_keys], axis=0)
    plt.imshow(avg_contact_matrix, cmap='viridis')
    plt.title(f"Average Contact Matrix (n={len(model_keys)})")
    plt.colorbar()
    
    # Plot contact frequency
    plt.subplot(133)
    contact_freq = np.mean([all_pair_matrices[pair][model]['is_contact'] > 0 for model in model_keys], axis=0)
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
            avg_contact_matrix = np.mean([all_pair_matrices[pair][model]['is_contact'] for model in cluster_models], axis=0)
            
            plt.subplot(1, n_clusters, cluster + 1)
            plt.imshow(avg_contact_matrix, cmap='viridis')
            plt.title(f"Cluster {cluster} (n={len(cluster_models)})")
            plt.colorbar()
        
        plt.tight_layout()
        plt.show()

# Main function to run the clustering
def cluster_and_visualize(all_pair_matrices, pair):
    model_keys, labels, reduced_features = cluster_models(all_pair_matrices, pair)
    if model_keys is not None:
        visualize_clusters(all_pair_matrices, pair, model_keys, labels, reduced_features)
    else:
        print(f"No valid models for pair {pair}")

# Usage
for pair in all_pair_matrices.keys():
    print(f"\nProcessing pair: {pair}")
    cluster_and_visualize(all_pair_matrices, pair)