import igraph
import numpy as np
import pandas as pd
import random
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from queue import PriorityQueue
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
import plotly.graph_objects as go
from plotly.offline import plot

###############################################################################
##################### Classes Stoichiometry & AssemblyPath ####################
###############################################################################

class Stoichiometry:
    def __init__(self, graph: igraph.Graph, parent: Optional['Stoichiometry'] = None):
        self.graph = graph.copy()
        self.parent = parent
        self.children: List[Stoichiometry] = []
        self.is_root = parent is None
        self.is_leaf = False
        self.stagnation_counter = 0
        self.score = 0  # New attribute to store the score

    def add_child(self, child: 'Stoichiometry'):
        self.children.append(child)

    def add_protein(self, protein: str):
        self.graph.add_vertex(name=protein)

    def add_ppi(self, protein1: str, protein2: str, ppi_data: Dict):
        edge_name = tuple(sorted((protein1, protein2)))
        if not self.graph.are_connected(protein1, protein2):
            self.graph.add_edge(protein1, protein2, name=edge_name, **ppi_data)

    def get_protein_count(self) -> Dict[str, int]:
        return dict(Counter(self.graph.vs['name']))

    def __eq__(self, other: 'Stoichiometry') -> bool:
        return self.get_protein_count() == other.get_protein_count() and self.graph.isomorphic(other.graph)

    def __hash__(self):
        return hash(tuple(sorted(self.get_protein_count().items())) + tuple(self.graph.get_edgelist()))

class AssemblyPath:
    def __init__(self, root: Stoichiometry):
        self.root = root
        self.path: List[Stoichiometry] = [root]

    def add_stoichiometry(self, stoichiometry: Stoichiometry):
        self.path.append(stoichiometry)
        
    def merge_contiguous_stoichiometries(self):
        i = 0
        while i < len(self.path) - 1:
            current = self.path[i]
            next_stoich = self.path[i + 1]
            
            if current.get_protein_count() == next_stoich.get_protein_count():
                self.path.pop(i + 1)
            else:
                current.children = [next_stoich]
                next_stoich.parent = current
                i += 1
                

###############################################################################
############################### Helper functions ##############################
###############################################################################

def select_weighted_ppi(protein: int, combined_graph: igraph.Graph, existing_stoichiometry: Stoichiometry) -> Tuple[str, Dict]:
    
    edges = combined_graph.incident(protein, mode="all")
    weights = []
    valid_edges = []
    
    # Compute the weights
    for edge in edges:
        edge_data = combined_graph.es[edge].attributes()
        
        two_mers_data = edge_data.get('2_mers_data', pd.DataFrame())
        n_mers_data   = edge_data.get('N_mers_data', pd.DataFrame())
        
        # 2-mers and N-mers dataframes verification
        if isinstance(two_mers_data, pd.DataFrame) and 'N_models' in two_mers_data.columns:
            two_mers_models = two_mers_data['N_models'].values
        else:
            two_mers_models = []
            
        if isinstance(n_mers_data, pd.DataFrame) and 'N_models' in n_mers_data.columns:
            n_mers_models = n_mers_data['N_models'].values
        else:
            n_mers_models = []
        
        # compute the N_models average
        n_mers_mean   = np.nanmean(n_mers_models)   if len(n_mers_models)   > 0 else 0
        two_mers_mean = np.nanmean(two_mers_models) if len(two_mers_models) > 0 else 0
        weight = (n_mers_mean + two_mers_mean) / 10
        
        # Adjust weight based on dynamics
        dynamics = edge_data.get('dynamics', 'Static')
        if dynamics == 'Strong Positive':
            weight *= 2
        elif dynamics == 'Weak Positive':
            weight *= 1.5
        elif dynamics == 'Weak Negative':
            weight *= 0.75
        elif dynamics == 'Strong Negative':
            weight *= 0.5

        # Penalize edges that would create homooligomers only if the protein is not involved in homooligomerization in the original graph
        source, target = combined_graph.es[edge].tuple
        partner = target if combined_graph.vs[source]['name'] == protein else source
        if partner in existing_stoichiometry.get_protein_count():
            if not is_homooligomer_in_original_graph(protein, combined_graph):
                weight *= 0.1
        
        if np.isfinite(weight) and weight > 0:
            weights.append(weight)
            valid_edges.append(edge)
    
    if not valid_edges:
        raise ValueError(f"No valid edges found for protein {protein}")
    
    # Select PPI edge and get its attributes
    selected_edge = random.choices(valid_edges, weights=weights)[0]
    edge_data = combined_graph.es[selected_edge].attributes()
    
    # 
    source, target = combined_graph.es[selected_edge].tuple
    partner = target if combined_graph.vs[source]['name'] == protein else source
    
    return combined_graph.vs[partner]['name'], edge_data

def is_homooligomer_in_original_graph(protein: str, combined_graph: igraph.Graph) -> bool:
    edges = combined_graph.incident(protein, mode="all")
    for edge in edges:
        source, target = combined_graph.es[edge].tuple
        if source == target == protein:
            return True
    return False

def initialize_stoichiometry(combined_graph: igraph.Graph) -> Stoichiometry:
    root_protein = random.choice(combined_graph.vs)['name']
    initial_graph = igraph.Graph()
    initial_graph.add_vertex(name=root_protein)
    initial_stoichiometry = Stoichiometry(initial_graph)
    
    partner, ppi_data = select_weighted_ppi(root_protein, combined_graph, initial_stoichiometry)
    
    initial_stoichiometry.add_protein(partner)
    edge_name = tuple(sorted((root_protein, partner)))
    
    ppi_data_copy = ppi_data.copy()
    ppi_data_copy.pop('name', None)
    
    initial_stoichiometry.add_ppi(root_protein, partner, ppi_data_copy)
    
    return initial_stoichiometry

def generate_child(parent: Stoichiometry,
                   combined_graph: igraph.Graph,
                   add_number = 1) -> Optional[Stoichiometry]:
    
    child = Stoichiometry(parent.graph, parent)
    growth_achieved = False
    
    for _ in range(add_number):  # Try to add up to to_add_number proteins/PPIs
        
        selected_protein = random.choice(child.graph.vs)['name']
        partner, ppi_data = select_weighted_ppi(selected_protein, combined_graph, child)
        
        edge_name = tuple(sorted((selected_protein, partner)))
        ppi_data_copy = ppi_data.copy()
        ppi_data_copy.pop('name', None)
        
        if partner not in child.graph.vs['name']:
            # Add new protein and PPI
            child.add_protein(partner)
            child.add_ppi(selected_protein, partner, ppi_data_copy)
            growth_achieved = True
        else:
            
            # Check if there's an existing PPI between the proteins
            existing_edges = child.graph.es.select(_between=([selected_protein], [partner]))
            if not existing_edges:
                # Add new PPI between existing proteins
                child.add_ppi(selected_protein, partner, ppi_data_copy)
                growth_achieved = True
            else:
                for existing_edge in existing_edges:
                    if 'valency' in existing_edge.attributes():

                        existing_cluster_n = existing_edge['valency']['cluster_n']
                        new_cluster_n = ppi_data_copy['valency']['cluster_n']
                        if existing_cluster_n != new_cluster_n:
                            # Add new interaction mode
                            child.add_ppi(selected_protein, partner, ppi_data_copy)
                            growth_achieved = True
                            break
    
    if growth_achieved:
        update_dynamic_ppis(child, combined_graph)
        parent.add_child(child)
        return child
    
    else:
        return None

# To remove or add dynamic ppis to the stoichiometry based on a probability
def update_dynamic_ppis(stoichiometry: Stoichiometry, combined_graph: igraph.Graph):
    config = tuple(sorted(stoichiometry.get_protein_count().items()))
    
    for edge in stoichiometry.graph.es:
        
        source, target = edge.tuple
        combined_edges = combined_graph.es.select(_between=([source], [target]))
        
        for combined_edge in combined_edges:
            if combined_edge['valency']['cluster_n'] == edge['valency']['cluster_n']:
                n_mers_data = combined_edge['N_mers_data']
                
                if isinstance(n_mers_data, pd.DataFrame) and 'proteins_in_model' in n_mers_data.columns:
                    matching_rows = n_mers_data[n_mers_data['proteins_in_model'] == config]
                    if not matching_rows.empty:
                        
                        n_models = matching_rows['N_models'].values[0]
                        p_add = n_models / 5
                        
                        dynamics = combined_edge['dynamics']
                        if dynamics in ['Weak Positive', 'Positive', 'Strong Positive']:
                            if random.random() < p_add:
                                # Add positive dynamic PPI
                                stoichiometry.add_ppi(source, target, combined_edge.attributes())
                        elif dynamics in ['Weak Negative', 'Negative', 'Strong Negative']:
                            p_rem = 1 - p_add
                            if random.random() < p_rem:
                                # Remove negative dynamic PPI
                                stoichiometry.graph.delete_edges(edge)
                                # Check if removal causes detachment
                                if not stoichiometry.graph.is_connected():
                                    components = stoichiometry.graph.components()
                                    if len(components) > 1:
                                        # Create new child with detached piece
                                        detached_graph = stoichiometry.graph.subgraph(components[1])
                                        new_child = Stoichiometry(detached_graph, stoichiometry.parent)
                                        stoichiometry.parent.add_child(new_child)
                                        # Remove detached piece from current stoichiometry
                                        stoichiometry.graph = stoichiometry.graph.subgraph(components[0])
                        else:
                            # Dynamics is not a dynamic edge
                            pass

def calculate_stoichiometry_score(stoichiometry: Stoichiometry, combined_graph: igraph.Graph) -> float:
    score = 0
    protein_count = stoichiometry.get_protein_count()
    
    # Penalize homooligomers only if they're not in the original graph
    for protein, count in protein_count.items():
        if count > 1 and not is_homooligomer_in_original_graph(protein, combined_graph):
            score -= count * 10
    
    # Reward for each unique protein
    score += len(protein_count) * 5
    
    # Reward for PPIs present in the combined graph
    for edge in stoichiometry.graph.es:
        if combined_graph.are_connected(edge.source, edge.target):
            score += 1
    
    # Penalize missing proteins
    missing_proteins = set(combined_graph.vs['name']) - set(protein_count.keys())
    score -= len(missing_proteins) * 5
    
    return score

###############################################################################
################################# Main function ###############################
###############################################################################

def explore_stoichiometric_space(combined_graph: igraph.Graph, max_iterations: int = 1000, max_path_length: int = 100) -> List[AssemblyPath]:
    assembly_paths = []
    
    print("INITIALIZE: Stoichiometric Space Exploration Algorithm...")
    
    for i in range(max_iterations):
        print(f"   Iteration {i}...")
        try:
            root = initialize_stoichiometry(combined_graph)
            path = AssemblyPath(root)
            
            pq = PriorityQueue()
            pq.put((-calculate_stoichiometry_score(root, combined_graph), root))
            
            while not pq.empty() and len(path.path) < max_path_length:
                _, current = pq.get()
                
                if current.stagnation_counter >= 5:
                    current.is_leaf = True
                    break
                
                child = generate_child(current, combined_graph)
                if child is None:
                    current.stagnation_counter += 1
                    pq.put((-calculate_stoichiometry_score(current, combined_graph), current))
                else:
                    child_score = calculate_stoichiometry_score(child, combined_graph)
                    child.score = child_score  # Store the score in the Stoichiometry object
                    path.add_stoichiometry(child)
                    pq.put((-child_score, child))
                    
                    if child.get_protein_count() == current.get_protein_count():
                        current.stagnation_counter += 1
                    else:
                        current.stagnation_counter = 0
            
            path.merge_contiguous_stoichiometries()
            assembly_paths.append(path)
        except ValueError as e:
            print(f"Error in iteration: {e}")
            continue
        
    print("FINISHED: Stoichiometric Space Exploration Algorithm")
    
    # Sort assembly paths by the score of their final stoichiometry
    assembly_paths.sort(key=lambda p: p.path[-1].score if p.path else 0, reverse=True)
    
    return assembly_paths


###############################################################################
####################### PCoA reduction and Visualization ######################
###############################################################################


def generate_protein_matrix(paths):
    all_proteins = set()
    for path in paths:
        for stoichiometry in path.path:
            all_proteins.update(stoichiometry.get_protein_count().keys())
    
    protein_list = sorted(list(all_proteins))
    
    matrix = []
    stoichiometry_dict = defaultdict(int)  # To group by protein counts
    for path in paths:
        for stoichiometry in path.path:
            protein_count = tuple(stoichiometry.get_protein_count().get(p, 0) for p in protein_list)
            stoichiometry_dict[protein_count] += 1  # Increment frequency count
            matrix.append(protein_count)
    
    return np.array(matrix), protein_list, stoichiometry_dict


def dimensionality_reduction_chunked(protein_matrix, chunk_size=1000, metric='manhattan', epsilon=1e-9, dims = "2D"):
    """
    Perform dimensionality reduction in chunks to avoid memory overload, 
    with a small epsilon added to the distance matrix to avoid division by zero.
    
    Args:
    protein_matrix (np.ndarray): The matrix of protein data.
    chunk_size (int): The size of each chunk to process.
    metric (str): The distance metric to use for pairwise distances.
    epsilon (float): Small value added to avoid division by zero.

    Returns:
    coords (list): 2D/3D coordinates for each protein combination (depending on
                   dims: "2D" or "3D").
    """
    if dims == "2D":
        n_samples = protein_matrix.shape[0]
        coords = np.zeros((n_samples, 2))  # To store 2D coordinates for each protein
    
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            chunk = protein_matrix[i:end, :]
    
            # Compute pairwise distances and add epsilon to avoid zero distances
            dist_matrix = pairwise_distances(chunk, metric=metric) + epsilon
    
            # Apply MDS on the chunk
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            coords_chunk = mds.fit_transform(dist_matrix)
    
            # Store the resulting 2D coordinates
            coords[i:end, :] = coords_chunk
    
        return coords
    
    elif dims == "3D":
        n_samples = protein_matrix.shape[0]
        coords = np.zeros((n_samples, 3))  # Changed to store 3D coordinates
    
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            chunk = protein_matrix[i:end, :]
    
            # Compute pairwise distances and add epsilon to avoid zero distances
            dist_matrix = pairwise_distances(chunk, metric=metric) + epsilon
    
            # Apply MDS on the chunk, now with 3 components
            mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
            coords_chunk = mds.fit_transform(dist_matrix)
    
            # Store the resulting 3D coordinates
            coords[i:end, :] = coords_chunk
    
        return coords


def visualize_paths_2d(coords, paths, protein_counts, stoichiometry_dict, protein_list):
    fig = go.Figure()

    # Plot stoichiometries as points
    unique_coords = {}  # To store aggregated coords by protein count
    for idx, stoichiometry in enumerate(protein_counts):
        protein_count = tuple(stoichiometry)
        if protein_count not in unique_coords:
            unique_coords[protein_count] = {
                'x': coords[idx, 0],
                'y': coords[idx, 1],
                'freq': stoichiometry_dict[protein_count],
                'total_count': sum(protein_count)
            }
    
    # Scale factors
    stoich_scale_factor = 5
    
    # Prepare for plotting points
    xs = [info['x'] for info in unique_coords.values()]
    ys = [info['y'] for info in unique_coords.values()]
    freq_sizes = [info['freq'] for info in unique_coords.values()]  # Size proportional to frequency
    stoich_sizes = [info['total_count'] * stoich_scale_factor for info in unique_coords.values()]
    texts = [f'Protein counts: {protein_count}<br>Stoichiometry:{"".join(["<br>   - " + protein_list[i] + ": " + str(protein_count[i]) for i in range(len(protein_count))])}<br>Frequency: {info["freq"]}' 
             for protein_count, info in zip(unique_coords.keys(), unique_coords.values())]
    
    # Add lines for paths connecting different protein counts
    max_freq = max(info['freq'] for info in unique_coords.values())
    for path in paths:
        for i in range(1, len(path.path)):
            start_protein_count = tuple(path.path[i-1].get_protein_count().get(p, 0) for p in protein_list)
            end_protein_count = tuple(path.path[i].get_protein_count().get(p, 0) for p in protein_list)
            
            if start_protein_count != end_protein_count:
                start_idx = list(stoichiometry_dict.keys()).index(start_protein_count)
                end_idx = list(stoichiometry_dict.keys()).index(end_protein_count)
                
                # Calculate line thickness based on frequency, with a maximum thickness
                freq = max(stoichiometry_dict[start_protein_count], stoichiometry_dict[end_protein_count])
                thickness = min(freq / max_freq * 10, 5)  # Max thickness of 5
                
                # Assign color based on frequency
                color_intensity = freq / max_freq  # Normalize frequency for coloring
                
                # Draw arrow to show direction
                fig.add_trace(go.Scatter(x=[xs[start_idx], xs[end_idx]], 
                                          y=[ys[start_idx], ys[end_idx]], 
                                          mode='lines+markers',
                                          line=dict(width=thickness, color=f'rgba({int(255 * color_intensity)}, 0, {255 - int(255 * color_intensity)}, 0.5)'),
                                          marker=dict(symbol='arrow', size=thickness * 4, angleref='previous'),
                                          hoverinfo='none'))
    
    # Add Stoichiometric points
    fig.add_trace(go.Scatter(x=xs, y=ys, 
                             mode='markers', 
                             marker=dict(size       = stoich_sizes,
                                         color      = freq_sizes,
                                         showscale  = True,
                                         colorscale = 'Viridis',
                                         colorbar   = dict(title="Stoich. Freq.")
                                    ),
                             text=texts,
                             hoverinfo='text',
                             name='Stoichiometries'))

    fig.update_layout(
        title=dict(
            text="Stoichiometric Space Exploration Plot",
            x=0.5,  # Centers the title
            xanchor="center"  # Anchors the title in the center
        ),
        xaxis_title="PCoA Dimension 1",
        yaxis_title="PCoA Dimension 2",
        showlegend=False,
        hovermode = "closest",
        hoverdistance = 500,
        xaxis=dict(scaleanchor="y", scaleratio=1),  # Lock aspect ratio
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    plot(fig)



def visualize_paths_3d(coords, paths, protein_counts, stoichiometry_dict, protein_list):
    fig = go.Figure()

    # Plot stoichiometries as points
    unique_coords = {}  # To store aggregated coords by protein count
    for idx, stoichiometry in enumerate(protein_counts):
        protein_count = tuple(stoichiometry)
        if protein_count not in unique_coords:
            unique_coords[protein_count] = {
                'x': coords[idx, 0],
                'y': coords[idx, 1],
                'z': coords[idx, 2],  # Added z-coordinate
                'freq': stoichiometry_dict[protein_count],
                'total_count': sum(protein_count)
            }
    
    # Scale factors
    stoich_scale_factor = 5
    
    # Prepare for plotting points
    xs = [info['x'] for info in unique_coords.values()]
    ys = [info['y'] for info in unique_coords.values()]
    zs = [info['z'] for info in unique_coords.values()]  # Added z-coordinates
    freq_sizes = [info['freq'] for info in unique_coords.values()]
    stoich_sizes = [info['total_count'] * stoich_scale_factor for info in unique_coords.values()]
    texts = [f'Protein counts: {protein_count}<br>Stoichiometry:{"".join(["<br>   - " + protein_list[i] + ": " + str(protein_count[i]) for i in range(len(protein_count))])}<br>Frequency: {info["freq"]}' 
             for protein_count, info in zip(unique_coords.keys(), unique_coords.values())]
    
    # Add lines for paths connecting different protein counts
    max_freq = max(info['freq'] for info in unique_coords.values())
    for path in paths:
        for i in range(1, len(path.path)):
            start_protein_count = tuple(path.path[i-1].get_protein_count().get(p, 0) for p in protein_list)
            end_protein_count = tuple(path.path[i].get_protein_count().get(p, 0) for p in protein_list)
            
            if start_protein_count != end_protein_count:
                start_idx = list(stoichiometry_dict.keys()).index(start_protein_count)
                end_idx = list(stoichiometry_dict.keys()).index(end_protein_count)
                
                # Calculate line thickness based on frequency, with a maximum thickness
                freq = max(stoichiometry_dict[start_protein_count], stoichiometry_dict[end_protein_count])
                thickness = min(freq / max_freq * 10, 5)  # Max thickness of 5
                
                # Assign color based on frequency
                color_intensity = freq / max_freq  # Normalize frequency for coloring
                
                # Draw arrow to show direction (3D)
                fig.add_trace(go.Scatter3d(x=[xs[start_idx], xs[end_idx]], 
                                           y=[ys[start_idx], ys[end_idx]], 
                                           z=[zs[start_idx], zs[end_idx]],  # Added z-coordinates
                                           mode='lines',
                                           line=dict(width=thickness, color=f'rgba({int(255 * color_intensity)}, 0, {255 - int(255 * color_intensity)}, 0.5)'),
                                           hoverinfo='none'))

    # Add Stoichiometric points (3D)
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs,  # Added z-coordinates
                               mode='markers', 
                               marker=dict(size=stoich_sizes,
                                           color=freq_sizes,
                                           showscale=True,
                                           colorscale='Viridis',
                                           colorbar=dict(title="Stoich. Freq.")
                                          ),
                               text=texts,
                               hoverinfo='text',
                               name='Stoichiometries'))

    fig.update_layout(
        scene=dict(
            xaxis_title="PCoA Dimension 1",
            yaxis_title="PCoA Dimension 2",
            zaxis_title="PCoA Dimension 3",
            aspectmode='cube'  # This ensures equal aspect ratio in 3D
        ),
        title=dict(
            text="3D Stoichiometric Space Exploration Plot",
            x=0.5,
            xanchor="center"
        ),
        showlegend=False,
        hovermode="closest",
    )
    
    plot(fig)


# Example usage with paths from your generated stoichiometries
combined_graph = mm_output['combined_graph']
paths = explore_stoichiometric_space(combined_graph, max_iterations=50, max_path_length=3)

# Analyze paths
for i, path in enumerate(paths):
    print(f"Path {i + 1}:")
    for j, stoichiometry in enumerate(path.path):
        print(f"  Stoichiometry {j + 1}: {stoichiometry.get_protein_count()} (Score: {stoichiometry.score})")

# Print the best stoichiometry
best_path = paths[0]
best_stoichiometry = best_path.path[-1]
print( "\nBest Stoichiometry:")
print(f"   - Proteins: {best_stoichiometry.get_protein_count()}")
print(f"   - Score: {best_stoichiometry.score}")


# Generate protein matrix
protein_matrix, protein_list, stoichiometry_dict = generate_protein_matrix(paths)

# Dimensionality reduction 
coords_2d = dimensionality_reduction_chunked(protein_matrix, chunk_size=100, metric='manhattan', dims = "2D")
coords_3d = dimensionality_reduction_chunked(protein_matrix, chunk_size=100, metric='manhattan', dims = "3D")

# Visualize paths in the 2D plane and in 3D space
visualize_paths_2d(coords_2d, paths, protein_matrix, stoichiometry_dict, protein_list)
visualize_paths_3d(coords_3d, paths, protein_matrix, stoichiometry_dict, protein_list)
