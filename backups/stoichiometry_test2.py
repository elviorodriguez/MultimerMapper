import igraph
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from collections import Counter

class Stoichiometry:
    def __init__(self, graph: igraph.Graph, parent: Optional['Stoichiometry'] = None):
        self.graph = graph.copy()
        self.parent = parent
        self.children: List[Stoichiometry] = []
        self.is_root = parent is None
        self.is_leaf = False

    def add_child(self, child: 'Stoichiometry'):
        self.children.append(child)

    def add_protein(self, protein: str):
        self.graph.add_vertex(name=protein)

    def add_ppi(self, protein1: str, protein2: str, ppi_data: Dict):
        edge_name = tuple(sorted((protein1, protein2)))
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
        """Merge contiguous Stoichiometry instances with the same protein count."""
        i = 0
        while i < len(self.path) - 1:
            current_stoichiometry = self.path[i]
            next_stoichiometry = self.path[i + 1]
            
            if current_stoichiometry.get_protein_count() == next_stoichiometry.get_protein_count():

                # Remove the next stoichiometry from the path
                self.path.pop(i + 1)
            else:
                # Update parent-children relationship
                current_stoichiometry.children = next_stoichiometry
                next_stoichiometry.parent = current_stoichiometry
                i += 1

def select_random_weighted_ppi(protein: str, combined_graph: igraph.Graph) -> Tuple[str, Dict]:
    edges = combined_graph.incident(protein, mode="all")
    weights = []
    valid_edges = []
    
    for edge in edges:
        edge_data = combined_graph.es[edge].attributes()
        n_mers_models = edge_data['N_mers_data']['N_models']
        two_mers_models = edge_data['2_mers_data']['N_models']
        
        if isinstance(n_mers_models, (int, float)):
            n_mers_mean = n_mers_models
        else:
            n_mers_mean = np.nanmean(n_mers_models) if len(n_mers_models) > 0 else 0
        
        if isinstance(two_mers_models, (int, float)):
            two_mers_mean = two_mers_models
        else:
            two_mers_mean = np.nanmean(two_mers_models) if len(two_mers_models) > 0 else 0
        
        weight = (n_mers_mean + two_mers_mean) / 10
        
        if np.isfinite(weight) and weight > 0:
            weights.append(weight)
            valid_edges.append(edge)
    
    if not valid_edges:
        raise ValueError(f"No valid edges found for protein {protein}")
    
    selected_edge = random.choices(valid_edges, weights=weights)[0]
    edge_data = combined_graph.es[selected_edge].attributes()
    
    source, target = combined_graph.es[selected_edge].tuple
    partner = target if source == protein else source
    
    return combined_graph.vs[partner]['name'], edge_data

def initialize_stoichiometry(combined_graph: igraph.Graph) -> Stoichiometry:
    root_protein = random.choice(combined_graph.vs)['name']
    partner, ppi_data = select_random_weighted_ppi(root_protein, combined_graph)
    
    initial_graph = igraph.Graph()
    initial_graph.add_vertices([root_protein, partner])
    edge_name = tuple(sorted((root_protein, partner)))
    
    # Remove 'name' from ppi_data if it exists
    ppi_data_copy = ppi_data.copy()
    ppi_data_copy.pop('name', None)
    
    initial_graph.add_edge(root_protein, partner, name=edge_name, **ppi_data_copy)
    
    return Stoichiometry(initial_graph)

def generate_child(parent: Stoichiometry, combined_graph: igraph.Graph) -> Optional[Stoichiometry]:
    selected_protein = random.choice(parent.graph.vs)['name']
    partner, ppi_data = select_random_weighted_ppi(selected_protein, combined_graph)
    
    child = Stoichiometry(parent.graph, parent)
    edge_name = tuple(sorted((selected_protein, partner)))
    
    # Remove 'name' from ppi_data if it exists
    ppi_data_copy = ppi_data.copy()
    ppi_data_copy.pop('name', None)
    
    if partner in child.graph.vs['name']:
        # Check if there's an existing PPI between the proteins
        existing_edges = child.graph.es.select(_between=([selected_protein], [partner]))
        if existing_edges:
            for existing_edge in existing_edges:
                existing_cluster_n = existing_edge['valency']['cluster_n']
                new_cluster_n = ppi_data_copy['valency']['cluster_n']
                if existing_cluster_n == new_cluster_n:
                    # Check multivalency state
                    multivalency_states = ppi_data_copy.get('multivalency_states', {})
                    if not any(multivalency_states.values()):
                        return None
            # Add new interaction mode
            child.add_ppi(selected_protein, partner, ppi_data_copy)
        else:
            # Add new PPI between existing proteins
            child.add_ppi(selected_protein, partner, ppi_data_copy)
    else:
        # Add new protein and PPI
        child.add_protein(partner)
        child.add_ppi(selected_protein, partner, ppi_data_copy)
    
    # Check for dynamic PPIs
    update_dynamic_ppis(child, combined_graph)
    
    parent.add_child(child)
    return child

def update_dynamic_ppis(stoichiometry: Stoichiometry, combined_graph: igraph.Graph):
    for edge in stoichiometry.graph.es:
        source, target = edge.tuple
        config = tuple(sorted(stoichiometry.get_protein_count().items()))
        
        combined_edges = combined_graph.es.select(_between=([source], [target]))
        for combined_edge in combined_edges:
            if combined_edge['valency']['cluster_n'] == edge['valency']['cluster_n']:
                n_mers_data = combined_edge['N_mers_data']
                
                if config in n_mers_data['proteins_in_model'].tolist():
                    n_models = n_mers_data.loc[n_mers_data['proteins_in_model'] == config, 'N_models'].values[0]
                    p_add = n_models / 5
                    
                    if edge['dynamics'] == 'Positive':
                        if random.random() < p_add:
                            # Add positive dynamic PPI
                            stoichiometry.add_ppi(source, target, combined_edge.attributes())
                    elif edge['dynamics'] == 'Negative':
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



def explore_stoichiometric_space(combined_graph: igraph.Graph, max_iterations: int = 1000, max_children: int = 10000) -> List[AssemblyPath]:
    assembly_paths = []
    
    for i in range(max_iterations):
        print(f"Iteration {i}")
        try:
            root = initialize_stoichiometry(combined_graph)
            path = AssemblyPath(root)
            
            current = root
            children_generated = 0
            while not current.is_leaf and children_generated < max_children:
                # print(f"   Children generated: {children_generated}")
                child = generate_child(current, combined_graph)
                if child is None:
                    print("   Reached leaf node")
                    current.is_leaf = True
                    break
                path.add_stoichiometry(child)
                current = child
                children_generated += 1
                
                print(children_generated)
            
            if children_generated == max_children:
                print(f"   Reached maximum number of children ({max_children})")
                current.is_leaf = True
            
            # This is to reduce the redundancy of the stoichiometries
            path.merge_contiguous_stoichiometries()
            
            assembly_paths.append(path)
        except ValueError as e:
            print(f"Error in iteration: {e}")
            continue
    
    return assembly_paths

# Explore the stoichiometric space
combined_graph = mm_output['combined_graph']
paths = explore_stoichiometric_space(combined_graph, max_iterations= 1, max_children= 10000)


# Analyze and merge paths
for i, path in enumerate(paths):

    print(f"After merging Path {i + 1}:")
    for j, stoichiometry in enumerate(path.path):
        print(f"  Stoichiometry {j + 1}: {stoichiometry.get_protein_count()}")



############################### Visualizations ################################
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
import plotly.graph_objects as go
from plotly.offline import plot
from collections import defaultdict


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


def dimensionality_reduction_chunked(protein_matrix, chunk_size=1000, metric='manhattan', epsilon=1e-9):
    """
    Perform dimensionality reduction in chunks to avoid memory overload, 
    with a small epsilon added to the distance matrix to avoid division by zero.
    
    Args:
    protein_matrix (np.ndarray): The matrix of protein data.
    chunk_size (int): The size of each chunk to process.
    metric (str): The distance metric to use for pairwise distances.
    epsilon (float): Small value added to avoid division by zero.

    Returns:
    coords (list): 2D coordinates for each protein.
    """
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


def visualize_paths(coords, paths, protein_counts, stoichiometry_dict):
    fig = go.Figure()

    # Plot stoichiometries as points
    unique_coords = {}  # To store aggregated coords by protein count
    for idx, stoichiometry in enumerate(protein_counts):
        protein_count = tuple(stoichiometry)
        if protein_count not in unique_coords:
            unique_coords[protein_count] = {
                'x': coords[idx, 0],
                'y': coords[idx, 1],
                'freq': stoichiometry_dict[protein_count]
            }
    
    # Prepare for plotting points
    xs = [info['x'] for info in unique_coords.values()]
    ys = [info['y'] for info in unique_coords.values()]
    sizes = [info['freq'] * 20 for info in unique_coords.values()]  # Size proportional to frequency
    texts = [f"Protein counts: {protein_count}<br>Frequency: {info['freq']}" 
             for protein_count, info in zip(unique_coords.keys(), unique_coords.values())]

    fig.add_trace(go.Scatter(x=xs, y=ys, 
                             mode='markers', 
                             marker=dict(size=10, color=sizes, showscale=True, colorscale='Viridis'),
                             text=texts,
                             name='Stoichiometries'))
    
    # Add lines for paths connecting different protein counts
    for path in paths:
        for i in range(1, len(path.path)):
            start_protein_count = tuple(path.path[i-1].get_protein_count().get(p, 0) for p in protein_counts[0])
            end_protein_count = tuple(path.path[i].get_protein_count().get(p, 0) for p in protein_counts[0])
            
            if start_protein_count != end_protein_count:
                start_idx = list(unique_coords.keys()).index(start_protein_count)
                end_idx = list(unique_coords.keys()).index(end_protein_count)
                
                # Draw line between them, using frequency as line width
                fig.add_trace(go.Scatter(x=[xs[start_idx], xs[end_idx]], 
                                         y=[ys[start_idx], ys[end_idx]], 
                                         mode='lines', 
                                         line=dict(width=max(stoichiometry_dict[start_protein_count], 
                                                            stoichiometry_dict[end_protein_count]) * 2)))

    fig.update_layout(title="Stoichiometries and Path Transitions",
                      xaxis_title="PCoA Dimension 1",
                      yaxis_title="PCoA Dimension 2",
                      showlegend=False)
    
    plot(fig)


# Example usage with paths from your generated stoichiometries
paths = explore_stoichiometric_space(combined_graph, max_iterations=100, max_children=10000)

# Generate protein matrix
protein_matrix, protein_list, stoichiometry_dict = generate_protein_matrix(paths)

# Dimensionality reduction
coords = dimensionality_reduction_chunked(protein_matrix, chunk_size=100, metric='manhattan')

# Visualize paths in the 2D plane
visualize_paths(coords, paths, protein_matrix, stoichiometry_dict)
