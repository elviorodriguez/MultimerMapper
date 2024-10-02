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



def explore_stoichiometric_space(combined_graph: igraph.Graph, max_iterations: int = 100) -> List[AssemblyPath]:
    assembly_paths = []
    
    for i in range(max_iterations):
        print(f"   Iteration {i}")
        try:
            root = initialize_stoichiometry(combined_graph)
            path = AssemblyPath(root)
            
            current = root
            while not current.is_leaf:
                child = generate_child(current, combined_graph)
                if child is None:
                    current.is_leaf = True
                    break
                path.add_stoichiometry(child)
                current = child
            
            assembly_paths.append(path)
        except ValueError as e:
            print(f"Error in iteration: {e}")
            continue
    
    return assembly_paths


# Explore the stoichiometric space
paths = explore_stoichiometric_space(combined_graph)

# Analyze the resulting paths
for i, path in enumerate(paths):
    print(f"Path {i + 1}:")
    for j, stoichiometry in enumerate(path.path):
        print(f"  Stoichiometry {j + 1}: {stoichiometry.get_protein_count()}")