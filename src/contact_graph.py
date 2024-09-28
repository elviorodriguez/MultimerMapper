
import numpy as np
import pandas as pd
import igraph
import py3Dmol
import itertools
import random
from typing import Dict, List, Tuple, Optional
from collections import Counter
from logging import Logger
from Bio import PDB
from copy import deepcopy
import io
import string
from itertools import cycle
import webbrowser
from scipy.spatial.distance import pdist, squareform
import plotly.graph_objects as go
from plotly.offline import plot
from collections import defaultdict

from cfg.default_settings import PT_palette, DOMAIN_COLORS_RRC, SURFACE_COLORS_RRC
from utils.pdb_utils import center_of_mass, rotate_points
from src.detect_domains import format_domains_df_as_no_loops_domain_clusters
from utils.progress_bar import print_progress_bar

##############################################################################################################
########################################### Contact classification ###########################################
##############################################################################################################

def generate_contact_classification_matrix(valency):
    # Extract necessary data from the valency dictionary
    was_tested_in_2mers = valency['was_tested_in_2mers']
    was_tested_in_Nmers = valency['was_tested_in_Nmers']
    average_2mers_matrix = valency['average_2mers_matrix']
    average_Nmers_matrix = valency['average_Nmers_matrix']

    # Create boolean matrices for presence in 2mers and Nmers
    is_present_in_2mers = average_2mers_matrix > 0
    is_present_in_Nmers = average_Nmers_matrix > 0

    # Initialize the classification matrix with 5 (No_Contact)
    shape = average_2mers_matrix.shape
    classification_matrix = np.full(shape, 0, dtype=np.int8)

    # Apply classification rules
    if was_tested_in_2mers and was_tested_in_Nmers:
        classification_matrix[is_present_in_2mers & is_present_in_Nmers]  = 1   # Static
        classification_matrix[~is_present_in_2mers & is_present_in_Nmers] = 2   # Positive
        classification_matrix[is_present_in_2mers & ~is_present_in_Nmers] = 3   # Negative

    elif was_tested_in_2mers and not was_tested_in_Nmers:
        classification_matrix[is_present_in_2mers] = 4                          # No_Nmers_Data
        
    elif not was_tested_in_2mers and was_tested_in_Nmers:
        classification_matrix[is_present_in_Nmers] = 5                          # No_2mers_Data
        
    # No need for else case as 0 (No_Contact) is the default

    # Add the new matrix to the valency dictionary
    valency['contact_classification_matrix'] = classification_matrix

    return valency

def add_contact_classification_matrix(combined_graph):
    '''
    Adds the key contact_classification_matrix to the edges attribute "valency"
    containing the classification of each contact as a matrix. The encoding is 
    stored in CLASSIFICATION_ENCODING variable of this module.
    '''
    for edge in combined_graph.es:
        edge['valency'] = generate_contact_classification_matrix(edge['valency'])

CLASSIFICATION_ENCODING = {
    0: 'No Contact',
    1: 'Static',
    2: 'Positive',
    3: 'Negative',
    4: 'No Nmers Data',
    5: 'No 2mers Data'
}

# Define color scheme for different residue-residue contact classifications
rrc_classification_colors = {
    1: PT_palette['gray'],      # Static
    2: PT_palette['green'],     # Positive
    3: PT_palette['red'],       # Negative
    4: PT_palette['orange'],    # No Nmers Data
    5: PT_palette['yellow']     # No 2mers Data
}

# Color palette for network representation (not implemented)
default_color_palette = {
    "Red":            ["#ffebee", "#ffcdd2", "#ef9a9a", "#e57373", "#ef5350", "#f44336", "#e53935", "#d32f2f", "#c62828", "#ff8a80", "#ff5252", "#d50000", "#f44336", "#ff1744", "#b71c1c"],
    "Green":          ["#e8f5e9", "#c8e6c9", "#a5d6a7", "#81c784", "#66bb6a", "#4caf50", "#43a047", "#388e3c", "#2e7d32", "#b9f6ca", "#69f0ae", "#00e676", "#4caf50", "#00c853", "#1b5e20"],
    "Blue":           ["#e3f2fd", "#bbdefb", "#90caf9", "#64b5f6", "#42a5f5", "#2196f3", "#1e88e5", "#1976d2", "#1565c0", "#82b1ff", "#448aff", "#2979ff", "#2962ff", "#2196f3", "#0d47a1"],
    "Yellow":         ["#fffde7", "#fff9c4", "#fff59d", "#fff176", "#ffee58", "#ffeb3b", "#fdd835", "#fbc02d", "#f9a825", "#ffff8d", "#ffff00", "#ffea00", "#ffd600", "#ffeb3b", "#f57f17"],
    "Lime":           ["#f9fbe7", "#f0f4c3", "#e6ee9c", "#dce775", "#d4e157", "#cddc39", "#c0ca33", "#afb42b", "#9e9d24", "#f4ff81", "#eeff41", "#c6ff00", "#aeea00", "#cddc39", "#827717"],
    "Orange":         ["#fff3e0", "#ffe0b2", "#ffcc80", "#ffb74d", "#ffa726", "#ff9800", "#fb8c00", "#f57c00", "#ef6c00", "#ffd180", "#ffab40", "#ff9100", "#ff6d00", "#ff9800", "#e65100"],
    "Purple":         ["#f3e5f5", "#e1bee7", "#ce93d8", "#ba68c8", "#ab47bc", "#9c27b0", "#8e24aa", "#7b1fa2", "#6a1b9a", "#ea80fc", "#e040fb", "#d500f9", "#aa00ff", "#9c27b0", "#4a148c"],
    "Light_Blue":     ["#e1f5fe", "#b3e5fc", "#81d4fa", "#4fc3f7", "#29b6f6", "#03a9f4", "#039be5", "#0288d1", "#0277bd", "#80d8ff", "#40c4ff", "#00b0ff", "#0091ea", "#03a9f4", "#01579b"],
    "Teal":           ["#e0f2f1", "#b2dfdb", "#80cbc4", "#4db6ac", "#26a69a", "#009688", "#00897b", "#00796b", "#00695c", "#a7ffeb", "#64ffda", "#1de9b6", "#00bfa5", "#009688", "#004d40"],
    "Light_Green":    ["#f1f8e9", "#dcedc8", "#c5e1a5", "#aed581", "#9ccc65", "#8bc34a", "#7cb342", "#689f38", "#558b2f", "#ccff90", "#b2ff59", "#76ff03", "#64dd17", "#8bc34a", "#33691e"],
    "Amber":          ["#fff8e1", "#ffecb3", "#ffe082", "#ffd54f", "#ffca28", "#ffc107", "#ffb300", "#ffa000", "#ff8f00", "#ffe57f", "#ffd740", "#ffc400", "#ffab00", "#ffc107", "#ff6f00"],
    "Deep_Orange":    ["#fbe9e7", "#ffccbc", "#ffab91", "#ff8a65", "#ff7043", "#ff5722", "#f4511e", "#e64a19", "#d84315", "#ff9e80", "#ff6e40", "#ff3d00", "#dd2c00", "#ff5722", "#bf360c"],
    "Pink":           ["#fce4ec", "#f8bbd0", "#f48fb1", "#f06292", "#ec407a", "#e91e63", "#d81b60", "#c2185b", "#ad1457", "#ff80ab", "#ff4081", "#f50057", "#c51162", "#e91e63", "#880e4f"],
    "Deep_Purple":    ["#ede7f6", "#d1c4e9", "#b39ddb", "#9575cd", "#7e57c2", "#673ab7", "#5e35b1", "#512da8", "#4527a0", "#b388ff", "#7c4dff", "#651fff", "#6200ea", "#673ab7", "#311b92"],
    "Cyan":           ["#e0f7fa", "#b2ebf2", "#80deea", "#4dd0e1", "#26c6da", "#00bcd4", "#00acc1", "#0097a7", "#00838f", "#84ffff", "#18ffff", "#00e5ff", "#00b8d4", "#00bcd4", "#006064"],
    "Indigo":         ["#e8eaf6", "#c5cae9", "#9fa8da", "#7986cb", "#5c6bc0", "#3f51b5", "#3949ab", "#303f9f", "#283593", "#8c9eff", "#536dfe", "#3d5afe", "#304ffe", "#3f51b5", "#1a237e"],
}


###############################################################################################################
############################################### Miscellaneous #################################################
###############################################################################################################

# Define a list of valid chain IDs
VALID_CHAIN_IDS = list(string.ascii_uppercase) + list(string.digits)

def clean_chain(original_chain):
    """
    Takes a PDB.Chain.Chain object and returns a new 'clean' chain object,
    detached from any previous structure.
    
    Parameters:
    original_chain (PDB.Chain.Chain): The original chain object to be cleaned.
    
    Returns:
    PDB.Chain.Chain: A new 'clean' chain object.
    """
    # Create a new structure and model
    new_structure = PDB.Structure.Structure("clean_structure")
    new_model = PDB.Model.Model(0)
    
    # Deepcopy the original chain to ensure it's detached from the original structure
    new_chain = deepcopy(original_chain)
    
    # Remove the chain from any prior associations
    if new_chain.parent:
        new_chain.parent.detach_child(new_chain.id)
    
    # Assign a new chain ID if needed
    new_chain.id = 'A'
    
    # Add the new chain to the model and the model to the structure
    new_model.add(new_chain)
    new_structure.add(new_model)
    
    return new_model['A']


def generate_unique_colors(n, palette):

    if not isinstance(palette, cycle):
        palette = cycle(palette)
    
    # Use cycle to repeat the palette if needed and select 'n' colors
    colors = [next(cycle(palette)) for _ in range(n)]
    
    return colors

def get_centroid(protein, residue_index, centroid_cache):
    if (protein.get_ID(), residue_index) not in centroid_cache:
        centroid_cache[(protein.get_ID(), residue_index)] = protein.get_res_centroids_xyz([residue_index])[0]
    return centroid_cache[(protein.get_ID(), residue_index)]

def get_ca(protein, residue_index, ca_cache):
    if (protein.get_ID(), residue_index) not in ca_cache:
        ca_cache[(protein.get_ID(), residue_index)] = protein.get_res_CA_xyz(res_list=[residue_index])[0]
    return ca_cache[(protein.get_ID(), residue_index)]


###############################################################################################################
################################### Helper functions for optimized layout #####################################
###############################################################################################################

def get_protein_radius(protein):
    """
    Calculate the radius of the protein's bounding sphere.
    
    Args:
        protein: A protein object with get_CM() and get_res_CA_xyz() methods.
    
    Returns:
        float: The radius of the protein's bounding sphere plus a 5 Angstrom buffer.
    """
    cm = protein.get_CM()
    residue_positions = protein.get_res_CA_xyz()
    distances = np.linalg.norm(residue_positions - cm, axis=1)
    return np.max(distances) + 5  # Add 5 Angstroms as buffer

def safe_normalize(v):
    """
    Safely normalize a vector, returning a zero vector if the norm is zero.
    
    Args:
        v (np.array): The vector to normalize.
    
    Returns:
        np.array: The normalized vector, or a zero vector if the input has zero norm.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.zeros_like(v)
    return v / norm

def get_connected_components(proteins, ppis, logger):
    adj_list = {p.get_ID(): set() for p in proteins}
    for ppi in ppis:
        p1, p2 = ppi.get_tuple_pair()
        adj_list[p1].add(p2)
        adj_list[p2].add(p1)
    
    visited = set()
    components = []
    for protein in proteins:
        if protein.get_ID() not in visited:
            component = set()
            stack = [protein.get_ID()]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    component.add(node)
                    stack.extend(adj_list[node] - visited)
            components.append(component)
    
    logger.info(f"Found {len(components)} connected components")
    return components

def initialize_positions(proteins, ppis, components):
    """
    Initialize the positions of proteins for the layout algorithm.
    
    Args:
        proteins (list): List of protein objects.
        ppis (list): List of protein-protein interactions.
        components (list): List of connected components in the network.
    
    Returns:
        tuple: A tuple containing:
            - np.array: Initial positions of proteins.
            - np.array: Radii of proteins.
    """
    n = len(proteins)
    positions = np.random.rand(n, 3) * 1000
    radii = np.array([get_protein_radius(protein) for protein in proteins])
    
    if len(components) == 1:
        # For a single component, use a 3D grid layout
        grid_size = int(np.ceil(n**(1/3)))
        for i, protein in enumerate(proteins):
            x = (i % grid_size) * 200
            y = ((i // grid_size) % grid_size) * 200
            z = (i // (grid_size**2)) * 200
            positions[i] = [x, y, z]
    else:
        # Use the existing component-based initialization
        for i, component in enumerate(components):
            component_indices = [proteins.index(p) for p in proteins if p.get_ID() in component]
            component_center = np.array([1000 * np.cos(2 * np.pi * i / len(components)),
                                         1000 * np.sin(2 * np.pi * i / len(components)),
                                         0])
            for idx in component_indices:
                positions[idx] = component_center + np.random.rand(3) * 100
    
    # Adjust positions to avoid initial overlaps
    for _ in range(100):
        distances = squareform(pdist(positions))
        for i in range(n):
            for j in range(i+1, n):
                min_distance = radii[i] + radii[j]
                if distances[i,j] < min_distance:
                    direction = positions[j] - positions[i]
                    move = (min_distance - distances[i,j]) * 0.5
                    positions[i] -= safe_normalize(direction) * move
                    positions[j] += safe_normalize(direction) * move
    
    return positions, radii


def compute_forces(positions, radii, proteins, ppis, components):
    """
    Compute forces acting on proteins based on their positions and interactions.
    
    Args:
        positions (np.array): Current positions of proteins.
        radii (np.array): Radii of proteins.
        proteins (list): List of protein objects.
        ppis (list): List of protein-protein interactions.
        components (list): List of connected components in the network.
    
    Returns:
        np.array: Forces acting on each protein.
    """
    n = len(proteins)
    forces = np.zeros((n, 3))
    
    distances = squareform(pdist(positions))
    
    # Repulsive forces between all proteins
    for i in range(n):
        for j in range(i+1, n):
            if distances[i,j] > 0:
                min_distance = radii[i] + radii[j]
                if distances[i,j] < min_distance:
                    force = 1000 * (min_distance - distances[i,j]) / min_distance
                    direction = safe_normalize(positions[i] - positions[j])
                    forces[i] += force * direction
                    forces[j] -= force * direction
    
    # Attractive forces for proteins connected by PPIs
    for ppi in ppis:
        p1 = proteins.index(ppi.get_protein_1())
        p2 = proteins.index(ppi.get_protein_2())
        
        distance = distances[p1, p2]
        if distance > 0:
            min_distance = radii[p1] + radii[p2]
            if distance > min_distance:
                force = 0.2 * (distance - min_distance)
                direction = safe_normalize(positions[p2] - positions[p1])
                forces[p1] += force * direction
                forces[p2] -= force * direction
    
    # Cohesive forces within components
    for component in components:
        component_indices = [proteins.index(p) for p in proteins if p.get_ID() in component]
        component_center = np.mean(positions[component_indices], axis=0)
        for idx in component_indices:
            direction = component_center - positions[idx]
            distance = np.linalg.norm(direction)
            if distance > 0:
                force = 0.01 * distance
                forces[idx] += force * safe_normalize(direction)
    
     # Improved force to separate PPI lines and avoid protein backbones
    for i, ppi1 in enumerate(ppis):
        p1_1 = proteins.index(ppi1.get_protein_1())
        p1_2 = proteins.index(ppi1.get_protein_2())
        line1 = positions[p1_2] - positions[p1_1]
        line1_dir = safe_normalize(line1)
        
        for j, ppi2 in enumerate(ppis[i+1:], start=i+1):
            p2_1 = proteins.index(ppi2.get_protein_1())
            p2_2 = proteins.index(ppi2.get_protein_2())
            line2 = positions[p2_2] - positions[p2_1]
            line2_dir = safe_normalize(line2)
            
            # Calculate the closest distance between the two lines
            v1 = line1
            v2 = line2
            w0 = positions[p1_1] - positions[p2_1]
            
            a = np.dot(v1, v1)
            b = np.dot(v1, v2)
            c = np.dot(v2, v2)
            d = np.dot(v1, w0)
            e = np.dot(v2, w0)
            
            denominator = a*c - b*b
            if abs(denominator) > 1e-6:  # Avoid division by zero
                sc = (b*e - c*d) / denominator
                tc = (a*e - b*d) / denominator
            else:
                sc = tc = 0
            
            sc = np.clip(sc, 0, 1)
            tc = np.clip(tc, 0, 1)
            
            closest_p1 = positions[p1_1] + sc * v1
            closest_p2 = positions[p2_1] + tc * v2
            
            separation = closest_p2 - closest_p1
            distance = np.linalg.norm(separation)
            
            min_separation = radii[p1_1] + radii[p1_2] + radii[p2_1] + radii[p2_2]
            if distance < min_separation:
                force = 20 * (min_separation - distance)  # Increased force strength
                direction = safe_normalize(separation)
                
                # Apply forces to the proteins
                forces[p1_1] -= force * direction * (1 - sc)
                forces[p1_2] -= force * direction * sc
                forces[p2_1] += force * direction * (1 - tc)
                forces[p2_2] += force * direction * tc
                
                # Additional force to separate lines at origin
                origin_separation = positions[p2_1] - positions[p1_1]
                origin_distance = np.linalg.norm(origin_separation)
                if origin_distance < min_separation:
                    origin_force = 10 * (min_separation - origin_distance)
                    origin_direction = safe_normalize(origin_separation)
                    forces[p1_1] -= origin_force * origin_direction
                    forces[p2_1] += origin_force * origin_direction
        
        # Force to prevent lines from passing through protein backbones
        for k, protein in enumerate(proteins):
            if k != p1_1 and k != p1_2:
                protein_radius = radii[k]
                protein_to_line = np.cross(line1, positions[k] - positions[p1_1])
                distance_to_line = np.linalg.norm(protein_to_line) / np.linalg.norm(line1)
                
                if distance_to_line < protein_radius + 15:  # Add some buffer
                    force = 15 * (protein_radius + 5 - distance_to_line)
                    direction = safe_normalize(np.cross(line1_dir, protein_to_line))
                    forces[p1_1] += force * direction
                    forces[p1_2] += force * direction
                    forces[k] -= force * direction
    
    return forces

def apply_layout(proteins, ppis, network, iterations=10000, logger=None):
    """
    Apply the layout algorithm to position proteins in 3D space.
    
    Args:
        proteins (list): List of protein objects.
        ppis (list): List of protein-protein interactions.
        network: Network object containing logger and other network information.
        iterations (int): Number of iterations for the layout algorithm.
        logger: Logger object for output messages.
    
    Returns:
        np.array: Optimized positions of proteins.
    """
    components = get_connected_components(proteins, ppis, logger)
    positions, radii = initialize_positions(proteins, ppis, components)
    network.logger.info(f'   - Nº of iterations to perform: {iterations}')
    network.logger.info(f'   - Number of connected components: {len(components)}')
    
    for i in range(iterations):
        if i % 1000 == 0:
            network.logger.info(f'   - Iteration {i}...')
        
        forces = compute_forces(positions, radii, proteins, ppis, components)
        
        # Use an adaptive damping factor with slower decay
        damping_factor = 1 - (i / iterations)**0.5
        positions += forces * 0.05 * damping_factor  # Reduced step size for finer control
        
        # Ensure components don't drift too far apart
        if len(components) > 1:
            for component in components:
                component_indices = [proteins.index(p) for p in proteins if p.get_ID() in component]
                component_center = np.mean(positions[component_indices], axis=0)
                global_center = np.mean(positions, axis=0)
                max_distance = 1000  # Maximum distance from global center
                if np.linalg.norm(component_center - global_center) > max_distance:
                    direction = safe_normalize(component_center - global_center)
                    for idx in component_indices:
                        positions[idx] = global_center + direction * max_distance + (positions[idx] - component_center)
    
    # Final adjustment to bring interacting proteins closer together
    buffer_distance = 20  # Angstroms
    network.logger.info(f'   - Performing final adjustment...')
    
    # Create a dictionary of interacting protein pairs
    interacting_pairs = set()
    for ppi in ppis:
        p1 = proteins.index(ppi.get_protein_1())
        p2 = proteins.index(ppi.get_protein_2())
        interacting_pairs.add((min(p1, p2), max(p1, p2)))
    
    for _ in range(100):  # Perform 100 iterations of final adjustment
        distances = squareform(pdist(positions))
        for i, j in interacting_pairs:
            current_distance = distances[i,j]
            min_distance = radii[i] + radii[j] + buffer_distance
            if current_distance > min_distance:
                direction = safe_normalize(positions[j] - positions[i])
                move = (current_distance - min_distance) * 0.1  # Move 10% of the excess distance
                positions[i] += direction * move * 0.5
                positions[j] -= direction * move * 0.5
    
    return positions

def optimize_layout(network):
    """
    Optimize the layout of proteins in the network.
    
    Args:
        network: Network object containing proteins, PPIs, and logger.
    """
    proteins = network.get_proteins()
    ppis = network.get_ppis()
    
    optimized_positions = apply_layout(proteins, ppis, network, logger=network.logger)
    
    for i, protein in enumerate(proteins):
        protein.translate(optimized_positions[i] - protein.get_CM())
    
    for protein in proteins:
        try:
            protein.rotate2all(ppis)
        except Exception as e:
            network.logger.error(f"Error rotating protein {protein.get_ID()}: {str(e)}")  

# ----------------------------------------------------------------------------------------------------------- #
###############################################################################################################
#                                                                                                             #
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ CLASSES @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
#                                                                                                             #
###############################################################################################################
# ----------------------------------------------------------------------------------------------------------- #

###############################################################################################################
################################################ Class Residue ################################################
###############################################################################################################

class Residue:
    """
    Represents a residue in a protein, along with its interactions and structural properties.

    Attributes:
        index (int): The index of the residue.
        name (str): The name of the residue.
        pLDDT (float): The predicted LDDT (Local Distance Difference Test) score of the residue.
        interactions (dict): Dictionary of interactions where the key is a tuple (ppi_id, cluster_n),
                             and the value is a set of partner protein IDs.
    """
    def __init__(self, index, name, pLDDT):
        """
        Initializes a Residue object.

        Parameters:
            index (int): Residue index.
            name (str): Residue name.
            pLDDT (float): Predicted LDDT score of the residue.
        """
        self.index = index
        self.name = name
        self.pLDDT = pLDDT
        self.interactions = {}  # Dict of {ppi_id: set(partner_protein_ids)}

    def get_index(self):
        """Returns the index of the residue."""
        return self.index
    
    def get_name(self):
        """Returns the name of the residue."""
        return self.name

    def get_plddt(self):
        """Returns the predicted LDDT score of the residue."""
        return self.pLDDT
    
    def get_interactions(self):
        """Returns the dictionary of interactions for this residue."""
        return self.interactions
    
    @property
    def surface_id(self):
        """
        Returns the surface identifier for the residue if it's part of interaction group A (not co-occupied by different PPIs).
        
        Returns:
            tuple: (ppi_id, cluster_n) for group A residues, None otherwise.
        """
        if self.interaction_group == "A":
            # Assuming self.interactions stores tuples of (partner_protein_id, ppi_id, cluster_n)
            return list(self.interactions.keys())[0]  # Return (ppi_id, cluster_n)
        return None  # For groups B and C, we don't assign a specific surface

    @property
    def interaction_group(self):
        """
        Determines the interaction group for the residue based on its interactions.

        Returns:
            str: "A" for single protein/single mode, "B" for single protein/multiple modes, 
                 "C" for multiple proteins, and "Non-interacting" if no interactions exist.
        """
        if not self.interactions:
            return "Non-interacting"
        unique_proteins = set(protein_id for protein_ids in self.interactions.values() for protein_id in protein_ids)
        if len(unique_proteins) > 1:
            return "C"  # Multiple proteins
        elif len(self.interactions) > 1:
            return "B"  # Single protein, multiple modes
        else:
            return "A"  # Single protein, single mode
    
    def add_interaction(self, partner_protein_id, ppi_id, cluster_n):
        """
        Adds an interaction between the residue and a partner protein.

        Parameters:
            partner_protein_id (str): The ID of the partner protein.
            ppi_id (str): The ID of the protein-protein interaction (PPI).
            cluster_n (int): The cluster number of the interaction.
        """
        key = (ppi_id, cluster_n)
        if key not in self.interactions:
            self.interactions[key] = set()
        self.interactions[key].add(partner_protein_id)


###############################################################################################################
################################################ Class Surface ################################################
###############################################################################################################

class Surface:
    """
    Represents the surface of a protein, including its residues and interactions.

    Attributes:
        protein_id (str): The ID of the protein.
        residues (dict): A dictionary where the key is the residue index and the value is a Residue object.
        surfaces (dict): A dictionary where the key is a tuple (ppi_id, cluster_n) and 
                         the value is a set of residue indices.
    """
    def __init__(self, protein_id):
        """
        Initializes a Surface object.

        Parameters:
            protein_id (str): The ID of the protein.
        """
        self.protein_id = protein_id
        self.residues = {}  # key: residue index, value: Residue object
        self.surfaces = {}  # key: ppi_id, value: set of residue indices

    def add_residue(self, index, name, pLDDT):
        """
        Adds a residue to the surface.

        Parameters:
            index (int): The index of the residue.
            name (str): The name of the residue.
            pLDDT (float): The predicted LDDT score of the residue.
        """
        if index not in self.residues:
            self.residues[index] = Residue(index, name, pLDDT)

    def add_interaction(self, residue_index, partner_protein_id, ppi_id, cluster_n):
        """
        Adds an interaction between a residue and a partner protein.

        Parameters:
            residue_index (int): The index of the residue.
            partner_protein_id (str): The ID of the partner protein.
            ppi_id (str): The ID of the protein-protein interaction (PPI).
            cluster_n (int): The cluster number of the interaction.
        """
        if residue_index in self.residues:
            self.residues[residue_index].add_interaction(partner_protein_id, ppi_id, cluster_n)
            key = (ppi_id, cluster_n)
            if key not in self.surfaces:
                self.surfaces[key] = set()
            self.surfaces[key].add(residue_index)

    def get_interacting_residues(self):
        """
        Returns all residues involved in interactions.

        Returns:
            dict: A dictionary where the key is the residue index and the value is the Residue object 
                  (for residues with interactions only).
        """
        return {idx: res for idx, res in self.residues.items() if res.interactions}

    def get_residues_by_group(self):
        """
        Returns residues grouped by interaction type (A, B, C).

        Returns:
            dict: A dictionary with keys "A", "B", "C", where each value is a list of Residue objects.
        """
        groups = {"A": [], "B": [], "C": []}
        for residue in self.residues.values():
            if residue.interaction_group in groups:
                groups[residue.interaction_group].append(residue)
        return groups

    def get_surfaces(self):
        """
        Returns all surfaces (interactions) on the protein.

        Returns:
            dict: A dictionary where the key is a tuple (ppi_id, cluster_n) and 
                  the value is a set of residue indices involved in the interaction.
        """
        return self.surfaces

###############################################################################################################
################################################ Class Protein ################################################
###############################################################################################################

class Protein(object):
    """
    Represents a protein and its structural and interaction properties.

    Attributes:
        unique_ID (int): Unique identifier for the protein.
        logger (Logger): Logger object to log operations.
        vertex (igraph.Vertex): Graph vertex representing the protein.
        ID (str): Protein ID.
        name (str): Protein name.
        PDB_chain (PDB.Chain.Chain): Chain object from the PDB structure.
        seq (str): Protein sequence.
        res_names (list[str]): List of residue names.
        res_pLDDT (list[float]): List of pLDDT scores for each residue.
        domains (tuple[int]): Domains of the protein.
        RMSD_df (pd.DataFrame): RMSD values for the protein.
        chain_ID (str): ID of the protein chain.
        surface (Surface): Surface object representing the protein's surface interactions.
    """

    def __init__(self, graph_vertex: igraph.Vertex, unique_ID: int, logger: Logger, verbose: bool = True):
        """
        Initializes a Protein object.

        Parameters:
            graph_vertex (igraph.Vertex): Vertex representing the protein in the combined_graph.
            unique_ID (int): Unique identifier for the protein.
            logger (Logger): Logger object for logging.
        """

        # Progress
        if verbose:
            logger.info(f"   - Creating object of class Protein: {graph_vertex['name']}")

        # Attributes
        self.unique_ID  : int               = unique_ID
        self.logger     : Logger            = logger
        self.vertex     : igraph.Vertex     = graph_vertex
        self.ID         : str               = graph_vertex['name']
        self.name       : str               = graph_vertex['IDs']
        self.PDB_chain  : PDB.Chain.Chain   = clean_chain(graph_vertex['ref_PDB_chain'])
        self.seq        : str               = graph_vertex['seq']
        self.res_names  : list[str]         = [AA + str(i + 1) for i, AA in enumerate(self.seq)]
        self.res_pLDDT  : list[float]       = [res["CA"].get_bfactor() for res in self.PDB_chain.get_residues()]
        self.domains    : tuple[int]        = format_domains_df_as_no_loops_domain_clusters(graph_vertex['domains_df'])
        self.RMSD_df    : pd.DataFrame      = self.vertex['RMSD_df']
        self.chain_ID   : str               = self.PDB_chain.id

        # Translate PDB to the origin
        self._translate_to_origin()

        self.surface = Surface(self.ID)
        for i, aa in enumerate(self.seq):
            self.surface.add_residue(i, aa + str(i + 1), self.res_pLDDT[i])
    
    # -------------------------------------------------------------------------------------
    # -------------------------------------- Adders ---------------------------------------
    # -------------------------------------------------------------------------------------

    def add_interaction(self, residue_index, partner_protein_id, ppi_id, cluster_n):
        """
        Adds an interaction between a residue of the protein and a partner protein.

        Parameters:
            residue_index (int): The index of the residue involved in the interaction.
            partner_protein_id (str): The ID of the partner protein.
            ppi_id (str): The PPI identifier.
            cluster_n (int): Cluster number of the interaction.
        """
        self.surface.add_interaction(residue_index, partner_protein_id, ppi_id, cluster_n)

    # -------------------------------------------------------------------------------------
    # -------------------------------------- Getters --------------------------------------
    # -------------------------------------------------------------------------------------
    
    def get_unique_ID(self):
        """Returns the unique ID of the protein."""
        return self.unique_ID
    
    def get_ID(self):
        """Returns the ID of the protein."""
        return self.ID
    
    def get_seq(self):
        """Returns the sequence of the protein."""
        return self.seq
    
    def get_name(self):
        """Returns the name of the protein."""
        return self.name
    
    def get_CM(self):
        """Returns the center of mass of the protein chain."""
        return self.PDB_chain.center_of_mass()
    
    def get_res_pLDDT(self, res_list = None) -> list[float]:
        """
        Returns the pLDDT values for residues.

        Parameters:
            res_list (list, optional): List of residue indices (zero-indexed). 
                                       If provided, returns the pLDDT values for these residues only.

        Returns:
            list[float]: List of pLDDT values.
        """
        if res_list is not None: 
            return[self.res_pLDDT[res] for res in res_list]
        return self.res_pLDDT
    
    def get_res_centroids_xyz(self, res_list = None) -> list[np.array]:
        """
        Returns the centroid coordinates for residues.

        Parameters:
            res_list (list, optional): List of residue indices (zero-indexed). 
                                       If provided, returns centroid coordinates for these residues only.

        Returns:
            list[np.array]: List of centroid coordinates for residues.
        """

        # Compute residue centroids
        res_xyz: list[np.array] = [ res.center_of_mass() for res  in self.PDB_chain.get_residues() ]

        # Return only the subset
        if res_list is not None: 
            return [res_xyz[res] for res in res_list]
        
        # Return everything
        return res_xyz
    
    def get_res_CA_xyz(self, res_list = None) -> list[np.array]:
        """
        Returns the CA atom coordinates for residues.

        Parameters:
            res_list (list, optional): List of residue indices (zero-indexed). 
                                       If provided, returns CA coordinates for these residues only.

        Returns:
            list[np.array]: List of CA coordinates for residues.
        """

        # Compute residue centroids
        CA_xyz: list[np.array] = [ np.array(atom.coord) for atom in self.PDB_chain.get_atoms() if atom.get_name() == "CA" ]

        # Return only the subset
        if res_list is not None: 
            return[ CA_xyz[res] for res in res_list ]
        
        # Return everything
        return CA_xyz
    
    def get_res_names(self, res_list = None):
        """
        Returns the names of residues.

        Parameters:
            res_list (list, optional): List of residue indices (zero-indexed). 
                                       If provided, returns names for these residues only.

        Returns:
            list[str]: List of residue names.
        """
        if res_list is not None:
            return[self.res_names[res] for res in res_list]
        return self.res_names
    
    def get_surface(self) -> Surface:
        return self.surface
    
    # -------------------------------------------------------------------------------------
    # --------------------------------------- Movers --------------------------------------
    # -------------------------------------------------------------------------------------

    def _translate_to_origin(self) -> None:
        """Translates the protein PDB_chain to the reference frame origin (0, 0, 0)."""

        # Translate the protein PDB_chain to the origin (0,0,0)
        CM: np.array = self.PDB_chain.center_of_mass()
        for atom in self.PDB_chain.get_atoms():
            atom.transform(np.identity(3), np.array(-CM))


    def translate(self, translation_vector: np.array) -> None:
        """
        Translates the protein PDB_chain by the given translation vector.

        Parameters:
            translation_vector (np.array): Translation vector to apply.
        """

        # Translate the protein PDB_chain in the direction of the translation_vector
        for atom in self.PDB_chain.get_atoms():
            atom.transform(np.identity(3), np.array(translation_vector))
    
    def rotate(self, reference_point, rotation_matrix) -> None:
        """
        Rotates the protein's atoms around a reference point using a rotation matrix.

        Parameters:
            reference_point (np.array): The reference point for the rotation.
            rotation_matrix (np.array): Rotation matrix to apply.
        """

        # Apply rotation to all atoms of PDB chain
        PDB_atoms = [atom.get_coord() for atom in self.PDB_chain.get_atoms()]
        rotated_PDB_atoms = np.dot(PDB_atoms - reference_point, rotation_matrix.T) + reference_point

        # Update the atom coordinates
        for A, atom in enumerate(self.PDB_chain.get_atoms()):
            atom.set_coord(rotated_PDB_atoms[A])


    # Update the Protein class method
    def rotate2all(self, network_ppis: list):
        subset_indices = []
        partners_CMs = []
        
        for ppi in network_ppis:
            if self.get_ID() not in ppi.get_tuple_pair() or ppi.is_homooligomeric():
                continue
            
            if self.get_ID() == ppi.get_prot_ID_1():
                surf_idxs = ppi.get_contacts_res_1()
                partner = ppi.get_protein_2()
            elif self.get_ID() == ppi.get_prot_ID_2():
                surf_idxs = ppi.get_contacts_res_2()
                partner = ppi.get_protein_1()
            else:
                self.logger.error(f'FATAL ERROR: PPI {ppi.get_tuple_pair()} does not contain the expected protein ID {self.get_ID()}...')
                continue

            partner_CM = partner.get_CM()
            if not np.isnan(partner_CM).any() and not np.isinf(partner_CM).any():
                partners_CMs.append(partner_CM)

            subset_indices.extend(surf_idxs)
        
        if not partners_CMs:
            self.logger.warning(f"No valid partner centroids found for protein: {self.get_ID()}")
            return
        
        partners_centroid = np.mean(partners_CMs, axis=0)
        
        if np.isnan(partners_centroid).any() or np.isinf(partners_centroid).any():
            self.logger.error(f"Invalid partners' centroid for protein: {self.get_ID()}")
            return
        
        _, rotation_matrix, _ = rotate_points(
            points=self.get_res_centroids_xyz(),
            reference_point=self.get_CM(),
            subset_indices=list(set(subset_indices)),
            target_point=partners_centroid
        )
        
        self.rotate(self.get_CM(), rotation_matrix)

    # -------------------------------------------------------------------------------------
    # --------------------------------------- Format --------------------------------------
    # -------------------------------------------------------------------------------------

    def convert_chain_to_pdb_in_memory(self) -> io.StringIO:
        """
        Saves the PDB.Chain.Chain object to an in-memory PDB file.

        Returns:
            io.StringIO: StringIO object containing the PDB data.
        """

        # Create an in-memory file
        pdb_mem_file = io.StringIO()

        # Create a PDBIO instance
        pdbio = PDB.PDBIO()
        
        # Set the structure to the Model
        pdbio.set_structure(self.PDB_chain)

        # Save the Model to a PDB file
        pdbio.save(pdb_mem_file)
        
        # Reset the cursor of the StringIO object to the beginning
        pdb_mem_file.seek(0)
                
        return pdb_mem_file
    
    def change_PDB_chain_id(self, new_chain_ID) -> None:
        """
        Changes the ID of the PDB chain.

        Parameters:
            new_chain_ID (str): The new chain ID.
        """

        self.PDB_chain.id = new_chain_ID
        self.chain_ID = new_chain_ID


    # -------------------------------------------------------------------------------------
    # -------------------------------------- Operators ------------------------------------
    # -------------------------------------------------------------------------------------

    def __str__(self):
        """Returns a string representation of the protein."""
        return f"Protein ID: {self.ID} --------------------------------------------\n   - Name: {self.name}\n   - Sequence: {self.seq}"


###############################################################################################################
################################################## Class PPI ##################################################
###############################################################################################################

class PPI(object):
    """
    Represents a Protein-Protein Interaction (PPI) between two proteins.

    Attributes:
        edge (igraph.Edge): The edge object representing the interaction in a graph.
        prot_ID_1 (str): ID of the first protein in the interaction.
        prot_ID_2 (str): ID of the second protein in the interaction.
        protein_1 (Protein): First protein object.
        protein_2 (Protein): Second protein object.
        cluster_n (int): Cluster number for the interaction.
        models (list[tuple]): List of interaction models.
        x_lab (str): X-axis label for interaction data.
        y_lab (str): Y-axis label for interaction data.
        freq_matrix (np.array): Frequency matrix of residue-residue contacts.
        class_matrix (np.array): Classification matrix for residue-residue contacts.
        contacts_res_1 (list[int]): Contact residues from the first protein.
        contacts_res_2 (list[int]): Contact residues from the second protein.
        contact_freq (list[float]): Frequency of contacts between residues.
        contacts_classification (list[int]): Classification of contacts between residues.
    """

    def __init__(self, proteins: list[Protein], graph_edge: igraph.Edge, logger: Logger, verbose: bool = True) -> None:
        """
        Initializes a PPI object.

        Parameters:
            proteins (list[Protein]): List of Protein objects.
            graph_edge (igraph.Edge): Edge representing the interaction.
            logger (Logger): Logger object for logging.
            verbose (bool): Generates a logging msg using logger
        """

        if verbose:
            logger.info(f'   - Creating object of class PPI: {graph_edge["name"]}')
        
        self.logger         : Logger        = logger
        self.edge           : igraph.Edge   = graph_edge
        self.prot_ID_1      : str           = sorted(self.edge['name'])[0]
        self.prot_ID_2      : str           = sorted(self.edge['name'])[1]
        self.protein_1      : Protein       = [ prot for prot in proteins if prot.get_ID() == self.prot_ID_1 ][0]
        self.protein_2      : Protein       = [ prot for prot in proteins if prot.get_ID() == self.prot_ID_2 ][0]
        self.cluster_n      : int           = self.edge['valency']['cluster_n']
        self.models         : list[tuple]   = self.edge['valency']['models']
        self.x_lab          : str           = self.edge['valency']['x_lab']
        self.y_lab          : str           = self.edge['valency']['y_lab']
        self.freq_matrix    : np.array      = self.edge['valency']['average_matrix']
        self.class_matrix   : np.array      = self.edge['valency']['contact_classification_matrix']

        # Contacts extracted from freq_matrix and class_matrix
        contact_indices                 : np.array      = np.nonzero(self.freq_matrix)                      # Helper computation
        self.contacts_res_1             : list[int]     = contact_indices[0].tolist()                       # Prot 1 contact residues indexes
        self.contacts_res_2             : list[int]     = contact_indices[1].tolist()                       # Prot 2 contact residues indexes
        self.contact_freq               : list[float]   = self.freq_matrix[contact_indices].tolist()        # Residue-Residue contact frequency
        self.contacts_classification    : list[int]     = self.class_matrix[contact_indices].tolist()       # Residue-Residue contact classification (encoded)

        self.add_interactions()

    # -------------------------------------------------------------------------------------
    # -------------------------------------- Getters --------------------------------------
    # -------------------------------------------------------------------------------------

    def get_edge(self):
        """Returns the edge object representing the interaction."""
        return self.edge

    def get_tuple_pair(self):
        """Returns a tuple of the protein IDs involved in the interaction."""
        return tuple(sorted(self.edge['name']))
    
    def get_cluster_n(self):
        """Returns the cluster number of the interaction."""
        return self.cluster_n
    
    def get_protein_1(self) -> Protein:
        """Returns the first Protein object involved in the interaction."""
        return self.protein_1
    
    def get_protein_2(self) -> Protein:
        """Returns the second Protein object involved in the interaction."""
        return self.protein_2
    
    def get_prot_ID_1(self) -> str:
        """Returns the ID of the first Protein object involved in the interaction."""
        return self.prot_ID_1
    
    def get_prot_ID_2(self) -> str:
        """Returns the ID of the second Protein object involved in the interaction."""
        return self.prot_ID_2
    
    def get_contacts_res_1(self) -> list[int]:
        """Returns the list of contact residues from the first protein involved in the interaction."""
        return self.contacts_res_1

    def get_contacts_res_2(self) -> list[int]:
        """Returns the list of contact residues from the second protein involved in the interaction."""
        return self.contacts_res_2

    def is_homooligomeric(self):
        """Checks if the interaction is a from a homooligomer (interaction between the same protein)."""
        return len(set(self.get_tuple_pair())) == 1
    
    def add_interactions(self):
        """Adds interactions between the contact residues of the two proteins."""
        for res1, res2 in zip(self.contacts_res_1, self.contacts_res_2):
            self.protein_1.add_interaction(res1, self.protein_2.get_ID(), self.get_tuple_pair(), self.get_cluster_n())
            self.protein_2.add_interaction(res2, self.protein_1.get_ID(), self.get_tuple_pair(), self.get_cluster_n())
    
    def get_contacts_classification(self) -> List[int]:
        return self.contacts_classification


###############################################################################################################
############################################# Class Stoichiometry #############################################
###############################################################################################################



class Stoichiometry:
    def __init__(self, network, protein_counts: Dict[str, int], parent=None):
        self.network = network
        self.protein_counts = protein_counts
        self.proteins = self._create_proteins()
        self.ppis = self._create_ppis()
        self.score = self._calculate_score()
        self.parent = parent
        self.child = None
        self.is_convergent = False

    def _create_proteins(self) -> List[Protein]:
        proteins = []
        for protein_id, count in self.protein_counts.items():
            original_protein = next(p for p in self.network.get_proteins() if p.get_ID() == protein_id)
            proteins.extend([Protein(original_protein.vertex, len(proteins) + i, original_protein.logger, verbose=False) for i in range(count)])
        return proteins

    def _create_ppis(self) -> List[PPI]:
        ppis = []
        for ppi in self.network.get_ppis():
            prot1_id, prot2_id = ppi.get_tuple_pair()
            for p1 in [p for p in self.proteins if p.get_ID() == prot1_id]:
                for p2 in [p for p in self.proteins if p.get_ID() == prot2_id]:
                    if p1 != p2:
                        ppis.append(PPI([p1, p2], ppi.get_edge(), ppi.logger, verbose=False))
        return ppis
    
    def _calculate_score(self) -> float:
        score = 0
        
        # 1) Homooligomer convergence
        for ppi in self.ppis:
            if ppi.is_homooligomeric():
                convergent_state = ppi.get_edge()['homooligomerization_states']['N_states'][-1]
                if convergent_state:
                    protein_id = ppi.get_prot_ID_1()
                    if self.protein_counts[protein_id] % convergent_state == 0:
                        score += 0.1
        
        # 2) Multivalent pair convergence
        multivalent_pairs = self.network.get_multivalent_pairs()
        for pair in multivalent_pairs:
            if pair[0] in self.protein_counts and pair[1] in self.protein_counts:
                convergent_state = self.network.get_multivalent_convergent_state(pair)
                if convergent_state:
                    current_state = (self.protein_counts[pair[0]], self.protein_counts[pair[1]])
                    if current_state == convergent_state or (current_state[0] % convergent_state[0] == 0 and current_state[1] % convergent_state[1] == 0):
                        score += 0.1
        
        # 3-5) Contact classifications
        static_contacts = 0
        neg_dynamic_contacts = 0
        pos_dynamic_contacts = 0
        total_contacts = 0
        
        for ppi in self.ppis:
            classifications = ppi.get_contacts_classification()
            total_contacts += len(classifications)
            static_contacts += classifications.count(1)
            neg_dynamic_contacts += classifications.count(3)
            pos_dynamic_contacts += classifications.count(2)
        
        score += 0.2 * (static_contacts / total_contacts) if total_contacts > 0 else 0
        score -= 0.1 * (neg_dynamic_contacts / total_contacts) if total_contacts > 0 else 0
        score += 0.1 * (pos_dynamic_contacts / total_contacts) if total_contacts > 0 else 0
        
        # 6) Co-occupied residues
        co_occupied_residues = sum(len(p.get_surface().get_residues_by_group()['B']) + len(p.get_surface().get_residues_by_group()['C']) for p in self.proteins)
        total_residues = sum(len(p.get_seq()) for p in self.proteins)
        score -= 0.1 * (co_occupied_residues / total_residues) if total_residues > 0 else 0
        
        # 7) Fully connected check
        if not self.is_fully_connected():
            score -= 0.5
        
        return max(min(score, 1), -1)  # Ensure score is between -1 and 1

    def is_fully_connected(self) -> bool:
        visited = set()
        stack = [self.proteins[0]]
        while stack:
            protein = stack.pop()
            if protein not in visited:
                visited.add(protein)
                for ppi in self.ppis:
                    if ppi.get_protein_1() == protein:
                        stack.append(ppi.get_protein_2())
                    elif ppi.get_protein_2() == protein:
                        stack.append(ppi.get_protein_1())
        return len(visited) == len(self.proteins)

    def generate_child(self) -> Optional['Stoichiometry']:
        if self.is_convergent:
            return None

        new_protein_counts = self.protein_counts.copy()
        
        # Select a random protein from the current Stoichiometry
        random_protein = random.choice(self.proteins)
        protein_id = random_protein.get_ID()
        
        # Select a random PPI from the combined graph for the selected protein
        available_ppis = [ppi for ppi in self.network.get_ppis() if protein_id in ppi.get_tuple_pair()]
        if not available_ppis:
            self.is_convergent = True
            return None
        
        random_ppi = random.choice(available_ppis)
        partner_id = random_ppi.get_prot_ID_1() if random_ppi.get_prot_ID_1() != protein_id else random_ppi.get_prot_ID_2()
        
        # Determine if we should connect to an existing protein or create a new one
        should_create_new = self._should_create_new_protein(random_ppi, partner_id)
        
        if should_create_new:
            new_protein_counts[partner_id] = new_protein_counts.get(partner_id, 0) + 1
        else:
            # Connect to an existing protein, no change in protein_counts
            pass
        
        # Check for conflicting configurations and potentially remove PPIs
        self._handle_conflicting_configurations(new_protein_counts, random_ppi)
        
        # Create a new Stoichiometry object
        child = Stoichiometry(self.network, new_protein_counts, parent=self)
        self.child = child
        
        return child

    def _should_create_new_protein(self, ppi: PPI, partner_id: str) -> bool:
        edge = ppi.get_edge()
        
        # If the partner is not present, always create a new one
        if partner_id not in self.protein_counts or self.protein_counts[partner_id] == 0:
            return True
        
        # Check for homooligomeric interactions
        if ppi.is_homooligomeric():
            return True
        
        # Check for multivalent interactions
        if edge['valency']['cluster_n'] > 0:
            return np.random.random() < 0.5  # 50% chance to create a new protein for multivalent interactions
        
        # By default, prefer connecting to existing proteins
        return np.random.random() < 0.1  # 10% chance to create a new protein in other cases

    def _handle_conflicting_configurations(self, new_protein_counts: Dict[str, int], ppi: PPI):
        edge = ppi.get_edge()
        
        # Check for negative dynamics
        if "Negative" in edge['dynamics']:
            removal_probability = np.mean(list(edge['N_mers_data']['N_models'])) / 5
            if np.random.random() < removal_probability:
                # Remove the PPI (don't add it to the new configuration)
                return
        
        # Check for conflicts with N_mers data
        for _, row in edge['N_mers_data'].iterrows():
            proteins_in_model = set(row['proteins_in_model'])
            if proteins_in_model.issubset(new_protein_counts.keys()):
                if row['N_models'] < self.network.combined_graph['cutoffs_dict']['N_models_cutoff']:
                    removal_probability = 1 - (row['N_models'] / self.network.combined_graph['cutoffs_dict']['N_models_cutoff'])
                    if np.random.random() < removal_probability:
                        # Remove the PPI (don't add it to the new configuration)
                        return

    def __eq__(self, other):
        if not isinstance(other, Stoichiometry):
            return NotImplemented
        return self.protein_counts == other.protein_counts

    def __hash__(self):
        return hash(tuple(sorted(self.protein_counts.items())))

def generate_stoichiometries(network, num_iterations: int = 50):

    # Progress
    network.logger.info( 'INITIALIZING: Stoichiometric Space Exploration Algorithm...')
    network.logger.info(f'   Nº of iterations per starting point: {num_iterations}')

    stoichiometries = []
    
    # Start from each protein in the network
    for point, start_protein in enumerate(network.get_proteins()):

        # Progress
        network.logger.info(f'   Starting point ({point + 1}/{len(network.get_proteins())}): {start_protein.get_ID()}')

        initial_counts = {start_protein.get_ID(): 1}
        root = Stoichiometry(network, initial_counts)
        
        current = root
        path = [current]
        
        for i in range(num_iterations):

            # Progress
            if i % 10 == 0:
                network.logger.info(f'      - Iteration: {i} | Nº of Stoichiometries in path: {len(path)}...')

            child = current.generate_child()
            if child is None:
                break
            path.append(child)
            current = child
        
        stoichiometries.extend(path)
    
    network.logger.info('FINISHED: Stoichiometric Space Exploration Algorithm')
    
    return stoichiometries

def analyze_stoichiometries(stoichiometries: List[Stoichiometry]):
    # Count occurrences of each unique stoichiometry
    stoich_counts = {}
    for s in stoichiometries:
        key = tuple(sorted(s.protein_counts.items()))
        stoich_counts[key] = stoich_counts.get(key, 0) + 1
    
    # Sort by frequency and score
    sorted_stoich = sorted(stoich_counts.items(), key=lambda x: (-x[1], -dict(x[0]).get('score', 0)))
    
    return sorted_stoich



###############################################################################################################
################################################ Class Network ################################################
###############################################################################################################

class Network(object):
    """
    Represents a network of proteins and their interactions (PPIs) using a graph structure.

    Attributes:
        logger (Logger): Logger object to log operations.
        combined_graph (igraph.Graph): Graph representing the combined protein-protein interactions.
        proteins (list[Protein]): List of Protein objects in the network.
        proteins_IDs (list[str]): List of protein IDs in the network.
        ppis (list[PPI]): List of Protein-Protein Interactions (PPIs) in the network.
        ppis_dynamics (list[str]): List of dynamics types for the PPIs in the network.
    """
    
    def __init__(self, combined_graph: igraph.Graph, logger: Logger, remove_interaction = ("Indirect",)):
        """
        Initializes a Network object.

        Parameters:
            combined_graph (igraph.Graph): Graph representing the combined protein-protein interactions.
            logger (Logger): Logger object for logging.
            remove_interaction (tuple): Interaction types to be removed from the network (default: ("Indirect",)).
        """

        logger.info("Creating object of class Network...")

        add_contact_classification_matrix(combined_graph)

        self.logger         : Logger            = logger
        self.combined_graph : igraph.Graph      = combined_graph
        self.proteins       : list[Protein]     = [Protein(vertex, unique_ID, logger)   for unique_ID, vertex   in enumerate(combined_graph.vs)]
        self.proteins_IDs   : list[str]         = [vertex['name']                       for vertex              in combined_graph.vs]
        self.ppis           : list[PPI]         = [PPI(self.proteins, edge, logger)     for edge                in combined_graph.es if (edge['dynamics'] not in remove_interaction) and ((edge['valency']['average_matrix'] > 0).sum() > 0)]
        self.ppis_dynamics  : list[PPI]         = [edge['dynamics']                     for edge                in combined_graph.es]
        self.edge_counts    : Counter           = Counter([tuple_edge for i, tuple_edge in enumerate(self.combined_graph.es['name']) if (self.combined_graph.es[i]['dynamics'] not in remove_interaction) and ((self.combined_graph.es[i]['valency']['average_matrix'] > 0).sum() > 0)])

        # Assign unique and incremental chain IDs to each PDB.Chain.Chain object of each protein
        for p, prot in enumerate(self.proteins):
            prot.change_PDB_chain_id(VALID_CHAIN_IDS[p])
    
    # -------------------------------------------------------------------------------------
    # -------------------------------------- Getters --------------------------------------
    # -------------------------------------------------------------------------------------
    
    def get_proteins(self) -> list[Protein]:
        """
        Returns the list of proteins in the network.

        Returns:
            list[Protein]: List of Protein objects.
        """
        return self.proteins
    
    def get_proteins_IDs(self) -> list[str]:
        """
        Returns the list of protein IDs in the network.

        Returns:
            list[str]: List of protein IDs.
        """
        return self.proteins_IDs
    
    def get_ppis(self) -> list[PPI]:
        """
        Returns the list of Protein-Protein Interactions (PPIs) in the network.

        Returns:
            list[PPI]: List of PPI objects.
        """
        return self.ppis

    ########################## For stoichiometry computations (greedy method) ##########################

    def generate_stoichiometries_greedy(self, num_stoichiometries: int = 100, max_units: int = 6, max_iterations: int = 1000, convergent_iterations = 10) -> List[Stoichiometry]:
        
        # Progress
        self.logger.info(f"INITIALIZING: Stoichiometry Exploration (Greedy)...")
        self.logger.info(f"   Parameters:")
        self.logger.info(f"      - num_stoichiometries = {num_stoichiometries}")
        self.logger.info(f"      - max_units = {max_units}")
        self.logger.info(f"      - convergent_iterations = {convergent_iterations}")
        self.logger.info(f"   Generating stoichiometries:")

        # Counters and results
        stoichiometries = []
        iterations = 0
        convergency_counter = 0

        while len(stoichiometries) < num_stoichiometries and iterations < max_iterations and convergency_counter < convergent_iterations:

            # Initial random guess
            protein_counts = self._generate_random_protein_counts(max_units)

            # Optimize the guess
            stoichiometry, optimized_protein_counts = self._optimize_stoichiometry(protein_counts, max_units)
            
            # If the iteration resulted in a valid stoichiometry
            if stoichiometry:

                # If the stoichiometry was not previously explored
                if stoichiometry not in stoichiometries:
                    stoichiometries.append(stoichiometry)
                    self.logger.info(f"      - Generated stoichiometry {len(stoichiometries)}/{num_stoichiometries}")

                    # Get back the convergency counter to zero
                    convergency_counter = 0
                
                # If it was previously explored
                else:
                    convergency_counter += 1
            
            iterations += 1

            # Progress
            if iterations % 50 == 0:
                self.logger.info(f"   - Reached iteration {iterations}:")
                self.logger.info(f"      Current valid stoichiometries: {len(stoichiometries)}/{num_stoichiometries}")
                self.logger.info(f"      Convergency counter: {convergency_counter}")


        self.logger.info(f"Greedy stoichiometry generation complete.")
        if convergency_counter < convergent_iterations:
            self.logger.info( "   - Algorithm reached convergency")
        self.logger.info(f"   - Created {len(stoichiometries)} stoichiometries in {iterations} iterations.")
        return sorted(stoichiometries, key=lambda s: s.score, reverse=True)

    def _generate_random_protein_counts(self, max_units: int) -> Dict[str, int]:
        return {protein_id: random.randint(1, max_units) for protein_id in self.proteins_IDs}

    def _optimize_stoichiometry(self, initial_counts: Dict[str, int], max_units: int) -> Stoichiometry:
        current_counts = initial_counts.copy()
        current_stoichiometry = Stoichiometry(self, current_counts)
        
        if not current_stoichiometry.is_fully_connected():
            return None

        for _ in range(100):  # Limit the number of optimization steps
            improved = False
            for protein_id in self.proteins_IDs:
                for delta in [-1, 1]:
                    new_counts = current_counts.copy()
                    new_counts[protein_id] = max(1, min(new_counts[protein_id] + delta, max_units))
                    
                    if new_counts != current_counts:
                        new_stoichiometry = Stoichiometry(self, new_counts)
                        if new_stoichiometry.is_fully_connected() and new_stoichiometry.score > current_stoichiometry.score:
                            current_counts = new_counts
                            current_stoichiometry = new_stoichiometry
                            improved = True
                            break
                
                if improved:
                    break
            
            if not improved:
                break

        return current_stoichiometry, current_counts

    ########################## For stoichiometry computations (full) ##########################
    
    def generate_stoichiometries(self, max_units: int = 6) -> List[Stoichiometry]:
        self.logger.info(f"Starting stoichiometry generation with max_units={max_units}")
        protein_ids = self.get_proteins_IDs()
        stoichiometries = []
        total_combinations = len(list(itertools.product(range(1, max_units + 1), repeat=len(protein_ids))))
        self.logger.info(f"Total possible combinations: {total_combinations}")

        # Progress
        current_stoichiometry = 0
        self.logger.info(print_progress_bar(current_stoichiometry, total_combinations, text = " (Stoichiometries)", progress_length = 40))

        filtered_combinations = 0
        created_stoichiometries = 0

        for counts in itertools.product(range(1, max_units + 1), repeat=len(protein_ids)):
            total_combinations += 1
            protein_counts = dict(zip(protein_ids, counts))
            
            if self._is_viable_stoichiometry(protein_counts):
                stoichiometry = Stoichiometry(self, protein_counts)
                if stoichiometry.is_fully_connected():
                    stoichiometries.append(stoichiometry)
                    created_stoichiometries += 1
                    if created_stoichiometries % 10 == 0:
                        self.logger.info(f"Created {created_stoichiometries} viable stoichiometries")
                        self.logger.info(print_progress_bar(current_stoichiometry, total_combinations, text = " (Stoichiometries)", progress_length = 40))

            else:
                filtered_combinations += 1
                if filtered_combinations % 1000000 == 0:
                    self.logger.info(f"Filtered out {filtered_combinations} unviable combinations")

        self.logger.info(f"Stoichiometry generation complete. Created {len(stoichiometries)} stoichiometries.")
        return sorted(stoichiometries, key=lambda s: s.score, reverse=True)

    def _is_viable_stoichiometry(self, protein_counts: Dict[str, int]) -> bool:
        # Check if there are at least two proteins without a connection between them
        for p1, p2 in itertools.combinations(protein_counts.keys(), 2):
            if (p1, p2) not in self.edge_counts and (p2, p1) not in self.edge_counts:
                return False

        # Check if the number of proteins exceeds the number of possible interactions
        for (p1, p2), count in self.edge_counts.items():
            if p1 in protein_counts and p2 in protein_counts:
                if p1 == p2:  # Homooligomeric interaction
                    if protein_counts[p1] > count + 1:
                        return False
                else:  # Heterooligomeric interaction
                    if min(protein_counts[p1], protein_counts[p2]) > count:
                        return False

        return True

    def get_multivalent_pairs(self) -> List[Tuple[str, str]]:
        multivalent_pairs = []
        for edge in self.combined_graph.es:
            if edge['multivalency_states'] is not None:
                multivalent_pairs.append(tuple(sorted(edge['name'])))
        return list(set(multivalent_pairs))

    def get_multivalent_convergent_state(self, pair: Tuple[str, str]) -> Tuple[int, int]:
        for edge in self.combined_graph.es:
            if set(edge['name']) == set(pair):
                if edge['multivalency_states'] is not None:
                    # Find the largest true state
                    true_states = [state for state, is_valid in edge['multivalency_states'].items() if is_valid]
                    if true_states:
                        largest_state = max(true_states, key=lambda x: len(x))
                        return (largest_state.count(pair[0]), largest_state.count(pair[1]))
        return None

    # -------------------------------------------------------------------------------------
    # --------------------------------------- Adders --------------------------------------
    # -------------------------------------------------------------------------------------

    def add_new_protein(self):
        """
        Adds a new protein to the network (currently not implemented).
        """
        raise NotImplementedError("Method not implemented yet...")

    # -------------------------------------------------------------------------------------
    # -------------------------------------- Plotting -------------------------------------
    # -------------------------------------------------------------------------------------

    def generate_layout(self,
                        algorithm      = ["optimized", "drl", "fr", "kk", "circle", "grid", "random"][0],
                        scaling_factor = [None       , 300  , 200 , 100 , 100     , 120   , 100     ][0]):
        """
        Generates a 3D layout for the network using different algorithms.

        Parameters:
            algorithm (str): Layout algorithm to use. Default is "optimized" (recommended).
            scaling_factor (int): Scaling factor for the layout (default is 300).
        """

        self.logger.info(f"INITIALIZING: 3D layout generation using {algorithm} method...")
        
        # When there are less than 3 proteins, use "drl" algorithm
        if len(self.proteins) < 3:
            self.logger.warning('   - Less than 3 proteins in the network. Using drl method...')
            algorithm = "drl"
            scaling_factor = 250

        if algorithm == "optimized":
            optimize_layout(self)

        else:

            self.logger.info(f"   - Scaling factor: {scaling_factor}...")

            # Ensure proteins are at the origin
            for prot in self.proteins:
                self.logger.info(f"   - Translating {prot.get_ID()} to the origin first...")
                prot._translate_to_origin()

            # Generate 3D coordinates for plot
            layt = [list(np.array(coord) * scaling_factor) for coord in list(self.combined_graph.layout(algorithm, dim=3))]

            # Translate the proteins to new positions
            for p, prot in enumerate(self.proteins):
                self.logger.info(f"   - Translating {prot.get_ID()} to new layout positions...")
                prot.translate(layt[p])

            # Rotate the proteins to best orient their surfaces 
            for prot in self.proteins:
                self.logger.info(f"   - Rotating {prot.get_ID()} with respect to their partners CMs...")
                prot.rotate2all(self.ppis)
            
        self.logger.info("FINISHED: 3D layout generation algorithm")


    def generate_py3dmol_plot(self, save_path: str = './3D_graph.html',
                              classification_colors = rrc_classification_colors,
                              domain_colors         = DOMAIN_COLORS_RRC,
                              res_centroid_colors   = SURFACE_COLORS_RRC,
                              show_plot = True):
        """
        Generates a 3D molecular visualization of the network using py3Dmol.

        Parameters:
            save_path (str): Path to save the HTML visualization file (default: './3D_graph.html').
            classification_colors (dict): Mapping of contact classification categories to colors.
            surface_residue_palette (dict): Color palette for surface residues.
            show_plot (bool): Whether to automatically display the plot (default: True).
        """
        
        # Progress
        self.logger.info("INITIALIZING: Generating py3Dmol visualization...")

        # Create a view object
        view = py3Dmol.view(width='100%', height='100%')

        # Create a global dictionary to map PPI IDs to colors
        ppi_colors = {}
        color_index = 0
        for ppi in self.ppis:
            ppi_id = (ppi.get_tuple_pair(), ppi.get_cluster_n())  # Include cluster number in the key
            if ppi_id not in ppi_colors:
                ppi_colors[ppi_id] = res_centroid_colors[color_index % len(res_centroid_colors)]
                color_index += 1

        ########################### Protein Backbones ###########################

        # Progress
        self.logger.info('   Adding protein backbones...')

        # Get the maximum number of domains of all proteins
        max_domain_value = max([max(prot.domains) for prot in self.proteins])

        # Add each protein backbone to the viewer
        for protein in self.proteins:

            # Get palette for the protein
            residue_groups = protein.surface.get_residues_by_group()

            # Get the maximum domain value across all proteins to generate enough unique colors
            domain_colors = generate_unique_colors(n = max_domain_value + 1, palette = domain_colors)

            # Progress
            self.logger.info(f'      - Adding {protein.get_ID()} backbone')

            # Add the protein PDB.Chain.Chain properly formatted to the view
            pdb_string = protein.convert_chain_to_pdb_in_memory()
            pdb_string_content = pdb_string.getvalue()
            view.addModel(pdb_string_content, 'pdb')

            # Color the backbone based on domain information
            for i, domain in enumerate(protein.domains):
                
                # Color each residue backbone of the protein  based on domain information
                view.setStyle(
                    {'chain': protein.chain_ID, 'resi': [i + 1]},
                    {'cartoon': {'color': domain_colors[domain]}}
                )

        ########################### Surface Residues ###########################

        # Progress
        self.logger.info('   Adding protein surface residue centroids...')

        # Add contact residue centroids for different Surface object in the protein
        for protein in self.proteins:

            # Memoization for centroids and CA positions
            centroid_cache = {}
            ca_cache = {}

            # Set to keep track of added residues
            added_residues = set()

            # Progress
            self.logger.info(f'      - Adding {protein.get_ID()} surface centroids')

            # Extract protein surface residues, classified by group (A, B or C)
            residue_groups = protein.surface.get_residues_by_group()
            
            for group, residues in residue_groups.items():
                for residue in residues:
                    if residue.index in added_residues: # Skip already added
                        continue

                    if group == "A":
                        tuple_pair, cluster_n = residue.surface_id
                        color = ppi_colors[(tuple_pair, cluster_n)]
                    elif group == "B":
                        color = "gray"
                    elif group == "C":
                        color = "black"
                    else:  # Non-interacting
                        continue
                    
                    # Get centroid CM and CA CM using dynamic programming
                    centroid = get_centroid(protein, residue.index, centroid_cache)
                    CA = get_ca(protein, residue.get_index(), ca_cache)

                    # Add the residue centroid as a Sphere
                    view.addSphere({
                        'center': {'x': float(centroid[0]), 'y': float(centroid[1]), 'z': float(centroid[2])},
                        'radius': 1.0,
                        'color': color
                    })

                    # Add a line connecting the sphere to the backbone CA atom
                    view.addCylinder({
                        'start': {'x': float(centroid[0]), 'y': float(centroid[1]), 'z': float(centroid[2])},
                        'end': {'x': float(CA[0]), 'y': float(CA[1]), 'z': float(CA[2])},
                        'radius': 0.1,
                        'color': color
                    })

                    added_residues.add(residue.index)

        ############################ Contact Lines ############################

        self.logger.info('   Adding contact lines...')

        # Add contact residues and contact lines dynamically
        for ppi in self.ppis:

            self.logger.info(f'      - Adding PPI {ppi.get_tuple_pair()} cluster {ppi.get_cluster_n()} contacts...')

            protein_1 = ppi.get_protein_1()
            protein_2 = ppi.get_protein_2()

            # Get contact residue indices and centroids for both proteins
            contact_residues_1 = ppi.get_contacts_res_1()
            contact_residues_2 = ppi.get_contacts_res_2()
            centroids_1 = protein_1.get_res_centroids_xyz(contact_residues_1)
            centroids_2 = protein_2.get_res_centroids_xyz(contact_residues_2)

            # Render contact cylinders between protein 1 and 2
            for i, (centroid_1, centroid_2) in enumerate(zip(centroids_1, centroids_2)):
                contact_classification = ppi.contacts_classification[i]
                contact_freq = ppi.contact_freq[i]
                cylinder_color = classification_colors.get(contact_classification, 'gray')
                cylinder_radius = (0.3 * contact_freq) * 0.5 # Scale radius based on frequency (0 to 1)

                view.addCylinder({
                    'start': {'x': float(centroid_1[0]), 'y': float(centroid_1[1]), 'z': float(centroid_1[2])},
                    'end': {'x': float(centroid_2[0]), 'y': float(centroid_2[1]), 'z': float(centroid_2[2])},
                    'radius': cylinder_radius,
                    'color': cylinder_color,
                    'opacity': 0.9
                })
        
        ############################# Some Labels #############################

        # Add N and C terminal labels, and protein IDs
        for protein in self.proteins:
            n_term = protein.get_res_CA_xyz([0])[0]
            c_term = protein.get_res_CA_xyz([-1])[0]
            cm = protein.get_CM()

            view.addLabel('N', {'position': {'x': float(n_term[0]), 'y': float(n_term[1]), 'z': float(n_term[2])},
                                'fontSize': 14, 'fontColor': 'black', 'backgroundOpacity': 0.0})
            view.addLabel('C', {'position': {'x': float(c_term[0]), 'y': float(c_term[1]), 'z': float(c_term[2])},
                                'fontSize': 14, 'fontColor': 'black', 'backgroundOpacity': 0.0})
            view.addLabel(protein.get_ID(), {'position': {'x': float(cm[0]), 'y': float(cm[1]), 'z': float(cm[2]) + 5},
                                            'fontSize': 18, 'color': 'black', 'backgroundOpacity': 0.6})
            
        # Set camera and background
        view.zoomTo()
        view.setBackgroundColor('white')

        # Save the visualization as an HTML file
        self.logger.info( "   Saving HTML file...")
        view.write_html(save_path)
        self.logger.info(f"   Visualization saved to {save_path}")

        self.logger.info("FINISHED: Generating py3Dmol visualization")

        if show_plot:
            webbrowser.open(f"{save_path}")


    def generate_plotly_3d_plot(self, save_path: str = './3D_graph.html',
                                classification_colors: Dict = rrc_classification_colors,
                                domain_colors: Dict         = {i: v[-5] for i,v in enumerate(default_color_palette.values())},
                                res_centroid_colors: Dict   = {i: v[-1] for i,v in enumerate(default_color_palette.values())},
                                show_plot: bool = True):
        """
        Generates a 3D molecular visualization of the network using Plotly.

        Parameters:
            save_path (str): Path to save the HTML visualization file (default: './3D_graph.html').
            classification_colors (dict): Mapping of contact classification categories to colors.
            domain_colors (dict): Mapping of domain numbers to colors.
            res_centroid_colors (dict): Mapping of residue types to colors for centroids.
            show_plot (bool): Whether to automatically display the plot (default: True).
        """
        
        # Progress
        self.logger.info("INITIALIZING: Generating Plotly 3D visualization...")

        # Create a figure object
        fig = go.Figure()

        # Create a global dictionary to map PPI IDs to colors
        ppi_colors = {}
        color_index = 0
        for ppi in self.ppis:
            ppi_id = (ppi.get_tuple_pair(), ppi.get_cluster_n())  # Include cluster number in the key
            if ppi_id not in ppi_colors:
                ppi_colors[ppi_id] = res_centroid_colors[color_index % len(res_centroid_colors)]
                color_index += 1

        ########################### Protein Backbones ###########################

        # Progress
        self.logger.info('   Adding protein backbones...')

        pLDDT_colors = ['#e13939', '#f18438', '#f1e66b', '#00b1b0', '#001a9c']

        plddt_traces = []
        domain_traces = []

        for protein in self.proteins:
            backbone_CAs = protein.get_res_CA_xyz()
            plddt_values = protein.get_res_pLDDT()
            resid_values = protein.get_res_names()

            plddt_trace = go.Scatter3d(
                x=[xyz[0] for xyz in backbone_CAs],
                y=[xyz[1] for xyz in backbone_CAs],
                z=[xyz[2] for xyz in backbone_CAs],
                mode='lines',
                line=dict(
                    color=plddt_values,
                    colorscale=pLDDT_colors,
                    width=5,
                ),
                name=f"{protein.get_ID()} (pLDDT)",
                showlegend=False,
                hoverinfo='text',
                hovertext=resid_values
            )
            plddt_traces.append(plddt_trace)

            domain_trace = go.Scatter3d(
                x=[xyz[0] for xyz in backbone_CAs],
                y=[xyz[1] for xyz in backbone_CAs],
                z=[xyz[2] for xyz in backbone_CAs],
                mode='lines',
                line=dict(
                    color=[domain_colors.get(domain, 'gray') for domain in protein.domains],
                    width=5
                ),
                name=f"{protein.get_ID()} (Domains)",
                showlegend=False,
                hoverinfo='text',
                hovertext=resid_values
            )
            domain_traces.append(domain_trace)

        # Add grouped traces
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], name="pLDDT Backbones", visible='legendonly'))
        for trace in plddt_traces:
            fig.add_trace(trace)

        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], name="Domain Backbones", visible='legendonly'))
        for trace in domain_traces:
            fig.add_trace(trace)

        # Update trace grouping
        for i in range(len(plddt_traces) + 1):
            fig.data[i].legendgroup = "pLDDT"
            fig.data[i].visible = 'legendonly'

        for i in range(len(plddt_traces) + 1, len(plddt_traces) + len(domain_traces) + 2):
            fig.data[i].legendgroup = "Domains"
            fig.data[i].visible = 'legendonly'

        ############################ Contact Lines ############################

        # Progress
        self.logger.info('   Adding contact lines...')

        # Dictionary to group by classification
        classified_traces = defaultdict(lambda: {'x': [], 'y': [], 'z': [], 'hovertext': [], 'freqs': []})

        def generate_intermediate_points(p1, p2, num_points=20):
            """
            Generates num_points equidistant points between p1 and p2 (excluding p1 and p2).
            """
            return [p1 + (p2 - p1) * t for t in np.linspace(0, 1, num_points + 2)][1:-1]

        for ppi in self.ppis:
            self.logger.info(f'      - Adding PPI {ppi.get_tuple_pair()} cluster {ppi.get_cluster_n()} contacts...')

            protein_1 = ppi.get_protein_1()
            protein_2 = ppi.get_protein_2()

            contact_residues_1 = ppi.get_contacts_res_1()
            contact_residues_2 = ppi.get_contacts_res_2()

            for i, (res1_index, res2_index) in enumerate(zip(contact_residues_1, contact_residues_2)):
                centroid_1 = np.array(protein_1.get_res_centroids_xyz([res1_index])[0])
                centroid_2 = np.array(protein_2.get_res_centroids_xyz([res2_index])[0])

                contact_classification = ppi.contacts_classification[i]
                contact_freq = ppi.contact_freq[i]
                color = classification_colors.get(contact_classification, 'gray')

                # Generate 5 equidistant intermediate points between centroid_1 and centroid_2
                x_intermediates = generate_intermediate_points(centroid_1[0], centroid_2[0])
                y_intermediates = generate_intermediate_points(centroid_1[1], centroid_2[1])
                z_intermediates = generate_intermediate_points(centroid_1[2], centroid_2[2])

                # Grouping coordinates and hovertext by classification, including intermediates
                classified_traces[contact_classification]['x'].extend([centroid_1[0]] + x_intermediates + [centroid_2[0], None])
                classified_traces[contact_classification]['y'].extend([centroid_1[1]] + y_intermediates + [centroid_2[1], None])
                classified_traces[contact_classification]['z'].extend([centroid_1[2]] + z_intermediates + [centroid_2[2], None])
                
                # Replicate hovertext for each segment, including intermediates
                hovertext = f"{protein_1.get_ID()}-{protein_1.get_res_names([res1_index])[0]} / " \
                            f"{protein_2.get_ID()}-{protein_2.get_res_names([res2_index])[0]}"
                classified_traces[contact_classification]['hovertext'].extend([hovertext] * (len(x_intermediates) + 2))
                classified_traces[contact_classification]['hovertext'].append(None)  # To handle the 'None' in coordinates
                
                classified_traces[contact_classification]['freqs'].append(contact_freq)

        # Create a single trace per classification
        for classification, data in classified_traces.items():
            fig.add_trace(go.Scatter3d(
                x=data['x'],
                y=data['y'],
                z=data['z'],
                mode='lines',
                line=dict(
                    color=classification_colors.get(classification, 'gray'),
                    width=2,  # Fixed width
                ),
                opacity=0.5,
                name=f"{CLASSIFICATION_ENCODING[classification]} Contacts",
                showlegend=True,
                hoverinfo='text',
                hovertext=data['hovertext']
            ))

        ################### Surface residues connected to the CM ###################
        
        # Progress
        self.logger.info('   Adding protein surface residue centroids and lines...')

        shared_residue_color = 'black'

        x, y, z     = [], [], []
        colors      = []
        hover_texts = []

        # Pack the data
        for protein in self.proteins:
            cm = protein.get_CM()
            surface_residues = protein.surface.get_interacting_residues()

            for index, residue in surface_residues.items():
                centroid = protein.get_res_centroids_xyz([index])[0]
                
                x.extend([centroid[0], cm[0], None])
                y.extend([centroid[1], cm[1], None])
                z.extend([centroid[2], cm[2], None])
                
                if len(residue.interactions) > 1:  # Shared residue
                    color = shared_residue_color
                elif residue.interactions:
                    ppi_id, cluster_n = next(iter(residue.interactions.keys()))
                    color = ppi_colors.get((ppi_id, cluster_n), 'gray')
                else:
                    color = 'gray'
                colors.extend([color, color, color])
                
                hover_text = f"{protein.get_ID()}-{residue.get_name()}"
                hover_texts.extend([hover_text, hover_text, hover_text])

        # Add lines
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            line=dict(color=colors, width=5),
            marker=dict(size=5,
                        color=colors,
                        opacity=0.5
                ),
            opacity=0.5,
            hoverinfo='text',
            hovertext=hover_texts,
            name=f"Surfaces",
            showlegend=True
        ))       

        ############################# Some Labels #############################

        nc_term_list_xyz  = []
        nc_term_list_text = []
        CM_list_xyz       = []
        CM_list_text      = []

        # Compute N and C terminal labels, and protein IDs
        for protein in self.proteins:
            # Add N-ter
            nc_term_list_xyz.append(protein.get_res_CA_xyz([0])[0])
            nc_term_list_text.append('N')

            # Add C-ter
            nc_term_list_xyz.append(protein.get_res_CA_xyz([len(protein.get_seq())-1])[0])
            nc_term_list_text.append('C')

            # Add CM
            CM_list_xyz.append(protein.get_CM())
            CM_list_text.append(protein.get_ID())

        # Add N/C-terminals
        fig.add_trace(go.Scatter3d(
            x=[nc_term[0] for nc_term in nc_term_list_xyz],
            y=[nc_term[1] for nc_term in nc_term_list_xyz],
            z=[nc_term[2] for nc_term in nc_term_list_xyz],
            mode='text',
            text=nc_term_list_text,
            textposition='top center',
            name = "N/C-terminal",
            textfont=dict(size=14, color='black'),
            showlegend=True,
            visible='legendonly'
        ))

        # Add ID labels over the CM
        CM_offset = 20
        fig.add_trace(go.Scatter3d(
            x=[cm[0] for cm in CM_list_xyz],
            y=[cm[1] for cm in CM_list_xyz],
            z=[cm[2] + CM_offset for cm in CM_list_xyz],  # Offset in z-axis
            mode='text',
            text=CM_list_text,
            name = "Protein IDs",
            textposition='top center',
            textfont=dict(size=18, color='black'),
            showlegend=True
        ))

        # Set camera and background
        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
                zaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
                aspectmode="data",
                # Allow free rotation along all axis
                dragmode="orbit",
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=0.6)
                )
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(r=0, l=0, b=0, t=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # Save the visualization as an HTML file
        self.logger.info("   Saving HTML file...")
        fig.write_html(save_path)
        self.logger.info(f"   Visualization saved to {save_path}")

        self.logger.info("FINISHED: Generating Plotly 3D visualization")

        if show_plot:
            plot(fig)

    # -------------------------------------------------------------------------------------
    # ------------------------------------- Operators -------------------------------------
    # -------------------------------------------------------------------------------------


    def __str__(self):

        result = f"Network with {len(self.proteins_IDs)} proteins and {len(self.ppis)} PPIs:"

        for prot in self.proteins:
            result += "\n   >" + prot.get_ID()+ ": " + prot.get_seq()
        return result

