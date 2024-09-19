
import numpy as np
import pandas as pd
import igraph
import py3Dmol
from logging import Logger
from Bio import PDB
from copy import deepcopy
import io
import string
from itertools import cycle
import webbrowser
from scipy.spatial.distance import pdist, squareform

from cfg.default_settings import PT_palette
from utils.pdb_utils import center_of_mass, rotate_points
from src.detect_domains import format_domains_df_as_no_loops_domain_clusters

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

# Define color scheme for different contact classifications
classification_colors = {
    1: PT_palette['gray'],      # Static
    2: PT_palette['green'],     # Positive
    3: PT_palette['red'],       # Negative
    4: PT_palette['orange'],    # No Nmers Data
    5: PT_palette['yellow']     # No 2mers Data
}

# Color palette for network representation
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


###############################################################################################################
################################### Helper functions for optimized layout #####################################
###############################################################################################################

def get_protein_radius(protein):
    """Calculate the radius of the protein's bounding sphere."""
    cm = protein.get_CM()
    residue_positions = protein.get_res_CA_xyz()
    distances = np.linalg.norm(residue_positions - cm, axis=1)
    return np.max(distances) + 5  # Add 5 Angstroms as buffer

def initialize_positions(proteins, ppis):
    """Initialize protein positions based on their interactions."""
    n = len(proteins)
    positions = np.random.rand(n, 3) * 1000  # Random initial positions
    radii = np.array([get_protein_radius(protein) for protein in proteins])
    
    # Adjust positions based on interactions
    for ppi in ppis:
        p1 = proteins.index(ppi.get_protein_1())
        p2 = proteins.index(ppi.get_protein_2())
        
        # Move interacting proteins closer, but not too close
        direction = positions[p2] - positions[p1]
        distance = np.linalg.norm(direction)
        min_distance = radii[p1] + radii[p2]
        if distance < min_distance:
            midpoint = (positions[p1] + positions[p2]) / 2
            positions[p1] = midpoint - direction / distance * radii[p1]
            positions[p2] = midpoint + direction / distance * radii[p2]
    
    return positions, radii

def compute_forces(positions, radii, proteins, ppis):
    """Compute forces between proteins based on their interactions and surfaces."""
    n = len(proteins)
    forces = np.zeros((n, 3))
    
    # Repulsive force between all proteins
    distances = squareform(pdist(positions))
    for i in range(n):
        for j in range(i+1, n):
            if distances[i,j] > 0:
                min_distance = radii[i] + radii[j]
                if distances[i,j] < min_distance:
                    force = 1000 * (min_distance - distances[i,j]) / min_distance
                    direction = (positions[i] - positions[j]) / distances[i,j]
                    forces[i] += force * direction
                    forces[j] -= force * direction
    
    # Attractive force between interacting proteins
    for ppi in ppis:
        p1 = proteins.index(ppi.get_protein_1())
        p2 = proteins.index(ppi.get_protein_2())
        
        # Compute centroids of interaction surfaces
        surface1 = np.mean(ppi.get_protein_1().get_res_centroids_xyz(ppi.get_contacts_res_1()), axis=0)
        surface2 = np.mean(ppi.get_protein_2().get_res_centroids_xyz(ppi.get_contacts_res_2()), axis=0)
        
        # Attractive force based on distance between interaction surfaces
        distance = distances[p1, p2]
        min_distance = radii[p1] + radii[p2]
        if distance > min_distance:
            force = 0.1 * (distance - min_distance)
            direction = (positions[p2] - positions[p1]) / distance
            forces[p1] += force * direction
            forces[p2] -= force * direction
    
    return forces

def apply_layout(proteins, ppis, network, iterations=200):
    """Apply force-directed layout algorithm."""
    positions, radii = initialize_positions(proteins, ppis)
    network.logger.info(f'   - Nº of iterations to perform: {iterations}')
    
    for i in range(iterations):
        if i % 10 == 0:
            network.logger.info(f'   - Iteration {i}...')
            for prot in proteins:
                prot.rotate2all(network.get_ppis())
        
        forces = compute_forces(positions, radii, proteins, ppis)
        
        # Dampen the forces over time
        damping_factor = 1 - (i / iterations)
        positions += forces * 0.1 * damping_factor  # Adjust step size as needed
        
        # Ensure proteins don't drift too far apart
        center = np.mean(positions, axis=0)
        for j, pos in enumerate(positions):
            direction = pos - center
            distance = np.linalg.norm(direction)
            max_distance = 1000  # Adjust as needed
            if distance > max_distance:
                positions[j] = center + direction / distance * max_distance
    
    return positions

def optimize_layout(network):
    """Optimize the layout of proteins in the network."""
    proteins = network.get_proteins()
    ppis = network.get_ppis()
    
    # Apply force-directed layout
    optimized_positions = apply_layout(proteins, ppis, network)
    
    # Update protein positions
    for i, protein in enumerate(proteins):
        protein.translate(optimized_positions[i] - protein.get_CM())
    
    # Final rotation of proteins to face their interaction partners
    for protein in proteins:
        protein.rotate2all(ppis)


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
    def __init__(self, index, name, pLDDT):
        self.index = index
        self.name = name
        self.pLDDT = pLDDT
        self.interactions = {}  # Dict of {ppi_id: set(partner_protein_ids)}

    def get_index(self):
        return self.index
    
    def get_name(self):
        return self.name

    def get_plddt(self):
        return self.pLDDT
    
    def get_interactions(self):
        return self.interactions
    
    def add_interaction(self, partner_protein_id, ppi_id):
        if ppi_id not in self.interactions:
            self.interactions[ppi_id] = set()
        self.interactions[ppi_id].add(partner_protein_id)

    @property
    def interaction_group(self):
        if not self.interactions:
            return "Non-interacting"
        unique_proteins = set(protein_id for protein_ids in self.interactions.values() for protein_id in protein_ids)
        if len(unique_proteins) > 1:
            return "C"  # Multiple proteins
        elif len(self.interactions) > 1:
            return "B"  # Single protein, multiple modes
        else:
            return "A"  # Single protein, single mode

    @property
    def surface_id(self):
        if self.interaction_group == "A":
            return list(self.interactions.keys())[0]  # Return the PPI ID
        return None  # For groups B and C, we don't assign a specific surface

###############################################################################################################
################################################ Class Surface ################################################
###############################################################################################################

class Surface:
    def __init__(self, protein_id):
        self.protein_id = protein_id
        self.residues = {}  # key: residue index, value: Residue object
        self.surfaces = {}  # key: ppi_id, value: set of residue indices

    def add_residue(self, index, name, pLDDT):
        if index not in self.residues:
            self.residues[index] = Residue(index, name, pLDDT)

    def add_interaction(self, residue_index, partner_protein_id, ppi_id):
        if residue_index in self.residues:
            self.residues[residue_index].add_interaction(partner_protein_id, ppi_id)
            if ppi_id not in self.surfaces:
                self.surfaces[ppi_id] = set()
            self.surfaces[ppi_id].add(residue_index)

    def get_interacting_residues(self):
        return {idx: res for idx, res in self.residues.items() if res.interactions}

    def get_residues_by_group(self):
        groups = {"A": [], "B": [], "C": []}
        for residue in self.residues.values():
            if residue.interaction_group in groups:
                groups[residue.interaction_group].append(residue)
        return groups

    def get_surfaces(self):
        return self.surfaces

###############################################################################################################
################################################ Class Protein ################################################
###############################################################################################################

class Protein(object):

    def __init__(self, graph_vertex: igraph.Vertex, unique_ID: int, logger: Logger):

        # Progress
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
    
    def add_interaction(self, residue_index, partner_protein_id, ppi_id):
        self.surface.add_interaction(residue_index, partner_protein_id, ppi_id)

    # -------------------------------------------------------------------------------------
    # -------------------------------------- Getters --------------------------------------
    # -------------------------------------------------------------------------------------
    
    def get_unique_ID(self) : return self.unique_ID
    def get_ID(self)        : return self.ID
    def get_seq(self)       : return self.seq
    def get_name(self)      : return self.name
    def get_CM(self)        : return self.PDB_chain.center_of_mass()
    
    def get_res_pLDDT(self, res_list = None) -> list[float]:
        '''Returns per residue pLDDT values as a list. If a res_list of residue
        indexes (zero index based) is passed, you will get only these pLDDT values.'''
        if res_list is not None: 
            return[self.res_pLDDT[res] for res in res_list]
        return self.res_pLDDT
    
    def get_res_centroids_xyz(self, res_list = None) -> list[np.array]:
        '''Pass a list of residue indexes as list (zero index based) to get their centroid coordinates.'''

        # Compute residue centroids
        res_xyz: list[np.array] = [ res.center_of_mass() for res  in self.PDB_chain.get_residues() ]

        # Return only the subset
        if res_list is not None: 
            return [res_xyz[res] for res in res_list]
        
        # Return everything
        return res_xyz
    
    def get_res_CA_xyz(self, res_list = None) -> list[np.array]:
        '''Pass a list of residue indexes as list (zero index based) to get their CA coordinates.'''

        # Compute residue centroids
        CA_xyz: list[np.array] = [ np.array(atom.coord) for atom in self.PDB_chain.get_atoms() if atom.get_name() == "CA" ]

        # Return only the subset
        if res_list is not None: 
            return[ CA_xyz[res] for res in res_list ]
        
        # Return everything
        return CA_xyz
    
    def get_res_names(self, res_list = None):
        '''Pass a list of residue indexes as list (zero index based) to get their residues names.'''
        if res_list is not None:
            return[self.res_names[res] for res in res_list]
        return self.res_names
    
    # -------------------------------------------------------------------------------------
    # --------------------------------------- Movers --------------------------------------
    # -------------------------------------------------------------------------------------

    def _translate_to_origin(self) -> None:

        # Translate the protein PDB_chain to the origin (0,0,0)
        CM: np.array = self.PDB_chain.center_of_mass()
        for atom in self.PDB_chain.get_atoms():
            atom.transform(np.identity(3), np.array(-CM))


    def translate(self, translation_vector: np.array) -> None:

        # Translate the protein PDB_chain in the direction of the translation_vector
        for atom in self.PDB_chain.get_atoms():
            atom.transform(np.identity(3), np.array(translation_vector))
    
    def rotate(self, reference_point, rotation_matrix) -> None:

        # Apply rotation to all atoms of PDB chain
        PDB_atoms = [atom.get_coord() for atom in self.PDB_chain.get_atoms()]
        rotated_PDB_atoms = np.dot(PDB_atoms - reference_point, rotation_matrix.T) + reference_point

        # Update the atom coordinates
        for A, atom in enumerate(self.PDB_chain.get_atoms()):
            atom.set_coord(rotated_PDB_atoms[A])


    def rotate2all(self, network_ppis: list):

        # Get protein surface residues and partners CMs
        subset_indices = []
        partners_CMs   = []
        for P, ppi in enumerate(network_ppis):

            # Verify in the PPI involves the protein and is not an homooligomeric PPI
            if self.get_ID() not in ppi.get_tuple_pair() or ppi.is_homooligomeric():
                continue
            
            # Get the correct protein partner and contact residues indexes
            if self.get_ID() == ppi.get_prot_ID_1():
                surf_idxs   : list[int] = ppi.get_contacts_res_1()
                partner     : Protein   = ppi.get_protein_2()
            elif self.get_ID() == ppi.get_prot_ID_2():
                surf_idxs   : list[int] = ppi.get_contacts_res_2()
                partner     : Protein   = ppi.get_protein_1()
            else:
                raise ValueError(f'FATAL ERROR: PPI {ppi.get_tuple_pair()} does not contain the expected protein ID {self.get_ID()}...')

            partners_CMs.append(partner.get_CM())

            for residue_index in surf_idxs:
                if residue_index not in subset_indices:
                    subset_indices.append(residue_index)

        if not partners_CMs:
            self.logger.error(f"   - No valid partner centroids found for protein: {self.get_ID()}")
            self.logger.error(f"      - partners_CMs: {partners_CMs}")
            return  # Early return if no valid partners are found
        
        # Compute partners' centroid
        if len(partners_CMs) == 1:
            partners_centroid = partners_CMs[0]
        else:
            partners_centroid = center_of_mass(partners_CMs)

        # Ensure the partners centroid is valid (not zero or NaN)
        if np.isnan(partners_centroid).any() or np.isinf(partners_centroid).any():
            self.logger.error(f"   - Invalid partners' centroid (NaN/Inf) for protein: {self.get_ID()}")
            self.logger.error(f"      - partners_CMs: {partners_CMs}")
            return
        
        # Get rotation matrix
        _, rotation_matrix, _ =\
            rotate_points(points            = self.get_res_centroids_xyz(),
                          reference_point   = self.get_CM(),
                          subset_indices    = subset_indices,
                          target_point      = partners_centroid)
        
        # Apply rotation to the PDB
        self.rotate(self.get_CM(), rotation_matrix)

    # -------------------------------------------------------------------------------------
    # --------------------------------------- Format --------------------------------------
    # -------------------------------------------------------------------------------------

    def convert_chain_to_pdb_in_memory(self) -> io.StringIO:
        """
        Saves a PDB.Chain.Chain object to an in-memory PDB file.

        Parameters:
        - chain: PDB.Chain.Chain object that you want to save

        Returns:
        - A StringIO object containing the PDB data
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

        self.PDB_chain.id = new_chain_ID
        self.chain_ID = new_chain_ID


    # -------------------------------------------------------------------------------------
    # -------------------------------------- Operators ------------------------------------
    # -------------------------------------------------------------------------------------

    def __str__(self):
        return f"Protein ID: {self.ID} --------------------------------------------\n   - Name: {self.name}\n   - Sequence: {self.seq}"


###############################################################################################################
################################################## Class PPI ##################################################
###############################################################################################################

class PPI(object):

    def __init__(self, proteins: list[Protein], graph_edge: igraph.Edge, logger: Logger) -> None:

        logger.info(f'   - Creating object of class PPI: {graph_edge["name"]}')
        
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

        for res1, res2 in zip(self.contacts_res_1, self.contacts_res_2):
            self.protein_1.add_interaction(res1, self.protein_2.get_ID(), self.get_tuple_pair())
            self.protein_2.add_interaction(res2, self.protein_1.get_ID(), self.get_tuple_pair())
    
    # -------------------------------------------------------------------------------------
    # -------------------------------------- Getters --------------------------------------
    # -------------------------------------------------------------------------------------

    def get_edge(self):
        return self.edge

    def get_tuple_pair(self):
        return tuple(sorted(self.edge['name']))
    
    def get_cluster_n(self):
        return self.cluster_n
    
    def get_protein_1(self) -> Protein:
        return self.protein_1
    
    def get_protein_2(self) -> Protein:
        return self.protein_2
    
    def get_prot_ID_1(self) -> str:
        return self.prot_ID_1
    
    def get_prot_ID_2(self) -> str:
        return self.prot_ID_2
    
    def get_contacts_res_1(self) -> list[int]:
        return self.contacts_res_1

    def get_contacts_res_2(self) -> list[int]:
        return self.contacts_res_2

    def is_homooligomeric(self):
        return len(set(self.get_tuple_pair())) == 1

    
###############################################################################################################
################################################ Class Network ################################################
###############################################################################################################

class Network(object):
    
    def __init__(self, combined_graph: igraph.Graph, logger: Logger, remove_interaction = ("Indirect",)):

        logger.info("Creating object of class Network...")

        add_contact_classification_matrix(combined_graph)

        self.logger         : Logger            = logger
        self.combined_graph : igraph.Graph      = combined_graph
        self.proteins       : list[Protein]     = [Protein(vertex, unique_ID, logger)   for unique_ID, vertex   in enumerate(combined_graph.vs)]
        self.proteins_IDs   : list[str]         = [vertex['name']                       for vertex              in combined_graph.vs]
        self.ppis           : list[PPI]         = [PPI(self.proteins, edge, logger)     for edge                in combined_graph.es if (edge['dynamics'] not in remove_interaction) and ((edge['valency']['average_matrix'] > 0).sum() > 0)]
        self.ppis_dynamics  : list[PPI]         = [edge['dynamics']                     for edge                in combined_graph.es]

        # Assign unique and incremental chain IDs to each PDB.Chain.Chain object of each protein
        for p, prot in enumerate(self.proteins):
            prot.change_PDB_chain_id(VALID_CHAIN_IDS[p])

    
    # -------------------------------------------------------------------------------------
    # -------------------------------------- Getters --------------------------------------
    # -------------------------------------------------------------------------------------
    
    def get_proteins(self) -> list[Protein]:
        '''Returns the list of proteins'''
        return self.proteins
    
    def get_proteins_IDs(self) -> list[str]:
        '''Returns the list of proteins IDs of the network'''
        return self.proteins_IDs
    
    def get_ppis(self) -> list[PPI]:
        '''Returns the list of ppis of the network'''
        return self.ppis

    # -------------------------------------------------------------------------------------
    # --------------------------------------- Adders --------------------------------------
    # -------------------------------------------------------------------------------------

    def add_new_protein(self):
        raise NotImplementedError("Method not implemented yet...")

    # -------------------------------------------------------------------------------------
    # -------------------------------------- Plotting -------------------------------------
    # -------------------------------------------------------------------------------------

    def generate_layout(self,
                        algorithm      = ["optimized", "drl", "fr", "kk", "circle", "grid", "random"][0],
                        scaling_factor = [None       , 300  , 200 , 100 , 100     , 120   , 100     ][0]):

        self.logger.info(f"INITIALIZING: 3D layout generation using {algorithm} method...")
            
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
                              classification_colors = classification_colors,
                              surface_residue_palette = default_color_palette,
                              show_plot = True):
        
        # Progress
        self.logger.info("INITIALIZING: Generating py3Dmol visualization...")
        
        # Set a different color palette for each protein
        protein_colors = cycle(surface_residue_palette.values())
        protein_color_map = {protein.get_unique_ID(): next(protein_colors) for protein in self.proteins}

        # Create a view object
        view = py3Dmol.view(width='100%', height='100%')

        # Progress
        self.logger.info('   Adding protein backbones and contact centroids...')

        # Add each protein to the viewer
        for idx, protein in enumerate(self.proteins):

            # Get palette for the protein
            base_color = protein_color_map[protein.get_unique_ID()]
            residue_groups = protein.surface.get_residues_by_group()

            # Get the maximum domain value across all proteins to generate enough unique colors
            max_domain_value = max(protein.domains)
            domain_colors = generate_unique_colors(n = max_domain_value + 1, palette = base_color)

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

            # Progress
            self.logger.info(f'      - Adding {protein.get_ID()} surface centroids')

            # Add contact residue centroids for different Surface object in the protein
            surface_colors = {}
            for i, surface_id in enumerate(protein.surface.get_surfaces().keys()):
                surface_colors[surface_id] = base_color[7 + i % (len(base_color) - 7)]
            
            for group, residues in residue_groups.items():
                if group == "A":
                    for residue in residues:
                        color = surface_colors[residue.surface_id]
                elif group == "B":
                    color = "gray"
                else:  # group C
                    color = "black"
                
                for residue in residues:
                    centroid = protein.get_res_centroids_xyz([residue.index])[0]
                    CA = protein.get_res_CA_xyz(res_list=[residue.get_index()])[0]

                    # Add the residue centroid as a Sphere
                    view.addSphere({
                        'center': {'x': float(centroid[0]), 'y': float(centroid[1]), 'z': float(centroid[2])},
                        'radius': 1.0,
                        'color': color
                    })

                    # Add a line connecting the sphere to the backbone
                    view.addCylinder({
                        'start': {'x': float(centroid[0]), 'y': float(centroid[1]), 'z': float(centroid[2])},
                        'end': {'x': float(CA[0]), 'y': float(CA[1]), 'z': float(CA[2])},
                        'radius': 0.1,
                        'color': color
                    })

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
                cylinder_radius = (0.1 + 0.4 * contact_freq) * 0.5 # Scale radius based on frequency (0 to 1)

                view.addCylinder({
                    'start': {'x': float(centroid_1[0]), 'y': float(centroid_1[1]), 'z': float(centroid_1[2])},
                    'end': {'x': float(centroid_2[0]), 'y': float(centroid_2[1]), 'z': float(centroid_2[2])},
                    'radius': cylinder_radius,
                    'color': cylinder_color,
                    'opacity': 0.9
                })

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

    # -------------------------------------------------------------------------------------
    # ------------------------------------- Operators -------------------------------------
    # -------------------------------------------------------------------------------------


    def __str__(self):

        result = f"Network with {len(self.proteins_IDs)} proteins and {len(self.ppis)} PPIs:"

        for prot in self.proteins:
            result += "\n   >" + prot.get_ID()+ ": " + prot.get_seq()
        return result

