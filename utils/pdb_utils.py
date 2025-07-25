
# import os
import numpy as np
import pandas as pd
from Bio.PDB.Chain import Chain
from Bio.SeqUtils import seq1
from Bio import PDB
from typing import List, Tuple, Set
import logging
# from scipy import stats
# from scipy.spatial.transform import Rotation

# from src.coordinate_analyzer import plot_traj_metadata

def get_chain_sequence(chain: Chain):
    
    chain_seq = seq1(''.join(residue.resname for residue in chain))
    
    return chain_seq

def get_domain_atoms(chain, start, end):
    return [atom for atom in chain.get_atoms() if atom.name == 'CA' and start <= atom.get_parent().id[1] <= end]


def get_domain_data(chain, start, end):
    
    domain_residues = [r for r in chain.get_residues()][start-1: end]
    
    domain_atoms = [atom for atom in chain.get_atoms() if atom.name == 'CB' or (atom.name == 'CA' and atom.get_parent().resname == 'GLY')]
    domain_atoms = [atom for atom in domain_atoms if start <= atom.get_parent().id[1] <= end]
    
    domain_coords = np.array([atom.coord for atom in domain_atoms])
    domain_plddts = np.array([atom.bfactor for atom in domain_atoms])
    
    return domain_residues, domain_atoms, domain_coords, domain_plddts


# Computes 3D distance
def calculate_distance(coord1, coord2):
    """
    Calculates and returns the Euclidean distance between two 3D coordinates.
    
    Parameters:
        - coord1 (list/array): xyz coordinates 1.
        - coord2 (list/array): xyz coordinates 2.
    
    Returns
        - distance (float): Euclidean distance between coord1 and coord2
    """
    return np.sqrt(np.sum((np.array(coord2) - np.array(coord1))**2))

def rotate_points(points, reference_point, subset_indices, target_point):
    # Extract subset of points
    subset_points = np.array([points[i] for i in subset_indices])

    # Calculate center of mass of the subset
    subset_center_of_mass = np.mean(subset_points, axis=0)

    # Calculate the vector from the reference point to the subset center of mass
    vector_to_subset_com = subset_center_of_mass - reference_point

    # Calculate the target vector
    target_vector = target_point - reference_point

    # Calculate the rotation axis using cross product
    rotation_axis = np.cross(vector_to_subset_com, target_vector)
    rotation_axis /= np.linalg.norm(rotation_axis)

    # Calculate the angle of rotation
    angle = np.arccos(np.dot(vector_to_subset_com, target_vector) /
                    (np.linalg.norm(vector_to_subset_com) * np.linalg.norm(target_vector)))

    # Perform rotation using Rodrigues' rotation formula
    rotation_matrix = np.array([[np.cos(angle) + rotation_axis[0]**2 * (1 - np.cos(angle)),
                                rotation_axis[0] * rotation_axis[1] * (1 - np.cos(angle)) - rotation_axis[2] * np.sin(angle),
                                rotation_axis[0] * rotation_axis[2] * (1 - np.cos(angle)) + rotation_axis[1] * np.sin(angle)],
                                [rotation_axis[1] * rotation_axis[0] * (1 - np.cos(angle)) + rotation_axis[2] * np.sin(angle),
                                np.cos(angle) + rotation_axis[1]**2 * (1 - np.cos(angle)),
                                rotation_axis[1] * rotation_axis[2] * (1 - np.cos(angle)) - rotation_axis[0] * np.sin(angle)],
                                [rotation_axis[2] * rotation_axis[0] * (1 - np.cos(angle)) - rotation_axis[1] * np.sin(angle),
                                rotation_axis[2] * rotation_axis[1] * (1 - np.cos(angle)) + rotation_axis[0] * np.sin(angle),
                                np.cos(angle) + rotation_axis[2]**2 * (1 - np.cos(angle))]])

    # Apply rotation to all points
    rotated_points = np.dot(points - reference_point, rotation_matrix.T) + reference_point

    return rotated_points, rotation_matrix, rotation_axis

####################################################################################

def center_of_mass(points):
   
    x_list = []
    y_list = []
    z_list = []
    
    for point in points:
        
        x_list.append(point[0])
        y_list.append(point[1])
        z_list.append(point[2])
        
    return np.array([np.mean(x_list), np.mean(y_list), np.mean(z_list)])


# ----------- Helper classes for selecting chains and domains -----------------

class ChainSelect(PDB.Select):
    def __init__(self, chain):
        self.chain = chain

    def accept_chain(self, chain):
        return chain.id == self.chain.id
    
class DomainSelect(PDB.Select):
    def __init__(self, chain, start, end):
        self.chain = chain
        self.start = start
        self.end = end

    def accept_residue(self, residue):
        return (residue.get_parent().id == self.chain.id and 
                self.start <= residue.id[1] <= self.end)
    
# ---------------------------------------------------------------------------



#################################################################################################
########################### Helper functions for core superimposition ###########################
#################################################################################################

def get_core_residues(pLDDTs: List[np.ndarray], 
                     percentile_threshold: float = 50.0,
                     cv_threshold_percentile: float = 50.0,
                     min_core_fraction: float = 0.3,
                     max_core_fraction: float = 0.8,
                     logger: logging.Logger = None) -> Set[int]:
    """
    Identify core residues based on high mean pLDDT and low coefficient of variation.
    
    Args:
        pLDDTs: List of numpy arrays, each containing per-residue pLDDT values for one model
        percentile_threshold: Percentile threshold for mean pLDDT (default: 75th percentile)
        cv_threshold_percentile: Percentile threshold for coefficient of variation (default: 25th percentile, meaning low CV)
        min_core_fraction: Minimum fraction of residues to include in core (safety check)
        max_core_fraction: Maximum fraction of residues to include in core (safety check)
    
    Returns:
        Set of residue indices (0-based) that are considered core residues
    """
    # Set up logging
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

    if not pLDDTs or len(pLDDTs) == 0:
        raise ValueError("pLDDTs list cannot be empty")
    
    # Convert to numpy array (models x residues)
    plddt_matrix = np.array(pLDDTs)
    n_models, n_residues = plddt_matrix.shape
    
    if n_models < 2:
        # If only one model, use all residues above median pLDDT
        mean_plddt = plddt_matrix[0]
        threshold = np.percentile(mean_plddt, 50)  # Use median as threshold
        core_residues = set(np.where(mean_plddt >= threshold)[0])
        return core_residues
    
    # Calculate statistics per residue across all models
    mean_plddt = np.mean(plddt_matrix, axis=0)  # Mean pLDDT per residue
    std_plddt = np.std(plddt_matrix, axis=0)    # Standard deviation per residue
    
    # Calculate coefficient of variation (CV = std/mean) to identify stable residues
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    cv_plddt = std_plddt / (mean_plddt + epsilon)
    
    # Define thresholds based on the distribution of the data
    mean_threshold = np.percentile(mean_plddt, percentile_threshold)
    cv_threshold = np.percentile(cv_plddt, cv_threshold_percentile)  # Low CV means stable
    
    # Identify core residues: high mean pLDDT AND low coefficient of variation
    high_plddt_mask = mean_plddt >= mean_threshold
    low_cv_mask = cv_plddt <= cv_threshold
    
    # Combine both criteria
    core_mask = high_plddt_mask & low_cv_mask
    core_residues = set(np.where(core_mask)[0])
    
    # Safety checks to ensure reasonable core size
    core_fraction = len(core_residues) / n_residues
    
    if core_fraction < min_core_fraction:
        # If too few core residues, relax the CV threshold
        logger.warning(f"      - Core fraction ({core_fraction:.2f}) below minimum ({min_core_fraction}). Relaxing CV threshold...")
        
        # Use only mean pLDDT threshold, but lower it
        relaxed_mean_threshold = np.percentile(mean_plddt, percentile_threshold - 20)
        core_residues = set(np.where(mean_plddt >= relaxed_mean_threshold)[0])
        core_fraction = len(core_residues) / n_residues
        
    elif core_fraction > max_core_fraction:
        # If too many core residues, make criteria more stringent
        logger.warning(f"      - Core fraction ({core_fraction:.2f}) above maximum ({max_core_fraction}). Making criteria more stringent...")
        
        # Use higher percentile thresholds
        stricter_mean_threshold = np.percentile(mean_plddt, min(95, percentile_threshold + 15))
        stricter_cv_threshold = np.percentile(cv_plddt, max(5, cv_threshold_percentile - 10))
        
        stricter_core_mask = (mean_plddt >= stricter_mean_threshold) & (cv_plddt <= stricter_cv_threshold)
        core_residues = set(np.where(stricter_core_mask)[0])
        core_fraction = len(core_residues) / n_residues
    
    logger.info(f"      - Identified {len(core_residues)} core residues ({core_fraction:.2f} of total) with mean pLDDT >= {mean_threshold:.1f} and CV <= {cv_threshold:.3f}")
    
    # Final safety check - ensure we have at least some core residues
    if len(core_residues) < 3:
        logger.warning("      - Very few core residues found. Using top 30% by mean pLDDT as fallback.")
        fallback_threshold = np.percentile(mean_plddt, 70)
        core_residues = set(np.where(mean_plddt >= fallback_threshold)[0])
    
    return core_residues


def core_superimposition(chain_atoms: List, ref_atoms: List, core_residues: Set[int],
                         fallback_rotation: np.ndarray, fallback_translation: np.ndarray,
                         logger: logging.Logger = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform superimposition using only the core residues.
    
    Args:
        chain_atoms: List of atoms from the chain to be aligned
        ref_atoms: List of atoms from the reference structure
        core_residues: Set of residue indices (0-based) to use for alignment
    
    Returns:
        Tuple of (rotation_matrix, translation_vector)
    """
    # Set up logging
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    
    if not core_residues:
        logger.error("      - No core residues provided for superimposition. Returning all atoms rotran as fallback")
        return fallback_rotation, fallback_translation
    
    if len(chain_atoms) != len(ref_atoms):
        logger.error(f"      - Atom lists have different lengths: {len(chain_atoms)} vs {len(ref_atoms)}. Returning all atoms rotran as fallback")
        return fallback_rotation, fallback_translation
    
    # Filter atoms to only include core residues
    core_chain_atoms = []
    core_ref_atoms = []
    
    for i, (chain_atom, ref_atom) in enumerate(zip(chain_atoms, ref_atoms)):
        if i in core_residues:
            core_chain_atoms.append(chain_atom)
            core_ref_atoms.append(ref_atom)
    
    if len(core_chain_atoms) < 3:
        logger.error(f"Too few core atoms for superimposition: {len(core_chain_atoms)}. "
                        "Need at least 3 atoms. Returning all atoms rotran as fallback")
        return fallback_rotation, fallback_translation
    
    logger.debug(f"      - Performing superimposition using {len(core_chain_atoms)} core atoms out of {len(chain_atoms)} total atoms")
    
    # Initialize superimposer
    super_imposer = PDB.Superimposer()
    
    # Set atoms for superimposition (reference first, then mobile)
    super_imposer.set_atoms(core_ref_atoms, core_chain_atoms)
    
    # Get rotation matrix and translation vector
    rotation_matrix, translation_vector = super_imposer.rotran
    
    return rotation_matrix, translation_vector


# Additional utility function to visualize core residue selection
def plot_core_residue_analysis(pLDDTs: List[np.ndarray], 
                              core_residues: Set[int],
                              protein_ID: str,
                              save_path: str = None,
                              logger: logging.Logger = None,
                              show_plot = False):
    """
    Create plots to visualize the core residue selection process.
    
    Args:
        pLDDTs: List of numpy arrays with per-residue pLDDT values
        core_residues: Set of core residue indices
        protein_ID: Protein identifier for plot title
        save_path: Path to save the plot (optional)
    """
    # Set up logging
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

    try:
        import matplotlib.pyplot as plt
        
        plddt_matrix = np.array(pLDDTs)
        mean_plddt = np.mean(plddt_matrix, axis=0)
        std_plddt = np.std(plddt_matrix, axis=0)
        cv_plddt = std_plddt / (mean_plddt + 1e-8)
        
        residue_indices = np.arange(len(mean_plddt))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot mean pLDDT
        ax1.plot(residue_indices, mean_plddt, 'b-', alpha=0.7, label='Mean pLDDT')
        ax1.scatter(list(core_residues), mean_plddt[list(core_residues)], 
                   c='red', s=20, label='Core residues', zorder=5)
        ax1.set_ylabel('Mean pLDDT')
        ax1.set_title(f'{protein_ID}: Core Residue Selection')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot coefficient of variation
        ax2.plot(residue_indices, cv_plddt, 'g-', alpha=0.7, label='Coefficient of Variation')
        ax2.scatter(list(core_residues), cv_plddt[list(core_residues)], 
                   c='red', s=20, label='Core residues', zorder=5)
        ax2.set_xlabel('Residue Index')
        ax2.set_ylabel('Coefficient of Variation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"      - Core residue analysis plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        plt.close()
        
    except ImportError:
        logger.error("      - Matplotlib not available for plotting")