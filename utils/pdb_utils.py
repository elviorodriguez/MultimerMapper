
import os
import numpy as np
import pandas as pd
from Bio.PDB.Chain import Chain
from Bio.SeqUtils import seq1
from Bio import PDB
from scipy.spatial.transform import Rotation

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
