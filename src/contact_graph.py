
import numpy as np
import pandas as pd
import igraph
import py3Dmol
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
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.transform import Rotation

from cfg.default_settings import PT_palette, DOMAIN_COLORS_RRC, SURFACE_COLORS_RRC, default_color_palette
from utils.pdb_utils import rotate_points
from src.detect_domains import format_domains_df_as_no_loops_domain_clusters


def compute_residue_forces(proteins, ppis, 
                          min_contact_distance=5.0, 
                          max_contact_distance=15.0,
                          contact_force_strength=1.0,
                          repulsion_strength=10.0,
                          global_repulsion_strength=1.0,
                          torque_strength=0.5):
    """
    Compute forces and torques on all residues in the network.
    
    Parameters:
    - min_contact_distance: Distance below which contacts become repulsive
    - max_contact_distance: Distance above which contacts become attractive
    - contact_force_strength: Strength of contact-based forces
    - repulsion_strength: Strength of non-contact repulsion
    - global_repulsion_strength: Strength of global protein repulsion
    - torque_strength: Strength of torque forces for protein orientation
    """
    
    # Initialize forces and torques
    forces = {}  # protein_id -> force vector
    torques = {}  # protein_id -> torque vector
    
    for protein in proteins:
        protein_id = protein.get_ID()
        forces[protein_id] = np.zeros(3)
        torques[protein_id] = np.zeros(3)
    
    # 1. Contact-based forces (attractive/repulsive based on frequency and distance)
    for ppi in ppis:
        p1, p2 = ppi.get_protein_1(), ppi.get_protein_2()
        p1_id, p2_id = p1.get_ID(), p2.get_ID()
        
        contacts_1 = ppi.get_contacts_res_1()
        contacts_2 = ppi.get_contacts_res_2()
        contact_freqs = ppi.contact_freq
        
        p1_centroids = p1.get_res_centroids_xyz()
        p2_centroids = p2.get_res_centroids_xyz()
        
        for i, (res1_idx, res2_idx, freq) in enumerate(zip(contacts_1, contacts_2, contact_freqs)):
            pos1 = p1_centroids[res1_idx]
            pos2 = p2_centroids[res2_idx]
            
            direction = pos2 - pos1
            distance = np.linalg.norm(direction)
            
            if distance < 1e-6:  # Avoid division by zero
                continue
                
            direction_normalized = direction / distance
            
            # Weight force by contact frequency
            weight = freq / max(contact_freqs) if contact_freqs else 1.0
            
            # Determine force magnitude based on distance
            if distance < min_contact_distance:
                # Repulsive force
                force_magnitude = repulsion_strength * weight * (min_contact_distance - distance) / min_contact_distance
                force_direction = -direction_normalized
            elif distance > max_contact_distance:
                # Attractive force
                force_magnitude = contact_force_strength * weight * (distance - max_contact_distance) / max_contact_distance
                force_direction = direction_normalized
            else:
                # Optimal range - small attractive force to maintain contact
                force_magnitude = contact_force_strength * weight * 0.1
                force_direction = direction_normalized
            
            # Apply forces
            force_vector = force_magnitude * force_direction
            forces[p1_id] += force_vector
            forces[p2_id] -= force_vector
            
            # Apply torques to orient proteins correctly
            p1_cm = p1.get_CM()
            p2_cm = p2.get_CM()
            
            # Torque on protein 1
            r1 = pos1 - p1_cm
            torque1 = np.cross(r1, force_vector) * torque_strength * weight
            torques[p1_id] += torque1
            
            # Torque on protein 2
            r2 = pos2 - p2_cm
            torque2 = np.cross(r2, -force_vector) * torque_strength * weight
            torques[p2_id] += torque2
    
    # 2. Global protein repulsion (prevent overlaps)
    for i, p1 in enumerate(proteins):
        for j, p2 in enumerate(proteins[i+1:], i+1):
            p1_id, p2_id = p1.get_ID(), p2.get_ID()
            
            p1_cm = p1.get_CM()
            p2_cm = p2.get_CM()
            
            direction = p2_cm - p1_cm
            distance = np.linalg.norm(direction)
            
            if distance < 1e-6:
                continue
                
            direction_normalized = direction / distance
            
            # Estimate protein radii
            p1_radius = get_protein_radius(p1)
            p2_radius = get_protein_radius(p2)
            min_separation = p1_radius + p2_radius + 10  # 10Å buffer
            
            if distance < min_separation:
                force_magnitude = global_repulsion_strength * (min_separation - distance) / min_separation
                force_vector = force_magnitude * direction_normalized
                
                forces[p1_id] -= force_vector
                forces[p2_id] += force_vector
    
    # 3. Non-contact residue repulsion (prevent clashes)
    for i, p1 in enumerate(proteins):
        for j, p2 in enumerate(proteins[i+1:], i+1):
            p1_id, p2_id = p1.get_ID(), p2.get_ID()
            
            # Skip if these proteins have contacts (handled above)
            has_contacts = any(ppi for ppi in ppis if 
                              set([p1_id, p2_id]) == set([ppi.get_prot_ID_1(), ppi.get_prot_ID_2()]))
            
            if has_contacts:
                continue
            
            p1_centroids = p1.get_res_centroids_xyz()
            p2_centroids = p2.get_res_centroids_xyz()
            
            for res1_idx, pos1 in enumerate(p1_centroids):
                for res2_idx, pos2 in enumerate(p2_centroids):
                    direction = pos2 - pos1
                    distance = np.linalg.norm(direction)
                    
                    if distance < 1e-6:
                        continue
                    
                    direction_normalized = direction / distance
                    
                    # Repulsive force for close non-contact residues
                    clash_distance = 8.0  # Minimum distance between non-contact residues
                    if distance < clash_distance:
                        force_magnitude = repulsion_strength * 0.5 * (clash_distance - distance) / clash_distance
                        force_vector = force_magnitude * direction_normalized
                        
                        forces[p1_id] -= force_vector
                        forces[p2_id] += force_vector
    
    return forces, torques

def apply_forces_and_torques(proteins, forces, torques, step_size=0.1, max_rotation=0.1):
    """
    Apply computed forces and torques to proteins.
    
    Parameters:
    - step_size: Translation step size
    - max_rotation: Maximum rotation angle per step (radians)
    """
    
    for protein in proteins:
        protein_id = protein.get_ID()
        
        # Apply translational forces
        if protein_id in forces:
            force = forces[protein_id]
            # Limit force magnitude to prevent instability
            force_magnitude = np.linalg.norm(force)
            if force_magnitude > 100:  # Arbitrary limit
                force = force / force_magnitude * 100
            
            translation = force * step_size
            protein.translate(translation)
        
        # Apply rotational torques
        if protein_id in torques:
            torque = torques[protein_id]
            torque_magnitude = np.linalg.norm(torque)
            
            if torque_magnitude > 1e-6:
                # Convert torque to rotation
                rotation_axis = torque / torque_magnitude
                rotation_angle = min(torque_magnitude * step_size, max_rotation)
                
                # Create rotation matrix
                rotation = Rotation.from_rotvec(rotation_angle * rotation_axis)
                rotation_matrix = rotation.as_matrix()
                
                # Apply rotation around center of mass
                protein.rotate(protein.get_CM(), rotation_matrix)

def apply_residue_level_layout(proteins, ppis, network, 
                              iterations=5000,
                              min_contact_distance=5.0,
                              max_contact_distance=15.0,
                              contact_force_strength=1.0,
                              repulsion_strength=10.0,
                              global_repulsion_strength=1.0,
                              torque_strength=0.5,
                              initial_step_size=0.5,
                              final_step_size=0.01,
                              initial_max_rotation=0.2,
                              final_max_rotation=0.01):
    """
    Apply residue-level force field layout with adaptive parameters.
    
    Parameters for optimization:
    - min_contact_distance: Distance below which contacts become repulsive
    - max_contact_distance: Distance above which contacts become attractive  
    - contact_force_strength: Strength of contact-based forces
    - repulsion_strength: Strength of non-contact repulsion
    - global_repulsion_strength: Strength of global protein repulsion
    - torque_strength: Strength of torque forces for protein orientation
    - initial_step_size: Initial translation step size (strong forces)
    - final_step_size: Final translation step size (fine tuning)
    - initial_max_rotation: Initial max rotation per step
    - final_max_rotation: Final max rotation per step
    """
    
    network.logger.info(f'Applying residue-level force field layout...')
    network.logger.info(f'   - Total iterations: {iterations}')
    network.logger.info(f'   - Contact distance range: {min_contact_distance}-{max_contact_distance}Å')
    
    # Phase 1: Strong forces for initial positioning (60% of iterations)
    phase1_iterations = int(iterations * 0.6)
    network.logger.info(f'   - Phase 1 (strong forces): {phase1_iterations} iterations')
    
    for i in range(phase1_iterations):
        if i % 500 == 0:
            network.logger.info(f'   - Phase 1 iteration {i}...')
        
        # Compute forces with higher strength
        forces, torques = compute_residue_forces(
            proteins, ppis,
            min_contact_distance=min_contact_distance,
            max_contact_distance=max_contact_distance,
            contact_force_strength=contact_force_strength * 2.0,  # Stronger initial forces
            repulsion_strength=repulsion_strength * 2.0,
            global_repulsion_strength=global_repulsion_strength * 2.0,
            torque_strength=torque_strength * 2.0
        )
        
        # Apply with larger step size
        progress = i / phase1_iterations
        current_step_size = initial_step_size * (1 - progress * 0.5)  # Gradually reduce
        current_max_rotation = initial_max_rotation * (1 - progress * 0.5)
        
        apply_forces_and_torques(proteins, forces, torques, 
                                current_step_size, current_max_rotation)
    
    # Phase 2: Medium forces for refinement (30% of iterations)  
    phase2_iterations = int(iterations * 0.3)
    network.logger.info(f'   - Phase 2 (medium forces): {phase2_iterations} iterations')
    
    for i in range(phase2_iterations):
        if i % 300 == 0:
            network.logger.info(f'   - Phase 2 iteration {i}...')
        
        forces, torques = compute_residue_forces(
            proteins, ppis,
            min_contact_distance=min_contact_distance,
            max_contact_distance=max_contact_distance,
            contact_force_strength=contact_force_strength,
            repulsion_strength=repulsion_strength,
            global_repulsion_strength=global_repulsion_strength,
            torque_strength=torque_strength
        )
        
        progress = i / phase2_iterations
        current_step_size = initial_step_size * 0.5 * (1 - progress * 0.7)
        current_max_rotation = initial_max_rotation * 0.5 * (1 - progress * 0.7)
        
        apply_forces_and_torques(proteins, forces, torques,
                                current_step_size, current_max_rotation)
    
    # Phase 3: Fine tuning (10% of iterations)
    phase3_iterations = iterations - phase1_iterations - phase2_iterations
    network.logger.info(f'   - Phase 3 (fine tuning): {phase3_iterations} iterations')
    
    for i in range(phase3_iterations):
        if i % 100 == 0:
            network.logger.info(f'   - Phase 3 iteration {i}...')
        
        forces, torques = compute_residue_forces(
            proteins, ppis,
            min_contact_distance=min_contact_distance,
            max_contact_distance=max_contact_distance,
            contact_force_strength=contact_force_strength * 0.3,  # Weaker forces
            repulsion_strength=repulsion_strength * 0.3,
            global_repulsion_strength=global_repulsion_strength * 0.3,
            torque_strength=torque_strength * 0.3
        )
        
        apply_forces_and_torques(proteins, forces, torques,
                                final_step_size, final_max_rotation)
    
    network.logger.info('Residue-level force field layout completed')

def optimize_residue_layout(network, iterations=5000, **kwargs):
    """
    Main function to replace the existing optimize_layout function.
    """
    proteins = network.get_proteins()
    ppis = network.get_ppis()
    
    # Initialize with basic spacing to avoid overlaps
    initialize_positions(proteins, ppis, get_connected_components(proteins, ppis, network.logger))
    
    # Apply residue-level layout
    apply_residue_level_layout(proteins, ppis, network, iterations=iterations, **kwargs)

# Modified Network.generate_layout method
def generate_layout_fine(self,
                   algorithm="residue_optimized",
                   iterations=5000,
                   # Residue-level parameters
                   min_contact_distance=5.0,
                   max_contact_distance=15.0,
                   contact_force_strength=1.0,
                   repulsion_strength=10.0,
                   global_repulsion_strength=1.0,
                   torque_strength=0.5,
                   initial_step_size=0.5,
                   final_step_size=0.01,
                   initial_max_rotation=0.2,
                   final_max_rotation=0.01,
                   # Legacy parameters
                   scaling_factor=None,
                   # New
                   min_interprotein_distance=150.0,  # Minimum distance between protein centers
                   surface_alignment_strength=8.0,   # Strength of surface face-to-face alignment
                   line_separation_strength=25.0,
                   n_contacts_sample=20):
    """
    Generate 3D layout with residue-level force field optimization.
    
    Optimizable parameters:
    - min_contact_distance: Distance below which contacts become repulsive (3.0-8.0Å)
    - max_contact_distance: Distance above which contacts become attractive (10.0-20.0Å)
    - contact_force_strength: Strength of contact-based forces (0.1-5.0)
    - repulsion_strength: Strength of non-contact repulsion (1.0-50.0)
    - global_repulsion_strength: Strength of global protein repulsion (0.1-10.0)
    - torque_strength: Strength of torque forces for protein orientation (0.1-2.0)
    - initial_step_size: Initial translation step size (0.1-1.0)
    - final_step_size: Final translation step size (0.001-0.1)
    - initial_max_rotation: Initial max rotation per step (0.05-0.5 radians)
    - final_max_rotation: Final max rotation per step (0.001-0.1 radians)
    """

    self.logger.info(f"INITIALIZING: 3D layout generation using {algorithm} method...")
    
    # if len(self.proteins) < 3:
    #     self.logger.warning('   - Less than 3 proteins in the network. Using legacy drl method...')
    #     algorithm = "drl"
    #     scaling_factor = 250

    if algorithm == "residue_optimized":
        optimize_residue_layout_fast(self, 
                               iterations=iterations,
                               min_contact_distance=min_contact_distance,
                               max_contact_distance=max_contact_distance,
                               contact_force_strength=contact_force_strength,
                               repulsion_strength=repulsion_strength,
                               global_repulsion_strength=global_repulsion_strength,
                               torque_strength=torque_strength,
                               initial_step_size=initial_step_size,
                               final_step_size=final_step_size,
                               initial_max_rotation=initial_max_rotation,
                               final_max_rotation=final_max_rotation,
                               min_interprotein_distance=min_interprotein_distance,  # Minimum distance between protein centers
                               surface_alignment_strength=surface_alignment_strength,   # Strength of surface face-to-face alignment
                               line_separation_strength=line_separation_strength,
                               n_contacts_sample=n_contacts_sample)
    
    elif algorithm == "optimized":
        optimize_layout(self, iterations=iterations)
    
    else:
        # Legacy methods
        self.logger.info(f"   - Scaling factor: {scaling_factor}...")

        for prot in self.proteins:
            self.logger.info(f"   - Translating {prot.get_ID()} to the origin first...")
            prot._translate_to_origin()

        layt = [list(np.array(coord) * scaling_factor) for coord in list(self.combined_graph.layout(algorithm, dim=3))]

        for p, prot in enumerate(self.proteins):
            self.logger.info(f"   - Translating {prot.get_ID()} to new layout positions...")
            prot.translate(layt[p])

        for prot in self.proteins:
            self.logger.info(f"   - Rotating {prot.get_ID()} with respect to their partners CMs...")
            prot.rotate2all(self.ppis)
        
    self.logger.info("FINISHED: 3D layout generation algorithm")

############################# Vectorized Methods #############################

# OLD METHOD
# def compute_residue_forces_vectorized(proteins, ppis, 
#                                      min_contact_distance=5.0, 
#                                      max_contact_distance=15.0,
#                                      contact_force_strength=1.0,
#                                      repulsion_strength=10.0,
#                                      global_repulsion_strength=1.0,
#                                      torque_strength=0.5):
#     """
#     Vectorized computation of forces and torques on all residues.
#     """
    
#     # Initialize forces and torques
#     forces = {}
#     torques = {}
    
#     for protein in proteins:
#         protein_id = protein.get_ID()
#         forces[protein_id] = np.zeros(3)
#         torques[protein_id] = np.zeros(3)
    
#     # 1. VECTORIZED Contact-based forces
#     for ppi in ppis:
#         p1, p2 = ppi.get_protein_1(), ppi.get_protein_2()
#         p1_id, p2_id = p1.get_ID(), p2.get_ID()
        
#         contacts_1 = np.array(ppi.get_contacts_res_1())
#         contacts_2 = np.array(ppi.get_contacts_res_2())
#         contact_freqs = np.array(ppi.contact_freq)
        
#         if len(contacts_1) == 0:
#             continue
            
#         # Get all relevant centroids at once
#         p1_centroids = np.array(p1.get_res_centroids_xyz())
#         p2_centroids = np.array(p2.get_res_centroids_xyz())
        
#         # Vectorized contact positions
#         pos1_contacts = p1_centroids[contacts_1]  # Shape: (n_contacts, 3)
#         pos2_contacts = p2_centroids[contacts_2]  # Shape: (n_contacts, 3)
        
#         # Vectorized distance calculations
#         directions = pos2_contacts - pos1_contacts  # Shape: (n_contacts, 3)
#         distances = np.linalg.norm(directions, axis=1)  # Shape: (n_contacts,)
        
#         # Avoid division by zero
#         valid_contacts = distances > 1e-6
#         if not np.any(valid_contacts):
#             continue
            
#         # Filter valid contacts
#         distances = distances[valid_contacts]
#         directions = directions[valid_contacts]
#         contact_freqs = contact_freqs[valid_contacts]
#         pos1_contacts = pos1_contacts[valid_contacts]
#         pos2_contacts = pos2_contacts[valid_contacts]
        
#         # Normalize directions
#         directions_normalized = directions / distances[:, np.newaxis]
        
#         # Vectorized force calculations
#         weights = contact_freqs / np.max(contact_freqs) if len(contact_freqs) > 0 else np.ones_like(contact_freqs)
        
#         # Determine force magnitudes based on distance ranges
#         repulsive_mask = distances < min_contact_distance
#         attractive_mask = distances > max_contact_distance
#         optimal_mask = ~(repulsive_mask | attractive_mask)
        
#         force_magnitudes = np.zeros_like(distances)
        
#         # Repulsive forces
#         if np.any(repulsive_mask):
#             force_magnitudes[repulsive_mask] = (
#                 repulsion_strength * weights[repulsive_mask] * 
#                 (min_contact_distance - distances[repulsive_mask]) / min_contact_distance
#             )
#             directions_normalized[repulsive_mask] *= -1  # Repulsive direction
        
#         # Attractive forces
#         if np.any(attractive_mask):
#             force_magnitudes[attractive_mask] = (
#                 contact_force_strength * weights[attractive_mask] * 
#                 (distances[attractive_mask] - max_contact_distance) / max_contact_distance
#             )
        
#         # Optimal range forces
#         if np.any(optimal_mask):
#             force_magnitudes[optimal_mask] = contact_force_strength * weights[optimal_mask] * 0.1
        
#         # Vectorized force vectors
#         force_vectors = force_magnitudes[:, np.newaxis] * directions_normalized
        
#         # Sum all forces for each protein
#         total_force_p1 = np.sum(force_vectors, axis=0)
#         total_force_p2 = -np.sum(force_vectors, axis=0)
        
#         forces[p1_id] += total_force_p1
#         forces[p2_id] += total_force_p2
        
#         # Vectorized torque calculations
#         p1_cm = p1.get_CM()
#         p2_cm = p2.get_CM()
        
#         # Torque vectors
#         r1_vectors = pos1_contacts - p1_cm  # Shape: (n_contacts, 3)
#         r2_vectors = pos2_contacts - p2_cm  # Shape: (n_contacts, 3)
        
#         # Vectorized cross products
#         torque1_vectors = np.cross(r1_vectors, force_vectors) * torque_strength * weights[:, np.newaxis]
#         torque2_vectors = np.cross(r2_vectors, -force_vectors) * torque_strength * weights[:, np.newaxis]
        
#         # Sum torques
#         total_torque_p1 = np.sum(torque1_vectors, axis=0)
#         total_torque_p2 = np.sum(torque2_vectors, axis=0)
        
#         torques[p1_id] += total_torque_p1
#         torques[p2_id] += total_torque_p2
    
#     # 2. VECTORIZED Global protein repulsion
#     n_proteins = len(proteins)
#     if n_proteins > 1:
#         # Get all protein centers at once
#         protein_cms = np.array([protein.get_CM() for protein in proteins])
#         protein_ids = [protein.get_ID() for protein in proteins]
        
#         # Vectorized distance matrix
#         cm_distances = squareform(pdist(protein_cms))
        
#         # Get protein radii
#         protein_radii = np.array([get_protein_radius(protein) for protein in proteins])
        
#         # Vectorized repulsion calculations
#         for i in range(n_proteins):
#             for j in range(i + 1, n_proteins):
#                 distance = cm_distances[i, j]
#                 if distance < 1e-6:
#                     continue
                
#                 direction = protein_cms[j] - protein_cms[i]
#                 direction_normalized = direction / distance
                
#                 min_separation = protein_radii[i] + protein_radii[j] + 10  # 10Å buffer
                
#                 if distance < min_separation:
#                     force_magnitude = global_repulsion_strength * (min_separation - distance) / min_separation
#                     force_vector = force_magnitude * direction_normalized
                    
#                     forces[protein_ids[i]] -= force_vector
#                     forces[protein_ids[j]] += force_vector
    
#     # 3. OPTIMIZED Non-contact residue repulsion (only for close proteins)
#     clash_distance = 8.0
    
#     for i in range(len(proteins)):
#         for j in range(i + 1, len(proteins)):
#             p1, p2 = proteins[i], proteins[j]
#             p1_id, p2_id = p1.get_ID(), p2.get_ID()
            
#             # Skip if these proteins have contacts (already handled)
#             has_contacts = any(ppi for ppi in ppis if 
#                               set([p1_id, p2_id]) == set([ppi.get_prot_ID_1(), ppi.get_prot_ID_2()]))
            
#             if has_contacts:
#                 continue
            
#             # Quick distance check - only process if proteins are close
#             p1_cm, p2_cm = p1.get_CM(), p2.get_CM()
#             protein_distance = np.linalg.norm(p2_cm - p1_cm)
#             p1_radius = get_protein_radius(p1)
#             p2_radius = get_protein_radius(p2)
            
#             # Only process if proteins are close enough to potentially clash
#             if protein_distance > (p1_radius + p2_radius + clash_distance * 2):
#                 continue
            
#             # Vectorized residue-residue repulsion
#             p1_centroids = np.array(p1.get_res_centroids_xyz())
#             p2_centroids = np.array(p2.get_res_centroids_xyz())
            
#             # Compute all pairwise distances at once
#             # This creates a (n_res1, n_res2, 3) array of difference vectors
#             diff_vectors = p2_centroids[np.newaxis, :, :] - p1_centroids[:, np.newaxis, :]  # Broadcasting
#             distances_matrix = np.linalg.norm(diff_vectors, axis=2)  # Shape: (n_res1, n_res2)
            
#             # Find clashing residue pairs
#             clash_mask = distances_matrix < clash_distance
            
#             if not np.any(clash_mask):
#                 continue
            
#             # Get indices of clashing pairs
#             clash_indices = np.where(clash_mask)
#             clash_distances = distances_matrix[clash_indices]
#             clash_directions = diff_vectors[clash_indices]
            
#             # Vectorized force calculations for clashing residues
#             clash_directions_normalized = clash_directions / clash_distances[:, np.newaxis]
#             force_magnitudes = repulsion_strength * 0.5 * (clash_distance - clash_distances) / clash_distance
#             clash_forces = force_magnitudes[:, np.newaxis] * clash_directions_normalized
            
#             # Sum forces for each protein
#             total_clash_force_p1 = -np.sum(clash_forces, axis=0)
#             total_clash_force_p2 = np.sum(clash_forces, axis=0)
            
#             forces[p1_id] += total_clash_force_p1
#             forces[p2_id] += total_clash_force_p2
    
#     return forces, torques

def compute_residue_forces_vectorized(proteins, ppis, 
                                     min_contact_distance=5.0, 
                                     max_contact_distance=15.0,
                                     contact_force_strength=1.0,
                                     repulsion_strength=10.0,
                                     global_repulsion_strength=1.0,
                                     torque_strength=0.5,
                                     # NEW PARAMETERS
                                     min_interprotein_distance=50.0,
                                     surface_alignment_strength=5.0,
                                     line_separation_strength=20.0,
                                     n_contacts_sample=15):
    forces = {}
    torques = {}
    
    for protein in proteins:
        protein_id = protein.get_ID()
        forces[protein_id] = np.zeros(3)
        torques[protein_id] = np.zeros(3)
    
    # 1. ENHANCED Contact-based forces with surface alignment
    for ppi in ppis:
        p1, p2 = ppi.get_protein_1(), ppi.get_protein_2()
        p1_id, p2_id = p1.get_ID(), p2.get_ID()
        
        contacts_1 = np.array(ppi.get_contacts_res_1())
        contacts_2 = np.array(ppi.get_contacts_res_2())
        contact_freqs = np.array(ppi.contact_freq)
        
        if len(contacts_1) == 0:
            continue

        # Sample contacts if n_contacts_sample is specified
        if n_contacts_sample is not None and len(contacts_1) > n_contacts_sample:
            sample_indices = np.random.choice(len(contacts_1), size=n_contacts_sample, replace=False)
            contacts_1 = contacts_1[sample_indices]
            contacts_2 = contacts_2[sample_indices]
            contact_freqs = contact_freqs[sample_indices]
            
        p1_centroids = np.array(p1.get_res_centroids_xyz())
        p2_centroids = np.array(p2.get_res_centroids_xyz())
        
        pos1_contacts = p1_centroids[contacts_1]
        pos2_contacts = p2_centroids[contacts_2]
        
        # Calculate surface centers
        surface1_center = np.mean(pos1_contacts, axis=0)
        surface2_center = np.mean(pos2_contacts, axis=0)
        
        directions = pos2_contacts - pos1_contacts
        distances = np.linalg.norm(directions, axis=1)
        
        valid_contacts = distances > 1e-6
        if not np.any(valid_contacts):
            continue
        
        distances = distances[valid_contacts]
        directions = directions[valid_contacts]
        contact_freqs = contact_freqs[valid_contacts]
        pos1_contacts = pos1_contacts[valid_contacts]
        pos2_contacts = pos2_contacts[valid_contacts]
        
        directions_normalized = directions / distances[:, np.newaxis]
        weights = contact_freqs / np.max(contact_freqs) if len(contact_freqs) > 0 else np.ones_like(contact_freqs)
        
        # Standard contact forces
        repulsive_mask = distances < min_contact_distance
        attractive_mask = distances > max_contact_distance
        optimal_mask = ~(repulsive_mask | attractive_mask)
        
        force_magnitudes = np.zeros_like(distances)
        
        if np.any(repulsive_mask):
            force_magnitudes[repulsive_mask] = (
                repulsion_strength * weights[repulsive_mask] * 
                (min_contact_distance - distances[repulsive_mask]) / min_contact_distance
            )
            directions_normalized[repulsive_mask] *= -1
        
        if np.any(attractive_mask):
            force_magnitudes[attractive_mask] = (
                contact_force_strength * weights[attractive_mask] * 
                (distances[attractive_mask] - max_contact_distance) / max_contact_distance
            )
        
        if np.any(optimal_mask):
            force_magnitudes[optimal_mask] = contact_force_strength * weights[optimal_mask] * 0.1
        
        force_vectors = force_magnitudes[:, np.newaxis] * directions_normalized
        
        total_force_p1 = np.sum(force_vectors, axis=0)
        total_force_p2 = -np.sum(force_vectors, axis=0)
        
        forces[p1_id] += total_force_p1
        forces[p2_id] += total_force_p2
        
        # Enhanced torque calculation with surface alignment
        p1_cm = p1.get_CM()
        p2_cm = p2.get_CM()
        
        # Standard torques
        r1_vectors = pos1_contacts - p1_cm
        r2_vectors = pos2_contacts - p2_cm
        
        torque1_vectors = np.cross(r1_vectors, force_vectors) * torque_strength * weights[:, np.newaxis]
        torque2_vectors = np.cross(r2_vectors, -force_vectors) * torque_strength * weights[:, np.newaxis]
        
        total_torque_p1 = np.sum(torque1_vectors, axis=0)
        total_torque_p2 = np.sum(torque2_vectors, axis=0)
        
        # NEW: Surface alignment torques (magnetic-like force)
        # Create vectors from protein centers to surface centers
        p1_to_surface1 = surface1_center - p1_cm
        p2_to_surface2 = surface2_center - p2_cm
        
        # Vector between protein centers
        protein_axis = p2_cm - p1_cm
        protein_axis_normalized = protein_axis / (np.linalg.norm(protein_axis) + 1e-8)
        
        # Calculate alignment torques to orient surfaces face-to-face
        # For p1: align surface1 to point toward p2
        desired_p1_surface_dir = protein_axis_normalized
        current_p1_surface_dir = p1_to_surface1 / (np.linalg.norm(p1_to_surface1) + 1e-8)
        alignment_torque_p1 = np.cross(current_p1_surface_dir, desired_p1_surface_dir) * surface_alignment_strength
        
        # For p2: align surface2 to point toward p1
        desired_p2_surface_dir = -protein_axis_normalized
        current_p2_surface_dir = p2_to_surface2 / (np.linalg.norm(p2_to_surface2) + 1e-8)
        alignment_torque_p2 = np.cross(current_p2_surface_dir, desired_p2_surface_dir) * surface_alignment_strength
        
        total_torque_p1 += alignment_torque_p1
        total_torque_p2 += alignment_torque_p2
        
        torques[p1_id] += total_torque_p1
        torques[p2_id] += total_torque_p2
    
    # 2. ENHANCED Global protein repulsion with minimum distance
    n_proteins = len(proteins)
    if n_proteins > 1:
        protein_cms = np.array([protein.get_CM() for protein in proteins])
        protein_ids = [protein.get_ID() for protein in proteins]
        
        cm_distances = squareform(pdist(protein_cms))
        protein_radii = np.array([get_protein_radius(protein) for protein in proteins])
        
        for i in range(n_proteins):
            for j in range(i + 1, n_proteins):
                distance = cm_distances[i, j]
                if distance < 1e-6:
                    continue
                
                direction = protein_cms[j] - protein_cms[i]
                direction_normalized = direction / distance
                
                # Use the larger of: radius-based separation or minimum distance parameter
                radius_based_separation = protein_radii[i] + protein_radii[j] + 10
                min_separation = max(radius_based_separation, min_interprotein_distance)
                
                if distance < min_separation:
                    # Stronger repulsion force
                    force_magnitude = global_repulsion_strength * 2.0 * (min_separation - distance) / min_separation
                    force_vector = force_magnitude * direction_normalized
                    
                    forces[protein_ids[i]] -= force_vector
                    forces[protein_ids[j]] += force_vector
    
    # 3. NEW: Line separation forces to prevent contact lines from crossing
    ppi_contact_lines = []
    for ppi in ppis:
        p1, p2 = ppi.get_protein_1(), ppi.get_protein_2()
        p1_id, p2_id = p1.get_ID(), p2.get_ID()
        
        contacts_1 = np.array(ppi.get_contacts_res_1())
        contacts_2 = np.array(ppi.get_contacts_res_2())
        
        if len(contacts_1) == 0:
            continue

        # Sample contacts if n_contacts_sample is specified
        if n_contacts_sample is not None and len(contacts_1) > n_contacts_sample:
            sample_indices = np.random.choice(len(contacts_1), size=n_contacts_sample, replace=False)
            contacts_1 = contacts_1[sample_indices]
            contacts_2 = contacts_2[sample_indices]
        
        p1_centroids = np.array(p1.get_res_centroids_xyz())
        p2_centroids = np.array(p2.get_res_centroids_xyz())
        
        pos1_contacts = p1_centroids[contacts_1]
        pos2_contacts = p2_centroids[contacts_2]
        
        # Store line segments for each contact
        for i in range(len(contacts_1)):
            ppi_contact_lines.append({
                'start': pos1_contacts[i],
                'end': pos2_contacts[i],
                'p1_id': p1_id,
                'p2_id': p2_id,
                'p1_res_idx': contacts_1[i],
                'p2_res_idx': contacts_2[i]
            })
    
    # Apply line separation forces
    for i, line1 in enumerate(ppi_contact_lines):
        for j, line2 in enumerate(ppi_contact_lines[i+1:], start=i+1):
            # Skip if lines are from the same PPI
            if (line1['p1_id'] == line2['p1_id'] and line1['p2_id'] == line2['p2_id']) or \
               (line1['p1_id'] == line2['p2_id'] and line1['p2_id'] == line2['p1_id']):
                continue
            
            # Calculate minimum distance between line segments
            line1_vec = line1['end'] - line1['start']
            line2_vec = line2['end'] - line2['start']
            w0 = line1['start'] - line2['start']
            
            a = np.dot(line1_vec, line1_vec)
            b = np.dot(line1_vec, line2_vec)
            c = np.dot(line2_vec, line2_vec)
            d = np.dot(line1_vec, w0)
            e = np.dot(line2_vec, w0)
            
            denominator = a*c - b*b
            if abs(denominator) < 1e-6:
                sc = 0.0
                tc = 0.0
            else:
                sc = (b*e - c*d) / denominator
                tc = (a*e - b*d) / denominator
            
            sc = np.clip(sc, 0, 1)
            tc = np.clip(tc, 0, 1)
            
            closest_p1 = line1['start'] + sc * line1_vec
            closest_p2 = line2['start'] + tc * line2_vec
            
            separation_vec = closest_p2 - closest_p1
            separation_distance = np.linalg.norm(separation_vec)
            
            min_line_separation = 15.0  # Minimum distance between contact lines
            if separation_distance < min_line_separation and separation_distance > 1e-6:
                force_magnitude = line_separation_strength * (min_line_separation - separation_distance) / min_line_separation
                force_direction = separation_vec / separation_distance
                
                # Apply forces to the proteins involved in both lines
                forces[line1['p1_id']] -= force_direction * force_magnitude * (1 - sc)
                forces[line1['p2_id']] -= force_direction * force_magnitude * sc
                forces[line2['p1_id']] += force_direction * force_magnitude * (1 - tc)
                forces[line2['p2_id']] += force_direction * force_magnitude * tc
    
    # 4. OPTIMIZED Non-contact residue repulsion (only for close proteins)
    clash_distance = 8.0
    
    for i in range(len(proteins)):
        for j in range(i + 1, len(proteins)):
            p1, p2 = proteins[i], proteins[j]
            p1_id, p2_id = p1.get_ID(), p2.get_ID()
            
            # Skip if these proteins have contacts (already handled)
            has_contacts = any(ppi for ppi in ppis if 
                              set([p1_id, p2_id]) == set([ppi.get_prot_ID_1(), ppi.get_prot_ID_2()]))
            
            if has_contacts:
                continue
            
            # Quick distance check
            p1_cm, p2_cm = p1.get_CM(), p2.get_CM()
            protein_distance = np.linalg.norm(p2_cm - p1_cm)
            p1_radius = get_protein_radius(p1)
            p2_radius = get_protein_radius(p2)
            
            if protein_distance > (p1_radius + p2_radius + clash_distance * 2):
                continue
            
            # Vectorized residue-residue repulsion
            p1_centroids = np.array(p1.get_res_centroids_xyz())
            p2_centroids = np.array(p2.get_res_centroids_xyz())
            
            diff_vectors = p2_centroids[np.newaxis, :, :] - p1_centroids[:, np.newaxis, :]
            distances_matrix = np.linalg.norm(diff_vectors, axis=2)
            
            clash_mask = distances_matrix < clash_distance
            
            if not np.any(clash_mask):
                continue
            
            clash_indices = np.where(clash_mask)
            clash_distances = distances_matrix[clash_indices]
            clash_directions = diff_vectors[clash_indices]
            
            clash_directions_normalized = clash_directions / clash_distances[:, np.newaxis]
            force_magnitudes = repulsion_strength * 0.5 * (clash_distance - clash_distances) / clash_distance
            clash_forces = force_magnitudes[:, np.newaxis] * clash_directions_normalized
            
            total_clash_force_p1 = -np.sum(clash_forces, axis=0)
            total_clash_force_p2 = np.sum(clash_forces, axis=0)
            
            forces[p1_id] += total_clash_force_p1
            forces[p2_id] += total_clash_force_p2
    
    return forces, torques

def apply_residue_level_layout_optimized(proteins, ppis, network, 
                                        iterations=5000,
                                        min_contact_distance=5.0,
                                        max_contact_distance=15.0,
                                        contact_force_strength=1.0,
                                        repulsion_strength=10.0,
                                        global_repulsion_strength=1.0,
                                        torque_strength=0.5,
                                        initial_step_size=0.5,
                                        final_step_size=0.01,
                                        initial_max_rotation=0.2,
                                        final_max_rotation=0.01,
                                        min_interprotein_distance=150.0,
                                        surface_alignment_strength=8.0,
                                        line_separation_strength=25.0,
                                        n_contacts_sample=20
                                        ):
    """
    Optimized residue-level force field layout with vectorized operations.
    """
    
    network.logger.info(f'Applying OPTIMIZED residue-level force field layout...')
    network.logger.info(f'   - Total iterations: {iterations}')
    
    # Reduce logging frequency to improve performance
    log_interval = max(100, iterations // 50)  # Log at most 50 times
    
    # Phase 1: Strong forces (60% of iterations)
    phase1_iterations = int(iterations * 0.6)
    network.logger.info(f'   - Phase 1: {phase1_iterations} iterations')
    
    for i in range(phase1_iterations):
        if i % log_interval == 0:
            network.logger.info(f'   - Phase 1 iteration {i}...')
        
        forces, torques = compute_residue_forces_vectorized(
            proteins, ppis,
            min_contact_distance=min_contact_distance,
            max_contact_distance=max_contact_distance,
            contact_force_strength=contact_force_strength * 2.0,
            repulsion_strength=repulsion_strength * 2.0,
            global_repulsion_strength=global_repulsion_strength * 2.0,
            torque_strength=torque_strength * 2.0,
            min_interprotein_distance=min_interprotein_distance,
            surface_alignment_strength=surface_alignment_strength,
            line_separation_strength=line_separation_strength,
            n_contacts_sample=n_contacts_sample
        )
        
        progress = i / phase1_iterations
        current_step_size = initial_step_size * (1 - progress * 0.5)
        current_max_rotation = initial_max_rotation * (1 - progress * 0.5)
        
        apply_forces_and_torques(proteins, forces, torques, 
                                current_step_size, current_max_rotation)
    
    # Phase 2: Medium forces (30% of iterations)
    phase2_iterations = int(iterations * 0.3)
    network.logger.info(f'   - Phase 2: {phase2_iterations} iterations')
    
    for i in range(phase2_iterations):
        if i % log_interval == 0:
            network.logger.info(f'   - Phase 2 iteration {i}...')
        
        forces, torques = compute_residue_forces_vectorized(
            proteins, ppis,
            min_contact_distance=min_contact_distance,
            max_contact_distance=max_contact_distance,
            contact_force_strength=contact_force_strength,
            repulsion_strength=repulsion_strength,
            global_repulsion_strength=global_repulsion_strength,
            torque_strength=torque_strength,
            min_interprotein_distance=min_interprotein_distance,
            surface_alignment_strength=surface_alignment_strength,
            line_separation_strength=line_separation_strength,
            n_contacts_sample=n_contacts_sample
        )
        
        progress = i / phase2_iterations
        current_step_size = initial_step_size * 0.5 * (1 - progress * 0.7)
        current_max_rotation = initial_max_rotation * 0.5 * (1 - progress * 0.7)
        
        apply_forces_and_torques(proteins, forces, torques,
                                current_step_size, current_max_rotation)
    
    # Phase 3: Fine tuning (10% of iterations)
    phase3_iterations = iterations - phase1_iterations - phase2_iterations
    network.logger.info(f'   - Phase 3: {phase3_iterations} iterations')
    
    for i in range(phase3_iterations):
        if i % log_interval == 0:
            network.logger.info(f'   - Phase 3 iteration {i}...')
        
        forces, torques = compute_residue_forces_vectorized(
            proteins, ppis,
            min_contact_distance=min_contact_distance,
            max_contact_distance=max_contact_distance,
            contact_force_strength=contact_force_strength * 0.3,
            repulsion_strength=repulsion_strength * 0.3,
            global_repulsion_strength=global_repulsion_strength * 0.3,
            torque_strength=torque_strength * 0.3,
            min_interprotein_distance=min_interprotein_distance,
            surface_alignment_strength=surface_alignment_strength,
            line_separation_strength=line_separation_strength,
            n_contacts_sample=n_contacts_sample
        )
        
        apply_forces_and_torques(proteins, forces, torques,
                                final_step_size, final_max_rotation)
    
    network.logger.info('Optimized residue-level layout completed')

def optimize_residue_layout_fast(network, iterations=100, **kwargs):
    """
    Fast version of residue layout optimization.
    """
    proteins = network.get_proteins()
    ppis = network.get_ppis()
    
    # Initialize with basic spacing
    components = get_connected_components(proteins, ppis, network.logger)
    initialize_positions(proteins, ppis, components)
    
    # Apply optimized residue-level layout
    apply_residue_level_layout_optimized(proteins, ppis, network, iterations=iterations, **kwargs)

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

def safe_normalize(v, epsilon=1e-8):
    """
    Safely normalize a vector, returning a zero vector if the norm is very small.
    
    Args:
        v (np.array): The vector to normalize.
        epsilon (float): Small value to avoid division by zero.
    
    Returns:
        np.array: The normalized vector, or a zero vector if the input has very small norm.
    """
    norm = np.linalg.norm(v)
    if norm < epsilon:
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
    np.fill_diagonal(distances, np.inf)  # Avoid self-interactions
    
    # Repulsive forces between all proteins
    for i in range(n):
        for j in range(i+1, n):
            direction = positions[i] - positions[j]
            distance = np.linalg.norm(direction)
            if distance > 0:
                min_distance = radii[i] + radii[j]
                if distance < min_distance:
                    force = 1000 * (min_distance - distance) / min_distance
                    direction = safe_normalize(direction)
                    forces[i] += force * direction
                    forces[j] -= force * direction
    
    # Attractive forces for proteins connected by PPIs
    for ppi in ppis:
        p1 = proteins.index(ppi.get_protein_1())
        p2 = proteins.index(ppi.get_protein_2())
        
        direction = positions[p2] - positions[p1]
        distance = np.linalg.norm(direction)
        if distance > 0:
            min_distance = radii[p1] + radii[p2]
            if distance > min_distance:
                force = 0.2 * (distance - min_distance)
                direction = safe_normalize(direction)
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
            if abs(denominator) < 1e-6:  # Near-parallel lines
                sc = 0.0
                tc = 0.0
            else:
                sc = (b*e - c*d) / denominator
                tc = (a*e - b*d) / denominator
            
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
    for ppi in ppis:
        p1 = proteins.index(ppi.get_protein_1())
        p2 = proteins.index(ppi.get_protein_2())
        line = positions[p2] - positions[p1]
        line_dir = safe_normalize(line)
        
        for k, protein in enumerate(proteins):
            if k != p1 and k != p2:
                protein_radius = radii[k]
                protein_to_line = np.cross(line, positions[k] - positions[p1])
                distance_to_line = np.linalg.norm(protein_to_line) / (np.linalg.norm(line) + 1e-8)
                
                if distance_to_line < protein_radius + 15:  # Add some buffer
                    force = 15 * (protein_radius + 15 - distance_to_line)
                    direction = safe_normalize(np.cross(line_dir, protein_to_line))
                    forces[p1] += force * direction
                    forces[p2] += force * direction
                    forces[k] -= force * direction
    
    # Add a small random perturbation to break symmetry
    forces += np.random.normal(0, 0.1, forces.shape)
    
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
        
        # Check for invalid forces
        if np.any(np.isnan(forces)) or np.any(np.isinf(forces)):
            network.logger.warning(f"Invalid forces encountered in iteration {i}")
            forces = np.nan_to_num(forces, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # Use an adaptive damping factor with slower decay
        damping_factor = 1 - (i / iterations)**0.5
        step_size = 0.05 * damping_factor
        
        # Update positions with collision detection
        new_positions = positions + forces * step_size
        
        # Check for collisions and adjust
        for j in range(len(proteins)):
            for k in range(j+1, len(proteins)):
                distance = np.linalg.norm(new_positions[j] - new_positions[k])
                min_distance = radii[j] + radii[j]
                if distance < min_distance:
                    # Move proteins apart to prevent overlap
                    direction = safe_normalize(new_positions[j] - new_positions[k])
                    overlap = min_distance - distance
                    new_positions[j] += direction * overlap * 0.5
                    new_positions[k] -= direction * overlap * 0.5
        
        positions = new_positions
        
        # Ensure components don't drift too far apart
        if len(components) > 1:
            global_center = np.mean(positions, axis=0)
            max_distance = 1000  # Maximum distance from global center
            for component in components:
                component_indices = [proteins.index(p) for p in proteins if p.get_ID() in component]
                component_center = np.mean(positions[component_indices], axis=0)
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

def optimize_layout(network, iterations):
    """
    Optimize the layout of proteins in the network.
    
    Args:
        network: Network object containing proteins, PPIs, and logger.
    """
    proteins = network.get_proteins()
    ppis = network.get_ppis()
    
    optimized_positions = apply_layout(proteins, ppis, network, iterations = iterations, logger=network.logger)
    
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

    def generate_layout_coarse_grain(self,
                        algorithm      = ["optimized", "drl", "fr", "kk", "circle", "grid", "random"][0],
                        scaling_factor = [None       , 300  , 200 , 100 , 100     , 120   , 100     ][0],
                        iterations     = 10000):
        """
        Generates a 3D layout for the network using different algorithms.

        Parameters:
            algorithm (str): Layout algorithm to use. Default is "optimized" (recommended).
            scaling_factor (int): Scaling factor for the layout (default is 300).
            iterations (int): Steps for the force-field directed "optimized" algorithm (default 1000).
         """

        self.logger.info(f"INITIALIZING: 3D layout generation using {algorithm} method...")
        
        # When there are less than 3 proteins, use "drl" algorithm
        if len(self.proteins) < 3:
            self.logger.warning('   - Less than 3 proteins in the network. Using drl method...')
            algorithm = "drl"
            scaling_factor = 250

        if algorithm == "optimized":
            optimize_layout(self, iterations=iterations)

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

    def generate_layout_fine_grain(self, **kwargs):

        return generate_layout_fine(self, **kwargs)

    def generate_interactive_3d_plot(self, save_path: str = './interactive_3D_graph.html',
                                    classification_colors=rrc_classification_colors,
                                    domain_colors=DOMAIN_COLORS_RRC,
                                    res_centroid_colors=SURFACE_COLORS_RRC,
                                    show_plot=True):
        """
        Generates an interactive 3D molecular visualization with side control panel.
        
        Parameters:
            save_path (str): Path to save the HTML visualization file
            classification_colors (dict): Mapping of contact classification categories to colors
            domain_colors (dict): Color palette for protein domains
            res_centroid_colors (dict): Color palette for surface residues
            show_plot (bool): Whether to automatically display the plot
        """
        
        import json
        import webbrowser
        
        # Progress
        self.logger.info("INITIALIZING: Generating interactive 3D visualization...")
        
        # Collect PDB data from all proteins - store separately for each protein
        proteins_pdb_data = {}
        for protein in self.proteins:
            pdb_string = protein.convert_chain_to_pdb_in_memory()
            proteins_pdb_data[protein.get_ID()] = pdb_string.getvalue()
        
        # Get the maximum number of domains for color generation
        max_domain_value = max([max(prot.domains) for prot in self.proteins])
        domain_colors_list = generate_unique_colors(n=max_domain_value + 1, palette=domain_colors)
        
        # Prepare data structures for JavaScript
        proteins_data = []
        surface_residues_data = []
        contacts_data = []
        
        # Create PPI color mapping
        ppi_colors = {}
        color_index = 0
        for ppi in self.ppis:
            ppi_id = (ppi.get_tuple_pair(), ppi.get_cluster_n())
            if ppi_id not in ppi_colors:
                ppi_colors[ppi_id] = res_centroid_colors[color_index % len(res_centroid_colors)]
                color_index += 1
        
        # Process proteins data
        for protein in self.proteins:
            # Get protein center of mass
            cm = protein.get_CM()
            n_term = protein.get_res_CA_xyz([0])[0]
            c_term = protein.get_res_CA_xyz([-1])[0]
            
            protein_info = {
                'id': protein.get_ID(),
                'chain': protein.chain_ID,
                'cm': {'x': float(cm[0]), 'y': float(cm[1]), 'z': float(cm[2])},
                'n_term': {'x': float(n_term[0]), 'y': float(n_term[1]), 'z': float(n_term[2])},
                'c_term': {'x': float(c_term[0]), 'y': float(c_term[1]), 'z': float(c_term[2])},
                'domains': protein.domains
            }
            proteins_data.append(protein_info)
            
            # Process surface residues
            residue_groups = protein.surface.get_residues_by_group()
            centroid_cache = {}
            ca_cache = {}
            added_residues = set()
            
            for group, residues in residue_groups.items():
                for residue in residues:
                    if residue.index in added_residues:
                        continue
                    
                    # Determine color based on group
                    if group == "A":
                        tuple_pair, cluster_n = residue.surface_id
                        color = ppi_colors[(tuple_pair, cluster_n)]
                    elif group == "B":
                        color = "gray"
                    elif group == "C":
                        color = "black"
                    else:
                        continue
                    
                    # Get centroid and CA positions
                    centroid = get_centroid(protein, residue.index, centroid_cache)
                    ca_pos = get_ca(protein, residue.get_index(), ca_cache)
                    
                    surface_residue = {
                        'protein_id': protein.get_ID(),
                        'chain': protein.chain_ID,
                        'residue_index': residue.index,
                        'group': group,
                        'color': color,
                        'centroid': {'x': float(centroid[0]), 'y': float(centroid[1]), 'z': float(centroid[2])},
                        'ca_position': {'x': float(ca_pos[0]), 'y': float(ca_pos[1]), 'z': float(ca_pos[2])}
                    }
                    surface_residues_data.append(surface_residue)
                    added_residues.add(residue.index)
        
        # Process contacts data
        for ppi in self.ppis:
            protein_1 = ppi.get_protein_1()
            protein_2 = ppi.get_protein_2()
            
            contact_residues_1 = ppi.get_contacts_res_1()
            contact_residues_2 = ppi.get_contacts_res_2()
            centroids_1 = protein_1.get_res_centroids_xyz(contact_residues_1)
            centroids_2 = protein_2.get_res_centroids_xyz(contact_residues_2)
            
            for i, (centroid_1, centroid_2) in enumerate(zip(centroids_1, centroids_2)):
                contact_classification = ppi.contacts_classification[i]
                contact_freq = ppi.contact_freq[i]
                
                contact = {
                    'ppi_id': ppi.get_tuple_pair(),
                    'cluster_n': ppi.get_cluster_n(),
                    'classification': contact_classification,
                    'frequency': contact_freq,
                    'color': classification_colors.get(contact_classification, 'gray'),
                    'start': {'x': float(centroid_1[0]), 'y': float(centroid_1[1]), 'z': float(centroid_1[2])},
                    'end': {'x': float(centroid_2[0]), 'y': float(centroid_2[1]), 'z': float(centroid_2[2])}
                }
                contacts_data.append(contact)
        
        # Create HTML content
        html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive Protein Network Visualization</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
                display: flex;
                height: 100vh;
            }}
            .main-container {{
                display: flex;
                width: 100%;
                height: 100%;
            }}
            .viewer-container {{
                flex: 1;
                background-color: white;
                margin: 10px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                position: relative;
            }}
            .control-panel {{
                width: 210px;
                background-color: white;
                margin: 10px 10px 10px 0;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                overflow-y: auto;
            }}
            .control-section {{
                margin-bottom: 25px;
                padding-bottom: 15px;
                border-bottom: 1px solid #eee;
            }}
            .control-section:last-child {{
                border-bottom: none;
            }}
            .control-section h4 {{
                margin: 0 0 15px 0;
                color: #333;
                font-size: 16px;
            }}
            .control-group {{
                margin-bottom: 15px;
            }}
            .control-group label {{
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #555;
                font-size: 14px;
            }}
            select {{
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
            }}
            button {{
                width: 100%;
                padding: 10px;
                margin-bottom: 8px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                transition: background-color 0.3s;
            }}
            button:hover {{
                background-color: #0056b3;
            }}
            button.active {{
                background-color: #28a745;
            }}
            button.active:hover {{
                background-color: #218838;
            }}
            .contact-buttons {{
                display: flex;
                flex-direction: column;
                gap: 5px;
            }}
            .contact-button {{
                margin-bottom: 5px;
            }}
            .contact-button.static {{
                background-color: #6c757d;
            }}
            .contact-button.positive {{
                background-color: #28a745;
            }}
            .contact-button.negative {{
                background-color: #dc3545;
            }}
            .contact-button.static:hover {{
                background-color: #545b62;
            }}
            .contact-button.positive:hover {{
                background-color: #1e7e34;
            }}
            .contact-button.negative:hover {{
                background-color: #bd2130;
            }}
            .contact-button.no-nmers {{
                background-color: #fd7e14;
            }}
            .contact-button.no-2mers {{
                background-color: #ffc107;
            }}
            .contact-button.no-nmers:hover {{
                background-color: #e8680e;
            }}
            .contact-button.no-2mers:hover {{
                background-color: #e0a800;
            }}
        </style>
    </head>
    <body>
        <div class="main-container">
            <div class="viewer-container" id="viewer-container"></div>
            
            <div class="control-panel">
                <div class="control-section">
                <h4>Protein Style</h4>
                <div class="control-group">
                    <select id="style-select">
                        <option value="cartoon">Cartoon</option>
                        <option value="line">Line</option>
                        <option value="stick">Stick</option>
                        <option value="sphere">Sphere</option>
                        <option value="surface">Surface</option>
                    </select>
                </div>
            </div>

            <div class="control-section">
                <h4>Color Scheme</h4>
                <div class="control-group">
                    <select id="color-select">
                        <option value="chain">Chain</option>
                        <option value="spectrum">Spectrum</option>
                        <option value="residue">Residue</option>
                        <option value="secondary">Secondary Structure</option>
                        <option value="plddt">pLDDT</option>
                        <option value="domain" selected>Domain</option>
                    </select>
                </div>
            </div>

            <div class="control-section">
                <h4>Surface Options</h4>
                <div class="control-group">
                    <select id="surface-select">
                        <option value="none">None</option>
                        <option value="VDW">Van der Waals</option>
                        <option value="SAS">Solvent Accessible</option>
                        <option value="MS">Molecular Surface</option>
                    </select>
                </div>
            </div>
                
                <div class="control-section">
                    <h4>Interface Features</h4>
                    <div class="contact-buttons">
                        <button id="surface-residues-toggle" class="contact-button">Show Surface Residues</button>
                    </div>
                </div>
                <div class="control-section">
                    <h4>Contact Features</h4>
                    <div class="contact-buttons">
                        <button id="static-contacts-toggle" class="contact-button static" style="display: none;">Show Static Contacts</button>
                        <button id="positive-contacts-toggle" class="contact-button positive" style="display: none;">Show Positive Contacts</button>
                        <button id="negative-contacts-toggle" class="contact-button negative" style="display: none;">Show Negative Contacts</button>
                        <button id="no-nmers-contacts-toggle" class="contact-button no-nmers" style="display: none;">Show No Nmers Data</button>
                        <button id="no-2mers-contacts-toggle" class="contact-button no-2mers" style="display: none;">Show No 2mers Data</button>
                    </div>
                </div>
                
                <div class="control-section">
                    <h4>Labels</h4>
                    <button id="protein-ids-toggle">Show Protein IDs</button>
                    <button id="terminals-toggle">Show N/C Terminals</button>
                </div>
            </div>
        </div>

        <script>
            // Data from Python
            const proteinsPdbData = {json.dumps(proteins_pdb_data)};
            const proteinsData = {json.dumps(proteins_data)};
            const surfaceResiduesData = {json.dumps(surface_residues_data)};
            const contactsData = {json.dumps(contacts_data)};
            const domainColors = {json.dumps(domain_colors_list)};
            
            // Global variables
            let viewer = null;
            let surfaceResiduesVisible = true;
            let staticContactsVisible = true;
            let positiveContactsVisible = true;
            let negativeContactsVisible = true;
            let noNmersContactsVisible = true;
            let no2mersContactsVisible = true;
            let proteinIdsVisible = true;
            let terminalsVisible = true;
            
            // pLDDT color scale
            const plddt_colorscale = [
                [0.0, "#FF0000"],
                [0.4, "#FFA500"],
                [0.6, "#FFFF00"],
                [0.8, "#ADD8E6"],
                [1.0, "#00008B"]
            ];
            
            function getColorFromScale(value, scale) {{
                let lowerIndex = 0;
                for (let i = 0; i < scale.length; i++) {{
                    if (value <= scale[i][0]) {{
                        break;
                    }}
                    lowerIndex = i;
                }}
                
                if (lowerIndex >= scale.length - 1) {{
                    return scale[scale.length - 1][1];
                }}
                
                const lowerValue = scale[lowerIndex][0];
                const upperValue = scale[lowerIndex + 1][0];
                const valueFraction = (value - lowerValue) / (upperValue - lowerValue);
                
                const lowerColor = hexToRgb(scale[lowerIndex][1]);
                const upperColor = hexToRgb(scale[lowerIndex + 1][1]);
                
                const r = Math.round(lowerColor.r + valueFraction * (upperColor.r - lowerColor.r));
                const g = Math.round(lowerColor.g + valueFraction * (upperColor.g - lowerColor.g));
                const b = Math.round(lowerColor.b + valueFraction * (upperColor.b - lowerColor.b));
                
                return rgbToHex(r, g, b);
            }}
            
            function hexToRgb(hex) {{
                const result = /^#?([a-f\\d]{{2}})([a-f\\d]{{2}})([a-f\\d]{{2}})$/i.exec(hex);
                return result ? {{
                    r: parseInt(result[1], 16),
                    g: parseInt(result[2], 16),
                    b: parseInt(result[3], 16)
                }} : {{r: 0, g: 0, b: 0}};
            }}
            
            function rgbToHex(r, g, b) {{
                return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
            }}
            
            function initViewer() {{
                const config = {{
                    backgroundColor: 'white',
                    antialias: true
                }};
                
                viewer = $3Dmol.createViewer($("#viewer-container"), config);
                
                // Add each protein as a separate model
                Object.keys(proteinsPdbData).forEach(proteinId => {{
                    viewer.addModel(proteinsPdbData[proteinId], 'pdb');
                }});
                
                // Check which contact types exist in the dataset
                const hasStaticData = contactsData.some(contact => contact.classification === 1);
                const hasPositiveData = contactsData.some(contact => contact.classification === 2);
                const hasNegativeData = contactsData.some(contact => contact.classification === 3);
                const hasNoNmersData = contactsData.some(contact => contact.classification === 4);
                const hasNo2mersData = contactsData.some(contact => contact.classification === 5);

                
                // Show buttons only if data exists
                if (hasStaticData) {{
                    document.getElementById('static-contacts-toggle').style.display = 'block';
                }} else {{
                    document.getElementById('static-contacts-toggle').style.display = 'none';
                }}

                if (hasPositiveData) {{
                    document.getElementById('positive-contacts-toggle').style.display = 'block';
                }} else {{
                    document.getElementById('positive-contacts-toggle').style.display = 'none';
                }}

                if (hasNegativeData) {{
                    document.getElementById('negative-contacts-toggle').style.display = 'block';
                }} else {{
                    document.getElementById('negative-contacts-toggle').style.display = 'none';
                }}

                if (hasNoNmersData) {{
                    document.getElementById('no-nmers-contacts-toggle').style.display = 'block';
                }} else {{
                    document.getElementById('no-nmers-contacts-toggle').style.display = 'none';
                }}

                if (hasNo2mersData) {{
                    document.getElementById('no-2mers-contacts-toggle').style.display = 'block';
                }} else {{
                    document.getElementById('no-2mers-contacts-toggle').style.display = 'none';
                }}
                
                applyCurrentStyle();
                viewer.zoomTo();
                
                // Initialize all features as visible
                updateButtonStates();
                initializeFeatures();
                
                viewer.render();
            }}

            function updateButtonStates() {{
                // Update button states to reflect initial visibility
                const surfaceButton = document.getElementById('surface-residues-toggle');
                const staticButton = document.getElementById('static-contacts-toggle');
                const positiveButton = document.getElementById('positive-contacts-toggle');
                const negativeButton = document.getElementById('negative-contacts-toggle');
                const noNmersButton = document.getElementById('no-nmers-contacts-toggle');
                const no2mersButton = document.getElementById('no-2mers-contacts-toggle');
                const proteinIdsButton = document.getElementById('protein-ids-toggle');
                const terminalsButton = document.getElementById('terminals-toggle');
                
                if (surfaceResiduesVisible) {{
                    surfaceButton.textContent = 'Hide Surface Residues';
                    surfaceButton.classList.add('active');
                }}
                
                if (staticContactsVisible && staticButton.style.display !== 'none') {{
                    staticButton.textContent = 'Hide Static Contacts';
                    staticButton.classList.add('active');
                }}

                if (positiveContactsVisible && positiveButton.style.display !== 'none') {{
                    positiveButton.textContent = 'Hide Positive Contacts';
                    positiveButton.classList.add('active');
                }}

                if (negativeContactsVisible && negativeButton.style.display !== 'none') {{
                    negativeButton.textContent = 'Hide Negative Contacts';
                    negativeButton.classList.add('active');
                }}
                
                if (noNmersContactsVisible && noNmersButton.style.display !== 'none') {{
                    noNmersButton.textContent = 'Hide No Nmers Data';
                    noNmersButton.classList.add('active');
                }}
                
                if (no2mersContactsVisible && no2mersButton.style.display !== 'none') {{
                    no2mersButton.textContent = 'Hide No 2mers Data';
                    no2mersButton.classList.add('active');
                }}
                
                if (proteinIdsVisible) {{
                    proteinIdsButton.textContent = 'Hide Protein IDs';
                    proteinIdsButton.classList.add('active');
                }}
                
                if (terminalsVisible) {{
                    terminalsButton.textContent = 'Hide N/C Terminals';
                    terminalsButton.classList.add('active');
                }}
            }}

            function initializeFeatures() {{
                // Initialize all features that should be visible
                if (surfaceResiduesVisible) {{
                    addSurfaceResidues();
                }}
                
                if (staticContactsVisible && document.getElementById('static-contacts-toggle').style.display !== 'none') {{
                    addStaticContacts();
                }}

                if (positiveContactsVisible && document.getElementById('positive-contacts-toggle').style.display !== 'none') {{
                    addPositiveContacts();
                }}

                if (negativeContactsVisible && document.getElementById('negative-contacts-toggle').style.display !== 'none') {{
                    addNegativeContacts();
                }}
                
                if (noNmersContactsVisible) {{
                    addNoNmersContacts();
                }}
                
                if (no2mersContactsVisible) {{
                    addNo2mersContacts();
                }}
                
                if (proteinIdsVisible) {{
                    addProteinIdLabels();
                }}
                
                if (terminalsVisible) {{
                    addTerminalLabels();
                }}
            }}
            
            function applyCurrentStyle() {{
                if (!viewer) return;
                
                const style = document.getElementById('style-select').value;
                const colorScheme = document.getElementById('color-select').value;
                const surfaceType = document.getElementById('surface-select').value;
                
                // Clear current styles
                viewer.setStyle({{}}, {{}});
                
                // Apply main style
                if (colorScheme === 'domain') {{
                    // Apply domain coloring
                    proteinsData.forEach(protein => {{
                        protein.domains.forEach((domain, index) => {{
                            const styleObj = {{}};
                            styleObj[style] = {{color: domainColors[domain]}};
                            viewer.setStyle(
                                {{chain: protein.chain, resi: [index + 1]}},
                                styleObj
                            );
                        }});
                    }});
                }} else {{
                    // Apply standard coloring
                    let styleObj = {{}};
                    
                    if (colorScheme === 'chain') {{
                        styleObj[style] = {{colorscheme: 'chainHetatm'}};
                    }} else if (colorScheme === 'spectrum') {{
                        styleObj[style] = {{color: 'spectrum'}};
                    }} else if (colorScheme === 'residue') {{
                        styleObj[style] = {{colorscheme: 'amino'}};
                    }} else if (colorScheme === 'secondary') {{
                        styleObj[style] = {{colorscheme: 'ssPyMOL'}};
                    }} else if (colorScheme === 'plddt') {{
                        styleObj[style] = {{colorfunc: function(atom) {{
                            const bfactor = Math.max(0, Math.min(100, atom.b));
                            const normalizedValue = bfactor / 100;
                            return getColorFromScale(normalizedValue, plddt_colorscale);
                        }}}};
                    }}
                    
                    viewer.setStyle({{}}, styleObj);
                }}
                
                // Apply surface if selected
                viewer.removeAllSurfaces();
                if (surfaceType !== 'none') {{
                    viewer.addSurface(surfaceType, {{opacity: 0.8}});
                }}
                
                viewer.render();
            }}
            
            function toggleSurfaceResidues() {{
                surfaceResiduesVisible = !surfaceResiduesVisible;
                const button = document.getElementById('surface-residues-toggle');
                
                if (surfaceResiduesVisible) {{
                    addSurfaceResidues();
                    button.textContent = 'Hide Surface Residues';
                    button.classList.add('active');
                }} else {{
                    removeSurfaceResidues();
                    button.textContent = 'Show Surface Residues';
                    button.classList.remove('active');
                }}
                
                viewer.render();
            }}
            
            function addSurfaceResidues() {{
                surfaceResiduesData.forEach(residue => {{
                    // Add centroid sphere
                    viewer.addSphere({{
                        center: residue.centroid,
                        radius: 1.0,
                        color: residue.color,
                        alpha: 1.0
                    }});
                    
                    // Add connection to backbone
                    viewer.addCylinder({{
                        start: residue.ca_position,
                        end: residue.centroid,
                        radius: 0.1,
                        color: residue.color,
                        alpha: 1.0
                    }});
                }});
            }}
            
            function removeSurfaceResidues() {{
                // Remove all shapes and re-add only the visible contacts
                viewer.removeAllShapes();
                if (staticContactsVisible) addStaticContacts();
                if (positiveContactsVisible) addPositiveContacts();
                if (negativeContactsVisible) addNegativeContacts();
            }}
            
            function toggleContacts(contactType) {{
                const buttonMap = {{
                    'static': 'static-contacts-toggle',
                    'positive': 'positive-contacts-toggle',
                    'negative': 'negative-contacts-toggle',
                    'no-nmers': 'no-nmers-contacts-toggle',
                    'no-2mers': 'no-2mers-contacts-toggle'
                }};
                
                const button = document.getElementById(buttonMap[contactType]);
                
                if (contactType === 'static') {{
                    staticContactsVisible = !staticContactsVisible;
                    if (staticContactsVisible) {{
                        addStaticContacts();
                        button.textContent = 'Hide Static Contacts';
                        button.classList.add('active');
                    }} else {{
                        removeStaticContacts();
                        button.textContent = 'Show Static Contacts';
                        button.classList.remove('active');
                    }}
                }} else if (contactType === 'positive') {{
                    positiveContactsVisible = !positiveContactsVisible;
                    if (positiveContactsVisible) {{
                        addPositiveContacts();
                        button.textContent = 'Hide Positive Contacts';
                        button.classList.add('active');
                    }} else {{
                        removePositiveContacts();
                        button.textContent = 'Show Positive Contacts';
                        button.classList.remove('active');
                    }}
                }} else if (contactType === 'negative') {{
                    negativeContactsVisible = !negativeContactsVisible;
                    if (negativeContactsVisible) {{
                        addNegativeContacts();
                        button.textContent = 'Hide Negative Contacts';
                        button.classList.add('active');
                    }} else {{
                        removeNegativeContacts();
                        button.textContent = 'Show Negative Contacts';
                        button.classList.remove('active');
                    }}
                }} else if (contactType === 'no-nmers') {{
                    noNmersContactsVisible = !noNmersContactsVisible;
                    if (noNmersContactsVisible) {{
                        addNoNmersContacts();
                        button.textContent = 'Hide No Nmers Data';
                        button.classList.add('active');
                    }} else {{
                        removeNoNmersContacts();
                        button.textContent = 'Show No Nmers Data';
                        button.classList.remove('active');
                    }}
                }} else if (contactType === 'no-2mers') {{
                    no2mersContactsVisible = !no2mersContactsVisible;
                    if (no2mersContactsVisible) {{
                        addNo2mersContacts();
                        button.textContent = 'Hide No 2mers Data';
                        button.classList.add('active');
                    }} else {{
                        removeNo2mersContacts();
                        button.textContent = 'Show No 2mers Data';
                        button.classList.remove('active');
                    }}
                }}
                
                viewer.render();
            }}

            function addNoNmersContacts() {{
                const filteredContacts = contactsData.filter(contact => contact.classification === 4);
                filteredContacts.forEach(contact => {{
                    const radius = (0.3 * contact.frequency) * 0.5;
                    viewer.addCylinder({{
                        start: contact.start,
                        end: contact.end,
                        radius: radius,
                        color: contact.color,
                        alpha: 1.0
                    }});
                }});
            }}

            function removeNoNmersContacts() {{
                viewer.removeAllShapes();
                if (surfaceResiduesVisible) addSurfaceResidues();
                if (staticContactsVisible) addStaticContacts();
                if (positiveContactsVisible) addPositiveContacts();
                if (negativeContactsVisible) addNegativeContacts();
                if (no2mersContactsVisible) addNo2mersContacts();
            }}

            function addNo2mersContacts() {{
                const filteredContacts = contactsData.filter(contact => contact.classification === 5);
                filteredContacts.forEach(contact => {{
                    const radius = (0.3 * contact.frequency) * 0.5;
                    viewer.addCylinder({{
                        start: contact.start,
                        end: contact.end,
                        radius: radius,
                        color: contact.color,
                        alpha: 1.0
                    }});
                }});
            }}

            function removeNo2mersContacts() {{
                viewer.removeAllShapes();
                if (surfaceResiduesVisible) addSurfaceResidues();
                if (staticContactsVisible) addStaticContacts();
                if (positiveContactsVisible) addPositiveContacts();
                if (negativeContactsVisible) addNegativeContacts();
                if (noNmersContactsVisible) addNoNmersContacts();
            }}
            
            function addStaticContacts() {{
                const filteredContacts = contactsData.filter(contact => contact.classification === 1);
                filteredContacts.forEach(contact => {{
                    const radius = (0.3 * contact.frequency) * 0.5;
                    viewer.addCylinder({{
                        start: contact.start,
                        end: contact.end,
                        radius: radius,
                        color: contact.color,
                        alpha: 1.0
                    }});
                }});
            }}
            
            function rebuildAllShapes() {{
                viewer.removeAllShapes();
                if (surfaceResiduesVisible) addSurfaceResidues();
                if (staticContactsVisible) addStaticContacts();
                if (positiveContactsVisible) addPositiveContacts();
                if (negativeContactsVisible) addNegativeContacts();
                if (noNmersContactsVisible) addNoNmersContacts();
                if (no2mersContactsVisible) addNo2mersContacts();
            }}

            function removeStaticContacts() {{
                rebuildAllShapes();
            }}

            function removePositiveContacts() {{
                rebuildAllShapes();
            }}

            function removeNegativeContacts() {{
                rebuildAllShapes();
            }}

            function removeNoNmersContacts() {{
                rebuildAllShapes();
            }}

            function removeNo2mersContacts() {{
                rebuildAllShapes();
            }}

            function removeSurfaceResidues() {{
                rebuildAllShapes();
            }}
            
            function addPositiveContacts() {{
                const filteredContacts = contactsData.filter(contact => contact.classification === 2);
                filteredContacts.forEach(contact => {{
                    const radius = (0.3 * contact.frequency) * 0.5;
                    viewer.addCylinder({{
                        start: contact.start,
                        end: contact.end,
                        radius: radius,
                        color: contact.color,
                        alpha: 1.0
                    }});
                }});
            }}
            
            function removePositiveContacts() {{
                viewer.removeAllShapes();
                if (surfaceResiduesVisible) addSurfaceResidues();
                if (staticContactsVisible) addStaticContacts();
                if (negativeContactsVisible) addNegativeContacts();
            }}
            
            function addNegativeContacts() {{
                const filteredContacts = contactsData.filter(contact => contact.classification === 3);
                filteredContacts.forEach(contact => {{
                    const radius = (0.3 * contact.frequency) * 0.5;
                    viewer.addCylinder({{
                        start: contact.start,
                        end: contact.end,
                        radius: radius,
                        color: contact.color,
                        alpha: 1.0
                    }});
                }});
            }}
            
            function removeNegativeContacts() {{
                viewer.removeAllShapes();
                if (surfaceResiduesVisible) addSurfaceResidues();
                if (staticContactsVisible) addStaticContacts();
                if (positiveContactsVisible) addPositiveContacts();
            }}
            
            function toggleProteinIds() {{
                proteinIdsVisible = !proteinIdsVisible;
                const button = document.getElementById('protein-ids-toggle');
                
                if (proteinIdsVisible) {{
                    addProteinIdLabels();
                    button.textContent = 'Hide Protein IDs';
                    button.classList.add('active');
                }} else {{
                    viewer.removeAllLabels();
                    if (terminalsVisible) {{
                        addTerminalLabels();
                    }}
                    button.textContent = 'Show Protein IDs';
                    button.classList.remove('active');
                }}
                
                viewer.render();
            }}
            
            function toggleTerminals() {{
                terminalsVisible = !terminalsVisible;
                const button = document.getElementById('terminals-toggle');
                
                if (terminalsVisible) {{
                    addTerminalLabels();
                    button.textContent = 'Hide N/C Terminals';
                    button.classList.add('active');
                }} else {{
                    viewer.removeAllLabels();
                    if (proteinIdsVisible) {{
                        addProteinIdLabels();
                    }}
                    button.textContent = 'Show N/C Terminals';
                    button.classList.remove('active');
                }}
                
                viewer.render();
            }}
            
            function addTerminalLabels() {{
                proteinsData.forEach(protein => {{
                    viewer.addLabel('N', {{
                        position: protein.n_term,
                        fontSize: 20,
                        fontColor: 'black',
                        backgroundOpacity: 0.0,
                        font: 'bold Arial'
                    }});
                    
                    viewer.addLabel('C', {{
                        position: protein.c_term,
                        fontSize: 20,
                        fontColor: 'black',
                        backgroundOpacity: 0.0,
                        font: 'bold Arial'
                    }});
                }});
            }}
            
            function addProteinIdLabels() {{
                proteinsData.forEach(protein => {{
                    viewer.addLabel(protein.id, {{
                        position: {{
                            x: protein.cm.x,
                            y: protein.cm.y,
                            z: protein.cm.z + 5
                        }},
                        fontSize: 18,
                        color: 'black',
                        backgroundOpacity: 0.6
                    }});
                }});
            }}
            
            // Event listeners
            document.getElementById('style-select').addEventListener('change', applyCurrentStyle);
            document.getElementById('color-select').addEventListener('change', applyCurrentStyle);
            document.getElementById('surface-select').addEventListener('change', applyCurrentStyle);
            document.getElementById('surface-residues-toggle').addEventListener('click', toggleSurfaceResidues);
            document.getElementById('static-contacts-toggle').addEventListener('click', () => toggleContacts('static'));
            document.getElementById('positive-contacts-toggle').addEventListener('click', () => toggleContacts('positive'));
            document.getElementById('negative-contacts-toggle').addEventListener('click', () => toggleContacts('negative'));
            document.getElementById('no-nmers-contacts-toggle').addEventListener('click', () => toggleContacts('no-nmers'));
            document.getElementById('no-2mers-contacts-toggle').addEventListener('click', () => toggleContacts('no-2mers'));
            document.getElementById('protein-ids-toggle').addEventListener('click', toggleProteinIds);
            document.getElementById('terminals-toggle').addEventListener('click', toggleTerminals);
            
            // Initialize when page loads
            $(document).ready(function() {{
                initViewer();
            }});
        </script>
    </body>
    </html>
        """
        
        # Save the HTML file
        self.logger.info("   Saving HTML file...")
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"   Visualization saved to {save_path}")
        self.logger.info("FINISHED: Generating interactive 3D visualization")
        
        if show_plot:
            webbrowser.open(save_path)


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
