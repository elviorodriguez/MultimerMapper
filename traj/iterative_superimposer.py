
import numpy as np

class IterativeSuperimposer:
    """
    Iterative structural alignment using Kabsch algorithm with outlier pruning.
    
    Attributes:
        rotran (tuple): (rotation_matrix, translation_vector) tuple
        rms (float): RMSD computed over all atoms after final alignment
    
    Parameters:
        cutoff (float): Distance cutoff for pruning outliers (Å)
        max_iterations (int): Maximum number of refinement iterations
    """
    
    def __init__(self, cutoff=2.0, max_iterations=10):
        self.cutoff = cutoff
        self.max_iterations = max_iterations
        self.rotran = None  # (rotation matrix, translation vector)
        self.rms = None

    def set_atoms(self, reference_atoms, mobile_atoms):
        """
        Set the atom lists to align and perform iterative superposition.
        
        Args:
            reference_atoms (list): List of reference Bio.PDB.Atom objects
            mobile_atoms (list): List of mobile Bio.PDB.Atom objects to align
        """
        # Convert atoms to coordinate arrays
        ref_coords = np.array([atom.coord for atom in reference_atoms], dtype=np.float64)
        mob_coords = np.array([atom.coord for atom in mobile_atoms], dtype=np.float64)

        if len(ref_coords) != len(mob_coords):
            raise ValueError("Atom lists must be same length")

        # Preserve original coordinates for final RMSD calculation
        orig_ref = ref_coords.copy()
        orig_mob = mob_coords.copy()

        # Initialize indices and working coordinates
        n_atoms = len(ref_coords)
        included = np.arange(n_atoms)  # Track included atom indices
        current_ref = ref_coords
        current_mob = mob_coords

        # Iterative refinement loop
        iteration = 0
        while iteration < self.max_iterations:
            # Calculate optimal transformation using current atoms
            rot, tran = self._kabsch(current_ref, current_mob)
            
            # Transform current mobile coordinates
            transformed = (current_mob - current_mob.mean(0)) @ rot.T + current_ref.mean(0)
            
            # Calculate distances and identify outliers
            distances = np.linalg.norm(transformed - current_ref, axis=1)
            outlier_mask = distances > self.cutoff
            
            # Check stopping conditions
            if not np.any(outlier_mask):
                break  # No more outliers
                
            remaining = len(distances) - np.sum(outlier_mask)
            if remaining < 3:
                break  # Insufficient points for alignment
                
            # Update included atoms for next iteration
            included = included[~outlier_mask]
            current_ref = ref_coords[included]
            current_mob = mob_coords[included]
            iteration += 1

        # Final alignment using best subset of atoms
        if len(included) < 3:  # Fallback to all atoms if needed
            current_ref = ref_coords
            current_mob = mob_coords
            
        final_rot, final_tran = self._kabsch(current_ref, current_mob)
        
        # Calculate final RMSD over all original atoms
        transformed_all = (orig_mob - current_mob.mean(0)) @ final_rot.T + current_ref.mean(0)
        self.rms = np.sqrt(np.mean(np.sum((transformed_all - orig_ref)**2, axis=1)))
        
        # Store transformation components
        self.rotran = (final_rot, final_tran)

    def _kabsch(self, ref_coords, mob_coords):
        """Kabsch algorithm for optimal rotation/translation between two coordinate sets"""
        ref_centroid = ref_coords.mean(0)
        mob_centroid = mob_coords.mean(0)
        
        # Center coordinates
        ref_centered = ref_coords - ref_centroid
        mob_centered = mob_coords - mob_centroid
        
        # Calculate covariance matrix
        H = mob_centered.T @ ref_centered
        
        # Singular Value Decomposition
        U, _, Vt = np.linalg.svd(H)
        rotation = Vt.T @ U.T
        
        # Ensure right-handed coordinate system
        if np.linalg.det(rotation) < 0:
            Vt[-1] *= -1
            rotation = Vt.T @ U.T
        
        # Calculate translation
        translation = ref_centroid - rotation @ mob_centroid
        
        return rotation, translation