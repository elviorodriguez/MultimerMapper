import numpy as np
from Bio.PDB.qcprot import QCPSuperimposer
from Bio.PDB.ccealign import run_cealign

class CESuperimposer:
    """
    A superimposer that uses the CEAlign algorithm to optimally align structures,
    especially useful for conformationally variable proteins.
    
    Attributes:
        rotran (tuple): Rotation matrix and translation vector.
        rms (float): Root-mean-square deviation after alignment.
    """
    
    def __init__(self, window_size=8, max_gap=30):
        """
        Initialize the CEAlign-based superimposer.
        
        Args:
            window_size (int): Length of fragments used in CE alignment.
            max_gap (int): Maximum allowed gap between aligned fragments.
        """
        self.window_size = window_size
        self.max_gap = max_gap
        self.rotran = None  # (rotation matrix, translation vector)
        self.rms = None

    def set_atoms(self, ref_atoms, mob_atoms):
        """
        Set the reference and mobile atoms for alignment.
        
        Args:
            ref_atoms (list): Reference atoms (from PDB.Chain.Chain).
            mob_atoms (list): Mobile atoms to align to the reference.
        
        Raises:
            ValueError: If atom counts differ.
            RuntimeError: If no valid alignment is found.
        """
        # Extract coordinates from atoms
        ref_coords = np.array([atom.coord for atom in ref_atoms])
        mob_coords = np.array([atom.coord for atom in mob_atoms])

        # Validate atom counts
        if len(ref_coords) != len(mob_coords):
            raise ValueError("Reference and mobile must have the same number of atoms")

        # Obtain alignment paths using CEAlign
        paths = run_cealign(ref_coords.tolist(), mob_coords.tolist(), 
                            self.window_size, self.max_gap)

        best_rmsd = float('inf')
        best_rot = None
        best_tran = None

        # Evaluate each alignment path
        for pathA, pathB in paths:
            # Convert indices to integers and validate
            try:
                pathA = [int(idx) for idx in pathA]
                pathB = [int(idx) for idx in pathB]
            except (ValueError, TypeError):
                continue  # Skip invalid paths

            # Check indices are within bounds
            if (max(pathA) >= len(ref_coords) or min(pathA) < 0 or
                max(pathB) >= len(mob_coords) or min(pathB) < 0):
                continue

            # Extract coordinates for the current path
            ref_subset = ref_coords[pathA]
            mob_subset = mob_coords[pathB]

            # Skip if insufficient points for alignment
            if len(ref_subset) < 3:
                continue

            # Compute transformation using the subset
            sup = QCPSuperimposer()
            sup.set(ref_subset, mob_subset)
            sup.run()

            # Apply transformation to all mobile coordinates
            transformed_mob = np.dot(mob_coords, sup.rot) + sup.tran

            # Calculate RMSD over all atoms
            rmsd = np.sqrt(np.mean(np.sum((ref_coords - transformed_mob)**2, axis=1)))

            # Update best transformation if improved
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_rot = sup.rot.copy()
                best_tran = sup.tran.copy()

        if best_rot is None:
            raise RuntimeError("No valid alignment found")

        self.rotran = (best_rot, best_tran)
        self.rms = best_rmsd