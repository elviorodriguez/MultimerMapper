
import os
from Bio.PDB import PDBIO

def save_reference_pdbs(sliced_PAE_and_pLDDTs: dict, out_path: str = ".", overwrite: bool = False):

    # Create a folder named "PDB_ref_monomers" if it doesn't exist
    save_folder = out_path + "/PDB_ref_monomers"
    os.makedirs(save_folder, exist_ok = overwrite)
    
    # Save each reference monomer chain
    for protein_ID in sliced_PAE_and_pLDDTs.keys():

        # Create a PDBIO instance
        pdbio = PDBIO()
        
        # Set the structure to the Model
        pdbio.set_structure(sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"])

        # Save the Model to a PDB file
        output_pdb_file = save_folder + f"/{protein_ID}_ref.pdb"
        pdbio.save(output_pdb_file)
