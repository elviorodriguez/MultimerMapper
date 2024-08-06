
from Bio import PDB
from copy import deepcopy

def save_bio_pdb_model_model_list(models_list: list[PDB.Model.Model],
                                   output_file = "combined_structure.pdb"):

    # List of Model objects (set here what you want to save)
    model_list = deepcopy()
    
    ###################
    # Create a new Structure object
    structure_id = "combined_structure"
    structure = PDB.Structure.Structure(structure_id)

    # Add each Model to the Structure with a unique ID
    for i, model in enumerate(model_list):
        try:
            model.id = i  # Assign a unique ID to each model
            structure.add(model)
        except PDB.Entity.PDBConstructionException as e:
            print(f'Exception: {e}')
            print(f'Warning: {model.id} is repeated in the list. It will be skipped.')
            

    # Write the Structure to a PDB file

    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_file)

    print(f"Combined PDB file saved as {output_file}")


# # Example
# save_bio_pdb_model_model_list(pairwise_domains_traj_dict['full_pdb_model'])


