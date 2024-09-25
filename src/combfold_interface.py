
import numpy as np
import pandas as pd
from itertools import product
from string import ascii_uppercase, digits
import os
import json

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# Format the input JSON files for CombFold using Q_values and sequences -------
# -----------------------------------------------------------------------------

# subunits.json - Defines five subunits (named: 2,A,C,T,b)
# pdb files - Structure models predicted by AlphaFold-Multimer of different
#   pairings of the subunits. Each pair have multiple models, as many different
#   pairwise interactions can be considered during assembly
# crosslinks.txt - Defines crosslinks, each line represents a single crosslink.
#   The format of each line is
#   <res1> <chain_ids1> <res2> <chain_ids2> <minimal_distance> <maximal_distance> <crosslink_confidence>

# Generates all possible combinations of the proteins
def create_dataframe_with_combinations(protein_names, Q_values):
    
    # Make sure that the Q_values are format as int
    Q_values = [int(Q) for Q in Q_values]
    
    # Create an empty DataFrame with the specified column names
    df = pd.DataFrame(columns=protein_names)

    # Generate all possible combinations for Pi values between 0 and Qi
    combinations = product(*(range(Q + 1) for Q in Q_values))

    # Filter combinations where the sum is at least 2
    valid_combinations = [comb for comb in combinations if sum(comb) >= 2]

    # Add valid combinations to the DataFrame
    df = pd.DataFrame(valid_combinations, columns=protein_names)

    return df



# DOES NOT CONTAIN CROSSLINKING FOR SEQUENCE CONTINUITY
def generate_json_subunits(sliced_PAE_and_pLDDTs, combination):
        
    json_dict = {}
    
    # Define all possible letters/numbers/symbols for chain IDs that can be used
    chain_letters = (ascii_uppercase + "αβγδεζηθικλμνξοπρστυφχψω" + digits + '!#$%&()+,-.;=@[]^_{}~`')
    # Tested: $!#,
    
    # Counter to select the chain letter
    chain_ID_counter = 0
    
    # Iterate over the proteins
    for protein_ID, Q in combination.items():
        
        [chain_ID_counter]

        # Ensure correct format of Q
        Q = int(Q)
        
        # Skip the protein if it is not included in the combination
        if Q == 0:
            continue
        
        # Generate chain ID(s) for the protein
        chain_names = []
        for chain_number in range(Q):
            chain_ID = chain_letters[chain_ID_counter]
            chain_names.append(chain_ID)
            chain_ID_counter += 1
        
        # Iterate over the domains
        for domain in set(sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"][1]):
            
            # Domain definitions
            domain_definition = sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"][1]
            
            # Extract residue positions that match the current domain
            positions = [position for position, value in enumerate(domain_definition) if value == domain]
            domain_sequence = sliced_PAE_and_pLDDTs[protein_ID]["sequence"][min(positions): max(positions)+1]
            
            # Give the domain a name
            domain_name = protein_ID + "__" + str(domain)
            
            # Start residue
            start_residue = min(positions) + 1
            
            
                
            # ------ Subunit definition (debug) ------
            # print("name:", domain_name)
            # print("chain_names:", chain_names)
            # print("start_res:", start_residue)
            # print("sequence:", domain_sequence)
            # ----------------------------------------
            
            json_dict[domain_name] = {
                "name": domain_name,
                "chain_names": chain_names,
                "start_res": start_residue,
                "sequence": domain_sequence
                }
    
    return json_dict

def generate_JSONs_for_CombFold(prot_IDs, Q_values, sliced_PAE_and_pLDDTs):
    
    # Create the DataFrame with combinations
    my_dataframe_with_combinations = create_dataframe_with_combinations(prot_IDs,
                                                                        Q_values)
    # # Number of possible combinations
    # comb_K = len(my_dataframe_with_combinations)
    
    # Display the DataFrame
    print("Dataframe with stoichiometric combinations:")
    print(my_dataframe_with_combinations)
    
    # Output folder for combinations
    out_folder = "combinations_definitions"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    # Generate a JSON file with subunit definitions for each combination:
    # Iterate over the combination
    for i, combination in my_dataframe_with_combinations.iterrows():
        
        # Progress
        print("Combination:", i)
        
        # Output file name and path
        out_file_name = "_".join([str(Q[1]) for Q in combination.items()]) + ".json"
        json_file_path = os.path.join(out_folder, out_file_name)
        
        # Generate the JSON file subunit definition for the current combination
        json_dict = generate_json_subunits(sliced_PAE_and_pLDDTs, combination)
        
        # Save the dictionary as a JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(json_dict, json_file, indent=4)
        
        print(f"JSON file saved at: {json_file_path}")
        
    return my_dataframe_with_combinations


def generate_json_subunits2(sliced_PAE_and_pLDDTs, combination, drop_low_plddt_domains = None):
        
    json_dict = {}
    
    # Define all possible letters/numbers/symbols for chain IDs that can be used
    chain_letters = (ascii_uppercase + digits + '!#$%&()+,-.;=@[]^_{}~`')
    # Tested: $!#,
    
    # Counter to select the chain letter
    chain_ID_counter = 0
    
    # Crosslink constraints to ensure sequence continuity
    txt_crosslinks = ""
    
    # Iterate over the proteins
    for protein_ID, Q in combination.items():
        
        [chain_ID_counter]

        # Ensure correct format of Q
        Q = int(Q)
        
        # Skip the protein if it is not included in the combination
        if Q == 0:
            continue
        
        # Generate chain ID(s) for the protein
        chain_names = []
        for chain_number in range(Q):
            chain_ID = chain_letters[chain_ID_counter]
            chain_names.append(chain_ID)
            chain_ID_counter += 1
        
        # Iterate over the domains
        total_domains = len(set(sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"][1]))
        for current_domain, domain in enumerate(set(sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"][1])):
            
            # Domain definitions
            domain_definition = sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"][1]
            
            # Extract residue positions that match the current domain
            positions = [position for position, value in enumerate(domain_definition) if value == domain]
            domain_sequence = sliced_PAE_and_pLDDTs[protein_ID]["sequence"][min(positions): max(positions)+1]
            
            # Give the domain a name
            domain_name = protein_ID + "__" + str(domain)
            
            # Start residue
            start_residue = min(positions) + 1
            end_residue   = max(positions) + 1
            
            # Remove disordered loops
            if drop_low_plddt_domains is not None:
                list_of_domain_mean_plddt = [np.mean(pdb_plddts[start_residue-1:end_residue-1]) for pdb_plddts in sliced_PAE_and_pLDDTs[protein_ID]["pLDDTs"]]
                
                if any(mean_plddt >= drop_low_plddt_domains for mean_plddt in list_of_domain_mean_plddt):
                    pass
                else:
                    continue
    
            if current_domain < total_domains - 1:
                for chain in chain_names:
                    txt_crosslinks += str(end_residue) + " " + str(chain) + " " + str(end_residue+1) + " " + str(chain) + " 0 12 1.00\n"
                
            # ------ Subunit definition (debug) ------
            # print("name:", domain_name)
            # print("chain_names:", chain_names)
            # print("start_res:", start_residue)
            # print("sequence:", domain_sequence)
            # ----------------------------------------
            
            json_dict[domain_name] = {
                "name": domain_name,
                "chain_names": chain_names,
                "start_res": start_residue,
                "sequence": domain_sequence
                }
    
    return json_dict, txt_crosslinks



            
def generate_filesystem_for_CombFold(xlsx_Qvalues, out_folder, sliced_PAE_and_pLDDTs,
                                     AF2_2mers, AF2_Nmers, use_symlinks= False,
                                     drop_low_plddt_domains = None):
    '''
    

    Parameters
    ----------
    xlsx_Qvalues : TYPE
        DESCRIPTION.
    out_folder : TYPE
        DESCRIPTION.
    sliced_PAE_and_pLDDTs : TYPE
        DESCRIPTION.
    AF2_2mers : TYPE
        DESCRIPTION.
    AF2_Nmers : TYPE
        DESCRIPTION.
    use_symlinks : TYPE, optional
        DESCRIPTION. The default is False.
    drop_low_plddt_domains : None (default), int/float
        Minimum cutoff value to consider a domain for combinatorial assembly.
        Lower that this value (disordered) will be dropped. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    pdb_files : TYPE
        DESCRIPTION.

    '''
    
    # Read the desired combination
    combination = pd.read_excel(xlsx_Qvalues)
    print(combination)

    # Output folder creation
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        os.makedirs(out_folder + "/pdbs")
    else:
        raise ValueError(f'{out_folder} already exists')

    # Generate a JSON file with subunit definitions for each combination ------
    # Output file name and path
    out_file_name = out_folder + ".json"
    out_crosslink = out_folder + "_crosslinks.txt"
    json_file_path = os.path.join(out_folder, out_file_name)
    
    print("out_file_name:", out_file_name)
    print("json_file_path:", json_file_path)
    
    # Generate the JSON file subunit definition for the current combination
    json_dict, txt_crosslinks = generate_json_subunits2(sliced_PAE_and_pLDDTs,
                                                        combination,
                                                        drop_low_plddt_domains)
    
    # Save the dictionary as a JSON file and crosslinks as txt file
    with open(json_file_path, 'w') as json_file:
        json.dump(json_dict, json_file, indent=4)
    with open(os.path.join(out_folder, out_crosslink), 'w') as txt_file:
        txt_file.write(txt_crosslinks.rstrip('\n'))
    
    print(f"JSON file saved at: {json_file_path}")
    
    # Create symlinks to PDB files --------------------------------------------
    
    def find_pdb_files2(root_folder):
        pdb_files = []
        
        for foldername, subfolders, filenames in os.walk(root_folder):
            for filename in filenames:
                # Select only one PDB file for each prediction (unrelaxed and rank 1)
                if filename.endswith(".pdb"):
                    pdb_files.append(os.path.join(foldername, filename))
        
        return pdb_files
    
    # List to store all PDB files
    all_pdb_files = []
    
    if AF2_Nmers != None:
        folders_to_search = [AF2_2mers, AF2_Nmers]
    else:
        folders_to_search = [AF2_2mers]
    
    # Find all PDB files in all folders
    print("Finding all PDB files in AF2 prediction folders...")
    for folder in folders_to_search:
        pdb_files_in_folder = find_pdb_files2(folder)
        all_pdb_files.extend(pdb_files_in_folder)
    print(f"   - Number of PDB files found: {len(all_pdb_files)}")
    
    # Create the symlinks or copy PDB files
    if use_symlinks: print("Creating symbolic links to all PDB files...")
    else: print("Copying PDB files to pdbs directory...")
    for pdb_file in all_pdb_files:
        if use_symlinks:
            
            target_path = "../" + pdb_file
            symlink_path = out_folder + "/pdbs"
            # Create a relative symlink
            os.symlink(os.path.relpath(target_path, os.path.dirname(symlink_path)), symlink_path)
        else:
            import shutil
            # Specify the source path of the PDB file
            source_path = pdb_file
            # Specify the destination path for the copy
            destination_path = os.path.join(out_folder, "pdbs", os.path.basename(pdb_file))
            # Copy the file to the destination folder
            shutil.copy(source_path, destination_path)