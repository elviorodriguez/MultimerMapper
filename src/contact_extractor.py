# -----------------------------------------------------------------------------
# Get contact information from 2-mers dataset ---------------------------------
# -----------------------------------------------------------------------------

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


def compute_contacts(pdb_filename, min_diagonal_PAE_matrix,
                     # Protein symbols/names/IDs
                     protein_ID_a, protein_ID_b,
                     # This dictionary is created on the fly in best_PAE_to_domains.py (contains best pLDDT models info)
                     sliced_PAE_and_pLDDTs, filtered_pairwise_2mers_df,
                     # Cutoff parameters
                     contact_distance = 8.0, PAE_cutoff = 3, pLDDT_cutoff = 70,
                     is_debug = False):
    '''
    Computes the interface contact residues and extracts several metrics for
    each residue-residue interaction. Returns a dataframe with this info.

    Parameters:
    - pdb_filename (str/Bio.PDB.Model.Model): PDB file path/Bio.PDB.Model.Model object of the interaction.
    - min_diagonal_PAE_matrix (np.array): PAE matrix for the interaction.
    - contact_distance (float):  (default: 8.0).
    PAE_cutoff (float): Minimum PAE value (Angstroms) between two residues in order to consider a contact (default = 5 ).
    pLDDT_cutoff (float): Minimum pLDDT value between two residues in order to consider a contact. 
        The minimum pLDDT value of residue pairs will be used (default = 70).
    is_debug (bool): Set it to True to print some debug parts (default = False).

    Returns:
    - contacts_2mers_df (pd.DataFrame): Contains all residue-residue contacts information for the protein pair (protein_ID_a,
        protein_ID_b, res_a, res_b, AA_a, AA_b,res_name_a, res_name_b, PAE, pLDDT_a, pLDDT_b, min_pLDDT, ipTM, min_PAE, N_models,
        distance, xyz_a, xyz_b, CM_a, CM_b, chimera_code)

    '''
    # Empty df to store results
    columns = ["protein_ID_a", "protein_ID_b", "res_a", "res_b", "AA_a", "AA_b",
               "res_name_a", "res_name_b", "PAE", "pLDDT_a", "pLDDT_b", "min_pLDDT",
               "ipTM", "min_PAE", "N_models", "distance", "xyz_a", "xyz_b", "CM_a", "CM_b", "chimera_code"]
    
    contacts_2mers_df = pd.DataFrame(columns=columns)
    
    # Create PDB parser instance
    parser = PDBParser(QUIET=True)
    
    # Chekc if Bio.PDB.Model.Model object was provided directly or it was the PDB path
    if type(pdb_filename) == PDB.Model.Model:
        structure = pdb_filename
    elif type(pdb_filename) == str:
        structure = parser.get_structure('complex', pdb_filename)[0]
    
    # Extract chain IDs
    chains_list = [chain_ID.id for chain_ID in structure.get_chains()]
    if len(chains_list) != 2: raise ValueError("PDB have a number of chains different than 2")
    chain_a_id, chain_b_id = chains_list

    # Extract chains
    chain_a = structure[chain_a_id]
    chain_b = structure[chain_b_id]

    # Length of proteins
    len_a = len(chain_a)
    len_b = len(chain_b)
    
    # Get the number of rows and columns
    PAE_num_rows, PAE_num_cols = min_diagonal_PAE_matrix.shape
        
    # Match matrix dimensions with chains
    if len_a == PAE_num_rows and len_b == PAE_num_cols:
        a_is_row = True
    elif len_b == PAE_num_rows and len_a == PAE_num_cols: 
        a_is_row = False
    else:
        raise ValueError("PAE matrix dimensions does not match chain lengths")
        
    # Extract Bio.PDB.Chain.Chain objects from highest mean pLDDT model
    highest_pLDDT_PDB_a = sliced_PAE_and_pLDDTs[protein_ID_a]["PDB_xyz"]
    highest_pLDDT_PDB_b = sliced_PAE_and_pLDDTs[protein_ID_b]["PDB_xyz"]
    
    # Check that sequence lengths are consistent
    if len(highest_pLDDT_PDB_a) != len_a: raise ValueError(f"Length of chain {chain_a} of {protein_ID_a} from PDB file do not match the length of its corresponding highest mean pLDDT structure (reference).")
    if len(highest_pLDDT_PDB_b) != len_b: raise ValueError(f"Length of chain {chain_b} of {protein_ID_a} from PDB file do not match the length of its corresponding highest mean pLDDT structure (reference).")
    
    # Center of mass of each chain extracted from lowest pLDDT model
    CM_a = highest_pLDDT_PDB_a.center_of_mass()
    CM_b = highest_pLDDT_PDB_b.center_of_mass()
    
    # Progress
    print("----------------------------------------------------------------------------")
    print(f"Computing interface residues for {protein_ID_a}__vs__{protein_ID_b} pair...")
    print("Protein A:", protein_ID_a)
    print("Protein B:", protein_ID_b)
    print("Length A:", len_a)
    print("Length B:", len_b)
    print("PAE rows:", PAE_num_rows)
    print("PAE cols:", PAE_num_cols)
    print("Center of Mass A:", CM_a)
    print("Center of Mass B:", CM_b)

    
    # Extract PAE for a pair of residue objects
    def get_PAE_for_residue_pair(res_a, res_b, PAE_matrix, a_is_row, is_debug = False):
        
        # Compute PAE
        if a_is_row:
            # Extract PAE value for residue pair
            PAE_value =  PAE_matrix[res_a.id[1] - 1, res_b.id[1] - 1]
        else:
            # Extract PAE value for residue pair
            PAE_value =   PAE_matrix[res_b.id[1] - 1, res_a.id[1] - 1]
            
        if is_debug: print("Parsing residue pair:", res_a.id[1],",", res_b.id[1], ") - PAE_value:", PAE_value)
        
        return PAE_value
    
    
    # Extract the minimum pLDDT for a pair of residue objects
    def get_min_pLDDT_for_residue_pair(res_a, res_b):
        
        # Extract pLDDTs for each residue
        plddt_a = next(res_a.get_atoms()).bfactor
        plddt_b = next(res_b.get_atoms()).bfactor
        
        # Compute the minimum
        min_pLDDT = min(plddt_a, plddt_b)
        
        return min_pLDDT
        
    
    # Compute the residue-residue contact (centroid)
    def get_centroid_distance(res_a, res_b):
        
        # Get residues centroids and compute distance
        centroid_res_a = res_a.center_of_mass()
        centroid_res_b = res_b.center_of_mass()
        distance = calculate_distance(centroid_res_a, centroid_res_b)
        
        return distance
    
    
    # Chimera code to select residues from interfaces easily
    chimera_code = "sel "
    
    # Compute contacts
    contacts = []
    for res_a in chain_a:
        
        # Normalized residue position from highest pLDDT model (subtract CM)
        residue_id_a = res_a.id[1]                    # it is 1 based indexing
        residue_xyz_a = highest_pLDDT_PDB_a[residue_id_a].center_of_mass() - CM_a
        
        # pLDDT value for current residue of A
        res_pLDDT_a = res_a["CA"].get_bfactor()
        
        for res_b in chain_b:
        
            # Normalized residue position from highest pLDDT model (subtract CM)
            residue_id_b = res_b.id[1]                    # it is 1 based indexing
            residue_xyz_b = highest_pLDDT_PDB_b[residue_id_b].center_of_mass() - CM_b
            
            # pLDDT value for current residue of B
            res_pLDDT_b = res_b["CA"].get_bfactor()
        
            # Compute PAE for the residue pair and extract the minimum pLDDT
            pair_PAE = get_PAE_for_residue_pair(res_a, res_b, min_diagonal_PAE_matrix, a_is_row)
            pair_min_pLDDT = get_min_pLDDT_for_residue_pair(res_a, res_b)
            
            # Check if diagonal PAE value is lower than cutoff and pLDDT is high enough
            if pair_PAE < PAE_cutoff and pair_min_pLDDT > pLDDT_cutoff:               
                
                # Compute distance between residue pair
                pair_distance = get_centroid_distance(res_a, res_b)
                
                if pair_distance < contact_distance:
                    print("Residue pair:", res_a.id[1], res_b.id[1], "\n",
                          "  - PAE =", pair_PAE, "\n",
                          "  - min_pLDDT =", pair_min_pLDDT, "\n",
                          "  - distance =", pair_distance, "\n",)
                    
                    # Add residue pairs to chimera code to select residues easily
                    chimera_code += f"/a:{res_a.id[1]} /b:{res_b.id[1]} "
                    
                    # Add contact pair to dict
                    contacts = pd.DataFrame({
                        # Save them as 0 base
                        "protein_ID_a": [protein_ID_a],
                        "protein_ID_b": [protein_ID_b],
                        "res_a": [residue_id_a - 1],
                        "res_b": [residue_id_b - 1],
                        "AA_a": [seq1(res_a.get_resname())],      # Get the aminoacid of chain A in the contact
                        "AA_b": [seq1(res_b.get_resname())],      # Get the aminoacid of chain B in the contact
                        "res_name_a": [seq1(res_a.get_resname()) + str(residue_id_a)],
                        "res_name_b": [seq1(res_b.get_resname()) + str(residue_id_b)],
                        "PAE": [pair_PAE],
                        "pLDDT_a": [res_pLDDT_a],
                        "pLDDT_b": [res_pLDDT_b],
                        "min_pLDDT": [pair_min_pLDDT],
                        "ipTM": "",
                        "min_PAE": "",
                        "N_models": "",
                        "distance": [pair_distance],
                        "xyz_a": [residue_xyz_a],
                        "xyz_b": [residue_xyz_b],
                        "CM_a": [np.array([0,0,0])],
                        "CM_b": [np.array([0,0,0])],
                        "chimera_code": ""})
                    
                    contacts_2mers_df = pd.concat([contacts_2mers_df, contacts], ignore_index = True)
    
    # Add the chimera code, ipTM and min_PAE
    contacts_2mers_df["chimera_code"] = chimera_code
    contacts_2mers_df["ipTM"] = float(filtered_pairwise_2mers_df[(filtered_pairwise_2mers_df['protein1'] == protein_ID_a) & (filtered_pairwise_2mers_df['protein2'] == protein_ID_b)]["ipTM"])
    contacts_2mers_df["min_PAE"] = float(filtered_pairwise_2mers_df[(filtered_pairwise_2mers_df['protein1'] == protein_ID_a) & (filtered_pairwise_2mers_df['protein2'] == protein_ID_b)]["min_PAE"])
    contacts_2mers_df["N_models"] = int(filtered_pairwise_2mers_df[(filtered_pairwise_2mers_df['protein1'] == protein_ID_a) & (filtered_pairwise_2mers_df['protein2'] == protein_ID_b)]["N_models"])
                    
    # Compute CM (centroid) for contact residues and append it to df
    CM_ab = np.mean(np.array(contacts_2mers_df["xyz_a"]), axis = 0)   # CM of A residues interacting with B
    CM_ba = np.mean(np.array(contacts_2mers_df["xyz_b"]), axis = 0)   # CM of B residues interacting with A
    # Add CM for contact residues
    contacts_2mers_df['CM_ab'] = [CM_ab] * len(contacts_2mers_df)
    contacts_2mers_df['CM_ba'] = [CM_ba] * len(contacts_2mers_df)
    
    
    # Calculate magnitude of each CM to then compute unitary vectors
    norm_ab = np.linalg.norm(CM_ab)
    norm_ba = np.linalg.norm(CM_ba)
    # Compute unitary vectors to know direction of surfaces and add them
    contacts_2mers_df['V_ab'] = [CM_ab / norm_ab] * len(contacts_2mers_df)  # Unitary vector AB
    contacts_2mers_df['V_ba'] = [CM_ba / norm_ba] * len(contacts_2mers_df)  # Unitary vector BA
    
    
    return contacts_2mers_df


# Wrapper for compute_contacts
def compute_contacts_batch(pdb_filename_list, min_diagonal_PAE_matrix_list,
                           # Protein symbols/names/IDs
                           protein_ID_a_list, protein_ID_b_list,
                           # This dictionary is created on the fly in best_PAE_to_domains.py (contains best pLDDT models info)
                           sliced_PAE_and_pLDDTs, filtered_pairwise_2mers_df,
                           # Cutoff parameters
                           contact_distance = 8.0, PAE_cutoff = 3, pLDDT_cutoff = 70,
                           is_debug = False):
    '''
    Wrapper for compute_contacts function, to allow computing contacts on many
    pairs.
    
    Parameters:
        - pdb_filename_list (str/Bio.PDB.Model.Model): list of paths or Biopython PDB models
        - min_diagonal_PAE_matrix_list
    '''
    
    # Empty df to store results
    columns = ["protein_ID_a", "protein_ID_b", "res_a", "res_b", "AA_a", "AA_b",
               "res_name_a", "res_name_b", "PAE", "pLDDT_a", "pLDDT_b", "min_pLDDT",
               "ipTM", "min_PAE", "N_models", "distance", "xyz_a", "xyz_b", "CM_a", "CM_b", "chimera_code"]
    contacts_2mers_df = pd.DataFrame(columns=columns)
        
    # Check if all lists have the same length
    if not (len(pdb_filename_list) == len(min_diagonal_PAE_matrix_list) == len(protein_ID_a_list) == len(protein_ID_b_list)):
        raise ValueError("Lists arguments for compute_contacts_batch function must have the same length")
    
    # For progress bar
    total_models = len(pdb_filename_list)
    model_num = 0    
    
    # Compute contacts one pair at a time
    for i in range(len(pdb_filename_list)):
        
        # Get data for i
        pdb_filename = pdb_filename_list[i]
        PAE_matrix = min_diagonal_PAE_matrix_list[i]
        protein_ID_a = protein_ID_a_list[i]
        protein_ID_b = protein_ID_b_list[i]
                
        # Compute contacts for pair
        contacts_2mers_df_i = compute_contacts(
            pdb_filename = pdb_filename,
            min_diagonal_PAE_matrix = PAE_matrix,
            protein_ID_a = protein_ID_a,
            protein_ID_b = protein_ID_b,
            sliced_PAE_and_pLDDTs = sliced_PAE_and_pLDDTs,
            filtered_pairwise_2mers_df = filtered_pairwise_2mers_df,
            # Cutoff parameters
            contact_distance = contact_distance, PAE_cutoff = PAE_cutoff, pLDDT_cutoff = pLDDT_cutoff,
            is_debug = False)
        
        contacts_2mers_df = pd.concat([contacts_2mers_df, contacts_2mers_df_i], ignore_index = True)
        
        # For progress bar
        model_num += 1
        print_progress_bar(model_num, total_models, text = " (2-mers contacts)", progress_length = 40)
        print("")
    
    return contacts_2mers_df




def compute_contacts_from_pairwise_2mers_df(filtered_pairwise_2mers_df, pairwise_2mers_df,
                                            sliced_PAE_and_pLDDTs,
                                            contact_distance = 8.0,
                                            contact_PAE_cutoff = 3,
                                            contact_pLDDT_cutoff = 70,
                                            is_debug = False):
    '''
    Computes contacts between interacting pairs of proteins defined in
    filtered_pairwise_2mers_df. It extracts the contacts from pairwise_2mers_df
    rank1 models (best ipTM) and returns the residue-residue contacts info as
    a dataframe (contacts_2mers_df).
    
    Parameters:
    - filtered_pairwise_2mers_df (): 
    - pairwise_2mers_df (pandas.DataFrame): 
    - sliced_PAE_and_pLDDTs (dict): 
    - contact_distance (float): maximum distance between residue centroids to consider a contact (Angstroms). Default = 8.
    - contact_PAE_cutoff (float): maximum PAE value to consider a contact (Angstroms). Default = 3.
    - contact_pLDDT_cutoff (float): minimum PAE value to consider a contact (0 to 100). Default = 70.
    - is_debug (bool): If True, shows some debug prints.
    
    Returns:
    - contacts_2mers_df (pandas.DataFrame): contains contact information. 
        columns = ["protein_ID_a", "protein_ID_b", "res_a", "res_b", "AA_a", "AA_b", "res_name_a", "res_name_b", "PAE",
                   "pLDDT_a", "pLDDT_b", "min_pLDDT", "ipTM", "min_PAE", "N_models", "distance", "xyz_a", "xyz_b", "CM_a",
                   "CM_b", "chimera_code"]
    '''
    
    # Check if pairwise_Nmers_df was passed by mistake
    if "proteins_in_model" in pairwise_2mers_df.columns:
        raise ValueError("Provided dataframe contains N-mers data. To compute contacts coming from N-mers models, please, use compute_contacts_from_pairwise_Nmers_df function.")
    
    # Convert necessary files to lists
    pdb_filename_list = []
    min_diagonal_PAE_matrix_list = []

    for i, row  in filtered_pairwise_2mers_df.iterrows():    
        pdb_model = pairwise_2mers_df.query(f'((protein1 == "{row["protein1"]}" & protein2 == "{row["protein2"]}") | (protein1 == "{row["protein2"]}" & protein2 == "{row["protein1"]}")) & rank == 1')["model"].reset_index(drop=True)[0]
        diag_sub_PAE = pairwise_2mers_df.query(f'((protein1 == "{row["protein1"]}" & protein2 == "{row["protein2"]}") | (protein1 == "{row["protein2"]}" & protein2 == "{row["protein1"]}")) & rank == 1')["diagonal_sub_PAE"].reset_index(drop=True)[0]
        
        pdb_filename_list.append(pdb_model)
        min_diagonal_PAE_matrix_list.append(diag_sub_PAE)
    
    contacts_2mers_df = compute_contacts_batch(
        pdb_filename_list = pdb_filename_list,
        min_diagonal_PAE_matrix_list = min_diagonal_PAE_matrix_list, 
        protein_ID_a_list = filtered_pairwise_2mers_df["protein1"],
        protein_ID_b_list = filtered_pairwise_2mers_df["protein2"],
        sliced_PAE_and_pLDDTs = sliced_PAE_and_pLDDTs,
        filtered_pairwise_2mers_df = filtered_pairwise_2mers_df,
        # Cutoffs
        contact_distance = contact_distance,
        PAE_cutoff = contact_PAE_cutoff,
        pLDDT_cutoff = contact_pLDDT_cutoff)
    
    return contacts_2mers_df


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# Get contact information from N-mers dataset ---------------------------------
# -----------------------------------------------------------------------------

def compute_contacts_Nmers(pairwise_Nmers_df_row, filtered_pairwise_Nmers_df, sliced_PAE_and_pLDDTs,
                           # Cutoff parameters
                           contact_distance = 8.0, PAE_cutoff = 3, pLDDT_cutoff = 70,
                           is_debug = False):
    '''
    
    '''
    # Empty df to store results
    columns = ["protein_ID_a", "protein_ID_b", "proteins_in_model", "res_a", "res_b", "AA_a", "AA_b",
                "res_name_a", "res_name_b", "PAE", "pLDDT_a", "pLDDT_b", "min_pLDDT", "pTM",
                "ipTM", "pDockQ", "min_PAE", "N_models", "distance", "xyz_a", "xyz_b", "CM_a", "CM_b", "chimera_code"]
    contacts_Nmers_df = pd.DataFrame(columns=columns)  
    
    # Get data frow df
    protein_ID_a            = pairwise_Nmers_df_row["protein1"]
    protein_ID_b            = pairwise_Nmers_df_row["protein2"]
    proteins_in_model       = pairwise_Nmers_df_row["proteins_in_model"]
    pdb_model               = pairwise_Nmers_df_row["model"]
    min_diagonal_PAE_matrix = pairwise_Nmers_df_row["diagonal_sub_PAE"]
    pTM                     = pairwise_Nmers_df_row["pTM"]
    ipTM                    = pairwise_Nmers_df_row["ipTM"]
    min_PAE                 = pairwise_Nmers_df_row["min_PAE"]
    pDockQ                  = pairwise_Nmers_df_row["pDockQ"]
    # PPV                     = pairwise_Nmers_df_row["PPV"]
    
    # Check if Bio.PDB.Model.Model object is OK
    if type(pdb_model) != PDB.Model.Model:
        raise ValueError(f"{pdb_model} is not of class Bio.PDB.Model.Model.")
    
    # Extract chain IDs
    chains_list = [chain_ID.id for chain_ID in pdb_model.get_chains()]
    if len(chains_list) != 2: raise ValueError(f"PDB model {pdb_model} have a number of chains different than 2")
    chain_a_id, chain_b_id = chains_list

    # Extract chains
    chain_a = pdb_model[chain_a_id]
    chain_b = pdb_model[chain_b_id]

    # Length of proteins
    len_a = len(chain_a)
    len_b = len(chain_b)
    
    # Get the number of rows and columns
    PAE_num_rows, PAE_num_cols = min_diagonal_PAE_matrix.shape
        
    # Match matrix dimensions with chains
    if len_a == PAE_num_rows and len_b == PAE_num_cols:
        a_is_row = True
    elif len_b == PAE_num_rows and len_a == PAE_num_cols: 
        a_is_row = False
    else:
        raise ValueError("PAE matrix dimensions does not match chain lengths")
            
    # Extract Bio.PDB.Chain.Chain objects from highest mean pLDDT model
    highest_pLDDT_PDB_a = sliced_PAE_and_pLDDTs[protein_ID_a]["PDB_xyz"]
    highest_pLDDT_PDB_b = sliced_PAE_and_pLDDTs[protein_ID_b]["PDB_xyz"]
    
    # Check that sequence lengths are consistent
    if len(highest_pLDDT_PDB_a) != len_a: raise ValueError(f"Length of chain {chain_a} of {protein_ID_a} from PDB file do not match the length of its corresponding highest mean pLDDT structure.")
    if len(highest_pLDDT_PDB_b) != len_b: raise ValueError(f"Length of chain {chain_b} of {protein_ID_a} from PDB file do not match the length of its corresponding highest mean pLDDT structure.")
    
    # Center of mass of each chain extracted from lowest pLDDT model
    CM_a = highest_pLDDT_PDB_a.center_of_mass()
    CM_b = highest_pLDDT_PDB_b.center_of_mass()
    
    # Progress
    print("----------------------------------------------------------------------------")
    print(f"Computing interface residues for ({protein_ID_a}, {protein_ID_b}) N-mer pair...")
    print(f"Model: {str(proteins_in_model)}")
    print("Protein A:", protein_ID_a)
    print("Protein B:", protein_ID_b)
    print("Length A:", len_a)
    print("Length B:", len_b)
    print("PAE rows:", PAE_num_rows)
    print("PAE cols:", PAE_num_cols)
    print("Center of Mass A:", CM_a)
    print("Center of Mass B:", CM_b)

    
    # Extract PAE for a pair of residue objects
    def get_PAE_for_residue_pair(res_a, res_b, PAE_matrix, a_is_row, is_debug = False):
        
        # Compute PAE
        if a_is_row:
            # Extract PAE value for residue pair
            PAE_value =  PAE_matrix[res_a.id[1] - 1, res_b.id[1] - 1]
        else:
            # Extract PAE value for residue pair
            PAE_value =   PAE_matrix[res_b.id[1] - 1, res_a.id[1] - 1]
            
        if is_debug: print("Parsing residue pair:", res_a.id[1],",", res_b.id[1], ") - PAE_value:", PAE_value)
        
        return PAE_value
    
    # Extract the minimum pLDDT for a pair of residue objects
    def get_min_pLDDT_for_residue_pair(res_a, res_b):
        
        # Extract pLDDTs for each residue
        plddt_a = next(res_a.get_atoms()).bfactor
        plddt_b = next(res_b.get_atoms()).bfactor
        
        # Compute the minimum
        min_pLDDT = min(plddt_a, plddt_b)
        
        return min_pLDDT
        
    
    # Compute the residue-residue contact (centroid)
    def get_centroid_distance(res_a, res_b):
        
        # Get residues centroids and compute distance
        centroid_res_a = res_a.center_of_mass()
        centroid_res_b = res_b.center_of_mass()
        distance = calculate_distance(centroid_res_a, centroid_res_b)
        
        return distance
    
    
    # Chimera code to select residues from interfaces easily
    chimera_code = "sel "
    
    # Compute contacts
    contacts = []
    for res_a in chain_a:
        
        # Normalized residue position from highest pLDDT model (subtract CM)
        residue_id_a = res_a.id[1]                    # it is 1 based indexing
        residue_xyz_a = highest_pLDDT_PDB_a[residue_id_a].center_of_mass() - CM_a
        
        # pLDDT value for current residue of A
        res_pLDDT_a = res_a["CA"].get_bfactor()
        
        for res_b in chain_b:
        
            # Normalized residue position (subtract CM)
            residue_id_b = res_b.id[1]                    # it is 1 based indexing
            residue_xyz_b = highest_pLDDT_PDB_b[residue_id_b].center_of_mass() - CM_b
            
            # pLDDT value for current residue of B
            res_pLDDT_b = res_b["CA"].get_bfactor()
        
            # Compute PAE for the residue pair and extract the minimum pLDDT
            pair_PAE = get_PAE_for_residue_pair(res_a, res_b, min_diagonal_PAE_matrix, a_is_row)
            pair_min_pLDDT = get_min_pLDDT_for_residue_pair(res_a, res_b)
            
            # Check if diagonal PAE value is lower than cutoff and pLDDT is high enough
            if pair_PAE < PAE_cutoff and pair_min_pLDDT > pLDDT_cutoff:               
                
                # Compute distance
                pair_distance = get_centroid_distance(res_a, res_b)
                
                if pair_distance < contact_distance:
                    print("Residue pair:", res_a.id[1], res_b.id[1], "\n",
                          "  - PAE =", pair_PAE, "\n",
                          "  - min_pLDDT =", pair_min_pLDDT, "\n",
                          "  - distance =", pair_distance, "\n",)
                    
                    # Add residue pairs to chimera code to select residues easily
                    chimera_code += f"/{chain_a_id}:{res_a.id[1]} /{chain_b_id}:{res_b.id[1]} "
                    
                    # Add contact pair to dict
                    contacts = pd.DataFrame({
                        # Save them as 0 base
                        "protein_ID_a": [protein_ID_a],
                        "protein_ID_b": [protein_ID_b],
                        "proteins_in_model": [proteins_in_model],
                        "res_a": [residue_id_a - 1],
                        "res_b": [residue_id_b - 1],
                        "AA_a": [seq1(res_a.get_resname())],      # Get the aminoacid of chain A in the contact
                        "AA_b": [seq1(res_b.get_resname())],      # Get the aminoacid of chain B in the contact
                        "res_name_a": [seq1(res_a.get_resname()) + str(residue_id_a)],
                        "res_name_b": [seq1(res_b.get_resname()) + str(residue_id_b)],
                        "PAE": [pair_PAE],
                        "pLDDT_a": [res_pLDDT_a],
                        "pLDDT_b": [res_pLDDT_b],
                        "min_pLDDT": [pair_min_pLDDT],
                        "pTM": "",
                        "ipTM": "",
                        "pDockQ": "",
                        "min_PAE": "",
                        "N_models": "",
                        "distance": [pair_distance],
                        "xyz_a": [residue_xyz_a],
                        "xyz_b": [residue_xyz_b],
                        "CM_a": [np.array([0,0,0])],
                        "CM_b": [np.array([0,0,0])],
                        "chimera_code": ""})
                    
                    contacts_Nmers_df = pd.concat([contacts_Nmers_df, contacts], ignore_index = True)
    
    # Add the chimera code, ipTM and min_PAE
    contacts_Nmers_df["chimera_code"] = chimera_code
    contacts_Nmers_df["pTM"] = pTM * len(contacts_Nmers_df)
    contacts_Nmers_df["ipTM"] = ipTM * len(contacts_Nmers_df)
    contacts_Nmers_df["min_PAE"] = min_PAE * len(contacts_Nmers_df)
    contacts_Nmers_df["pDockQ"] = pDockQ * len(contacts_Nmers_df)
    try:
        contacts_Nmers_df["N_models"] = int(
        filtered_pairwise_Nmers_df[
            (filtered_pairwise_Nmers_df['protein1'] == protein_ID_a)
            & (filtered_pairwise_Nmers_df['protein2'] == protein_ID_b)
            & (filtered_pairwise_Nmers_df['proteins_in_model'] == proteins_in_model)
            ]["N_models"]) * len(contacts_Nmers_df)
    except:
        print(filtered_pairwise_Nmers_df[
            (filtered_pairwise_Nmers_df['protein1'] == protein_ID_a)
            & (filtered_pairwise_Nmers_df['protein2'] == protein_ID_b)
            & (filtered_pairwise_Nmers_df['proteins_in_model'] == proteins_in_model)
            ])
        raise TypeError
                        
    # Compute CM (centroid) for contact residues and append it to df
    CM_ab = np.mean(np.array(contacts_Nmers_df["xyz_a"]), axis = 0)   # CM of A residues interacting with B
    CM_ba = np.mean(np.array(contacts_Nmers_df["xyz_b"]), axis = 0)   # CM of B residues interacting with A
    # Add CM for contact residues
    contacts_Nmers_df['CM_ab'] = [CM_ab] * len(contacts_Nmers_df)
    contacts_Nmers_df['CM_ba'] = [CM_ba] * len(contacts_Nmers_df)
    
    
    # Calculate magnitude of each CM to then compute unitary vectors
    norm_ab = np.linalg.norm(CM_ab)
    norm_ba = np.linalg.norm(CM_ba)
    # Compute unitary vectors to know direction of surfaces and add them
    contacts_Nmers_df['V_ab'] = [CM_ab / norm_ab] * len(contacts_Nmers_df)  # Unitary vector AB
    contacts_Nmers_df['V_ba'] = [CM_ba / norm_ba] * len(contacts_Nmers_df)  # Unitary vector BA
    
    
    return contacts_Nmers_df


def compute_contacts_from_pairwise_Nmers_df(pairwise_Nmers_df, filtered_pairwise_Nmers_df, sliced_PAE_and_pLDDTs,
                                            # Cutoffs
                                            contact_distance_cutoff = 8.0, contact_PAE_cutoff = 3, contact_pLDDT_cutoff = 70):
    
    print("")
    print("INITIALIZING: Compute residue-residue contacts for N-mers dataset...")
    print("")
    
    # Check if pairwise_2mers_df was passed by mistake
    if "proteins_in_model" not in pairwise_Nmers_df.columns:
        raise ValueError("Provided dataframe seems to come from 2-mers data. To compute contacts coming from 2-mers models, please, use compute_contacts_from_pairwise_2mers_df function.")
    
    # Empty df to store results
    columns = ["protein_ID_a", "protein_ID_b", "proteins_in_model", "res_a", "res_b", "AA_a", "AA_b",
                "res_name_a", "res_name_b", "PAE", "pLDDT_a", "pLDDT_b", "min_pLDDT", "pTM",
                "ipTM", "pDockQ", "min_PAE", "N_models", "distance", "xyz_a", "xyz_b", "CM_a", "CM_b", "chimera_code"]
    contacts_Nmers_df = pd.DataFrame(columns=columns)
    
    models_that_surpass_cutoff = [tuple(row) for i, row in filtered_pairwise_Nmers_df.filter(["protein1", "protein2", "proteins_in_model"]).iterrows()]
    
    # For progress bar
    total_models = len(models_that_surpass_cutoff)
    model_num = 0    
    
    for i, pairwise_Nmers_df_row in pairwise_Nmers_df.query("rank == 1").iterrows():
        
        # Skip models that do not surpass cutoffs
        row_prot1 = str(pairwise_Nmers_df_row["protein1"])
        row_prot2 = str(pairwise_Nmers_df_row["protein2"])
        row_prot_in_mod = tuple(pairwise_Nmers_df_row["proteins_in_model"])
        if (row_prot1, row_prot2, row_prot_in_mod) not in models_that_surpass_cutoff:
            continue
        
        # Compute contacts for those models that surpass cutoff
        contacts_Nmers_df_i = compute_contacts_Nmers(
            pairwise_Nmers_df_row, filtered_pairwise_Nmers_df, sliced_PAE_and_pLDDTs,
            # Cutoff parameters
            contact_distance = contact_distance_cutoff, PAE_cutoff = contact_PAE_cutoff, pLDDT_cutoff = contact_pLDDT_cutoff,
            is_debug = False)
        
        contacts_Nmers_df = pd.concat([contacts_Nmers_df, contacts_Nmers_df_i], ignore_index = True)
        
        # For progress bar
        model_num += 1
        print_progress_bar(model_num, total_models, text = " (N-mers contacts)", progress_length = 40)
        print("")
    
    return contacts_Nmers_df