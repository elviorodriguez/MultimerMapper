
import os                       # To save ChimeraX codes
import pandas as pd
from Bio.SeqUtils import seq1

# def compute_contacts_2mers(pdb_filename,
#                            min_diagonal_PAE_matrix: np.array,
#                            model_rank             : int,
#                            # Protein symbols/names/IDs
#                            protein_ID_a, protein_ID_b,
#                            # This dictionary is created on the fly in best_PAE_to_domains.py (contains best pLDDT models info)
#                            sliced_PAE_and_pLDDTs, filtered_pairwise_2mers_df,
#                            # Cutoff parameters
#                            contact_distance = 8.0, PAE_cutoff = 3, pLDDT_cutoff = 70,
#                            logger: Logger | None = None):
#     '''
#     Computes the interface contact residues and extracts several metrics for
#     each residue-residue interaction. Returns a dataframe with this info.

#     Parameters:
#     - pdb_filename (str/Bio.PDB.Model.Model): PDB file path/Bio.PDB.Model.Model object of the interaction.
#     - min_diagonal_PAE_matrix (np.array): PAE matrix for the interaction.
#     - contact_distance (float):  (default: 8.0).
#     PAE_cutoff (float): Minimum PAE value (Angstroms) between two residues in order to consider a contact (default = 5 ).
#     pLDDT_cutoff (float): Minimum pLDDT value between two residues in order to consider a contact. 
#         The minimum pLDDT value of residue pairs will be used (default = 70).

#     Returns:
#     - contacts_2mers_df (pd.DataFrame): Contains all residue-residue contacts information for the protein pair (protein_ID_a,
#         protein_ID_b, res_a, res_b, AA_a, AA_b,res_name_a, res_name_b, PAE, pLDDT_a, pLDDT_b, min_pLDDT, ipTM, min_PAE, N_models,
#         distance, xyz_a, xyz_b, CM_a, CM_b)

#     '''
#     if logger is None:
#         logger = configure_logger()(__name__)

#     # Empty df to store results
#     columns = ["protein_ID_a", "protein_ID_b", "res_a", "res_b", "AA_a", "AA_b",
#                "res_name_a", "res_name_b", "PAE", "pLDDT_a", "pLDDT_b", "min_pLDDT",
#                "ipTM", "min_PAE", "N_models", "distance", "xyz_a", "xyz_b", "CM_a", "CM_b"]
#     contacts_2mers_df = pd.DataFrame(columns=columns)
    
#     # Create PDB parser instance
#     parser = PDB.PDBParser(QUIET=True)
    
#     # Check if Bio.PDB.Model.Model object was provided directly or it was the PDB path
#     if type(pdb_filename) == PDB.Model.Model:
#         structure = pdb_filename
#     elif type(pdb_filename) == str:
#         structure = parser.get_structure('complex', pdb_filename)[0]
    
#     # Extract chain IDs
#     chains_list = [chain_ID.id for chain_ID in structure.get_chains()]
#     if len(chains_list) != 2: raise ValueError("PDB have a number of chains different than 2")
#     chain_a_id, chain_b_id = chains_list

#     # Extract chains
#     chain_a = structure[chain_a_id]
#     chain_b = structure[chain_b_id]

#     # Length of proteins
#     len_a = len(chain_a)
#     len_b = len(chain_b)
    
#     # Get the number of rows and columns
#     PAE_num_rows, PAE_num_cols = min_diagonal_PAE_matrix.shape
        
#     # Match matrix dimensions with chains
#     if len_a == PAE_num_rows and len_b == PAE_num_cols:
#         a_is_row = True
#     elif len_b == PAE_num_rows and len_a == PAE_num_cols: 
#         a_is_row = False
#     else:
#         raise ValueError("PAE matrix dimensions does not match chain lengths")
        
#     # Extract Bio.PDB.Chain.Chain objects from highest mean pLDDT model
#     highest_pLDDT_PDB_a = sliced_PAE_and_pLDDTs[protein_ID_a]["PDB_xyz"]
#     highest_pLDDT_PDB_b = sliced_PAE_and_pLDDTs[protein_ID_b]["PDB_xyz"]
    
#     # Check that sequence lengths are consistent
#     if len(highest_pLDDT_PDB_a) != len_a: raise ValueError(f"Length of chain {chain_a} of {protein_ID_a} from PDB file do not match the length of its corresponding highest mean pLDDT structure (reference).")
#     if len(highest_pLDDT_PDB_b) != len_b: raise ValueError(f"Length of chain {chain_b} of {protein_ID_a} from PDB file do not match the length of its corresponding highest mean pLDDT structure (reference).")
    
#     # Center of mass of each chain extracted from lowest pLDDT model
#     CM_a = highest_pLDDT_PDB_a.center_of_mass()
#     CM_b = highest_pLDDT_PDB_b.center_of_mass()
    
#     # Progress
#     logger.debug(f'Protein A: {protein_ID_a}')
#     logger.debug(f'Protein B: {protein_ID_b}')
#     logger.debug(f'Length A: {len_a}')
#     logger.debug(f'Length B: {len_b}')
#     logger.debug(f'PAE rows: {PAE_num_rows}')
#     logger.debug(f'PAE cols: {PAE_num_cols}')
#     logger.debug(f'Center of Mass A: {CM_a}')
#     logger.debug(f'Center of Mass B: {CM_b}')
    
#     # Chimera code to select residues from interfaces easily
#     chimera_code = "sel "

#     # Initialize matrices for clustering
#     model_data: dict = {
#         'PAE'       : np.zeros((len_a, len_b)),
#         'min_pLDDT' : np.zeros((len_a, len_b)),
#         'distance'  : np.zeros((len_a, len_b)),
#         'is_contact': np.zeros((len_a, len_b), dtype=bool)
#     }

#     # Compute contacts
#     for i, res_a in enumerate(chain_a):
        
#         # Normalized residue position from highest pLDDT model (subtract CM)
#         residue_id_a = res_a.id[1]                    # it is 1 based indexing
#         residue_xyz_a = highest_pLDDT_PDB_a[residue_id_a].center_of_mass() - CM_a
        
#         # pLDDT value for current residue of A
#         res_pLDDT_a = res_a["CA"].get_bfactor()
        
#         for j, res_b in enumerate(chain_b):
        
#             # Normalized residue position from highest pLDDT model (subtract CM)
#             residue_id_b = res_b.id[1]                    # it is 1 based indexing
#             residue_xyz_b = highest_pLDDT_PDB_b[residue_id_b].center_of_mass() - CM_b
            
#             # pLDDT value for current residue of B
#             res_pLDDT_b = res_b["CA"].get_bfactor()
        
#             # Compute PAE for the residue pair and extract the minimum pLDDT
#             pair_PAE = get_PAE_for_residue_pair(res_a, res_b, min_diagonal_PAE_matrix, a_is_row)
#             pair_min_pLDDT = get_min_pLDDT_for_residue_pair(res_a, res_b)
#             pair_distance = get_centroid_distance(res_a, res_b)

#             # Store data in matrices
#             model_data['PAE']       [i, j] = pair_PAE
#             model_data['min_pLDDT'] [i, j] = pair_min_pLDDT
#             model_data['distance']  [i, j] = pair_distance
#             model_data['is_contact'][i, j] = (pair_PAE       < PAE_cutoff         and 
#                                               pair_min_pLDDT > pLDDT_cutoff       and 
#                                               pair_distance  < contact_distance)
            
#             # Check if diagonal PAE value is lower than cutoff, pLDDT is high enough and residues are closer enough
#             if model_data['is_contact'][i, j]:
                
#                 # Debug
#                 logger.debug(f'Residue pair: {res_a.id[1]} {res_b.id[1]}')
#                 logger.debug(f'  - PAE       = {pair_PAE}')
#                 logger.debug(f'  - min_pLDDT = {pair_min_pLDDT}')
#                 logger.debug(f'  - distance  = {pair_distance}')
                
#                 # Add residue pairs to chimera code to select residues easily
#                 chimera_code += f"/a:{res_a.id[1]} /b:{res_b.id[1]} "
                
#                 # Add contact pair to dict
#                 contacts = pd.DataFrame({
#                     # Save them as 0 base
#                     "protein_ID_a": [protein_ID_a],
#                     "protein_ID_b": [protein_ID_b],
#                     "rank"        : [model_rank],
#                     "res_a"       : [residue_id_a - 1],
#                     "res_b"       : [residue_id_b - 1],
#                     "AA_a"        : [seq1(res_a.get_resname())],      # Get the amino acid of chain A in the contact
#                     "AA_b"        : [seq1(res_b.get_resname())],      # Get the amino acid of chain B in the contact
#                     "res_name_a"  : [seq1(res_a.get_resname()) + str(residue_id_a)],
#                     "res_name_b"  : [seq1(res_b.get_resname()) + str(residue_id_b)],
#                     "PAE"         : [pair_PAE],
#                     "pLDDT_a"     : [res_pLDDT_a],
#                     "pLDDT_b"     : [res_pLDDT_b],
#                     "min_pLDDT"   : [pair_min_pLDDT],
#                     "ipTM"        : "",
#                     "min_PAE"     : "",
#                     "N_models"    : "",
#                     "distance"    : [pair_distance],
#                     "xyz_a"       : [residue_xyz_a],
#                     "xyz_b"       : [residue_xyz_b],
#                     "CM_a"        : [np.array([0,0,0])],
#                     "CM_b"        : [np.array([0,0,0])]}
#                 )
                
#                 contacts_2mers_df = pd.concat([contacts_2mers_df, contacts], ignore_index = True)

#     # Add ipTM and min_PAE to df
#     contacts_2mers_df["ipTM"]     = float(filtered_pairwise_2mers_df[(filtered_pairwise_2mers_df['protein1'] == protein_ID_a) & (filtered_pairwise_2mers_df['protein2'] == protein_ID_b)]["ipTM"])
#     contacts_2mers_df["min_PAE"]  = float(filtered_pairwise_2mers_df[(filtered_pairwise_2mers_df['protein1'] == protein_ID_a) & (filtered_pairwise_2mers_df['protein2'] == protein_ID_b)]["min_PAE"])
#     contacts_2mers_df["N_models"] = int(filtered_pairwise_2mers_df[(filtered_pairwise_2mers_df['protein1'] == protein_ID_a) & (filtered_pairwise_2mers_df['protein2'] == protein_ID_b)]["N_models"])
                    
#     # Compute CM (centroid) for contact residues and append it to df
#     CM_ab = np.mean(np.array(contacts_2mers_df["xyz_a"]), axis = 0)   # CM of A residues interacting with B
#     CM_ba = np.mean(np.array(contacts_2mers_df["xyz_b"]), axis = 0)   # CM of B residues interacting with A
#     # Add CM for contact residues
#     contacts_2mers_df['CM_ab'] = [CM_ab] * len(contacts_2mers_df)
#     contacts_2mers_df['CM_ba'] = [CM_ba] * len(contacts_2mers_df)
    
    
#     # Calculate magnitude of each CM to then compute unitary vectors
#     norm_ab = np.linalg.norm(CM_ab)
#     norm_ba = np.linalg.norm(CM_ba)
#     # Compute unitary vectors to know direction of surfaces and add them
#     contacts_2mers_df['V_ab'] = [CM_ab / norm_ab] * len(contacts_2mers_df)  # Unitary vector AB
#     contacts_2mers_df['V_ba'] = [CM_ba / norm_ba] * len(contacts_2mers_df)  # Unitary vector BA
    
#     # Progress
#     number_of_contacts: int = contacts_2mers_df.shape[0]
#     logger.info(f'   - Nº of contacts found (rank_{model_rank}): {number_of_contacts}')

#     return contacts_2mers_df, chimera_code, model_data



'''

    # DataFrame initialization outside the loop
    results = []
    CM_a = highest_pLDDT_PDB_a.center_of_mass()
    CM_b = highest_pLDDT_PDB_b.center_of_mass()
    highest_pLDDT_PDB_a[res.id[1]].center_of_mass() - CM_a
    highest_pLDDT_PDB_b[res.id[1]].center_of_mass() - CM_b

            if model_data['is_contact'][i, j]:
                result = {
                    "protein_ID_a": protein_ID_a,
                    "protein_ID_b": protein_ID_b,
                    "rank": model_rank,
                    "res_a": res_a.id[1] - 1,
                    "res_b": res_b.id[1] - 1,
                    "AA_a": seq1(res_a.get_resname()),
                    "AA_b": seq1(res_b.get_resname()),
                    "res_name_a": f"{seq1(res_a.get_resname())}{res_a.id[1]}",
                    "res_name_b": f"{seq1(res_b.get_resname())}{res_b.id[1]}",
                    "PAE": pair_PAE,
                    "pLDDT_a": res_pLDDT_a,
                    "pLDDT_b": res_pLDDT_b,
                    "min_pLDDT": pair_min_pLDDT,
                    "ipTM": "",
                    "min_PAE": "",
                    "N_models": "",
                    "distance": pair_distance,
                    "xyz_a": residue_xyz_a,
                    "xyz_b": residue_xyz_b,
                    "CM_a": np.array([0, 0, 0]),
                    "CM_b": np.array([0, 0, 0])
                }
                results.append(result)

    # Create the DataFrame once after the loop
    contacts_2mers_df = pd.DataFrame(results)

        # Add ipTM, min_PAE, and N_models
    data_row = filtered_pairwise_2mers_df[(filtered_pairwise_2mers_df['protein1'] == protein_ID_a) & (filtered_pairwise_2mers_df['protein2'] == protein_ID_b)]
    contacts_2mers_df["ipTM"] = float(data_row["ipTM"])
    contacts_2mers_df["min_PAE"] = float(data_row["min_PAE"])
    contacts_2mers_df["N_models"] = int(data_row["N_models"])

    if not contacts_2mers_df.empty:
        # Compute CM (centroid) for contact residues
        CM_ab = np.mean(np.array(contacts_2mers_df["xyz_a"].tolist()), axis=0)
        CM_ba = np.mean(np.array(contacts_2mers_df["xyz_b"].tolist()), axis=0)

        contacts_2mers_df['CM_ab'] = [CM_ab] * len(contacts_2mers_df)
        contacts_2mers_df['CM_ba'] = [CM_ba] * len(contacts_2mers_df)

        # Calculate magnitude and unitary vectors
        norm_ab = np.linalg.norm(CM_ab)
        norm_ba = np.linalg.norm(CM_ba)

        contacts_2mers_df['V_ab'] = [CM_ab / norm_ab] * len(contacts_2mers_df)
        contacts_2mers_df['V_ba'] = [CM_ba / norm_ba] * len(contacts_2mers_df)
    else:
        # Handle the case when no contacts are found
        # logger.info(f'No contacts found for {protein_ID_a} vs {protein_ID_b} at rank {model_rank}')
        contacts_2mers_df['CM_ab'] = []
        contacts_2mers_df['CM_ba'] = []
        contacts_2mers_df['V_ab'] = []
        contacts_2mers_df['V_ba'] = []
    
    # # Precompute residue positions and pLDDT values for faster access in loops
    # residues_a = [(res, res["CA"].get_bfactor()) for res in chain_a]
    # residues_b = [(res, res["CA"].get_bfactor()) for res in chain_b]

    # for i, (res_a, res_pLDDT_a) in enumerate(residues_a):
    #     for j, (res_b, res_pLDDT_b) in enumerate(residues_b):
    #         pair_PAE = get_PAE_for_residue_pair(res_a, res_b, min_diagonal_PAE_matrix, True)
    #         pair_min_pLDDT = get_min_pLDDT_for_residue_pair(res_a, res_b)
    #         pair_distance = get_centroid_distance(res_a, res_b)

    #         model_data['PAE'][i, j] = pair_PAE
    #         model_data['min_pLDDT'][i, j] = pair_min_pLDDT
    #         model_data['distance'][i, j] = pair_distance
    #         model_data['is_contact'][i, j] = (pair_PAE < PAE_cutoff and 
    #                                           pair_min_pLDDT > pLDDT_cutoff and 
    #                                           pair_distance < contact_distance)





   # Empty df and dicts to store results
    columns = ["protein_ID_a", "protein_ID_b", "rank", "res_a", "res_b", "AA_a", "AA_b",
               "res_name_a", "res_name_b", "PAE", "pLDDT_a", "pLDDT_b", "min_pLDDT",
               "ipTM", "min_PAE", "N_models", "distance", "xyz_a", "xyz_b", "CM_a", "CM_b"]
    contacts_2mers_df = pd.DataFrame(columns=columns)
    chimera_code_dict: dict = {}

    
    contacts_2mers_df_i, chimera_code, 

        chimera_code_dict[sorted_tuple_model] = chimera_code
        contacts_2mers_df = pd.concat([contacts_2mers_df, contacts_2mers_df_i], ignore_index = True)


    return contacts_2mers_df, chimera_code_dict, 

    












    contacts_2mers_df, chimera_code_2mers_dict, 

    














    contacts_2mers_df['rank'] = contacts_2mers_df['rank'].astype(int)
    contacts_2mers_df['tuple_pair'] = [tuple(sorted([p1, p2])) for p1, p2 in zip(contacts_2mers_df['protein_ID_a'], contacts_2mers_df['protein_ID_b']) ]
            "chimera_code_2mers": chimera_code_2mers_dict,
        "contacts_2mers_df" : contacts_2mers_df,

'''






















'''
    # Empty df to store results
    columns = ["protein_ID_a", "protein_ID_b", "proteins_in_model", "rank", "res_a", "res_b", "AA_a", "AA_b",
                "res_name_a", "res_name_b", "PAE", "pLDDT_a", "pLDDT_b", "min_pLDDT", "pTM",
                "ipTM", "pDockQ", "min_PAE", "N_models", "distance", "xyz_a", "xyz_b", "CM_a", "CM_b"]
    contacts_Nmers_df = pd.DataFrame(columns=columns)  
    proteins_in_model       = pairwise_Nmers_df_row["proteins_in_model"]
    pTM                     = pairwise_Nmers_df_row["pTM"]
    ipTM                    = pairwise_Nmers_df_row["ipTM"]
    min_PAE                 = pairwise_Nmers_df_row["min_PAE"]
    pDockQ                  = pairwise_Nmers_df_row["pDockQ"]
    
    # Center of mass of each chain extracted from lowest pLDDT model
    CM_a = highest_pLDDT_PDB_a.center_of_mass()
    CM_b = highest_pLDDT_PDB_b.center_of_mass()
    logger.debug(f'Center of Mass A: {CM_a}')
    logger.debug(f'Center of Mass B: {CM_b}')
    
  # Chimera code to select residues from interfaces easily
    chimera_code = "sel "

    # Initialize matrices for clustering
    model_data: dict = {
        'PAE': np.zeros((len_a, len_b)),
        'min_pLDDT': np.zeros((len_a, len_b)),
        'distance': np.zeros((len_a, len_b)),
        'is_contact': np.zeros((len_a, len_b), dtype=bool)
    }
    
    # Compute contacts
    for i, res_a in enumerate(chain_a):
        
        # Normalized residue position from highest pLDDT model (subtract CM)
        residue_id_a = res_a.id[1]                    # it is 1 based indexing
        residue_xyz_a = highest_pLDDT_PDB_a[residue_id_a].center_of_mass() - CM_a
        
        # pLDDT value for current residue of A
        res_pLDDT_a = res_a["CA"].get_bfactor()
        
        for j, res_b in enumerate(chain_b):
        
            # Normalized residue position (subtract CM)
            residue_id_b = res_b.id[1]                    # it is 1 based indexing
            residue_xyz_b = highest_pLDDT_PDB_b[residue_id_b].center_of_mass() - CM_b
            
            # pLDDT value for current residue of B
            res_pLDDT_b = res_b["CA"].get_bfactor()
        
            # Compute PAE for the residue pair and extract the minimum pLDDT and distance
            pair_PAE = get_PAE_for_residue_pair(res_a, res_b, min_diagonal_PAE_matrix, a_is_row)
            pair_min_pLDDT = get_min_pLDDT_for_residue_pair(res_a, res_b)
            pair_distance = get_centroid_distance(res_a, res_b)

                    
            # Store data in matrices
            model_data['PAE']       [i, j] = pair_PAE
            model_data['min_pLDDT'] [i, j] = pair_min_pLDDT
            model_data['distance']  [i, j] = pair_distance
            model_data['is_contact'][i, j] = (pair_PAE       < PAE_cutoff         and 
                                              pair_min_pLDDT > pLDDT_cutoff       and 
                                              pair_distance  < contact_distance)
            
            # Check if diagonal PAE value is lower than cutoff, pLDDT is high enough and residues are closer enough
            if model_data['is_contact'][i, j]:

                logger.debug(f'Residue pair: {residue_id_a} {residue_id_b}')
                logger.debug(f'  - PAE       = {pair_PAE}')
                logger.debug(f'  - min_pLDDT = {pair_min_pLDDT}')
                logger.debug(f'  - distance  = {pair_distance}')
                
                # Add residue pairs to chimera code to select residues easily
                chimera_code += f"/{chain_a_id}:{residue_id_a} /{chain_b_id}:{residue_id_b} "
                
                # Add contact pair to dict
                contacts = pd.DataFrame({
                    # Save them as 0 base
                    "protein_ID_a"      : [protein_ID_a],
                    "protein_ID_b"      : [protein_ID_b],
                    "proteins_in_model" : [proteins_in_model],
                    "rank"              : [model_rank],
                    "res_a"             : [residue_id_a - 1],
                    "res_b"             : [residue_id_b - 1],
                    "AA_a"              : [seq1(res_a.get_resname())],      # Get the amino acid of chain A in the contact
                    "AA_b"              : [seq1(res_b.get_resname())],      # Get the amino acid of chain B in the contact
                    "res_name_a"        : [seq1(res_a.get_resname()) + str(residue_id_a)],
                    "res_name_b"        : [seq1(res_b.get_resname()) + str(residue_id_b)],
                    "PAE"               : [pair_PAE],
                    "pLDDT_a"           : [res_pLDDT_a],
                    "pLDDT_b"           : [res_pLDDT_b],
                    "min_pLDDT"         : [pair_min_pLDDT],
                    "pTM"               : "",
                    "ipTM"              : "",
                    "pDockQ"            : "",
                    "min_PAE"           : "",
                    "N_models"          : "",
                    "distance"          : [pair_distance],
                    "xyz_a"             : [residue_xyz_a],
                    "xyz_b"             : [residue_xyz_b],
                    "CM_a"              : [np.array([0,0,0])],
                    "CM_b"              : [np.array([0,0,0])]})
                
                contacts_Nmers_df = pd.concat([contacts_Nmers_df, contacts], ignore_index = True)
    
    # Add pTM, ipTM, min_PAE, pDockQ and N_models to the dataframe
    contacts_Nmers_df["pTM"]     = [pTM]     * len(contacts_Nmers_df)
    contacts_Nmers_df["ipTM"]    = [ipTM]    * len(contacts_Nmers_df)
    contacts_Nmers_df["min_PAE"] = [min_PAE] * len(contacts_Nmers_df)
    contacts_Nmers_df["pDockQ"]  = [pDockQ]  * len(contacts_Nmers_df)
    try:
        contacts_Nmers_df["N_models"] = [int(
        filtered_pairwise_Nmers_df[
            (filtered_pairwise_Nmers_df['protein1'] == protein_ID_a)
            & (filtered_pairwise_Nmers_df['protein2'] == protein_ID_b)
            & (filtered_pairwise_Nmers_df['proteins_in_model'] == proteins_in_model)
            ]["N_models"])] * len(contacts_Nmers_df)
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
    
    # Progress
    number_of_contacts: int = contacts_Nmers_df.shape[0]

    return contacts_Nmers_df, chimera_code,










    contacts_Nmers_df_i, chimera_code, 
    contacts_Nmers_df = pd.concat([contacts_Nmers_df, contacts_Nmers_df_i], ignore_index = True)
    chimera_code_Nmers_dict[tuple(sorted(row_prot_in_mod))] = chimera_code
    return contacts_Nmers_df, chimera_code_Nmers_dict, 






        # Empty df to store results
    columns = ["protein_ID_a", "protein_ID_b", "proteins_in_model", "rank", "res_a", "res_b", "AA_a", "AA_b",
                "res_name_a", "res_name_b", "PAE", "pLDDT_a", "pLDDT_b", "min_pLDDT", "pTM",
                "ipTM", "pDockQ", "min_PAE", "N_models", "distance", "xyz_a", "xyz_b", "CM_a", "CM_b"]
    contacts_Nmers_df = pd.DataFrame(columns=columns)
    # Chimera code dict to store contacts
    chimera_code_Nmers_dict: dict = {}

    




        contacts_Nmers_df, chimera_code_Nmers_dict, 
    # Force converting rank columns to ints
    contacts_Nmers_df['rank'] = contacts_Nmers_df['rank'].astype(int)

    # Add columns with unique tuple identifiers for the pair (useful for clustering)

    contacts_Nmers_df['tuple_pair'] = [tuple(sorted([p1, p2])) for p1, p2 in zip(contacts_Nmers_df['protein_ID_a'], contacts_Nmers_df['protein_ID_b']) ]

        "contacts_Nmers_df" : contacts_Nmers_df,
        "chimera_code_Nmers": chimera_code_Nmers_dict,

'''



