# -*- coding: utf-8 -*-

import pandas as pd

from src.convergency import does_nmer_is_fully_connected_network
from src.interpret_dynamics import classify_edge_dynamics, classification_df
from cfg.default_settings import min_PAE_cutoff_Nmers, pDockQ_cutoff_Nmers, N_models_cutoff, Nmer_stability_method, Nmers_contacts_cutoff_convergency, N_models_cutoff_conv_soft, miPAE_cutoff_conv_soft
from cfg.default_settings import use_dynamic_conv_soft_func, miPAE_cutoff_conv_soft_list, dynamic_conv_start, dynamic_conv_end

def does_2mer_homodimerize(query_protein: str, pairwise_2mers_df: pd.DataFrame, pairwise_2mers_df_F3: pd.DataFrame):
    '''Returns True if homo-2-mer forms, False if not, and None if it was not tested'''

    for i, row in pairwise_2mers_df_F3.iterrows():
        
        protein1 = str(row["protein1"])
        protein2 = str(row["protein2"])
               
        if (protein1 == protein2) and (protein1 == query_protein):
            return True
    
    for i, row in pairwise_2mers_df.iterrows():
        
        protein1 = str(row["protein1"])
        protein2 = str(row["protein2"])
               
        if (protein1 == protein2) and (protein1 == query_protein):
            return False

    return None

# Get proteins that dimerize (mm_output["pairwise_2mers_df_F3"]):
def get_proteins_that_homodimerize(pairwise_2mers_df_F3: pd.DataFrame,
                                   pairwise_Nmers_df_F3: pd.DataFrame):
    '''
    Analyzes pairwise_2mers_df_F3 and return the set of proteins that
    forms homodimers.

    Parameters
    ----------
    pairwise_2mers_df_F3 : pd.DataFrame
        Dataframe of interacting proteins from 2-mers.

    Returns
    -------
    homodim_prots : set
        Set of proteins that homodimerize.

    '''
    
    homodim_prots: set = set()
    
    for i, row in pairwise_2mers_df_F3.iterrows():
        
        protein1 = str(row["protein1"])
        protein2 = str(row["protein2"])
               
        if (protein1 == protein2) and (protein1 not in homodim_prots):
            homodim_prots.add(protein1)

    for i, row in pairwise_Nmers_df_F3.iterrows():
        
        protein1 = str(row["protein1"])
        protein2 = str(row["protein2"])
               
        if (protein1 == protein2) and (protein1 not in homodim_prots):
            homodim_prots.add(protein1)
            
    return set(homodim_prots)


def get_homo_N_mers_pairwise_df(pairwise_Nmers_df):
    
    # Filter for homooligomers
    homooligomers = pairwise_Nmers_df[pairwise_Nmers_df['proteins_in_model'].apply(lambda x: len(set(x)) == 1)]
    
    # Filter for models larger than dimers
    homo_N_mers_df = homooligomers[homooligomers['proteins_in_model'].str.len() > 2]
    
    return homo_N_mers_df


def check_if_skipped_homo_N_mers(homo_Nmers_models: set,
                                 protein,
                                 logger) -> list[bool]:
    '''
    Analyzes available homo-N-mers in search of inconsistencies with the
    pipeline. As homo_N_mers must be incremental (trimer -> tetramer -> etc),
    this function returns a boolean list that indicates true if the predictions
    were performed correctly. 
    '''

    
    models = sorted(list(homo_Nmers_models), key = len)
    # In the case that not even the homo-3-mer is present:
    if len(models) == 0:
        logger.warning(f"   No homooligomers bigger with N>2 for {protein}")
        return [False]
    biggest_homo_N_mer  = len(models[-1])

    # try:
    #     biggest_homo_N_mer  = len(models[-1])
    # # In the case that not even the homo-3-mer is present, this will rise an exception (models == [])
    # except IndexError():
    #     logger.warning(f"No homooligomerization states for {protein}")
    #     return [False]
    
    # Progress
    logger.info(f'   Computing homooligomerization stability for {models[0][0]}:')
    
    is_ok = []
    
    # All homo_N_mers must be incremental (trimer -> tetramer -> etc)
    for N in range(3, biggest_homo_N_mer + 1):
        
         # Verify if there is a model with length == N
        if any(len(model) == N for model in models):
            is_ok.append(True)
        else:
            logger.info(f'   Missing homo-{N}-mer for {models[0][0]}!')
            logger.info( '   Homo-N-mers must be incremental (trimer -> tetramer -> etc)')
            logger.info( '   You need to provide them all in order to properly extract homooligomerization states!')
            is_ok.append(False)
         
    return is_ok

# ok_set = set([("A", "A", "A"), ("A", "A", "A", "A"), ("A", "A", "A", "A", "A")])
# ok_set2 =  set( [("A", "A", "A", "A"), ("A", "A", "A"), ("A", "A", "A", "A", "A")])

# not_ok_set1 = set( [("A", "A", "A", "A"), ("A", "A", "A", "A", "A")])

# check_if_skipped_homo_N_mers(not_ok_set1)

# sorted(list(not_ok_set1), key=len)


def get_set_of_chains_in_model(model_pairwise_df: pd.DataFrame) -> set:
    """
    Extract all unique chain IDs from the model_pairwise_df.
    
    Args:
        model_pairwise_df (pd.DataFrame): DataFrame containing pairwise interactions.
    
    Returns:
        set: Set of all unique chain IDs.
    """
    chains_set = set()
    
    for i, row in model_pairwise_df.iterrows():
        model_chains = list(row['model'].get_chains())
        chain_ID1 = model_chains[0].get_id()
        chain_ID2 = model_chains[1].get_id()
        
        chains_set.add(chain_ID1)
        chains_set.add(chain_ID2)
    
    return chains_set

def extract_chain_ids(row):
    chains = list(row['model'].get_chains())
    chain_ID1 = chains[0].get_id()
    chain_ID2 = chains[1].get_id()
    return pd.Series([chain_ID1, chain_ID2])

def add_chain_information_to_df(model_pairwise_df):

    # Suppress the SettingWithCopyWarning (Crossing fingers for this to not brake the software in the future...)
    pd.options.mode.chained_assignment = None

    # Apply the extract_chain_ids function to each row
    model_pairwise_df[['chain_ID1', 'chain_ID2']] = model_pairwise_df.apply(extract_chain_ids, axis=1)
    return model_pairwise_df
        
def does_all_have_at_least_one_interactor(model_pairwise_df: pd.DataFrame,
                                          min_PAE_cutoff_Nmers: int | float,
                                          pDockQ_cutoff_Nmers: int | float,
                                          N_models_cutoff: int) -> bool:
    

    for i, row in model_pairwise_df.iterrows():
        
        for chain_ID in get_set_of_chains_in_model(model_pairwise_df):
            
            # Variable to count the number of times the chains surpass the cutoffs
            models_in_which_chain_surpass_cutoff = 0
            
            # Count one by one
            for rank in range(1, 6):
                chain_df = (model_pairwise_df
                                 .query('rank == @rank')
                                 .query('chain_ID1 == @chain_ID | chain_ID2 == @chain_ID')
                            )
                for c, chain_pair in chain_df.iterrows():
                    if chain_pair["min_PAE"] <= min_PAE_cutoff_Nmers and chain_pair["pDockQ"] >= pDockQ_cutoff_Nmers:
                        models_in_which_chain_surpass_cutoff += 1
                        break
                
            if not models_in_which_chain_surpass_cutoff >= N_models_cutoff:
                return False
    return True

# does_all_have_at_least_one_interactor(model_pairwise_df,
#                                           min_PAE_cutoff_Nmers,
#                                           pDockQ_cutoff_Nmers,
#                                           N_models_cutoff)


def find_homooligomerization_breaks(pairwise_2mers_df, pairwise_Nmers_df, pairwise_2mers_df_F3, pairwise_Nmers_df_F3, mm_output,
                                    logger,
                                    min_PAE_cutoff_Nmers,
                                    pDockQ_cutoff_Nmers,
                                    N_models_cutoff,
                                    Nmer_stability_method = Nmer_stability_method,
                                    N_models_cutoff_conv_soft = N_models_cutoff_conv_soft,
                                    miPAE_cutoff_conv_soft = miPAE_cutoff_conv_soft,
                                    use_dynamic_conv_soft_func = use_dynamic_conv_soft_func,
                                    miPAE_cutoff_conv_soft_list = miPAE_cutoff_conv_soft_list,
                                    dynamic_conv_start = dynamic_conv_start,
                                    dynamic_conv_end = dynamic_conv_end,):
    
    # Proteins that homodimerize
    homodim_prots: set = get_proteins_that_homodimerize(pairwise_2mers_df_F3, pairwise_Nmers_df_F3)
    
    # Initialize dict to store homooligomerization states of each protein
    homooligomerization_states: dict = {protein_ID: {"is_ok": [], "N_states": [], "2mer_interact": []} for protein_ID in homodim_prots}
    
    # This manages the case in which there is no N-mers models for the homooligomerization states
    try:
        # Subset of Nmers_df corresponding to homo-N-mers
        homo_N_mers_pairwise_df = get_homo_N_mers_pairwise_df(pairwise_Nmers_df)

    except KeyError as e:
        
        if pairwise_Nmers_df.empty:
            logger.warning("No N-mers passed... continuing.")
            
            # Set the dataframe as empty
            homo_N_mers_pairwise_df = pairwise_Nmers_df
            
            # # Before it was returning the homooligomerization_states without modifications, which caused many problems later
            # return homooligomerization_states

        else:
            logger.error( "Unknown KeyError encountered inside find_homooligomerization_breaks")
            logger.error(f"   - KeyError encountered: {e}")
            logger.error( "   - pairwise_Nmers_df was expecting to be empty:")
            logger.error(f"   - pairwise_Nmers_df content:n\ {pairwise_Nmers_df}")
            logger.error( '   - MultimerMapper will continue...')
            logger.error( '   - Results may be unreliable or it will crash later...')

    except Exception as e:
        logger.error(f"An unexpected error occurred inside find_homooligomerization_breaks:")
        logger.error(f"   - Error: {e}")
        logger.error( "   - get_homo_N_mers_pairwise_df() failed analyzing pairwise_Nmers_df")
        logger.error(f"   - pairwise_Nmers_df content:n\ {pairwise_Nmers_df}")
        logger.error( '   - MultimerMapper will continue anyways...')
        logger.error( '   - Results may be unreliable or the program will crash later...')
           
    for protein in homodim_prots:
        
        protein_homo_N_mers_pairwise_df = homo_N_mers_pairwise_df.query('protein1 == @protein')
        
        homo_Nmers_models: set = set(protein_homo_N_mers_pairwise_df['proteins_in_model'])
        
        # Skipped homo-N-mers will show False
        is_ok: list[bool] = check_if_skipped_homo_N_mers(homo_Nmers_models, protein, logger)
        
        # If any state was skipped
        if any(not ok for ok in is_ok):
            
            # State that there is a problem with the models filling N_states with Nones
            homooligomerization_states[protein]["is_ok"] = is_ok
            homooligomerization_states[protein]["N_states"] = [None] * len(is_ok)
            homooligomerization_states[protein]["2mer_interact"] = does_2mer_homodimerize(query_protein        = protein,
                                                                                          pairwise_2mers_df    = pairwise_2mers_df,
                                                                                          pairwise_2mers_df_F3 = pairwise_2mers_df_F3)
            
            # Skip to the next protein
            continue
        
        # If everything is OK, continue the computation
        else:
            
            # Add information to dict
            homooligomerization_states[protein]["is_ok"] = is_ok
            homooligomerization_states[protein]["2mer_interact"] = does_2mer_homodimerize(query_protein        = protein,
                                                                                          pairwise_2mers_df    = pairwise_2mers_df,
                                                                                          pairwise_2mers_df_F3 = pairwise_2mers_df_F3)
            
            # # Progress
            # logger.info(f'      Found {len(is_ok)} homooligomerization state(s):')
        
        # For each homooligomerization state (starting with homo-3-mers)
        for N_state, model in enumerate(sorted(list(homo_Nmers_models), key = len), start = 3):
            
            # Separate only data for the current homooligomerization state and add chain info
            model_pairwise_df = protein_homo_N_mers_pairwise_df.query('proteins_in_model == @model')
            add_chain_information_to_df(model_pairwise_df)
            
            # Select which method to use: PAE
            if Nmer_stability_method == "pae":
                # Make the verification
                all_have_at_least_one_interactor: bool = does_all_have_at_least_one_interactor(
                                                            model_pairwise_df = model_pairwise_df,
                                                            min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers,
                                                            pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers,
                                                            N_models_cutoff = N_models_cutoff)
                
                Nmer_is_stable = all_have_at_least_one_interactor
                
            # Select which method to use: Contact Network (recommended)
            elif Nmer_stability_method == "contact_network":
                # logger.info("--------------------- USING CONTACT NETWORK METHOD ---------------------")

                # Make the verification using the new function
                is_fully_connected_network = does_nmer_is_fully_connected_network(
                                            model_pairwise_df = model_pairwise_df,
                                            mm_output         = mm_output,
                                            pair              = (protein, protein),
                                            Nmers_contacts_cutoff = Nmers_contacts_cutoff_convergency,
                                            N_models_cutoff = N_models_cutoff,
                                            N_models_cutoff_conv_soft = N_models_cutoff_conv_soft,
                                            miPAE_cutoff_conv_soft = miPAE_cutoff_conv_soft,
                                            use_dynamic_conv_soft_func = use_dynamic_conv_soft_func,
                                            miPAE_cutoff_conv_soft_list = miPAE_cutoff_conv_soft_list,
                                            dynamic_conv_start = dynamic_conv_start,
                                            dynamic_conv_end = dynamic_conv_end)
                
                Nmer_is_stable = is_fully_connected_network

            # Select which method to use: Falls back to default method (Contact Network)
            else:
                logger.error(f"   - Something went wrong! Provided Nmer_stability_method is unknown: {Nmer_stability_method}")
                logger.error(f"      - Using default method: contact_network")

                # Make the verification using the new function
                is_fully_connected_network = does_nmer_is_fully_connected_network(
                                            model_pairwise_df = model_pairwise_df,
                                            mm_output         = mm_output,
                                            pair              = (protein, protein),
                                            Nmers_contacts_cutoff = Nmers_contacts_cutoff_convergency,
                                            N_models_cutoff = N_models_cutoff,
                                            N_models_cutoff_conv_soft = N_models_cutoff_conv_soft,
                                            miPAE_cutoff_conv_soft = miPAE_cutoff_conv_soft,
                                            use_dynamic_conv_soft_func = use_dynamic_conv_soft_func,
                                            miPAE_cutoff_conv_soft_list = miPAE_cutoff_conv_soft_list,
                                            dynamic_conv_start = dynamic_conv_start,
                                            dynamic_conv_end = dynamic_conv_end)
                
                Nmer_is_stable = is_fully_connected_network
            
            # Add if it surpass cutoff to N_states
            homooligomerization_states[protein]["N_states"].append(Nmer_is_stable)
        
        # For proteins that the last state computed (N) is positive, add the suggestion to compute N+1
        if all(ok for ok in is_ok) and (homooligomerization_states[protein]["N_states"][-1] == True):

            homooligomerization_states[protein]["is_ok"]   .append(False)
            homooligomerization_states[protein]["N_states"].append(None)


        # Log the results for the homooligomer
        if homooligomerization_states[protein]['2mer_interact']:
            dimer_classification = "Stable"
        elif homooligomerization_states[protein]['2mer_interact']:
            dimer_classification = "Unstable"
        elif homooligomerization_states[protein]['2mer_interact'] is None:
            dimer_classification = "Not tested"
        else:
            dimer_classification = f"Unexpected classification: {homooligomerization_states[protein]['2mer_interact']}"
        logger.info(f'      - 2 x {protein}: {dimer_classification}')
        for N_state_idx, no_skip in enumerate(homooligomerization_states[protein]["is_ok"]):
            if no_skip and homooligomerization_states[protein]["N_states"][N_state_idx]:
                nmer_classification = "Stable"
            elif no_skip and not homooligomerization_states[protein]["N_states"][N_state_idx]:
                nmer_classification = "Unstable"
            elif not no_skip:
                nmer_classification = "Not tested"
            else:
                nmer_classification = f"Unexpected classification: {no_skip}"

            logger.info(f'      - {N_state_idx + 3} x {protein}: {nmer_classification}')
    
    return homooligomerization_states
            
            
def add_homooligomerization_state(graph, pairwise_2mers_df, pairwise_Nmers_df, pairwise_2mers_df_F3, pairwise_Nmers_df_F3,
                                  edges_g1_sort, edges_g2_sort, untested_edges_tuples, tested_Nmers_edges_sorted,
                                  logger,
                                  mm_output,
                                  min_PAE_cutoff_Nmers  = min_PAE_cutoff_Nmers,
                                  pDockQ_cutoff_Nmers   = pDockQ_cutoff_Nmers,
                                  N_models_cutoff       = N_models_cutoff,
                                  Nmer_stability_method = Nmer_stability_method,
                                  N_models_cutoff_conv_soft = N_models_cutoff_conv_soft,
                                  miPAE_cutoff_conv_soft = miPAE_cutoff_conv_soft,
                                  use_dynamic_conv_soft_func = use_dynamic_conv_soft_func,
                                  miPAE_cutoff_conv_soft_list = miPAE_cutoff_conv_soft_list,
                                  dynamic_conv_start = dynamic_conv_start,
                                  dynamic_conv_end = dynamic_conv_end,):

    # Compute homooligomerization data
    homooligomerization_states = find_homooligomerization_breaks(
                                        pairwise_2mers_df = pairwise_2mers_df,
                                        pairwise_Nmers_df = pairwise_Nmers_df,
                                        pairwise_2mers_df_F3 = pairwise_2mers_df_F3,
                                        pairwise_Nmers_df_F3 = pairwise_Nmers_df_F3,
                                        mm_output = mm_output,
                                        logger = logger,
                                        min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers,
                                        pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers,
                                        N_models_cutoff = N_models_cutoff,
                                        Nmer_stability_method = Nmer_stability_method,
                                        N_models_cutoff_conv_soft = N_models_cutoff_conv_soft,
                                        miPAE_cutoff_conv_soft = miPAE_cutoff_conv_soft,
                                        use_dynamic_conv_soft_func = use_dynamic_conv_soft_func,
                                        miPAE_cutoff_conv_soft_list = miPAE_cutoff_conv_soft_list,
                                        dynamic_conv_start = dynamic_conv_start,
                                        dynamic_conv_end = dynamic_conv_end
                                        )

    # Initialize edge attribute
    graph.es["homooligomerization_states"] = None

    for edge in graph.es:
        source_name = graph.vs[edge.source]["name"]
        target_name = graph.vs[edge.target]["name"]

        # If it is a homooligomer
        if source_name == target_name:
            
            try:
                # Add its homooligomerization state data
                edge["homooligomerization_states"] = homooligomerization_states[source_name]

            # The edge was added because it was not tested?
            except KeyError:
                
                # Get dynamics in this case to verify if it is indirect
                e_dynamic = classify_edge_dynamics(tuple_edge = tuple(sorted(edge['name'])),
                                                    true_edge = edge,
                                                    
                                                    # Cutoffs
                                                    N_models_cutoff = N_models_cutoff,
                                                    
                                                    # Sorted tuple edges lists
                                                    sorted_edges_2mers_graph  = edges_g1_sort, 
                                                    sorted_edges_Nmers_graph  = edges_g2_sort,
                                                    untested_edges_tuples     = untested_edges_tuples,
                                                    tested_Nmers_edges_sorted = tested_Nmers_edges_sorted,
                                                    
                                                    classification_df = classification_df,
                                                    logger = logger)
                                
                if e_dynamic == 'Indirect':
                    logger.warning(f'Homooligomerization of indirect edge: ({source_name}, {target_name})')
                    logger.warning( '   homooligomerization_states will be set as empty indicating that it comes from an indirect interaction...')
                    # Add its homooligomerization state data
                    edge["homooligomerization_states"] = {"is_ok": [], "N_states": [],
                                                            "error"     : True,
                                                            "error_type": "Indirect edge"}

                else:
                    logger.error(f'KeyError appeared during homooligomerization state detection of {source_name}')
                    logger.error( '   MultimerMapper will continue anyways...')
                    logger.error( '   combined_graph will contain the key generated_from_key_error in "homooligomerization_states" attribute')
                    logger.error(f'   You can check the edge {edge["name"]} for more info...')

                    # Add its homooligomerization state data as error
                    edge["homooligomerization_states"] = {"is_ok": [], "N_states": [],
                                                            "error"     : True,
                                                            "error_type": "KeyError on edge not classified as Indirect"}


            except Exception as e:
                logger.error(f'An unexpected error appeared during homooligomerization state detection of {source_name}')
                logger.error(f'   Exception: {e}')
                logger.error( '   MultimerMapper will continue anyways...')
                logger.error( '   Results may be unreliable or it may crash later...')
    
    return homooligomerization_states