# -*- coding: utf-8 -*-

import pandas as pd

# Get proteins that dimerize (mm_output["pairwise_2mers_df_F3"]):
def get_proteins_that_homodimerize(pairwise_2mers_df_F3: pd.DataFrame):
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
            
    return homodim_prots


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
        logger.warning(f"No homooligomerization states for {protein}")
        return [False]
    biggest_homo_N_mer  = len(models[-1])

    # try:
    #     biggest_homo_N_mer  = len(models[-1])
    # # In the case that not even the homo-3-mer is present, this will rise an exception (models == [])
    # except IndexError():
    #     logger.warning(f"No homooligomerization states for {protein}")
    #     return [False]
    
    # Progress
    logger.info(f'Verifying homo-N-mers for {models[0][0]} (3-mers or bigger)')
    
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


def find_homooligomerization_breaks(pairwise_2mers_df_F3, pairwise_Nmers_df,
                                    logger,
                                    min_PAE_cutoff_Nmers,
                                    pDockQ_cutoff_Nmers,
                                    N_models_cutoff):
    
    # Proteins that homodimerize
    homodim_prots: set = get_proteins_that_homodimerize(pairwise_2mers_df_F3)
    
    # Initialize dict to store homooligomerization states of each protein
    homooligomerization_states: dict = {protein_ID: {"is_ok": [], "N_states": []} for protein_ID in homodim_prots}
    
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
        
        is_ok: list[bool] = check_if_skipped_homo_N_mers(homo_Nmers_models, protein, logger)
        
        # If any state was skipped
        if any(not ok for ok in is_ok):
            
            # State that there is a problem with the models filling N_states with Nones
            homooligomerization_states[protein]["is_ok"] = is_ok
            homooligomerization_states[protein]["N_states"] = [None] * len(is_ok)
            
            # Skip to the next protein
            continue
        
        # If everything is OK, continue the computation
        else:
            
            # Add information to dict
            homooligomerization_states[protein]["is_ok"] = is_ok
            
            # Progress
            logger.info(f'   - DONE: {len(is_ok)} homooligomerization state(s) found')
        
        # For each homooligomerization state (starting with homo-3-mers)
        for N_state, model in enumerate(sorted(list(homo_Nmers_models), key = len), start = 3):
            
            # Separate only data for the current homooligomerization state and add chain info
            model_pairwise_df = protein_homo_N_mers_pairwise_df.query('proteins_in_model == @model')
            add_chain_information_to_df(model_pairwise_df)
            
            # Make the verification
            all_have_at_least_one_interactor: bool = does_all_have_at_least_one_interactor(
                                                        model_pairwise_df,
                                                        min_PAE_cutoff_Nmers,
                                                        pDockQ_cutoff_Nmers,
                                                        N_models_cutoff)
            
            # Add if it surpass cutoff to N_states
            homooligomerization_states[protein]["N_states"].append(all_have_at_least_one_interactor)
        
        # For proteins that the last state computed (N) is positive, add the suggestion to compute N+1
        if all(ok for ok in is_ok) and (homooligomerization_states[protein]["N_states"][-1] == True):

            homooligomerization_states[protein]["is_ok"]   .append(False)
            homooligomerization_states[protein]["N_states"].append(None)
    
    return homooligomerization_states
            
            
        