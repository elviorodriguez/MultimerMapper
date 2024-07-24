
###############################################################################
# --------------------------------------------------------------------------- #
# ------------- Filter out non-interacting pairs using cutoffs -------------- #
# --------------------------------------------------------------------------- #
# -------------------------------- (2-mers) --------------------------------- #
# ---------- (ipTM < cutoff, min_pAE > cutoff, N_models > cutoff) ----------- #
# --------------------------------------------------------------------------- #
# -------------------------------- (N-mers) --------------------------------- #
# --------- (pDockQ < cutoff, min_pAE > cutoff, N_models > cutoff) ---------- #
# --------------------------------------------------------------------------- #
###############################################################################

import pandas as pd

# -----------------------------------------------------------------------------
# Get PPI from 2-mers dataset -------------------------------------------------
# -----------------------------------------------------------------------------

def filter_non_int_2mers_df(pairwise_2mers_df: pd.DataFrame,
                            min_PAE_cutoff: float | int = 4.5,
                            ipTM_cutoff: float | int = 0.4,
                            N_models_cutoff: int = 3):
    '''
    This part searches for pairwise interactions inside each combination and
    filters out those combinations that do not have fully connected networks
    using igraph and AF2 metrics (ipTM and PAE).
    ''' 
    
    # Pre-process pairwise interactions
    pairwise_2mers_df_F1 = (pairwise_2mers_df[

        # Filter the DataFrame based on the ipTM and min_PAE
        (pairwise_2mers_df['min_PAE'] <= min_PAE_cutoff) &
        (pairwise_2mers_df['ipTM'] >= ipTM_cutoff)]

      # Group by pairwise interaction
      .groupby(['protein1', 'protein2'])

      # Compute the number of models for each pair that are kept after the filter
      .size().reset_index(name='N_models')

      # Remove pairs with less that N_models_cutoff
      .query('N_models >= @N_models_cutoff')

      # Drop the index
      .reset_index(drop=True)
      )
    
    # Extract best min_PAE and ipTM
    pairwise_2mers_df_F2 = (pairwise_2mers_df
                            
      # Group by pairwise interaction
      .groupby(['protein1', 'protein2'])

      # Extract min_PAE
      .agg({'ipTM': 'max',
            'min_PAE': 'min'}
           )
      )
    
    # Merge both dfs
    pairwise_2mers_df_F3 = (
        pairwise_2mers_df_F1
        .merge(pairwise_2mers_df_F2, on=["protein1", "protein2"])
    )
    
    pairwise_2mers_df_F3 
    
    # unique_proteins
    unique_proteins = list(set(list(pairwise_2mers_df_F3.protein1) + list(pairwise_2mers_df_F3.protein2)))
    len(unique_proteins)
    
    return pairwise_2mers_df_F3, unique_proteins


# -----------------------------------------------------------------------------
# Get PPI from N-mers dataset -------------------------------------------------
# -----------------------------------------------------------------------------

def filter_non_int_Nmers_df(pairwise_Nmers_df: pd.DataFrame,
                            min_PAE_cutoff_Nmers: float | int = 4.5,
                            # As ipTM lose sense in N-mers data, we us pDockQ values instead
                            pDockQ_cutoff_Nmers: float | int = 0.15,
                            N_models_cutoff: int = 3,
                            is_debug: bool = False):
    
    # Pre-process N-mers pairwise interactions to count how many models surpass cutoff
    pairwise_Nmers_df_F1 = (pairwise_Nmers_df
                            
        # Unify the values on pDockQ and min_PAE the N-mer models with homooligomers
        .groupby(["protein1", "protein2", "proteins_in_model", "rank"])
        .agg({
            'min_PAE': 'min',   # keep only the minimum value of min_PAE
            'pDockQ': 'max'     # keep only the maximum value of pDockQ
        })
        .reset_index())
    
    pairwise_Nmers_df_F1 = (pairwise_Nmers_df_F1[

        # Filter the DataFrame based on the ipTM and min_PAE
        (pairwise_Nmers_df_F1['min_PAE'] <= min_PAE_cutoff_Nmers) &
        (pairwise_Nmers_df_F1['pDockQ'] >= pDockQ_cutoff_Nmers)]

        # Group by pairwise interaction of each N-mer model
        .groupby(['protein1', 'protein2', 'proteins_in_model'])

        # Compute the number of models for each pair that are kept after the filter
        .size().reset_index(name='N_models')

        # Remove pairs with less that N_models_cutoff
        .query('N_models >= @N_models_cutoff')
        .reset_index(drop=True)
        )
        
    if is_debug: print("\n","------------------ F1:\n", pairwise_Nmers_df_F1)
    
    # Extract best min_PAE and ipTM
    pairwise_Nmers_df_F2 = (pairwise_Nmers_df
                            
        # Group by pairwise interaction
        .groupby(['protein1', 'protein2', 'proteins_in_model'])

        # Extract min_PAE
        .agg({'min_PAE': 'min',
              'pDockQ': 'max'}
             )
        )
    
    if is_debug: print("\n","------------------ F2:\n", pairwise_Nmers_df_F2)
    
    # Merge both dfs
    pairwise_Nmers_df_F3 = (
        pairwise_Nmers_df_F1
        .merge(pairwise_Nmers_df_F2, on=["protein1", "protein2", "proteins_in_model"])
    )
    
    if is_debug: print("\n","------------------ F3:\n", pairwise_Nmers_df_F3.head(20))
    
    # unique_proteins
    unique_Nmers_proteins = list(set(list(pairwise_Nmers_df_F3.protein1) + list(pairwise_Nmers_df_F3.protein2)))
    
    return pairwise_Nmers_df_F3, unique_Nmers_proteins
