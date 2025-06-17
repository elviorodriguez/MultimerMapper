import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any


def analyze_protein_interactions(mm_output: Dict[str, Any], N_contacts_cutoff = 5,
                                 contacts_cutoff_reduction_step = 1) -> pd.DataFrame:
    """
    Analyze protein-protein interactions from AlphaFold multimer predictions.
    
    This function processes the mm_output dictionary containing pairwise contact matrices
    and generates a comprehensive table counting interactions between protein chains.
    
    Parameters:
    -----------º
    mm_output : dict
        Dictionary containing pairwise contact matrices with the following structure:
        - 'pairwise_contact_matrices': dict with protein pairs as keys
        - Each pair contains sub-models with contact information including:
          - 'PAE': Predicted Aligned Error matrix
          - 'min_pLDDT': Minimum predicted Local Distance Difference Test
          - 'distance': Distance matrix
          - 'is_contact': Boolean contact matrix
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - 'protein': Protein entity name (e.g., 'RuvBL1', 'RuvBL2')
        - 'proteins_in_model': Sorted tuple of all proteins in the model
        - 'rank': Model rank (1-5)
        - 'chain': Chain identifier ('A', 'B', 'C', etc.)
        - One column per unique protein entity counting contacts
    
    Examples:
    ---------
    >>> df = analyze_protein_interactions(mm_output)
    >>> print(df.head())
    """
    
    # # For recursion scape
    # if N_contacts_cutoff == 0:
    #     return pd.DataFrame()
    
    # Get all protein pairs that have matrices
    pairs = list(mm_output['pairwise_contact_matrices'].keys())
    
    # Extract all unique protein entities from the pairs
    unique_proteins = set()
    for pair in pairs:
        unique_proteins.update(pair)
    unique_proteins = sorted(list(unique_proteins))
    
    print(f"Found {len(unique_proteins)} unique protein entities: {unique_proteins}")
    
    # Initialize list to store all interaction data
    interaction_data = []
    
    # Process each protein pair
    for pair in pairs:
        print(f"Processing pair: {pair}")
        
        # Get all sub-models for this pair
        if pair not in mm_output['pairwise_contact_matrices']:
            continue
            
        sub_models = list(mm_output['pairwise_contact_matrices'][pair].keys())
        
        # Process each sub-model
        for sub_model_key in sub_models:
            # Parse sub-model key: (proteins_tuple, chains_tuple, rank)
            proteins_in_model, chain_pair, rank = sub_model_key
            chain_a, chain_b = chain_pair
            
            # Get contact matrix for this sub-model
            contact_data = mm_output['pairwise_contact_matrices'][pair][sub_model_key]
            
            # Check if there's interaction (>= 3 contacts as per your criterion)
            is_interacting = np.sum(contact_data['is_contact']) >= N_contacts_cutoff
            
            if not is_interacting:
                continue  # Skip non-interacting pairs
            
            # Convert chain IDs to protein indices
            chain_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
            
            # Get protein names for each chain
            protein_a_idx = chain_to_index.get(chain_a, 0)
            protein_b_idx = chain_to_index.get(chain_b, 1)
            
            # Ensure indices are within bounds
            if protein_a_idx >= len(proteins_in_model) or protein_b_idx >= len(proteins_in_model):
                continue
                
            protein_a = proteins_in_model[protein_a_idx]
            protein_b = proteins_in_model[protein_b_idx]
            
            # Sort proteins_in_model tuple for consistency
            proteins_in_model_sorted = tuple(sorted(proteins_in_model))
            
            # Create entries for both chains involved in the interaction
            # Entry for chain A
            interaction_data.append({
                'protein': protein_a,
                'proteins_in_model': proteins_in_model_sorted,
                'rank': rank,
                'chain': chain_a,
                'interacting_with_protein': protein_b,
                'PAE_min': np.min(contact_data['PAE']),
                'PAE_mean': np.mean(contact_data['PAE']),
                'contact_count': np.sum(contact_data['is_contact'])
            })
            
            # Entry for chain B (if different from chain A)
            if chain_a != chain_b:
                interaction_data.append({
                    'protein': protein_b,
                    'proteins_in_model': proteins_in_model_sorted,
                    'rank': rank,
                    'chain': chain_b,
                    'interacting_with_protein': protein_a,
                    'PAE_min': np.min(contact_data['PAE']),
                    'PAE_mean': np.mean(contact_data['PAE']),
                    'contact_count': np.sum(contact_data['is_contact'])
                })
    
    # # This is when recursion is implemented
    # if not interaction_data:
    #     new_N_contacts_cutoff =  N_contacts_cutoff - contacts_cutoff_reduction_step
    #     print("Warning: No interactions found in the data")
    #     print(f"   - Reducing N_contacts_cutoff from {N_contacts_cutoff} to {new_N_contacts_cutoff}")
    #     return analyze_protein_interactions(mm_output, new_N_contacts_cutoff)
    
    if not interaction_data:
        print("Warning: No interactions found in the data")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df_interactions = pd.DataFrame(interaction_data)
    
    # Create the final aggregated DataFrame
    final_data = []
    
    # Group by protein, proteins_in_model, rank, and chain
    grouped = df_interactions.groupby(['protein', 'proteins_in_model', 'rank', 'chain'])
    
    for (protein, proteins_in_model, rank, chain), group in grouped:
        # Initialize count dictionary for this chain
        contact_counts = {prot: 0 for prot in unique_proteins}
        
        # Count interactions with each protein type
        for _, row in group.iterrows():
            interacting_protein = row['interacting_with_protein']
            contact_counts[interacting_protein] += 1
        
        # Create row for final DataFrame
        row_data = {
            'protein': protein,
            'proteins_in_model': proteins_in_model,
            'rank': rank,
            'chain': chain
        }
        
        # Add contact count columns
        for prot in unique_proteins:
            row_data[f'contacts_with_{prot}'] = contact_counts[prot]
        
        final_data.append(row_data)
    
    # Create final DataFrame
    result_df = pd.DataFrame(final_data)
    
    # Sort the DataFrame for better readability
    sort_columns = ['proteins_in_model', 'rank', 'protein', 'chain']
    result_df = result_df.sort_values(sort_columns).reset_index(drop=True)
    
    print(f"Generated interaction table with {len(result_df)} rows")
    print(f"Columns: {list(result_df.columns)}")
    
    return result_df


def print_interaction_summary(df: pd.DataFrame) -> None:
    """
    Print a summary of the interaction analysis results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame returned by analyze_protein_interactions()
    """
    if df.empty:
        print("No interaction data to summarize.")
        return
    
    print("\n" + "="*50)
    print("PROTEIN INTERACTION ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"Total number of chain entries: {len(df)}")
    print(f"Unique protein combinations: {df['proteins_in_model'].nunique()}")
    print(f"Unique proteins: {df['protein'].nunique()}")
    print(f"Model ranks analyzed: {sorted(df['rank'].unique())}")
    
    print("\nProtein combinations found:")
    for combo in df['proteins_in_model'].unique():
        count = len(df[df['proteins_in_model'] == combo])
        print(f"  {combo}: {count} chain entries")
    
    # Find contact columns
    contact_columns = [col for col in df.columns if col.startswith('contacts_with_')]
    
    if contact_columns:
        print("\nTotal contacts per protein type:")
        for col in contact_columns:
            protein_name = col.replace('contacts_with_', '')
            total_contacts = df[col].sum()
            print(f"  {protein_name}: {total_contacts} total contacts")
    
    print("\n" + "="*50)


# df = analyze_protein_interactions(mm_output)
# print_interaction_summary(df)

# df.head()
