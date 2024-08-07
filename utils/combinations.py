
import itertools
import pandas as pd
import igraph

# -----------------------------------------------------------------------------
# ------------------ To catch lacking 2-mers predictions ----------------------
# -----------------------------------------------------------------------------

def find_all_possible_2mers_combinations(prot_IDs: list[str]) -> set:
        
    combinations: list[tuple] = list(itertools.combinations_with_replacement(prot_IDs, 2))
    
    sorted_combinations: list[tuple] = [tuple(sorted(comb)) for comb in combinations]
    
    return set(sorted(sorted_combinations))
    


def get_user_2mers_combinations(pairwise_2mers_df: pd.DataFrame):
        
    combinations = []
    
    for _, model in pairwise_2mers_df.iterrows():
        
        protein1 = model["protein1"]
        protein2 = model["protein2"]
        tuple_pair = (protein1, protein2)
        sorted_tuple_pair = tuple(sorted(tuple_pair))
        
        if sorted_tuple_pair not in combinations:
            combinations.append(sorted_tuple_pair)
            
    sorted_combinations = set(sorted(combinations))
    
    return sorted_combinations



def find_untested_2mers(prot_IDs: list[str], pairwise_2mers_df: pd.DataFrame):
    
    untested = []
    
    all_possible_comb = find_all_possible_2mers_combinations(prot_IDs)
    user_comb         = get_user_2mers_combinations(pairwise_2mers_df)
    
    for possible_comb in all_possible_comb:
        if possible_comb not in user_comb:
            untested.append(possible_comb)
    
    return set(sorted(untested))
    

# -----------------------------------------------------------------------------
# ------------------ To catch lacking N-mers predictions ----------------------
# ------------------          or suggest them            ----------------------
# -----------------------------------------------------------------------------


def suggest_combinations(combined_graph: igraph.Graph):

    # Get edges classified as "No 2-mers Data"
    list_of_untested_2mers = []

    pass