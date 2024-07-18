# To match JSON files with PDB files
from difflib import SequenceMatcher

# To select pdbs that match the .json filename      
def find_most_similar(query_string, string_list):
    '''
    Find the most similar string in a list based on token-based similarity.

    Parameters:
    - query_string (str): The query string to find a match for.
    - string_list (list of str): A list of strings to compare against the query string.

    Returns:
    str: The most similar string in the provided list.

    Example:
    >>> file_list = [
    ...    'YNG2__vs__YNG2L_relaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb',
    ...    'YNG2__vs__YNG2L_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb',
    ...    'YNG2__vs__YNG2L_unrelaxed_rank_002_alphafold2_multimer_v3_model_3_seed_000.pdb',
    ...    'YNG2__vs__YNG2L_unrelaxed_rank_003_alphafold2_multimer_v3_model_5_seed_000.pdb',
    ...    'YNG2__vs__YNG2L_unrelaxed_rank_004_alphafold2_multimer_v3_model_2_seed_000.pdb',
    ...    'YNG2__vs__YNG2L_unrelaxed_rank_005_alphafold2_multimer_v3_model_4_seed_000.pdb'
    ... ]
    >>> query_file = 'YNG2__vs__YNG2L_scores_rank_005_alphafold2_multimer_v3_model_4_seed_000.json'
    >>> most_similar = find_most_similar(query_file, file_list)
    >>> print(f"The most similar file to '{query_file}' is: {most_similar}")
    
    NOTE: It always will prefer relaxed PDBs over unrelaxed (smaller difference).
    '''
    similarities = [SequenceMatcher(None, query_string, s).ratio() for s in string_list]
    most_similar_index = similarities.index(max(similarities))
    return string_list[most_similar_index]