
import os
import itertools
import numpy as np
import pandas as pd
import igraph
from logging import Logger
from itertools import combinations
from typing import Set, Tuple
from copy import deepcopy
from collections import Counter

from utils.logger_setup import configure_logger


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

# Get untested in 2-mer pairs
def get_untested_2mer_pairs(mm_output):

    # Unpack necessary data
    prot_IDs          = mm_output['prot_IDs']
    pairwise_2mers_df = mm_output['pairwise_2mers_df']

    # Get untested pairs
    untested_2mers_edges_tuples = sorted(list(find_untested_2mers(prot_IDs = prot_IDs, pairwise_2mers_df = pairwise_2mers_df)), key=lambda x: x[0])

    return untested_2mers_edges_tuples
    

# -----------------------------------------------------------------------------
# ------------------ To catch lacking N-mers predictions ----------------------
# ------------------          or suggest them            ----------------------
# -----------------------------------------------------------------------------


def compute_node_combinations(
    graph: igraph.Graph,
    min_N: int = 3,
    max_N: int = 4,
    remove_interactions: Tuple[str, ...] = ("Indirect", "No 2-mers Data"),
    logger: Logger | None = None
) -> Set[Tuple[str, ...]]:
    
    if logger is None:
        logger = configure_logger()(__name__)
    
    # Deep copy the graph to avoid affecting the original
    graph = deepcopy(graph)
    
    # Remove edges with specified dynamics
    graph.delete_edges(
        graph.es.select(lambda e: e["dynamics"] in remove_interactions)
    )
    
    if min_N < 3:
        logger.warn(f'min_N (={min_N}) cannot be set below 3. Setting min_N = 3.')
        min_N = 3
    if min_N > max_N:
        logger.warn(f'min_N (={min_N}) cannot be bigger than max_N (={max_N}). Setting min_N = max_N.')
        min_N = max_N
    
    result_combinations = set()
    
    for edge in graph.es:
        source, target = edge.tuple
        
        # Get names of the source and target nodes
        source_name = graph.vs[source]["name"]
        target_name = graph.vs[target]["name"]
        
        # Get neighbors of both source and target nodes
        neighbors = set(graph.neighbors(source) + graph.neighbors(target))
        neighbor_names = [graph.vs[v]["name"] for v in neighbors]
        
        if source_name == target_name:
            # For self-edges, include combinations with the node repeated twice plus one or more neighbors
            base_combo = (source_name, source_name)
            for N in range(min_N, max_N + 1):
                for combo in combinations(set(neighbor_names), N - 2):
                    if len(set(combo)) == N - 2:  # Ensure we're not adding the same neighbor twice
                        result_combinations.add(base_combo + combo)

        else:
            # For regular edges, include the edge nodes and one or more neighbors
            base_combo = (source_name, target_name)
            for N in range(min_N, max_N + 1):
                for combo in combinations(set(neighbor_names), N - 2):
                    if all(n not in base_combo for n in combo):  # Ensure we're not adding source or target again
                        result_combinations.add(tuple(sorted(base_combo + combo)))
    
    # Remove potential duplicates
    result_combinations = set(sorted(set([tuple(sorted(c)) for c in result_combinations])))
    
    # Remove combinations > 3 with proteins repeated more than twice
    filtered_combinations = set()
    for combo in result_combinations:
        if len(combo) == 3:
            filtered_combinations.add(combo)
        else:
            protein_counts = Counter(combo)
            if all(count <= 2 for count in protein_counts.values()):
                filtered_combinations.add(combo)
    
    return filtered_combinations

# # Example usage:
# graph = igraph.Graph.Full(3)
# graph.vs["name"] = ["EAF6", "EPL1", "PHD1"]
# graph.add_edge("EAF6", "EAF6")  # Add the self-edge
# # Add some more edges
# graph.add_vertex("AAA")
# graph.add_vertex("BBB")
# graph.add_vertex("CCC")
# graph.add_edge("EPL1", "AAA")
# graph.add_edge("PHD1", "BBB")
# graph.add_edge("BBB", "CCC")
# graph.es["dynamics"] = "Direct"  # Assume all edges are direct for this example
# result = compute_node_combinations(graph, min_N=5, max_N=5)
# igraph.plot(graph, vertex_label = graph.vs["name"], vertex_size = 40, bbox=(0, 0, 300, 300))
# print(result)


def get_user_Nmers_combinations(pairwise_Nmers_df: pd.DataFrame):

    tested_Nmers: set[tuple] = set(sorted([ tuple(sorted(m)) for m in pairwise_Nmers_df['proteins_in_model'] ]))

    return tested_Nmers

def find_untested_Nmers(combined_graph: igraph.Graph, pairwise_Nmers_df: pd.DataFrame,
                        min_N: int = 3, max_N: int = 4, logger: Logger | None = None) -> list[tuple[str]]:
    
    if logger is None:
        logger = configure_logger()(__name__)
    
    untested = []
    
    all_possible_comb = compute_node_combinations(combined_graph, min_N = min_N, max_N = max_N, logger = logger)
    user_comb         = get_user_Nmers_combinations(pairwise_Nmers_df)
    
    for possible_comb in all_possible_comb:
        if possible_comb not in user_comb:
            untested.append(possible_comb)
    
    return set(sorted(untested))


def get_tested_Nmer_pairs(mm_output):

    # Unpack necessary data
    pairwise_Nmers_df = mm_output['pairwise_Nmers_df']

    tested_Nmers_edges_df = pd.DataFrame(np.sort(pairwise_Nmers_df[['protein1', 'protein2']], axis=1),
                 columns=['protein1', 'protein2']).drop_duplicates().reset_index(drop = True)
    tested_Nmers_edges_tuples   = [tuple(sorted(tuple(row))) for i, row in tested_Nmers_edges_df.iterrows()]

    return tested_Nmers_edges_tuples
    


# -----------------------------------------------------------------------------
# ------------------- To generate files with suggestions ----------------------
# -----------------------------------------------------------------------------

def suggest_combinations(mm_output: dict, out_path: str = None, min_N: int = 3, max_N: int = 4, log_level: str = "info"):

    logger = configure_logger(out_path = mm_output['out_path'], log_level = log_level)(__name__)

    # Unpack data
    combined_graph   : igraph.Graph = mm_output['combined_graph']
    prot_IDs         : list[str]    = mm_output['prot_IDs']
    prot_names       : list[str]    = mm_output['prot_names']
    prot_seqs        : list[str]    = mm_output['prot_seqs']
    pairwise_2mers_df: pd.DataFrame = mm_output['pairwise_2mers_df']
    pairwise_Nmers_df: pd.DataFrame = mm_output['pairwise_Nmers_df']


    # Get edges classified as "No 2-mers Data" 
    list_of_untested_2mers: list[tuple[str]] = find_untested_2mers(prot_IDs, pairwise_2mers_df)
    list_of_untested_2mers = list(list_of_untested_2mers)

    # List homooligomeric edges
    # homo_oligomeric_edges: list[igraph.Edge] = [ e for e in combined_graph.es if e['homooligomerization_states'] is not None]
    homo_oligomeric_edges: list[igraph.Edge] = [
        e for e in combined_graph.es
        if e['homooligomerization_states'] is not None
        and e['homooligomerization_states'].get('error') is None
    ]

    # Get homooligomeric last N_state for which the last computed homooligomeric state is positive and repeat it +1 times (it is necessary to increment the homo-N-mer)
    list_of_homo_oligomeric_Nstates_plus_one:  list[tuple[str]] = [ (e["name"][0],) * (2 + len(e['homooligomerization_states']['N_states']) + 1) for e in homo_oligomeric_edges if e['homooligomerization_states']['N_states'][-1] == True ]
    list_of_homo_oligomeric_Nstates_plus_one = list(list_of_homo_oligomeric_Nstates_plus_one)

    # Get homooligomeric edges for which there is problems (it is necessary to compute intermediate homo-N-mer or the homo-3-mer)
    homo_oligomeric_edges_inconsistent: list[igraph.Edge] = [ e for e in homo_oligomeric_edges if False in e['homooligomerization_states']['is_ok'] ]
    list_of_homo_oligomeric_Nstates_inconsistent:  list[tuple[str]] = [ (e["name"][0],) * (2 + e['homooligomerization_states']['is_ok'].index(False) + 1 ) for e in homo_oligomeric_edges_inconsistent ]
    list_of_homo_oligomeric_Nstates_inconsistent = list(list_of_homo_oligomeric_Nstates_inconsistent)

    # Get all possible combinations > 2, and remove those that the user have already computed
    list_of_untested_Nmers: list[tuple[str]] = find_untested_Nmers(combined_graph, pairwise_Nmers_df,
                                                                   min_N = min_N, max_N = max_N,
                                                                   logger = logger)
    list_of_untested_Nmers = list(list_of_untested_Nmers)
    

    # Combine all suggested combinations and remove duplicates (if any)
    suggested_combinations: list[tuple[str]] = list_of_untested_2mers + list_of_homo_oligomeric_Nstates_plus_one + list_of_homo_oligomeric_Nstates_inconsistent + list_of_untested_Nmers
    suggested_combinations: list[tuple[str]] = list(set(suggested_combinations))
    
    if out_path is not None:

        combination_suggestions_path = os.path.join(out_path, "combinations_suggestions")
        os.makedirs(combination_suggestions_path, exist_ok = True)

        fasta_names_save_path       = os.path.join(combination_suggestions_path, "combinations_suggestions_names.fasta")
        fasta_IDs_save_path         = os.path.join(combination_suggestions_path, "combinations_suggestions_IDs.fasta")
        TSV_names_list_save_path    = os.path.join(combination_suggestions_path, "sug_names_list.txt")
        TSV_IDs_list_save_path      = os.path.join(combination_suggestions_path, "sug_IDs_list.txt")
        csv_save_path               = os.path.join(combination_suggestions_path, "combinations_suggestions.csv")

        suggest_combinations_names          = [ [prot_names[prot_IDs.index(p)] for p in comb] for comb in suggested_combinations ]
        suggest_combinations_seqs           = [ [prot_seqs[prot_IDs.index(p)] for p in comb] for comb in suggested_combinations ]
        suggest_combinations_fasta_names    = [ '__vs__'.join(comb) for comb in suggest_combinations_names ]
        suggest_combinations_fasta_IDs      = [ '__vs__'.join(comb) for comb in suggested_combinations ]
        suggest_combinations_fasta_seqs     = [ ':'.join(comb) for comb in suggest_combinations_seqs ]
        suggest_combinations_txt_names      = [ '\t'.join(comb) for comb in suggest_combinations_names ]
        suggest_combinations_txt_IDs        = [ '\t'.join(comb) for comb in suggested_combinations ]

        with open(fasta_names_save_path, 'w') as fasta_names, open(fasta_IDs_save_path, 'w') as fasta_IDs:
            for names, IDs, seqs in zip(suggest_combinations_fasta_names, suggest_combinations_fasta_IDs, suggest_combinations_fasta_seqs):
                fasta_names.write(f'>{names}\n')
                fasta_names.write(f'{seqs}\n')
                fasta_IDs.write(f'>{IDs}\n')
                fasta_IDs.write(f'{seqs}\n')
        
        with open(TSV_names_list_save_path, 'w') as tsv_names, open(TSV_IDs_list_save_path, 'w') as tsv_IDs:
            for names, IDs in zip(suggest_combinations_txt_names, suggest_combinations_txt_IDs):
                tsv_names.write(f'#{names}\n')
                tsv_names.write(f'{IDs}\n')
                tsv_IDs.write(f'#{IDs}\n')
                tsv_IDs.write(f'{names}\n')

        with open(csv_save_path, 'w') as csv_file:
            csv_file.write(str(suggested_combinations))

    return suggested_combinations