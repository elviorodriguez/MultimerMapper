
import os
import itertools
import numpy as np
import pandas as pd
import igraph
import json
from logging import Logger
from typing import Set, Tuple
from copy import deepcopy
from collections import Counter
from typing import List, Tuple, Dict, Any

from utils.logger_setup import configure_logger, default_error_msgs
from src.stoich.stoich_space_exploration import generate_stoichiometric_space_graph


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
                for combo in itertools.combinations(set(neighbor_names), N - 2):
                    if len(set(combo)) == N - 2:  # Ensure we're not adding the same neighbor twice
                        result_combinations.add(base_combo + combo)

        else:
            # For regular edges, include the edge nodes and one or more neighbors
            base_combo = (source_name, target_name)
            for N in range(min_N, max_N + 1):
                for combo in itertools.combinations(set(neighbor_names), N - 2):
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


def get_expanded_3mer_suggestions(combined_graph, pairwise_Nmers_df):

    are_interactions: list[str] = ["Static", "Strong Positive", "Positive", "Weak Positive", "Weak Negative", "Negative", "Strong Negative", "No N-mers Data"]

    # Get pairs that engage PPIs
    ppi_pairs: list[igraph.Edge] = [e for e in combined_graph.es if e['dynamics'] in are_interactions]

    # Discard homooligomeric edges
    hetero_ppi_pairs: list[igraph.Edge] = [e for e in ppi_pairs if e.source != e.target]

    # Discard multivalent edges
    monovalent_pairs: list[tuple] = set([e['name'] for e in hetero_ppi_pairs])
    for e in hetero_ppi_pairs:
        if e['valency']['cluster_n'] > 0 and e['name'] in monovalent_pairs:
            monovalent_pairs.discard(e['name'])
    
    # Generate expanded hetero3mer combinations for remaining edges
    expanded_3mer_suggestions: list[tuple] = []
    for pair in monovalent_pairs:

        # Generate both expanded 3mers
        p2_q1: tuple = tuple(sorted((pair[0], pair[0], pair[1])))
        p1_q2: tuple = tuple(sorted((pair[0], pair[1], pair[1])))

        # Append data to suggestions
        expanded_3mer_suggestions.append(p2_q1)
        expanded_3mer_suggestions.append(p1_q2)

    # Remove those that the user already computed
    user_Nmer_combinations = get_user_Nmers_combinations(pairwise_Nmers_df)
    expanded_3mer_suggestions = [ pair for pair in expanded_3mer_suggestions if pair not in user_Nmer_combinations ]
    
    return expanded_3mer_suggestions

# -----------------------------------------------------------------------------
# ------------------- To suggest multivalent combinations ---------------------
# -----------------------------------------------------------------------------

def generate_multivalent_combinations(graph, multivalency_states):
    """
    Generates suggestions for possible combinations of multivalent proteins.

    Args:
        graph (igraph.Graph): Combined graph with the multivalent protein interactions.
        max_stoichiometry (int): The maximum stoichiometry to consider.

    Returns:
        list: A list of suggested multivalent protein combinations.
    """
    suggestions = []

    # Get the multivalent protein pairs
    multivalent_pairs = get_multivalent_pairs(graph)

    # Generate combinations for each multivalent pair
    for pair in multivalent_pairs:
        pair_multivalency_states = multivalency_states[tuple(sorted(pair))]
        pair_combinations = generate_multivalent_pair_suggestions(pair, pair_multivalency_states)
        suggestions.extend(pair_combinations)

    # Remove monomeric suggestions and duplicates
    suggestions = list(set([tuple(sorted(s)) for s in suggestions if len(set(s)) != 1]))

    return suggestions

def get_multivalent_pairs(graph):
    """
    Retrieves the multivalent protein pairs from the input graph.

    Args:
        graph (igraph.Graph): The graph representing the multivalent protein interactions.

    Returns:
        list: A list of tuples representing the multivalent protein pairs.
    """
    multivalent_pairs = []
    for edge in graph.es:
        if edge['multivalency_states'] is not None:
            multivalent_pairs.append((edge.source_vertex['name'], edge.target_vertex['name']))

    # Remove duplicates
    multivalent_pairs = list(set([ tuple(sorted(pair)) for pair in multivalent_pairs ]))

    return multivalent_pairs


def get_max_continuous_mer_size(pair_multivalency_states: dict) -> int:
    """
    Finds the maximum continuous mer size in the configurations.
    Only considers consecutive mer sizes.
    """
    mer_sizes = sorted(set(len(config) for config in pair_multivalency_states.keys()))
    max_continuous = min(mer_sizes)

    for i, mer in enumerate(mer_sizes, start= max_continuous):
        if i == mer:
            max_continuous = i
        else:
            break
            
    return max_continuous


def generate_multivalent_pair_suggestions(pair, pair_multivalency_states):
    """
    Generates suggestions for possible combinations of a multivalent protein pair.

    Args:
        pair (tuple): A tuple of protein IDs representing a multivalent pair.
        max_stoichiometry (int): The maximum stoichiometry to consider.

    Returns:
        list: A list of suggested multivalent protein combinations for the given pair.
    """

    suggestions = []
    max_continuous_size = get_max_continuous_mer_size(pair_multivalency_states)

    for state in pair_multivalency_states.keys():

        if pair_multivalency_states[state] and len(state) <= max_continuous_size:

            # Generate two children (one +P and the other +Q)
            sug_p = tuple(sorted(state + (pair[0],)))
            sug_q = tuple(sorted(state + (pair[1],)))

            # Add them to suggestions
            if sug_p not in pair_multivalency_states.keys() and sug_p not in suggestions:
                suggestions.append(sug_p)
            if sug_q not in pair_multivalency_states.keys() and sug_q not in suggestions:
                suggestions.append(sug_q)

    # Remove potential repetitions
    suggestions = [ s for s in suggestions if s not in [ tuple(sorted(k)) for k in pair_multivalency_states.keys() ] ] 

    return list(set(suggestions))


# -----------------------------------------------------------------------------
# ------------------- To generate files with suggestions ----------------------
# -----------------------------------------------------------------------------


def generate_af3_json_jobs(suggested_combinations: List[Tuple[str]], 
                          prot_IDs: List[str], 
                          prot_names: List[str], 
                          prot_seqs: List[str], 
                          use_names: bool = False,
                          base_job_name: str = "Multimer_Job") -> List[Dict[str, Any]]:
    """
    Generate AlphaFold Server compatible JSON job descriptions for protein combinations.
    
    Args:
        suggested_combinations: List of protein ID/name combinations
        prot_IDs: List of protein IDs
        prot_names: List of protein names  
        prot_seqs: List of protein sequences
        use_names: Whether to use protein names instead of IDs in job names
        base_job_name: Base name for the jobs
        
    Returns:
        List of job dictionaries compatible with AlphaFold Server
    """
    
    af3_jobs = []
    
    for i, combination in enumerate(suggested_combinations):
        # Create job name
        if use_names:
            # Convert IDs to names for job naming
            combo_names = [prot_names[prot_IDs.index(prot_id)] for prot_id in combination]
            job_name = f"{'_vs_'.join(combo_names)}"
        else:
            job_name = f"{'_vs_'.join(combination)}"
        
        # Build sequences list for this job
        sequences = []
        
        # Count occurrences of each protein in the combination
        from collections import Counter
        protein_counts = Counter(combination)
        
        # Add each unique protein with its count
        for protein_id in protein_counts:
            protein_index = prot_IDs.index(protein_id)
            protein_sequence = prot_seqs[protein_index]
            count = protein_counts[protein_id]
            
            sequences.append({
                "proteinChain": {
                    "sequence": protein_sequence,
                    "count": count
                }
            })
        
        # Create job dictionary
        job = {
            "name": job_name,
            "modelSeeds": [],  # Empty list for automated random seed assignment
            "sequences": sequences,
            "dialect": "alphafoldserver",
            "version": 1
        }
        
        af3_jobs.append(job)
    
    return af3_jobs

def save_af3_json_files(af3_jobs: List[Dict[str, Any]], 
                       combination_suggestions_path: str,
                       single_file: bool = True,
                       max_jobs_per_file: int = 100,
                       use_names: bool = False) -> None:
    """
    Save AlphaFold Server compatible JSON files.
    
    Args:
        af3_jobs: List of job dictionaries
        combination_suggestions_path: Path to save the JSON files
        single_file: If True, save all jobs in one file. If False, create separate files.
        max_jobs_per_file: Maximum number of jobs per file when single_file=False
    """
    
    os.makedirs(combination_suggestions_path, exist_ok=True)
    
    prefix = "ids"
    if use_names:
        prefix = "names"

    if single_file:
        # Save all jobs in a single file
        json_file_path = os.path.join(combination_suggestions_path, f"{prefix}_af3_jobs_all.json")
        with open(json_file_path, 'w') as f:
            json.dump(af3_jobs, f, indent=2)
        # print(f"Saved {len(af3_jobs)} jobs to {json_file_path}")
    
    else:
        # Save jobs in separate files or batches
        if max_jobs_per_file >= len(af3_jobs):
            # Create individual files for each job
            for i, job in enumerate(af3_jobs):
                json_file_path = os.path.join(combination_suggestions_path, f"{prefix}_af3_job_{i+1:03d}.json")
                with open(json_file_path, 'w') as f:
                    json.dump([job], f, indent=2)  # AF3 expects a list even for single jobs
            # print(f"Saved {len(af3_jobs)} individual job files")
        
        else:
            # Create batch files
            for i in range(0, len(af3_jobs), max_jobs_per_file):
                batch = af3_jobs[i:i + max_jobs_per_file]
                batch_num = i // max_jobs_per_file + 1
                json_file_path = os.path.join(combination_suggestions_path, f"{prefix}_af3_jobs_batch_{batch_num:03d}.json")
                with open(json_file_path, 'w') as f:
                    json.dump(batch, f, indent=2)
            # print(f"Saved {len(af3_jobs)} jobs in {(len(af3_jobs) + max_jobs_per_file - 1) // max_jobs_per_file} batch files")


# When no 2-mers and no N-mers are passed
def initialize_multimer_mapper(fasta_file, out_path, use_names, logger):

    from src.input_check import seq_input_from_fasta
    
    # Parse FASTA file
    prot_IDs, prot_names, prot_seqs, prot_lens, prot_N = seq_input_from_fasta(fasta_file_path = fasta_file,
                                                                              use_names = use_names,
                                                                              logger = logger)
    
    # Generate 2mers combinations
    suggested_combinations = find_all_possible_2mers_combinations(prot_IDs = prot_IDs)

    # Save the suggestions
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

        # Generate and save AF3 JSON files
        for b in [True, False]:
            af3_jobs = generate_af3_json_jobs(
                suggested_combinations=suggested_combinations,
                prot_IDs=prot_IDs,
                prot_names=prot_names,
                prot_seqs=prot_seqs,
                use_names=b,
                base_job_name=""
            )
            
            save_af3_json_files(
                af3_jobs=af3_jobs,
                combination_suggestions_path=combination_suggestions_path,
                single_file=True,  # Change to False if you want individual files
                use_names=b
            )

    return suggested_combinations
    
# When at least 2-mers or N-mers where passed
def suggest_combinations(mm_output: dict, out_path: str = None, min_N: int = 3, max_N: int = 4, log_level: str = "info"):

    logger = configure_logger(out_path = mm_output['out_path'], log_level = log_level)(__name__)

    # Unpack data
    combined_graph   : igraph.Graph = mm_output['combined_graph']
    prot_IDs         : list[str]    = mm_output['prot_IDs']
    prot_names       : list[str]    = mm_output['prot_names']
    prot_seqs        : list[str]    = mm_output['prot_seqs']
    pairwise_2mers_df: pd.DataFrame = mm_output['pairwise_2mers_df']
    pairwise_Nmers_df: pd.DataFrame = mm_output['pairwise_Nmers_df']
    multivalency_states: dict       = mm_output['multivalency_states']

    # Get edges classified as "No 2-mers Data" 
    list_of_untested_2mers: list[tuple[str]] = find_untested_2mers(prot_IDs, pairwise_2mers_df)
    list_of_untested_2mers = list(list_of_untested_2mers)

    # List homooligomeric edges
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

    # Get expanded 3mer combinations for monovalent edges (to increase multivalency detection sensitivity)
    list_of_expanded_3mer_suggestions: list[tuple[str]] = list(get_expanded_3mer_suggestions(combined_graph, pairwise_Nmers_df))

    # Get suggestions for multivalent pairs
    list_of_multivalent_suggestions: list[tuple[str]] = generate_multivalent_combinations(combined_graph, multivalency_states)

    # Combine all suggested combinations and remove duplicates (if any)
    suggested_combinations: list[tuple[str]] = list_of_untested_2mers + list_of_homo_oligomeric_Nstates_plus_one + list_of_homo_oligomeric_Nstates_inconsistent + list_of_untested_Nmers + list_of_expanded_3mer_suggestions + list_of_multivalent_suggestions
    suggested_combinations: list[tuple[str]] = list(set(suggested_combinations))

    # Remove homooligomeric combinations that have reached a fallback
    fall_back_edges = []
    for e in combined_graph.es:
        try:
            if e['symmetry_fallback']['fallback_detected'] is True:
                 fall_back_edges.append(tuple(sorted(set(e['name']))))
        except KeyError:
            logger.warning(f'   Edge {e["name"]} does not have symmetry fallback')
            logger.warning(f'   This is an unexpected behavior...')
            continue
        except Exception as e:
            logger.error(f'   Unknown exception occurred when searching for edge {e["name"]} symmetry fallback')
            logger.error(default_error_msgs[0])
            logger.error(default_error_msgs[1])
            continue
    suggested_combinations = [ comb for comb in suggested_combinations if tuple(sorted(set(comb))) not in fall_back_edges ]

    # Remove already computed suggestions
    already_computed: list[tuple[str]] = list(get_user_2mers_combinations(pairwise_2mers_df)) + list(get_user_Nmers_combinations(pairwise_Nmers_df))
    suggested_combinations: list[tuple[str]] = [ sug for sug in suggested_combinations if sug not in already_computed ]

    # Explore the stoichiometric space and remove uninformative suggestions
    stoich_dict, stoich_graph, uninformative_suggestions = generate_stoichiometric_space_graph(mm_output, suggested_combinations)
    suggested_combinations: list[tuple[str]] = [ sug for sug in suggested_combinations if sug not in uninformative_suggestions ]

    # Save the suggestions
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

        for b in [True, False]:
            af3_jobs = generate_af3_json_jobs(
                suggested_combinations=suggested_combinations,
                prot_IDs=prot_IDs,
                prot_names=prot_names,
                prot_seqs=prot_seqs,
                use_names=b,  # Set to True if you want to use names in job titles
                base_job_name=""
            )
            
            save_af3_json_files(
                af3_jobs=af3_jobs,
                combination_suggestions_path=combination_suggestions_path,
                single_file=True,  # Change to False if you want individual files
                use_names = b
            )

    return suggested_combinations, stoich_dict, stoich_graph