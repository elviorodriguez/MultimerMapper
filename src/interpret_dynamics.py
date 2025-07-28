
import numpy as np
import pandas as pd
import igraph
import logging
import math
from scipy.stats import pointbiserialr, chi2_contingency

from cfg.default_settings import edge_default_weight, edge_scaling_factor, edge_min_weight, edge_max_weight, edge_midpoint_PAE, edge_weight_sigmoidal_sharpness, use_cluster_aware_Nmers_variation
from utils.logger_setup import configure_logger

# -------------------------------------------------------------------------------------
# --------------------------- Classification df import --------------------------------
# -------------------------------------------------------------------------------------

def read_classification_df(path: str = "cfg/interaction_classification.tsv"):

    try:
        from multimer_mapper import mm_path
    except ImportError:
        from __main__ import mm_path


    classification_df = pd.read_csv(mm_path + '/' + path, sep= "\t")
    return classification_df

classification_df = read_classification_df()


# -------------------------------------------------------------------------------------
# -------------------------------- Helper functions -----------------------------------
# -------------------------------------------------------------------------------------

def find_edge_by_vertex_attributes(graph: igraph.Graph, vertex_attr_name: str, value1, value2):
    """
    Find an edge in the graph based on the attribute values of its vertices.
    
    Parameters:
    - graph (igraph.Graph): The graph to search in.
    - vertex_attr_name (str): The name of the vertex attribute to check.
    - value1: The value of the attribute for the first vertex.
    - value2: The value of the attribute for the second vertex.
    
    Returns:
    - igraph.Edge or None: The edge if found, otherwise None.
    """
    # Find the vertex indices
    vertex1 = graph.vs.find(**{vertex_attr_name: value1}).index
    vertex2 = graph.vs.find(**{vertex_attr_name: value2}).index

    # Find the edge connecting these vertices
    edge_id = graph.get_eid(vertex1, vertex2, directed=False, error=False)
    if edge_id != -1:
        return graph.es[edge_id]
    else:
        return None

# Example usage
# edge = find_edge_by_vertex_attributes(g, 'name', 'A', 'C')

# -----------------------

# To parse intervals
def preprocess_intervals(df, interval_name: str):
    """_summary_

    Args:
        df (pd.Dataframe): _description_
        interval_name (str): Column name representing an interval: [p;q) for example

    Returns:
        interpretable_df: df that allows interval interpretation
    """ 

    # Split the interval string into components
    df[['left_bracket', 'lower', 'upper', 'right_bracket']] = df[interval_name].str.extract(r'([\[\(])([-\d.]+);([-\d.]+)([\]\)])')
    
    # Convert bounds to float
    df['lower'] = df['lower'].astype(float)
    df['upper'] = df['upper'].astype(float)
    
    # Create boolean columns for inclusive bounds
    df['lower_inclusive'] = df['left_bracket'] == '['
    df['upper_inclusive'] = df['right_bracket'] == ']'
    
    return df

# To search if a value is in an interval
def is_in_interval(x, lower, upper, lower_inclusive, upper_inclusive):
    """Returns True if x is in the interval [ lower ; upper ]

    Args:
        x (float): number to know if it is in an interval
        lower (float): lower value of the interval
        upper (float): upper value of the interval
        lower_inclusive (bool): include lower value? "(" is False and "[" is True
        upper_inclusive (bool): include upper value? "(" is False and "[" is True

    Returns:
        bool: True if belongs, False if not
    """    
    if lower_inclusive:
        lower_check = x >= lower
    else:
        lower_check = x > lower
    
    if upper_inclusive:
        upper_check = x <= upper
    else:
        upper_check = x < upper
    
    return lower_check and upper_check

# To get the rows that 
def find_rows_that_contains_interval(df, interval_name, lambda_val):
    """Returns filtered df with the rows that contains lambda_val in interval_name
    column.

    Args:
        df (pd.DataFrame): dataframe to filter
        interval_name (str): column with intervals
        lambda_val (float): value to search in interval_name intervals

    Returns:
        pd.DataFrame: filtered df with the rows that contains lambda_val in 
        interval_name column.
    """    


    df = preprocess_intervals(df, interval_name)
    return df[df.apply(lambda row: is_in_interval(lambda_val, 
                                                  row['lower'], 
                                                  row['upper'], 
                                                  row['lower_inclusive'], 
                                                  row['upper_inclusive']), axis=1)]

# -------------------------------------------------------------------------------------
# ------------------------------- Classify the edge -----------------------------------
# -------------------------------------------------------------------------------------

# debug logger
def log_debug(logger, tuple_edge, is_present_in_2mers, is_present_in_Nmers,
              was_tested_in_2mers, was_tested_in_Nmers,
              e_dynamics, true_edge,
              Nmers_variation = "Not Reached", Nmers_mean_pdockq = "Not Reached"):
    logger.debug(f"Found dynamic classification of edge: {tuple_edge}")
    logger.debug(f"   - is_present_in_2mers: {is_present_in_2mers}")
    logger.debug(f"   - is_present_in_Nmers: {is_present_in_Nmers}")
    logger.debug(f"   - was_tested_in_2mers: {was_tested_in_2mers}")
    logger.debug(f"   - was_tested_in_Nmers: {was_tested_in_Nmers}")
    logger.debug(f"   - Nmers_variation    : {Nmers_variation}")
    logger.debug(f"   - Nmers_mean_pdockq  : {Nmers_mean_pdockq}")
    logger.debug(f"   - Classification     : <--- {e_dynamics} --->")
    logger.debug(f"   - True edge          : {true_edge}")


def classify_edge_dynamics(tuple_edge: tuple,
                           
                           # Combined Graph
                           true_edge: igraph.Edge,

                           # Cutoff
                           N_models_cutoff: int,

                           sorted_edges_2mers_graph  : list[tuple], 
                           sorted_edges_Nmers_graph  : list[tuple],
                           untested_edges_tuples     : list[tuple],
                           tested_Nmers_edges_sorted : list[tuple],

                           classification_df: pd.DataFrame|None = classification_df,

                           logger = None
    ):
    
    # Read classification df if needed
    if classification_df is None:
        classification_df = read_classification_df()

    # Configure logger
    if logger == None:
        logger = configure_logger()(__name__)
    
    # Get parameters for classification
    valency_contains_2mers = any(mer_number == 2 for mer_number in [len(model[0]) for model in true_edge['valency']['models']])
    valency_contains_Nmers = any(mer_number  > 2 for mer_number in [len(model[0]) for model in true_edge['valency']['models']])
    is_present_in_2mers = tuple_edge in sorted_edges_2mers_graph or valency_contains_2mers
    is_present_in_Nmers = tuple_edge in sorted_edges_Nmers_graph or valency_contains_Nmers
    was_tested_in_2mers = tuple_edge not in untested_edges_tuples
    was_tested_in_Nmers = tuple_edge in tested_Nmers_edges_sorted

    # --------------------------------------- (1) ---------------------------------------

    # Classify rows untested in 2-mers
    e_dynamics_rows = (
        classification_df
        .query(f'Tested_in_2_mers == {was_tested_in_2mers}')
    )
    
    # If the info is enough to classify
    if e_dynamics_rows.shape[0] == 1:
        e_dynamics = str(e_dynamics_rows["Classification"].iloc[0])
        log_debug(logger, tuple_edge, is_present_in_2mers, is_present_in_Nmers, was_tested_in_2mers, was_tested_in_Nmers, e_dynamics, true_edge)
        return e_dynamics
    # This happens for indirectly interacting proteins (they have no contacts and both is_present_in_2/Nmers end up as false)
    elif e_dynamics_rows.shape[0] == 0:
        return "Indirect"
    
    # --------------------------------------- (2) ---------------------------------------

    # Classify the rest of possibilities
    e_dynamics_rows = (
        classification_df
        .query(f'AF_2mers == {is_present_in_2mers and valency_contains_2mers}')
        .query(f'AF_Nmers == {is_present_in_Nmers and valency_contains_Nmers}')
        .query(f'Tested_in_N_mers == {was_tested_in_Nmers}')
    )

    # If the info is enough to classify
    if e_dynamics_rows.shape[0] == 1:
        e_dynamics = str(e_dynamics_rows["Classification"].iloc[0])
        log_debug(logger, tuple_edge, is_present_in_2mers, is_present_in_Nmers, was_tested_in_2mers, was_tested_in_Nmers, e_dynamics, true_edge)
        return e_dynamics
    # This happens for indirectly interacting proteins (they have no contacts and both is_present_in_2/Nmers end up as false)
    elif e_dynamics_rows.shape[0] == 0:
        return "Indirect"
    
    # --------------------------------------- (3) ---------------------------------------

    # If not, get more info
    Nmers_variation = get_edge_Nmers_variation(edge = true_edge, N_models_cutoff = N_models_cutoff)
    
    # Classify using N_mers_variation
    e_dynamics_rows = find_rows_that_contains_interval(df = e_dynamics_rows,
                                                       interval_name = "N_mers_variation",
                                                       lambda_val = Nmers_variation)
    
    # If the info is enough to classify
    if e_dynamics_rows.shape[0] == 1:
        e_dynamics = str(e_dynamics_rows["Classification"].iloc[0])
        log_debug(logger, tuple_edge, is_present_in_2mers, is_present_in_Nmers, was_tested_in_2mers, was_tested_in_Nmers, e_dynamics, true_edge, Nmers_variation)
        return e_dynamics
    # This happens for indirectly interacting proteins (they have no contacts and both is_present_in_2/Nmers end up as false)
    elif e_dynamics_rows.shape[0] == 0:
        return "Indirect"
    
    # --------------------------------------- (4) ---------------------------------------
    
    # If not, get more info
    Nmers_mean_pdockq = get_edge_Nmers_pDockQ(edge = true_edge, N_models_cutoff = N_models_cutoff)

    # Classify using N_mers_variation
    e_dynamics_rows = find_rows_that_contains_interval(df = e_dynamics_rows,
                                                        interval_name = "N_mers_pDockQ",
                                                        lambda_val = Nmers_mean_pdockq)

    # Info must be enough to classify at this point
    if e_dynamics_rows.shape[0] == 1:
        e_dynamics = str(e_dynamics_rows["Classification"].iloc[0])
        return e_dynamics
    
    # --------------------------------------- (5) ---------------------------------------

    # If not, something went wrong
    else:
        logger.error(f"Something went wrong with dynamics classification of edge: {tuple_edge}")
        logger.error(f"  - Edge: {true_edge}")
        logger.error(f"  - Filtered classification_df (e_dynamics_rows:\n{e_dynamics_rows}")
        logger.error( "  - MultimerMapper will continue...")
        e_dynamics = "ERROR"
        logger.error(f"  - Edge dynamics will be classified as {e_dynamics}")
        logger.error( "  - Results may be unreliable or the program may crash later...")
        return e_dynamics


# -------------------------------------------------------------------------------------
# -------------------- Getters based on dynamic classification ------------------------
# -------------------------------------------------------------------------------------

def get_edge_Nmers_variation(edge, N_models_cutoff: int, use_cluster_aware_variation = use_cluster_aware_Nmers_variation):

    # ---------------- Get Full N-mers variation (at cluster level) ----------------

    if use_cluster_aware_variation and edge['valency']['is_multivalent']:

        total_models = len(edge["N_mers_data"]['cluster'])
        predictions_that_surpass_cutoffs = len([1 for i in edge["N_mers_data"]['cluster'] if "✔" in i])

        Nmers_variation = predictions_that_surpass_cutoffs / total_models
        
        return Nmers_variation
    
    # ------------------------- Get normal N-mers variation ------------------------

    total_models = len(list(edge["N_mers_data"]["N_models"]))
    predictions_that_surpass_cutoffs = sum(edge["N_mers_data"]["N_models"] >= N_models_cutoff)

    Nmers_variation = predictions_that_surpass_cutoffs / total_models

    return Nmers_variation

def get_edge_Nmers_pDockQ(edge, N_models_cutoff):
    '''
    Computes the mean pDockQ using only N-mers predictions that surpass cutoffs
    '''
    return np.mean(edge["N_mers_data"].query(f'N_models >= {N_models_cutoff}')["pDockQ"])

# Color
def get_edge_color_hex(graph_edge: igraph.Edge, classification_df: pd.DataFrame):

    edge_dynamics = graph_edge["dynamics"]
    edge_color_hex = classification_df.query(f'Classification == "{edge_dynamics}"')["Color_hex"].iloc[0]
    return edge_color_hex

# Linetype
def get_edge_linetype(graph_edge: igraph.Edge, classification_df: pd.DataFrame):
    
    edge_dynamics = graph_edge["dynamics"]
    edge_line_type = classification_df.query(f'Classification == "{edge_dynamics}"')["Line_type"].iloc[0]
    return edge_line_type

# Weight using sigmoidal scaling
def sigmoid_rescale(x, min_val=1, max_val=6, midpoint=1, sharpness=1):
    """Sigmoid function to scale x between min_val and max_val."""
    return min_val + (max_val - min_val) / (1 + np.exp(-sharpness * (x - midpoint)))

def get_edge_weight(
    graph_edge: igraph.Edge,
    classification_df: pd.DataFrame,
    default_edge_weight = edge_default_weight,
    scaling_factor = edge_scaling_factor,
    min_edge_weight = edge_min_weight,
    max_edge_weight = edge_max_weight,
    midpoint_PAE = edge_midpoint_PAE,
    sharpness = edge_weight_sigmoidal_sharpness
):
    
    # Compute the midpoint based on midpoint_PAE
    midpoint = 1 / midpoint_PAE * scaling_factor

    # Get the edge classification
    edge_dynamics = graph_edge["dynamics"]
    edge_width_is_variable = classification_df.query(
        f'Classification == "{edge_dynamics}"'
    )["Variable_Edge_width"].iloc[0]

    if edge_width_is_variable:
        
        # Compute raw edge weight
        min_pae_2mers = graph_edge["2_mers_data"]["min_PAE"].to_numpy()
        min_pae_Nmers = graph_edge["N_mers_data"]["min_PAE"].to_numpy()
        all_min_pae = np.concatenate([min_pae_2mers, min_pae_Nmers])
        edge_weight_PAE = 1 / np.mean(all_min_pae)
        raw_weight = edge_weight_PAE * scaling_factor

        # Apply sigmoid rescaling
        edge_weight = sigmoid_rescale(
            raw_weight,
            min_val=min_edge_weight,
            max_val=max_edge_weight,
            midpoint=midpoint,
            sharpness=sharpness
        )

        # # Debug
        # print("EDGE:", graph_edge['name'])
        # print("  - Weight:", edge_weight, "!!!!!!!!!!!!!!!!!!!")
        # print("  - graph_edge[2_mers_data][min_PAE]:", graph_edge["2_mers_data"]["min_PAE"])
        # print("  - graph_edge[N_mers_data][min_PAE]:", graph_edge["N_mers_data"]["min_PAE"])
        # print("  - edge_weight_PAE:", edge_weight_PAE)
        # print("  - raw_weight:", raw_weight)
        # print("  - Valency:", graph_edge['valency']['cluster_n'])

        return edge_weight

    else:

        # # Debug
        # print("EDGE:", graph_edge['name'])
        # print("  - Weight:", default_edge_weight)
        # print("  - Valency:", graph_edge['valency']['cluster_n'])

        return default_edge_weight

# Oscillation
def get_edge_oscillation(graph_edge: igraph.Edge, classification_df: pd.DataFrame):
    
    edge_dynamics = graph_edge["dynamics"]
    edge_line_type = classification_df.query(f'Classification == "{edge_dynamics}"')["Edge_oscillates"].iloc[0]
    return edge_line_type



# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -------------------------------------------------------------------------------------
# --------------- Associations between protein appearance and dynamics ----------------
# -------------------------------------------------------------------------------------


def compute_phi_coef_from_N_mers_data(
        N_mers_data: pd.DataFrame, pair: tuple[str], protein: str, dynamics: str, logger: logging.Logger
    ):
    
    # No protein affects the manifestation of the interaction
    if dynamics == "Static":
        return {"phi_coef": 0, "pval": 1.0}
    # All tested proteins in all combinations affect the manifestation of the interaction in a negative way
    elif dynamics == "Strong Negative":
        return {"phi_coef": -1, "pval": 0.0}
    # All tested proteins in all combinations affect the manifestation of the interaction in a positive way
    elif dynamics == "Strong Positive":
        return {"phi_coef": 1, "pval": 0.0}
    
    # For homo-interactions, the protein must be in the combination at least three times
    if protein == pair[0] and protein == pair[1]:
        count_threshold = 3
    # For hetero-interactions that contain the protein, it must be in the combination at least two times
    elif protein in pair:
        count_threshold = 2
    # Everything else, it must be present at least once
    else:
        count_threshold = 1
    
    # Counts outcomes
    n11=0
    n10=0
    n01=0
    n00=0
    
    # Assign each combination to an outcome
    for idx, n_mer_row in N_mers_data.iterrows():
        
        # ----- Does the protein is in the combination? True or False (X) -----
        is_protein_in_combination  = False
        if n_mer_row['proteins_in_model'].count(protein) >= count_threshold:
            is_protein_in_combination = True
        
        # ------------- Does the pair interact? True or False (Y) -------------
        if "✔" in n_mer_row['cluster']:
            is_ppi_detected=True
        elif "✗" in n_mer_row['cluster'] or "X" in n_mer_row['cluster'] or "x" in n_mer_row['cluster']:
            is_ppi_detected=False
        else:
            is_ppi_detected=False
            logger.error( "Something went wrong during the computation of Phi Coefficient:")
            logger.error( "   - None of the following characters ✔, ✗, X or x is in N_mers_data")
            logger.error(f"   - Pair: {pair}")
            logger.error(f"   - Protein: {protein}")
            logger.error(f"   - N_mers_data row: {n_mer_row}")
            logger.error( "MultimerMapper will continue, but the results may be unreliable or the program might crash later...")
        
        # --------------------- Classify the count ----------------------------
        if is_protein_in_combination and is_ppi_detected:
            n11+=1
        if is_protein_in_combination and not is_ppi_detected:
            n10+=1
        if not is_protein_in_combination and is_ppi_detected:
            n01+=1
        if not is_protein_in_combination and not is_ppi_detected:
            n00+=1
        
    # Compute marginal totals
    n1_=n11+n10
    n0_=n01+n00
    n_1=n11+n01
    n_0=n10+n00

    
    # Compute Phi Coefficient and p-value using chi2_contingency
    if n1_ == 0 or n0_ == 0 or n_1 == 0 or n_0 == 0:
        return {"phi_coef": float('nan'), "pval": float('nan')}
    else:
        # Create contingency table
        contingency_table = np.array([[n11, n10], [n01, n00]])
        
        # Compute chi-square test
        chi2, pval, dof, expected = chi2_contingency(contingency_table)
        
        # Compute phi coefficient manually (same as before)
        phi_coef = (n11*n00-n10*n01)/math.sqrt(n1_*n0_*n_1*n_0)
        
        return {"phi_coef": phi_coef, "pval": pval}
    

def add_phi_coefficients_to_combined_graph(combined_graph, logger: None | logging.Logger = None):
    
    # Configure logger
    if logger == None:
        logger = configure_logger()(__name__)

    for edge in combined_graph.es:
        
        # Unpack necessary data
        pair = edge['name']
        prot1 = pair[0]
        prot2 = pair[1]
        dynamics = edge['dynamics']
        n_mers_df = edge['N_mers_data']
        possible_proteins = set([prot for combination in n_mers_df["proteins_in_model"] for prot in combination])

        all_phi_scores = {}
        
        for protein in possible_proteins:            
            protein_phi_result = compute_phi_coef_from_N_mers_data(
                n_mers_df, pair, protein, dynamics, logger)
                        
            all_phi_scores[protein] = protein_phi_result
        
        # Add phi scores to the edge (PPI)
        edge['phi_coef'] = all_phi_scores

        # Add phi coefficients as ASCII plots
        phi_ascii_plot = plot_phi_coefficients_ascii(edge['phi_coef'])
        html_phi_ascii_plot = convert_ascii_plot_to_html(phi_ascii_plot)
        edge['phi_coef_ascii_plot'] = phi_ascii_plot
        edge['phi_coef_ascii_plot_html'] = html_phi_ascii_plot


# -------------------------------------------------------------------------------------
# -------------- Convert phi coefficient to ASCII plots HTML compatible ---------------
# -------------------------------------------------------------------------------------


def plot_phi_coefficients_ascii(phi_coef_dict, edge_index=None, title=None):
    """
    Create an ASCII bar graph for phi coefficients.
    
    Args:
        phi_coef_dict: Dictionary with protein names as keys and phi coefficient data as values
                      (can be either float values or dict with 'phi_coef' and 'pval' keys)
        edge_index: Optional edge index for labeling
        title: Optional title for the plot
    
    Returns:
        String containing the ASCII plot
    """
    # Filter out NaN values and convert to appropriate format
    filtered_data = {}
    for protein, coef_data in phi_coef_dict.items():
        # Handle both old format (float) and new format (dict)
        if isinstance(coef_data, dict):
            coef = coef_data.get('phi_coef')
            pval = coef_data.get('pval')
        else:
            # Backward compatibility with old float format
            coef = coef_data
            pval = None
            
        if not (isinstance(coef, float) and math.isnan(coef)) and coef is not None:
            filtered_data[protein] = {'phi_coef': float(coef), 'pval': pval}
    
    if not filtered_data:
        return f"Edge {edge_index}: No valid phi coefficients to plot\n"
    
    # Configuration
    resolution = 0.05  # 0.05 per character
    scale_width = int(2.0 / resolution)  # Total width for -1 to +1 range
    zero_pos = scale_width // 2  # Position of zero
    max_protein_name_len = max(len(name) for name in filtered_data.keys())
    
    # Characters for different bar intensities
    bar_chars = {
        'light': '░',
        'medium': '▒', 
        'heavy': '█',
        'zero': '│'
    }
    
    # Build the plot
    lines = []
    
    # Add title
    if title:
        lines.append(title)
    elif edge_index is not None:
        lines.append(f"Edge {edge_index} - Phi Coefficients")
    else:
        lines.append("Phi Coefficients")
    lines.append("")
    
    # Add numerical scale: place "-1" at left, "0" at center, "1" at right
    prefix = " " * (max_protein_name_len + 1)
    nums = [" "] * (scale_width + 1)

    # left end "-1"
    nums[0] = "-"
    nums[1] = "1"

    # zero "0"
    nums[zero_pos + 1] = "0"

    # right end "1"
    nums[-1] = "1"

    lines.append(prefix + "".join(nums))
    
    # Create scale header with p-value reference
    scale_line = " " * (max_protein_name_len + 2)
    for i in range(scale_width):
        value = -1.0 + (i * resolution)
        if abs(value) < 0.025:  # Close to zero
            scale_line += "│"
        elif abs(value - (-1.0)) < 0.025:  # Close to -1
            scale_line += "┤"
        elif abs(value - 1.0) < 0.025:  # Close to +1
            scale_line += "├"
        elif i % 10 == 0:  # Every 0.5 units
            scale_line += "┼"
        else:
            scale_line += "─"
    scale_line = scale_line[:-1] + "├ (p-val)"
    
    lines.append(scale_line)
    
    # Add a separator
    lines.append("")
    
    # Sort proteins by name for consistent output
    sorted_proteins = sorted(filtered_data.items())
    
    # Create bars for each protein
    for protein, coef_data in sorted_proteins:
        coef = coef_data['phi_coef']
        pval = coef_data['pval']
        
        # Calculate bar length and direction
        bar_length = int(abs(coef) / resolution)
        bar_length = min(bar_length, zero_pos)  # Cap at maximum possible length
        
        # Choose bar character based on coefficient magnitude
        if abs(coef) < 0.1:
            bar_char = bar_chars['light']
        elif abs(coef) < 0.5:
            bar_char = bar_chars['medium']
        else:
            bar_char = bar_chars['heavy']
        
        # Build the line
        line = f"{protein:<{max_protein_name_len}} │"
        
        # Create the bar
        bar_line = [' '] * scale_width
        
        # Add zero line
        bar_line[zero_pos] = bar_chars['zero']
        
        if coef < 0:
            # Negative bar (goes left from zero)
            start_pos = max(0, zero_pos - bar_length)
            for i in range(start_pos, zero_pos):
                bar_line[i] = bar_char
        elif coef > 0:
            # Positive bar (goes right from zero)
            end_pos = min(scale_width, zero_pos + bar_length + 1)
            for i in range(zero_pos + 1, end_pos):
                bar_line[i] = bar_char
        
        line += ''.join(bar_line)
        
        # Format p-value for display
        if isinstance(pval, float) and not math.isnan(pval):
            pval_str = f"{pval:.3f}"
        else:
            pval_str = "N/A"
        
        line += f"│ {coef:>6.3f} ({pval_str})"
        
        lines.append(line)
    
    lines.append("")
    return "\n".join(lines)


def plot_all_edges_phi_coefficients(combined_graph):
    """
    Create ASCII bar graphs for all edges in the combined graph.
    
    Args:
        combined_graph: Graph object with edges containing phi_coef data
    
    Returns:
        String containing all plots
    """
    all_plots = []
    
    for i, edge in enumerate(combined_graph.es):
        if 'phi_coef' in edge.attributes():
            phi_coef_dict = edge['phi_coef']
            plot = plot_phi_coefficients_ascii(phi_coef_dict, edge_index=i)
            all_plots.append(plot)
    
    return "\n" + "="*80 + "\n".join(all_plots)


def convert_ascii_plot_to_html(ascii_plot: str):
    html_ascii_plot = ascii_plot.replace("\n", "<br>")
    return html_ascii_plot

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -------------------------------------------------------------------------------------
# ------------- Associations between protein appearance and RMSD change ---------------
# -------------------------------------------------------------------------------------

def compute_point_biserial_corr_from_rmsd_df(rmsd_df, ref_protein, query_protein, domain):
    
    # Determine how many times ref_protein must appear
    count_threshold = 2 if ref_protein == query_protein else 1
    
    # Collect rows of (RMSD, is_present)
    records = []
    domain_mask = rmsd_df['Domain'] == domain
    for _, row in rmsd_df[domain_mask].iterrows():
        rmsd = row['RMSD']
        model = row['Model']
        is_present = model.count(query_protein) >= count_threshold
        records.append({'RMSD': rmsd, 'is_present': is_present})

    # Turn into DataFrame
    rmsd_protein_counts_df = pd.DataFrame.from_records(records, columns=['RMSD', 'is_present'])
    
    # need both classes
    if rmsd_protein_counts_df["is_present"].nunique() < 2:
        return float("nan"), float("nan")

    # compute point‑biserial
    r, p_value = pointbiserialr(
        rmsd_protein_counts_df["RMSD"],
        rmsd_protein_counts_df["is_present"]
    )
    
    return r, p_value


def add_point_biserial_corr_for_rmsd_and_partners(combined_graph):
    
    for node in combined_graph.vs:
        
        # Extract necessary data
        ref_protein = node['name']
        rmsd_df = node['RMSD_df']
        available_domains = set(rmsd_df['Domain'])
        available_proteins = set([prot for combination in rmsd_df['Model'] for prot in combination])
        
        # Store results
        point_biserial_corr_dict = {d: {} for d in available_domains}
        
        for domain in available_domains:
            
            for query_protein in available_proteins:
                
                rpb, pval = compute_point_biserial_corr_from_rmsd_df(
                    rmsd_df, ref_protein, query_protein, domain
                )
                
                point_biserial_corr_dict[domain][query_protein] = {
                    "rpb": rpb, "pval": pval
                }
        
        # Add results to the node
        node['RMSD_point_biserial_corr'] = point_biserial_corr_dict

        # Add ASCII plots
        point_biserial_corr_ascii_plot = plot_point_biserial_corr_ascii(point_biserial_corr_dict)
        point_biserial_corr_ascii_plot_html = convert_ascii_plot_to_html(point_biserial_corr_ascii_plot)
        node['RMSD_point_biserial_corr_ascii_plot'] = point_biserial_corr_ascii_plot
        node['RMSD_point_biserial_corr_ascii_plot_html'] = point_biserial_corr_ascii_plot_html


def plot_point_biserial_corr_ascii(point_biserial_corr_dict, node_index=None, title=None):
    """
    Create ASCII bar graphs for point-biserial correlations across multiple domains.
    
    Args:
        point_biserial_corr_dict: Nested dictionary with domains as keys, and protein names 
                                 as sub-keys with 'rpb' and 'pval' values
        node_index: Optional node index for labeling
        title: Optional title for the plot
    
    Returns:
        String containing the ASCII plots for all domains
    """
    if not point_biserial_corr_dict:
        return f"Node {node_index}: No point-biserial correlation data to plot\n"
    
    # Configuration
    resolution = 0.05  # 0.05 per character
    scale_width = int(2.0 / resolution)  # Total width for -1 to +1 range
    zero_pos = scale_width // 2  # Position of zero
    
    # Characters for different bar intensities
    bar_chars = {
        'light': '░',
        'medium': '▒', 
        'heavy': '█',
        'zero': '│'
    }
    
    # Build the plot
    lines = []
    
    # Add title
    if title:
        lines.append(title)
    elif node_index is not None:
        lines.append(f"Node {node_index} - Point-Biserial Correlations (RMSD vs Protein Presence)")
    else:
        lines.append("Point-Biserial Correlations (RMSD vs Protein Presence)")
    lines.append("")
    
    # Sort domains for consistent output
    sorted_domains = sorted(point_biserial_corr_dict.keys())
    
    for domain_idx, domain in enumerate(sorted_domains):
        domain_data = point_biserial_corr_dict[domain]
        
        # Filter out NaN values and convert to float
        filtered_data = {}
        for protein, corr_data in domain_data.items():
            rpb = corr_data.get('rpb')
            pval = corr_data.get('pval')
            if not (isinstance(rpb, float) and math.isnan(rpb)) and rpb is not None:
                filtered_data[protein] = {'rpb': float(rpb), 'pval': pval}
        
        if not filtered_data:
            lines.append(f"Domain {domain}: No valid correlations to plot")
            lines.append("")
            continue
        
        # Add domain header
        lines.append(f"Domain: {domain}")
        lines.append("-" * (domain + 8))
        
        # Calculate max protein name length for this domain
        max_protein_name_len = max(len(name) for name in filtered_data.keys())
        
        # Add numerical scale: place "-1" at left, "0" at center, "1" at right
        prefix = " " * (max_protein_name_len + 1)
        nums = [" "] * (scale_width + 1)

        # left end "-1"
        nums[0] = "-"
        nums[1] = "1"

        # zero "0"
        nums[zero_pos + 1] = "0"

        # right end "1"
        nums[-1] = "1"

        lines.append(prefix + "".join(nums))
        
        # Create scale header with p-value reference
        scale_line = " " * (max_protein_name_len + 2)
        for i in range(scale_width):
            value = -1.0 + (i * resolution)
            if abs(value) < 0.025:  # Close to zero
                scale_line += "│"
            elif abs(value - (-1.0)) < 0.025:  # Close to -1
                scale_line += "┤"
            elif abs(value - 1.0) < 0.025:  # Close to +1
                scale_line += "├"
            elif i % 10 == 0:  # Every 0.5 units
                scale_line += "┼"
            else:
                scale_line += "─"
        scale_line = scale_line[:-1] + "├ (p-val)"
        
        lines.append(scale_line)
        
        # Add a separator
        lines.append("")
        
        # Sort proteins by name for consistent output
        sorted_proteins = sorted(filtered_data.items())
        
        # Create bars for each protein
        for protein, corr_data in sorted_proteins:
            rpb = corr_data['rpb']
            pval = corr_data['pval']
            
            # Calculate bar length and direction
            bar_length = int(abs(rpb) / resolution)
            bar_length = min(bar_length, zero_pos)  # Cap at maximum possible length
            
            # Choose bar character based on coefficient magnitude
            if abs(rpb) < 0.1:
                bar_char = bar_chars['light']
            elif abs(rpb) < 0.5:
                bar_char = bar_chars['medium']
            else:
                bar_char = bar_chars['heavy']
            
            # Build the line
            line = f"{protein:<{max_protein_name_len}} │"
            
            # Create the bar
            bar_line = [' '] * scale_width
            
            # Add zero line
            bar_line[zero_pos] = bar_chars['zero']
            
            if rpb < 0:
                # Negative bar (goes left from zero)
                start_pos = max(0, zero_pos - bar_length)
                for i in range(start_pos, zero_pos):
                    bar_line[i] = bar_char
            elif rpb > 0:
                # Positive bar (goes right from zero)
                end_pos = min(scale_width, zero_pos + bar_length + 1)
                for i in range(zero_pos + 1, end_pos):
                    bar_line[i] = bar_char
            
            line += ''.join(bar_line)
            
            # Format p-value for display
            if isinstance(pval, float) and not math.isnan(pval):
                pval_str = f"{pval:.3f}"
            else:
                pval_str = "N/A"
            
            line += f"│ {rpb:>6.3f} ({pval_str})"
            
            lines.append(line)
        
        # Add spacing between domains (except for the last one)
        if domain_idx < len(sorted_domains) - 1:
            lines.append("")
    
    lines.append("")
    return "\n".join(lines)