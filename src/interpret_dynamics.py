
import numpy as np
import pandas as pd
import igraph

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
        .query(f'AF_2mers == {is_present_in_2mers}')
        .query(f'AF_Nmers == {is_present_in_Nmers}')
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

def get_edge_Nmers_variation(edge, N_models_cutoff: int):

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

# Weight
def get_edge_weight(graph_edge: igraph.Edge, classification_df: pd.DataFrame, default_edge_weight = 0.5):

    edge_dynamics = graph_edge["dynamics"]
    edge_width_is_variable = classification_df.query(f'Classification == "{edge_dynamics}"')["Variable_Edge_width"].iloc[0]

    if edge_width_is_variable:
        
        # Use mean number of models that surpass the cutoff and 1/mean(miPAE) to construct a weight
        edge_weight_Nmers = int(np.mean(list(graph_edge["2_mers_data"]["N_models"]) + list(graph_edge["N_mers_data"]["N_models"])))
        edge_weight_PAE = int(1/ np.mean(list(graph_edge["2_mers_data"]["min_PAE"]) + list(graph_edge["N_mers_data"]["min_PAE"])))
        edge_weight = edge_weight_Nmers * edge_weight_PAE

        # Limit to reasonable values
        if edge_weight < 1:
            return 1
        elif edge_weight > 8:
            return 8
        return edge_weight
    
    # If it has fixed length
    else:
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
# ------------------- Residue-Residue contacts matrix dynamics ------------------------
# -------------------------------------------------------------------------------------