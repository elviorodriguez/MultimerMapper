
import os
import igraph
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot

from utils.logger_setup import configure_logger
from src.analyze_homooligomers import find_homooligomerization_breaks
from utils.oscillations import oscillate_line, oscillate_circle
from utils.combinations import find_untested_2mers
from src.interpret_dynamics import read_classification_df, classify_edge_dynamics, classification_df,  get_edge_color_hex, get_edge_linetype, get_edge_weight, get_edge_oscillation
from src.coordinate_analyzer import add_domain_RMSD_against_reference

# -----------------------------------------------------------------------------
# PPI graph for 2-mers --------------------------------------------------------
# -----------------------------------------------------------------------------

def generate_2mers_graph(pairwise_2mers_df_F3: pd.DataFrame,
                               out_path: str = ".",
                               overwrite: bool = False):
    
    # Extract unique nodes from both 'protein1' and 'protein2'
    nodes = list(set(pairwise_2mers_df_F3['protein1']) | set(pairwise_2mers_df_F3['protein2']))
    
    # Create an undirected graph
    graph = igraph.Graph()
    
    # Add vertices (nodes) to the graph
    graph.add_vertices(nodes)
    
    # Add edges to the graph
    edges = list(zip(pairwise_2mers_df_F3['protein1'], pairwise_2mers_df_F3['protein2']))
    graph.add_edges(edges)
    
    # Set the edge weight modifiers
    N_models_W = pairwise_2mers_df_F3.groupby(['protein1', 'protein2'])['N_models'].max().reset_index(name='weight')['weight']
    ipTM_W = pairwise_2mers_df_F3.groupby(['protein1', 'protein2'])['ipTM'].max().reset_index(name='weight')['weight']
    min_PAE_W = pairwise_2mers_df_F3.groupby(['protein1', 'protein2'])['min_PAE'].max().reset_index(name='weight')['weight']
    
    # Set the weights with custom weight function
    graph.es['weight'] = round(N_models_W * ipTM_W * (1/min_PAE_W) * 2, 2)
    
    # Add ipTM, min_PAE and N_models as attributes to the graph
    graph.es['ipTM'] = ipTM_W
    graph.es['min_PAE'] = min_PAE_W
    graph.es['N_models'] = N_models_W
    
    # Create the directory if it doesn't exist
    out_2d_dir = out_path + "/2D_graphs"
    os.makedirs(out_2d_dir, exist_ok = overwrite)
    
    # Plot full graph
    igraph.plot(graph, 
                layout = graph.layout("fr"),
                
                # Nodes (vertex) characteristics
                vertex_label = graph.vs["name"],
                vertex_size = 40,
                # vertex_color = 'lightblue',
                
                # Edges characteristics
                edge_width = graph.es['weight'],
                # edge_label = graph.es['ipTM'],
                
                # Plot size
                bbox=(400, 400),
                margin = 50,
                
                # Save it in out_path
                target = out_2d_dir + "/2D_graph_2mers-full.png")
    
    return graph

# -----------------------------------------------------------------------------
# (This is BETA) Keep only fully connected network combinations ---------------
# -----------------------------------------------------------------------------

def find_sub_graphs(graph: igraph.Graph,
                     out_path: str = ".",
                     overwrite: bool = False):
    
    # Find fully connected subgraphs ----------------------------------------------
    fully_connected_subgraphs = [graph.subgraph(component) for component in graph.components() if graph.subgraph(component).is_connected()]
    
    # Create the directory if it doesn't exist
    out_2d_dir = out_path + "/2D_graphs"
    os.makedirs(out_2d_dir, exist_ok = overwrite)

    # Print the fully connected subgraphs
    print("\nFully connected subgraphs:")
    for i, subgraph in enumerate(fully_connected_subgraphs, start = 1):
        print(f"   - Subgraph {i}: {subgraph.vs['name']}")
        
        # prot_num = len(subgraph.vs["name"])
        
        # Plot sub-graph
        igraph.plot(subgraph, 
                    layout = subgraph.layout("fr"),
                    
                    # Nodes (vertex) characteristics
                    vertex_label = subgraph.vs["name"],
                    vertex_size = 40,
                    # vertex_color = 'lightblue',
                    
                    # Edges characteristics
                    edge_width = subgraph.es['weight'],
                    # edge_label = graph.es['ipTM'],
                    
                    # Plot size
                    # bbox=(100 + 40 * prot_num, 100 + 40 * prot_num),
                    bbox = (400, 400),
                    margin = 50,
                    
                    # Save subplot
                    target = out_2d_dir + f"/2D_sub_graph_Nº{i}-" + '_'.join(subgraph.vs["name"]) + ".png")
        
    return fully_connected_subgraphs


# Filter pairwise_2mers_df to get pairwise data for each sub_graph
def get_fully_connected_subgraphs_pairwise_2mers_dfs(pairwise_2mers_df_F3: pd.DataFrame,
                                                      fully_connected_subgraphs: list[igraph.Graph]):
    
    fully_connected_subgraphs_pairwise_2mers_dfs = []
    
    for sub_graph in fully_connected_subgraphs:
        
        proteins_in_subgraph = sub_graph.vs["name"]
        
        sub_pairwise_2mers_df = pairwise_2mers_df_F3.query(f"protein1 in {proteins_in_subgraph} and protein2 in {proteins_in_subgraph}")
        sub_pairwise_2mers_df.reset_index(drop=True, inplace=True)
        
        fully_connected_subgraphs_pairwise_2mers_dfs.append(sub_pairwise_2mers_df)
        
    return fully_connected_subgraphs_pairwise_2mers_dfs

# -----------------------------------------------------------------------------
# PPI graph for N-mers --------------------------------------------------------
# -----------------------------------------------------------------------------

def generate_Nmers_graph(pairwise_Nmers_df_F3: pd.DataFrame,
                               out_path: str = ".",
                               overwrite: bool = False):
    
    graph_df = pd.DataFrame(np.sort(pairwise_Nmers_df_F3[['protein1', 'protein2']], axis=1), columns=['protein1', 'protein2']).drop_duplicates()
    
    # Create a graph from the DataFrame
    graph = igraph.Graph.TupleList(graph_df.itertuples(index=False), directed=False)
      
    # Set the edge weight modifiers
    N_models_W = pairwise_Nmers_df_F3.groupby(['protein1', 'protein2'])['N_models'].max().reset_index(name='weight')['weight']
    pDockQ_W = pairwise_Nmers_df_F3.groupby(['protein1', 'protein2'])['pDockQ'].max().reset_index(name='weight')['weight']
    min_PAE_W = pairwise_Nmers_df_F3.groupby(['protein1', 'protein2'])['min_PAE'].min().reset_index(name='weight')['weight']
    
    # Set the weights with custom weight function
    graph.es['weight'] = round(N_models_W * pDockQ_W * (1/min_PAE_W) * 2, 2)
    
    # Add ipTM, min_PAE and N_models as attributes to the graph
    graph.es['ipTM'] = pDockQ_W
    graph.es['min_PAE'] = min_PAE_W
    graph.es['N_models'] = N_models_W
    
    # Create the directory if it doesn't exist
    out_2d_dir = out_path + "/2D_graphs"
    os.makedirs(out_2d_dir, exist_ok = overwrite)

    # Plot full graph
    igraph.plot(graph, 
                layout = graph.layout("fr"),
                
                # Nodes (vertex) characteristics
                vertex_label = graph.vs["name"],
                vertex_size = 40,
                # vertex_color = 'lightblue',
                
                # Edges characteristics
                edge_width = graph.es['weight'],
                # edge_label = graph.es['ipTM'],
                
                # Plot size
                bbox=(400, 400),
                margin = 50,
                
                # 
                target = out_2d_dir + "/2D_graph_Nmers-full.png")
    
    return graph


# -----------------------------------------------------------------------------
# Combined PPI graph ----------------------------------------------------------
# -----------------------------------------------------------------------------

def add_edges_data(graph,
                   pairwise_2mers_df: pd.DataFrame,
                   pairwise_Nmers_df: pd.DataFrame,
                   untested_edges_tuples: list[tuple],
                   
                   min_PAE_cutoff_2mers: int | float = 8.99,
                   ipTM_cutoff_2mers: int | float = 0.240,
                   
                   # N-mers cutoffs
                   min_PAE_cutoff_Nmers = 8.99,
                   pDockQ_cutoff_Nmers = 0.022):
    '''Adds 2-mers and N-mers data to integrate it as hovertext in plotly graph plots'''

    # ----------------------------------------------------------------------------------------
    # ---------------------------------- N-mers data -----------------------------------------
    # ----------------------------------------------------------------------------------------
    
    # Pre-process N-mers pairwise interactions:
    
    # Initialize dataframe to store N_models
    pairwise_Nmers_df_F1 = (pairwise_Nmers_df
        .groupby(['protein1', 'protein2', 'proteins_in_model'])
        # Compute the number of models on each pair
        .size().reset_index(name='N_models')
        .reset_index(drop=True)
        )
    # Count the number of models that surpass both cutoffs
    for model_tuple, group in (pairwise_Nmers_df
            # Unify the values on pDockQ and min_PAE the N-mer models with homooligomers
            .groupby(["protein1", "protein2", "proteins_in_model", "rank"])
            .agg({
                'min_PAE': 'min',   # keep only the minimum value of min_PAE
                'pDockQ': 'max'     # keep only the maximum value of pDockQ
            }).reset_index()).groupby(['protein1', 'protein2', 'proteins_in_model']):
        # Lists with models that surpass each cutoffs
        list1 = list(group["min_PAE"] < min_PAE_cutoff_Nmers)
        list2 = list(group["pDockQ"] > pDockQ_cutoff_Nmers)
        # Compares both lists and see how many are True
        N_models = sum([a and b for a, b in zip(list1, list2)])
        pairwise_Nmers_df_F1.loc[
            (pairwise_Nmers_df_F1["proteins_in_model"] == model_tuple[2]) &
            (pairwise_Nmers_df_F1["protein1"] == model_tuple[0]) &
            (pairwise_Nmers_df_F1["protein2"] == model_tuple[1]), "N_models"] = N_models
    
    # Extract best min_PAE and ipTM
    pairwise_Nmers_df_F2 = (pairwise_Nmers_df
        # Group by pairwise interaction
        .groupby(['protein1', 'protein2', 'proteins_in_model'])
        # Extract min_PAE
        .agg({'pTM': 'max',
                'ipTM': 'max',
                'min_PAE': 'min',
                'pDockQ': 'max'}
                )
        ).reset_index().merge(pairwise_Nmers_df_F1.filter(['protein1', 'protein2', 'proteins_in_model', 'N_models']), on=["protein1", "protein2", "proteins_in_model"])
    
    pairwise_Nmers_df_F2["extra_Nmer_proteins"] = ""
    
    for i, row in pairwise_Nmers_df_F2.iterrows():
        extra_proteins = tuple(e for e in row["proteins_in_model"] if e not in (row["protein1"], row["protein2"]))
        # Count how many times prot1 and 2 appears (are they modelled as dimers/trimers/etc)
        count_prot1 = list(row["proteins_in_model"]).count(str(row["protein1"]))
        count_prot2 = list(row["proteins_in_model"]).count(str(row["protein2"]))
        if str(row["protein1"]) == str(row["protein2"]):
            if count_prot1 > 1:
                extra_proteins = extra_proteins + (f'{str(row["protein1"])} as {count_prot1}-mer',)
        else:
            if count_prot1 > 1:
                extra_proteins = extra_proteins + (f'{str(row["protein1"])} as {count_prot1}-mer',)
            if count_prot2 > 1:
                extra_proteins = extra_proteins + (f'{str(row["protein2"])} as {count_prot1}-mer',)
        pairwise_Nmers_df_F2.at[i, "extra_Nmer_proteins"] = extra_proteins
    

    # Initialize edge attribute to avoid AttributeError
    graph.es["N_mers_data"] = None
    
    # Get protein info for each protein pair
    for pair, data in pairwise_Nmers_df_F2.groupby(['protein1', 'protein2']):
        
        # Pair coming from the dataframe (sorted)
        df_pair = sorted(pair)
        
        # Add info to the edges when the graph_pair matches the df_pair
        for edge in graph.es:
            source_name = graph.vs[edge.source]["name"]
            target_name = graph.vs[edge.target]["name"]
            
            graph_pair = sorted((source_name, target_name))
            
            # Add the data when it is a match
            if df_pair == graph_pair:
                
                filtered_data = data.filter(["pTM", "ipTM", "min_PAE", "pDockQ",
                                                "N_models", "proteins_in_model", "extra_Nmer_proteins"])
                
                # If no info was added previously
                if edge["N_mers_data"] is None:
                    edge["N_mers_data"] = filtered_data
                    
                # If the edge contains N_mers data
                else:
                    # Append the data
                    edge["N_mers_data"] = pd.concat([edge["N_mers_data"], filtered_data], ignore_index = True)
    
    # Add No data tag as N_mers_info in those pairs not explored in N-mers
    for edge in graph.es:
        if edge["N_mers_data"] is None:
            edge["N_mers_info"] = "No data"
            edge["N_mers_data"] = pd.DataFrame(columns=["pTM", "ipTM", "min_PAE", "pDockQ", "N_models", "proteins_in_model", "extra_Nmer_proteins"])
        else:
            # Convert data to a string
            data_str = edge["N_mers_data"].filter(["pTM", "ipTM", "min_PAE", "pDockQ", "N_models", "proteins_in_model"]).to_string(index=False).replace('\n', '<br>')
            edge["N_mers_info"] = data_str
    
    
    
    # ----------------------------------------------------------------------------------------
    # ---------------------------------- 2-mers data -----------------------------------------
    # ----------------------------------------------------------------------------------------
    
    # Pre-process pairwise interactions
    pairwise_2mers_df_F1 = (pairwise_2mers_df[
        # Filter the DataFrame based on the ipTM and min_PAE
        (pairwise_2mers_df['min_PAE'] <= min_PAE_cutoff_2mers) &
        (pairwise_2mers_df['ipTM'] >= ipTM_cutoff_2mers)]
        # Group by pairwise interaction
        .groupby(['protein1', 'protein2'])
        # Compute the number of models for each pair that are kept after the filter
        .size().reset_index(name='N_models')
        .reset_index(drop=True)
        )
    
    pairwise_2mers_df_F2 = (pairwise_2mers_df
        # Group by pairwise interaction
        .groupby(['protein1', 'protein2'])
        # Extract min_PAE
        .agg({'pTM': 'max',
                'ipTM': 'max',
                'min_PAE': 'min',
                'pDockQ': 'max'}
                )
        ).reset_index().merge(pairwise_2mers_df_F1.filter(['protein1', 'protein2', 'N_models']), on=["protein1", "protein2"])
    
    
    # Initialize 2_mers_data edge attribute to avoid AttributeErrors
    graph.es["2_mers_data"] = None
    
    for pair, data in pairwise_2mers_df_F2.groupby(['protein1', 'protein2']):
    
        df_pair = sorted(pair)
    
        for edge in graph.es:
            source_name = graph.vs[edge.source]["name"]
            target_name = graph.vs[edge.target]["name"]
            
            graph_pair = sorted((source_name,target_name))
            
            # If the pair from the df is the same as the edge pair
            if df_pair == graph_pair:
                
                # Extract interaction data
                filtered_data = data.filter(["pTM", "ipTM", "min_PAE", "pDockQ", "N_models"])
                
                # If no info was added previously
                if edge["2_mers_data"] is None:
                    edge["2_mers_data"] = filtered_data
                    
                # If the edge contains 2_mers data (Which is not possible, I think...)
                else:
                    # DEBUG
                    # print("WARNING: SOMETHING IS WRONG WITH AN EDGE!")
                    # print("WARNING: There is an unknown inconsistency with the data...")
                    # print("WARNING: Have you modelled by mistake a protein pair twice???")
                    # print("WARNING: Edge that produced the warning:", (graph.vs[edge.source]["name"], graph.vs[edge.target]["name"]))
                    
                    # Append the data
                    edge["2_mers_data"] = pd.concat([edge["2_mers_data"], filtered_data], ignore_index= True)
        
    for edge in graph.es:
        
        # If no data was found for the edge
        if edge["2_mers_data"] is None:
            
            # DEBUG
            # print("WARNING: SOMETHING IS WRONG WITH AN EDGE!")
            # print("WARNING: There is an unknown inconsistency with the data...")
            # print("WARNING: Did you left a protein pair without exploring its interaction???")
            # print("WARNING: Edge that produced the warning:", (graph.vs[edge.source]["name"], graph.vs[edge.target]["name"]))

            # And add an empty dataframe as 2_mers_data attribute
            edge["2_mers_data"] = pd.DataFrame(columns=["pTM", "ipTM", "min_PAE", "pDockQ", "N_models"])
            
            if tuple(sorted((graph.vs[edge.source]["name"], graph.vs[edge.target]["name"]))) in untested_edges_tuples:

                # Add a label for missing data
                edge["2_mers_info"] = "No data"


            else:

                # Add a label for missing data
                edge["2_mers_info"] = "No rank surpass cutoff"
                
        else:
            # Convert data to a string and add the attribute on the edge
            edge["2_mers_info"] = edge["2_mers_data"].to_string(index=False).replace('\n', '<br>')

# Function to add the IDs of the proteins
def add_vertices_IDs(graph, prot_IDs, prot_names):
    
    for ID, name in zip(prot_IDs, prot_names):
        for vertex in graph.vs:
            if vertex["name"] == ID:
                vertex["IDs"] = name
                break

# Function to add meaning column to vertex based on color
# The logic based on color first is horrible, but it works...
# Will be changed in a future version.
def add_vertices_meaning(graph, vertex_color1='red', vertex_color2='green', vertex_color3 = 'orange', vertex_color_both='gray'):
    
    # Function to determine the meaning based on color
    def get_meaning(row):
        if row['color'] == vertex_color_both:
            return 'Static protein'
        elif row['color'] == vertex_color1:
            return 'Dynamic protein (disappears in N-mers)'
        elif row['color'] == vertex_color2:
            return 'Dynamic protein (appears in N-mers)'
        elif row['color'] == vertex_color3:
            return 'Protein dynamics not explored in N-mers'
        else:
            return 'unknown'
        
    vertex_df = graph.get_vertex_dataframe()
    # Apply the function to create the new 'meaning' column
    vertex_df['meaning'] = vertex_df.apply(get_meaning, axis=1)
    
    graph.vs["meaning"] = vertex_df['meaning']



# Combine 2-mers and N-mers graphs
def generate_combined_graph(
        graph1, graph2,
        pairwise_2mers_df, pairwise_Nmers_df,
        domains_df, sliced_PAE_and_pLDDTs,
        pairwise_2mers_df_F3,

        # Prot IDs and prot names to add them to the graph as hovertext later on
        prot_IDs, prot_names,

        # 2-mers cutoffs
        min_PAE_cutoff_2mers = 8.99, ipTM_cutoff_2mers = 0.240,

        # N-mers cutoffs
        min_PAE_cutoff_Nmers = 8.99, pDockQ_cutoff_Nmers = 0.022,

        # General cutoff
        N_models_cutoff = 3,

        # For RMSD calculations
        domain_RMSD_plddt_cutoff = 60, trimming_RMSD_plddt_cutoff = 70,

        # Other parameters
        edge_color1='red', edge_color2='green', edge_color3 = 'orange', edge_color4 = 'purple',  edge_color5 = "pink",
        edge_color6 = "blue", edge_color_both='black',
        vertex_color1='red', vertex_color2='green', vertex_color3='orange', vertex_color_both='gray',
        
        pdockq_indirect_interaction_cutoff = 0.23, predominantly_static_cutoff = 0.6,
        remove_indirect_interactions = True,
        
        is_debug = False, logger = None):
    """
    Compares two graphs and create a new graph with colored edges and vertices based on their differences.

    Parameters:
    - graph1 (2mers), graph2 (Nmers): igraph.Graph objects representing the two graphs to compare.
    - edge_color1, edge_color2, edge_color3, edge_color4, edge_color5, edge_color6, edge_color_both: Colors for edges in 
        (1) graph1 only (2mers only),
        (2) graph2 only (Nmers only), 
        (3) graph1 only but not tested in Nmers (lacks dynamic information), 
        (4) ambiguous (Some Nmers have it), 
        (5) indirect interactions (Nmers mean pDockQ < 0.23),
        (6) ambiguous but predominantly static and 
        (both: static) both graphs, respectively.
    - vertex_color1, vertex_color2, vertex_color_both: Colors for vertices in
        (1) graph1 only,
        (2) graph2 only,
        (both) both graphs, respectively.
    - pdockq_indirect_interaction_cutoff:
    - predominantly_static_cutoff (float 0->1): for ambiguous N-mer interactions, the fraction of models that need to be positive to consider it a predominantly static interaction. Default=0.6.

    Returns:
    - Combined igraph.Graph object with colored edges and vertices.
    """
    # Configure logger
    if logger is None:
        configure_logger()(__name__)

    # To check if the computation was performed or not
    tested_Nmers_edges_df = pd.DataFrame(np.sort(pairwise_Nmers_df[['protein1', 'protein2']], axis=1),
                 columns=['protein1', 'protein2']).drop_duplicates().reset_index(drop = True)
    tested_Nmers_edges_sorted = [tuple(sorted(tuple(row))) for i, row in tested_Nmers_edges_df.iterrows()]
    tested_Nmers_nodes = list(set(list(tested_Nmers_edges_df["protein1"]) + list(tested_Nmers_edges_df["protein2"])))
    
    # Get edges from both graphs
    edges_g1 = [(graph1.vs["name"][edge[0]], graph1.vs["name"][edge[1]]) for edge in graph1.get_edgelist()]
    edges_g2 = [(graph2.vs["name"][edge[0]], graph2.vs["name"][edge[1]]) for edge in graph2.get_edgelist()]

    # Sorted list of edges
    edges_g1_sort = sorted([tuple(sorted(t)) for t in edges_g1], key=lambda x: x[0])
    edges_g2_sort = sorted([tuple(sorted(t)) for t in edges_g2], key=lambda x: x[0])
    # Get untested 2-mers combinations
    untested_edges_tuples = sorted(list(find_untested_2mers(prot_IDs = prot_IDs, pairwise_2mers_df = pairwise_2mers_df)), key=lambda x: x[0])
        
    # Make a combined edges set
    edges_comb = sorted(list(set(edges_g1_sort + edges_g2_sort + untested_edges_tuples)), key=lambda x: x[0])

    # Create a graph with the data
    graphC = igraph.Graph.TupleList(edges_comb, directed=False)
    
    # Extract its vertices and edges
    edges_gC = [(graphC.vs["name"][edge[0]], graphC.vs["name"][edge[1]]) for edge in graphC.get_edgelist()]
    edges_gC_sort = [tuple(sorted(edge)) for edge in edges_gC]
    
    # ----------------------------------------------------------------------------------------
    # --------------------- Add vertex colors & meaning of the color -------------------------
    # ----------------------------------------------------------------------------------------

    # Create empty df to store dynamic proteins
    columns = ["protein", "only_in"]
    dynamic_proteins = pd.DataFrame(columns = columns)

    # Give each vertex a color
    vertex_colors = []
    for v in graphC.vs["name"]:

        # Shared edges in both in g1 and g2 (2-mers & N-mers)
        if v in graph1.vs["name"] and v in graph2.vs["name"]:
            vertex_colors.append(vertex_color_both)
        
        # Edges only in g1 (2-mers)
        elif v in graph1.vs["name"] and v not in graph2.vs["name"]:

            # Only in 2-mers, but not tested in N-mers (lacks info)
            if v not in tested_Nmers_nodes:
                vertex_colors.append(vertex_color3)
                dynamic_proteins = pd.concat([dynamic_proteins,
                                              pd.DataFrame({"protein": [v],
                                                            "only_in": ["2mers-but_not_tested_in_Nmers"]})
                                              ], ignore_index = True)
            
            # Only in 2-mers (with N-mers info)
            else:
                vertex_colors.append(vertex_color1)
                dynamic_proteins = pd.concat([dynamic_proteins,
                                              pd.DataFrame({"protein": [v],
                                                            "only_in": ["2mers"]})
                                              ], ignore_index = True)
        
        # Edges only in g2 (N-mers)
        elif v not in graph1.vs["name"] and v in graph2.vs["name"]:
            vertex_colors.append(vertex_color2)
            dynamic_proteins = pd.concat([dynamic_proteins,
                                          pd.DataFrame({"protein": [v],
                                                        "only_in": ["Nmers"]})
                                          ], ignore_index = True)
        
        # This if something happens
        else:
            raise ValueError("For some reason a node that comes from the graphs to compare is not in either graphs...")
        
    graphC.vs['color'] = vertex_colors
    add_vertices_meaning(graphC, vertex_color1, vertex_color2, vertex_color3, vertex_color_both)


    # ----------------------------------------------------------------------------------------
    # -------------- Add 2/N-mers data, homooligomeric states, RMSD, etc ---------------------
    # ----------------------------------------------------------------------------------------

    def add_homooligomerization_state(graph,
                                      pairwise_2mers_df_F3 = pairwise_2mers_df_F3,
                                      pairwise_Nmers_df = pairwise_Nmers_df,
                                      logger = logger,
                                      min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers,
                                      pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers,
                                      N_models_cutoff = N_models_cutoff):
        
        # Compute homooligomerization data
        homooligomerization_states = find_homooligomerization_breaks(
                                            pairwise_2mers_df_F3 = pairwise_2mers_df_F3,
                                            pairwise_Nmers_df = pairwise_Nmers_df,
                                            logger = logger,
                                            min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers,
                                            pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers,
                                            N_models_cutoff = N_models_cutoff)

        # Initialize edge attribute
        graph.es["homooligomerization_states"] = None

        for edge in graph.es:
            source_name = graph.vs[edge.source]["name"]
            target_name = graph.vs[edge.target]["name"]

            # If it is a homooligomer
            if source_name == target_name:

                # Add its homooligomerization state data
                edge["homooligomerization_states"] = homooligomerization_states[source_name]

                
    # Add data to the combined graph to allow hovertext display later
    add_edges_data(graphC, pairwise_2mers_df, pairwise_Nmers_df,
                   untested_edges_tuples = untested_edges_tuples, 
                   # 2-mers cutoffs
                   min_PAE_cutoff_2mers = min_PAE_cutoff_2mers,
                   ipTM_cutoff_2mers = ipTM_cutoff_2mers,
                   # N-mers cutoffs
                   min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers,
                   pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers)
    
    add_vertices_IDs(graphC, prot_IDs, prot_names)

    add_domain_RMSD_against_reference(graphC, domains_df, sliced_PAE_and_pLDDTs,pairwise_2mers_df, pairwise_Nmers_df,
                                      domain_RMSD_plddt_cutoff, trimming_RMSD_plddt_cutoff)
    
    add_homooligomerization_state(graphC,
                                  pairwise_2mers_df_F3 = pairwise_2mers_df_F3,
                                  pairwise_Nmers_df = pairwise_Nmers_df,
                                  logger = logger,
                                  min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers,
                                  pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers,
                                  N_models_cutoff = N_models_cutoff)
    

    # ----------------------------------------------------------------------------------------
    # -------------------------- Add edges dynamic classifications ---------------------------
    # ----------------------------------------------------------------------------------------

    # Dynamic classification of each edge is performed based on cfg/interaction_classification.tvs table

    # Create a df to track dynamic interactions
    edges_dynamics = []

    # Analyze one edge of the combined graph at a time
    for edge in edges_gC_sort:
        
        e_dynamic = classify_edge_dynamics(tuple_edge = edge,
                                           
                                           # Graph
                                           graph_Comb = graphC,
                                           
                                           # Cutoffs
                                           N_models_cutoff = N_models_cutoff,
                                           
                                           # Sorted tuple edges lists
                                           sorted_edges_2mers_graph  = edges_g1_sort, 
                                           sorted_edges_Nmers_graph  = edges_g2_sort,
                                           untested_edges_tuples     = untested_edges_tuples,
                                           tested_Nmers_edges_sorted = tested_Nmers_edges_sorted,

                                           classification_df = classification_df,
                                           
                                           logger = logger)

        edges_dynamics.append(e_dynamic)

    graphC.es['dynamics'] = edges_dynamics

    # ----------------------------------------------------------------------------------------
    # -------------------------- Add dict with the used cutoffs ------------------------------
    # ----------------------------------------------------------------------------------------

    # add cutoffs dict to the graph
    graphC["cutoffs_dict"] = dict(
        # 2-mers cutoffs
        min_PAE_cutoff_2mers = min_PAE_cutoff_2mers, ipTM_cutoff_2mers = ipTM_cutoff_2mers,
        # N-mers cutoffs
        min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers, pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers,
        # General cutoff
        N_models_cutoff = N_models_cutoff,
        # For RMSD calculations
        domain_RMSD_plddt_cutoff = domain_RMSD_plddt_cutoff, trimming_RMSD_plddt_cutoff = trimming_RMSD_plddt_cutoff)
    

    # Add edges "name"
    graphC.es["name"] = [(graphC.vs["name"][tuple_edge[0]], graphC.vs["name"][tuple_edge[1]]) for tuple_edge in graphC.get_edgelist()]
    
    
    return graphC, dynamic_proteins


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# 2D graph (protein level): interactive representation ------------------------
# -----------------------------------------------------------------------------

def format_homooligomerization_states(homooligomerization_states, logger):

    # Style definitions
    def apply_style(N, style: int):
        
        # OK and positive homooligomerization style (BLACK)
        if style == 0:
            return '<b style="color:black;">' + str(N) + '</b>'

        # OK, but negative homooligomerization style (RED)
        if style == 1:
            return '<b style="color:red;">' + str(N) + '</b>'

        # NOT OK homooligomerization style (ORANGE)
        if style == 2:
            return '<b style="color:orange;">' + str(N) + '</b>'

    formatted_N_states = '<b style="color:black;">2</b>|'

    for N, (is_ok, N_state) in enumerate(zip(homooligomerization_states["is_ok"], homooligomerization_states["N_states"]), start = 3):
        
        # If there is lacking predictions (error in the protocol)
        if N_state is None:
            formatted_N_states += apply_style(N, 2) + '|'
        # If the homooligomerization state is negative
        elif N_state is False:
            formatted_N_states += apply_style(N, 1) + '|'
        # If the homooligomerization state is positive
        elif N_state is True:
            formatted_N_states += apply_style(N, 0) + '|'
        else:
            logger.error('UNEXPECTED BEHAVIOR:')
            logger.error(f'   - Something is wrong with homooligomerization state! ("is_ok" value: {is_ok})')
            logger.error(f'   - Something is wrong with homooligomerization state! ("N_state" value: {N_state})')
    
    return formatted_N_states.rstrip("|")
        

# Generate a layout (using only static edges)
def generate_layout_for_combined_graph(
        combined_graph,
        edge_attribute_value = ['Static interaction', 'Ambiguous Dynamic (In some N-mers appear and in others disappear)'],
        vertex_attribute_value = 'Dynamic protein (disappears in N-mers)',
        layout_algorithm = "fr"):
    """
    Generate a layout for a combined graph based on specified edge and vertex attributes.
    
    Parameters:
    - combined_graph: igraph.Graph
        The combined 2/N-mers graph.
    - edge_attribute_value: list of str
        The values of the "meaning" attribute for edges to be included in the subgraph.
    - vertex_attribute_value: str
        The value of the "meaning" attribute for vertices to be included as isolated nodes in the subgraph.
    - layout_algorithm: str
        The layout algorithm to use (default is "fr").
    
    Returns:
    - igraph.Layout
        The layout for the combined graph based on specified edge and vertex attributes.
    """
    
    # Find vertices with the specified attribute value
    vertices_with_attribute = [v.index for v in combined_graph.vs.select(meaning=vertex_attribute_value)]

    # Create a subgraph with edges having the specified attribute value
    subgraph_edges = combined_graph.es.select(meaning_in=edge_attribute_value).indices
    subgraph = combined_graph.subgraph_edges(subgraph_edges)

    # Add isolated vertices with the specified attribute to the subgraph
    subgraph.add_vertices(vertices_with_attribute)

    # Generate layout for the subgraph using the specified algorithm
    layout = subgraph.layout(layout_algorithm)

    return layout


# ----------------------- Edge removal based in conditions --------------------------
def remove_edges_by_condition(graph: igraph.Graph, attribute: str, condition):
    """
    Remove edges from the graph that meet the given condition on an attribute.
    
    Parameters:
    - graph (igraph.Graph): The graph from which edges will be removed.
    - attribute (str): The edge attribute to check.
    - condition (function): A function that takes an attribute value and returns True if the edge should be removed.
    """
    edges_to_remove = [e.index for e in graph.es if condition(e[attribute])]
    graph.delete_edges(edges_to_remove)

def indirect_condition(value):
    return value == "Indirect"
# ------------------------------------------------------------------------------------

# Convert igraph graph to interactive plotly plot
def igraph_to_plotly(
        
        # Inputs
        graph: igraph.Graph,
        layout: igraph.Layout | str | None = "fr",
        save_html: str | None = None,
        
        # Edges visualization
        edge_width: int = 2,
        self_loop_orientation: float = 0,
        self_loop_size: float | int = 2.5,
        
        # Nodes visualization
        node_border_color: str = "black",
        node_border_width: int = 1,
        node_size: float | int = 4.5,
        node_names_fontsize: int = 12,
        use_bold_protein_names: bool = True,
        add_bold_RMSD_cutoff: int | float = 5,
        add_RMSD: bool = True,
        
        # General visualization options
        hovertext_size: int = 12,
        showlegend: bool = True,
        showgrid: bool = False,
        show_axis: bool = False,
        margin: dict = dict(l=0, r=0, b=0, t=0),
        legend_position: dict = dict(x=1.02, y=0.5),
        plot_graph: bool = True,
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        add_cutoff_legend: bool = True,
        oscillation_width: int = 1,
        oscillation_amplitude: float = 0.02,
        oscillation_lines_frequency: int = 8, 
        oscillation_circles_frequency: int = 20,
        remove_indirect_interactions: bool = True,
        
        logger = None):
    
    """
    Convert an igraph.Graph to an interactive Plotly plot. Used to visualize combined_graph.
    
    Parameters:
    - graph: igraph.Graph, the input graph.
    - layout: layout of the graph (e.g., layout = graph.layout_kamada_kawai()).
        if None (default), a layout will be produced using "fr" algorithm
        if str, a layout with layout algorithm will be created (eg: "kk" or "fr")
    - save_html (str): path to html file to be created.
    - edge_width (float): thickness of edges lines.
    - self_loop_orientation (float): rotates self-loop edges around the corresponding vertex (0.25 a quarter turn, 0.5 half, etc).
    - self_loop_size (float): self-loops circumferences size.
    - node_border_color (str): color for nodes borders (default = "black")
    - node_border_width (float): width of nodes borders (set to 0 to make them disappear)
    - node_size (float): size of nodes (vertices).
    - node_names_fontsize: size of protein names.
    - use_bold_protein_names (bool): display protein names in nodes as bold?
    - add_bold_RMSD_cutoff (float): cutoff value to highlight high RMSD domains in bold. To remove this option, set it to None.
    - add_RMSD (bool): add domain RMSD information in nodes hovertext?
    - hovertext_size (float): font size of hovertext.
    - showlegend (bool): display the legend?
    - showgrid (bool): display background grid?
    - showaxis (bool): display x and y axis?
    - margin (dict): left (l), right (r), bottom (b) and top (t) margins sizes. Default: dict(l=0, r=0, b=0, t=0).
    - legend_position (dict): x and y positions of the legend. Default: dict(x=1.02, y=0.5)
    - plot_graph (bool): display the plot in your browser?
    
    Returns:
    - fig: plotly.graph_objects.Figure, the interactive plot.
    """
    
    # ----------------------------------------------------------------------------------
    # --------------------------------- Initial setups ---------------------------------
    # ----------------------------------------------------------------------------------

    # Adjust the scale of the values
    node_size = node_size * 10
    self_loop_size = self_loop_size / 10
    default_edge_width = edge_width

    # Remove indirect interactions?
    if remove_indirect_interactions:

        # DeepCopy the graph to avoid affecting the original
        from copy import deepcopy
        graph = deepcopy(graph)
            
        # Remove edges that meet the condition
        remove_edges_by_condition(graph, attribute = 'dynamics', condition = indirect_condition)
    
    # Generate layout if if was not provided
    if layout == None:
        layout = graph.layout("fr")
    elif type(layout) == str:
        layout = graph.layout(layout)
    
    # Extract node and edge positions from the layout
    pos = {vertex.index: layout[vertex.index] for vertex in graph.vs}
    
    # Get the interaction classification dataframe
    classification_df = read_classification_df()

    # ----------------------------------------------------------------------------------
    # --------------------------- Edges/interaction/vertex -----------------------------
    # ----------------------------------------------------------------------------------

    # Create Scatter objects for edges, including self-loops
    edge_traces = []
    for edge in graph.es:
        
        try:
            # Get edge style based on meaning
            edge_dynamics   = edge["dynamics"]
            edge_color      = get_edge_color_hex(edge, classification_df)
            edge_linetype   = get_edge_linetype(edge, classification_df)
            edge_weight     = get_edge_weight(edge, classification_df)
            edge_oscillates = get_edge_oscillation(edge, classification_df)
        
        except:
            # Use default values by now
            edge_color      = "black"
            edge_linetype   = "solid"
            edge_weight     = 0.5
            edge_oscillates = True
            edge_dynamics   = "Static"
            
        # ----------------- Draw a circle for homooligomerization edges (self-loops) ----------------------
        if edge.source == edge.target:

            resolution = 200 # (number of points)
            theta = np.linspace(0, 2*np.pi, resolution)
            radius = self_loop_size
            
            # Adjust the position of the circle
            circle_x = pos[edge.source][0] + radius * np.cos(theta)
            circle_y = pos[edge.source][1] + radius * np.sin(theta) + radius

            # Apply rotation?
            if self_loop_orientation != 0:
                # Reference point to rotate the circle
                center_x = pos[edge.source][0]
                center_y = pos[edge.source][1]

                # Degrees to rotate the circle
                θ = self_loop_orientation * 2 * np.pi

                # New circle points
                circle_x_rot = center_x + (circle_x - center_x) * np.cos(θ) - (circle_y - center_y) * np.sin(θ)
                circle_y_rot = center_y + (circle_x - center_x) * np.sin(θ) + (circle_y - center_y) * np.cos(θ)

                # Overwrite values
                circle_x = circle_x_rot
                circle_y = circle_y_rot
            
            if edge_oscillates:
                # Get the center of the circle oscillation line is performed with that point as reference
                reference_point = np.array([np.mean(circle_x), np.mean(circle_y)])
                oscillated_circle_x, oscillated_circle_y = oscillate_circle(reference_point = reference_point,
                                                                            radius = radius,
                                                                            amplitude = oscillation_amplitude,
                                                                            frequency = oscillation_circles_frequency)
                
                # Add the oscillated edge trace
                oscillated_edge_trace = go.Scatter(
                    x = oscillated_circle_x.tolist() + [None],
                    y = oscillated_circle_y.tolist() + [None],
                    mode = "lines",
                    line = dict(color = edge_color, 
                                width = oscillation_width,
                                dash  = edge_linetype),
                    hoverinfo   = 'skip',
                    text        = None,
                    showlegend  = False
                )

                # add text trace
                edge_traces.append(oscillated_edge_trace)
            
            # Generate self-loop edge trace
            edge_trace = go.Scatter(
                x = circle_x.tolist() + [None],
                y = circle_y.tolist() + [None],
                mode = "lines",
                line = dict(color = edge_color,
                            width = int(edge_width * edge_weight),
                            dash  = edge_linetype),
                hoverinfo  ="text",
                text       = [edge_dynamics + "<br><br>-------- 2-mers data --------<br>" + edge["2_mers_info"] + "<br><br>-------- N-mers data --------<br>" + edge["N_mers_info"]] * len(circle_x),
                hovertext  = [edge_dynamics + "<br><br>-------- 2-mers data --------<br>" + edge["2_mers_info"] + "<br><br>-------- N-mers data --------<br>" + edge["N_mers_info"]] * len(circle_x),
                hoverlabel = dict(font=dict(family='Courier New', size=hovertext_size)),
                showlegend = False
            )

            # ------------- Add homooligomerization state data -----------------------

            # Calculate the center of the circle (this is to add text in the middle)
            circle_center_x = np.mean(circle_x)
            circle_center_y = np.mean(circle_y)
            formatted_N_states = format_homooligomerization_states(edge["homooligomerization_states"], logger = logger)

            # Add text in the center with homooligomerization state info
            text_trace = go.Scatter(
                x=[circle_center_x],
                y=[circle_center_y],
                mode='text',
                text=[formatted_N_states],
                textposition="middle center",
                hoverinfo='none',
                showlegend=False
            )

            # add text trace
            edge_traces.append(text_trace)

        # ----------------- Draw a line for heteromeric edges ----------------------
        else:
            
            # Generate additional points along the edge
            resolution = 200
            intermediate_x = np.linspace(pos[edge.source][0], pos[edge.target][0], resolution + 2)
            intermediate_y = np.linspace(pos[edge.source][1], pos[edge.target][1], resolution + 2)

            if edge_oscillates:
                oscillated_x, oscillated_y = oscillate_line(start_point = pos[edge.source],
                                                            end_point   = pos[edge.target],
                                                            amplitude = oscillation_amplitude,
                                                            frequency = oscillation_lines_frequency)
                
                # Add the oscillated edge trace
                oscillated_edge_trace = go.Scatter(
                    x = oscillated_x.tolist() + [None],
                    y = oscillated_y.tolist() + [None],
                    mode = "lines",
                    line = dict(color = edge_color, 
                                width = oscillation_width,
                                dash  = edge_linetype),
                    hoverinfo   = 'skip',
                    text        = None,
                    showlegend  = False
                )
                # Add traces
                edge_traces.append(oscillated_edge_trace)
            
            # Add the edge trace
            edge_trace = go.Scatter(
                x=intermediate_x.tolist() + [None],
                y=intermediate_y.tolist() + [None],
                mode = "lines",
                line = dict(color = edge_color,
                            width = int(edge_width * edge_weight),
                            dash  = edge_linetype),
                hoverinfo   = "text",  # Add hover text
                text        = [edge_dynamics + "<br><br>-------- 2-mers data --------<br>" + edge["2_mers_info"] + "<br><br>-------- N-mers data --------<br>" + edge["N_mers_info"]] * (resolution + 2),
                hovertext   = [edge_dynamics + "<br><br>-------- 2-mers data --------<br>" + edge["2_mers_info"] + "<br><br>-------- N-mers data --------<br>" + edge["N_mers_info"]] * (resolution + 2),
                hoverlabel  = dict(font=dict(family='Courier New', size=hovertext_size)),
                showlegend  = False
            )
        
        # Add traces
        edge_traces.append(edge_trace)
    
    
    # ----------------------------------------------------------------------------------
    # ------------------------ Nodes/proteins/vertex traces ----------------------------
    # ----------------------------------------------------------------------------------

    nodes_df = graph.get_vertex_dataframe()
    nodes_df["x_coord"] = [c[0] for c in layout.coords]
    nodes_df["y_coord"] = [c[1] for c in layout.coords]
    nodes_number = len(graph.get_vertex_dataframe())

    # Color
    try:
        nodes_df["color"]
    except:
        # graph.vs["color"] = ["gray"] * nodes_number
        nodes_df["color"] = ["gray"] * nodes_number
    
    # Meaning
    try:
        nodes_df["meaning"]
    except:
        # graph.vs["meaning"] = ["Proteins"] * nodes_number
        nodes_df["meaning"] = ["Proteins"] * nodes_number

    # ID
    try:
        nodes_df["IDs"]
    except:
        nodes_df["IDs"] = ["No ID data"] * nodes_number
    
    if use_bold_protein_names:
        bold_names = ["<b>" + name + "</b>" for name in nodes_df["name"]]
        nodes_df["name"] = bold_names
        
    nodes_hovertext =  [mng + f" (ID: {ID})" for mng, ID in zip(nodes_df["meaning"].tolist(), nodes_df["IDs"].tolist())]
    
    
    if add_RMSD:
        
        if add_bold_RMSD_cutoff != None:
            
            # Function to format RMSD values (adds bold HTML tags for RMSD > add_bold_RMSD_cutoff)
            def format_rmsd(value, threshold=5):
                rounded_value = round(value, 2)
                formatted_value = f'<b>{formatted(rounded_value)}</b>' if rounded_value > threshold else f'<b></b>{formatted(rounded_value)}'
                return formatted_value

            # Function to format a float with two decimal places as str
            def formatted(value):
                return '{:.2f}'.format(value)
            
            # Create empty list to store formatted RMSD dataframes
            RMSD_dfs = [""] * nodes_number
            
            # Apply the function to the 'RMSD' column
            for DF, sub_df in enumerate(nodes_df["RMSD_df"]):
                # nodes_df["RMSD_df"][DF]['RMSD'] = nodes_df["RMSD_df"][DF]['RMSD'].apply(lambda x: format_rmsd(x, threshold=add_bold_RMSD_cutoff))
                # nodes_df["RMSD_df"][DF].rename(columns={'RMSD': '<b></b>RMSD'}, inplace = True)
                RMSD_data = nodes_df["RMSD_df"][DF]['RMSD'].apply(lambda x: format_rmsd(x, threshold=add_bold_RMSD_cutoff))
                RMSD_dfs[DF] = nodes_df["RMSD_df"][DF].drop(columns="RMSD")
                RMSD_dfs[DF]['<b></b>RMSD'] = RMSD_data
        
        # Modify the hovertext to contain domains and RMSD values
        nodes_hovertext = [
            hovertext +
            "<br><br>-------- Reference structure domains --------<br>" +
            domain_data.to_string(index=False).replace('\n', '<br>')+
            "<br><br>-------- Domain RMSD against highest pLDDT structure --------<br>" +
            RMSD_data.to_string(index=False).replace('\n', '<br>') +
            f'<br><br>*Domains with mean pLDDT < {graph["cutoffs_dict"]["domain_RMSD_plddt_cutoff"]} (disordered) were not used for RMSD calculations.<br>'+
            f'**Only residues with pLDDT > {graph["cutoffs_dict"]["trimming_RMSD_plddt_cutoff"]} were considered for RMSD calculations.'
            # for hovertext, domain_data, RMSD_data in zip(nodes_hovertext, nodes_df["domains_df"], nodes_df["RMSD_df"])
            for hovertext, domain_data, RMSD_data in zip(nodes_hovertext, nodes_df["domains_df"], RMSD_dfs)
        ]
    
    node_trace = go.Scatter(
        x=nodes_df["x_coord"],
        y=nodes_df["y_coord"],
        mode="markers+text",
        marker=dict(size=node_size, color=nodes_df["color"],
                    line=dict(color=node_border_color, width= node_border_width)),
        text=nodes_df["name"],
        textposition='middle center',
        textfont=dict(size=node_names_fontsize),
        hoverinfo="text",
        hovertext=nodes_hovertext,
        hoverlabel=dict(font=dict(family='Courier New', size=hovertext_size)),
        showlegend=False
    )
    

    # Create the layout for the plot
    layout = go.Layout(        
        legend = legend_position,
        showlegend=showlegend,
        hovermode="closest",
        margin=margin,
        xaxis=dict(showgrid=showgrid,
                   # Keep aspect ratio
                   scaleanchor='y', scaleratio=1),
        yaxis=dict(showgrid=showgrid),
        xaxis_visible = show_axis, yaxis_visible = show_axis,
        plot_bgcolor=plot_bgcolor
    )
    
    
    # Create the Figure and add traces
    fig = go.Figure(data=[*edge_traces, node_trace], layout=layout)
    
    # ----------------------------------------------------------------------------------
    # ------------------------------- Legend Labels ------------------------------------
    # ----------------------------------------------------------------------------------

    # Extract the colors (col) and label text (meaning: mng) from the combined graph
    set_vertex_colors_meanings = set([(col, mng) for col, mng in zip(graph.vs["color"], graph.vs["meaning"])])
    
        # Add labels for vertex dynamics
    for col, mng in set_vertex_colors_meanings:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(symbol='circle', size=node_size, color=col),
            name=mng,
            showlegend=True
            ))
        
    
    set_edges_meanings = set([mng for mng in  graph.es["dynamics"]])

    # Add labels for edges dynamics
    for mng in set_edges_meanings:
        mng_color_hex  = classification_df.query(f'Classification == "{mng}"')["Color_hex"].iloc[0]
        mng_oscillates = classification_df.query(f'Classification == "{mng}"')["Edge_oscillates"].iloc[0]
        mng_linetype   = classification_df.query(f'Classification == "{mng}"')["Line_type"].iloc[0]
        if mng_oscillates:
            oscillation_tag = " (~)"
        else:
            oscillation_tag = ""
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color = mng_color_hex,
                      width = default_edge_width,
                      dash  = mng_linetype),
            name = mng + oscillation_tag,
            showlegend = True
        ))


    # Add cutoff labels
    if add_cutoff_legend:
        
        # Add empty space between labels
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(width=0),
            name= "",
            showlegend=True
            ))
        
        for cutoff_label, value in graph["cutoffs_dict"].items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(width=0),
                name= cutoff_label + " = " + str(value),
                showlegend=True
                ))
    
    # -----------------------------------------------------------------------------
        
    if plot_graph: plot(fig)
    
    # Save the plot?
    if save_html is not None:
        fig.write_html(save_html)
    
    return fig
