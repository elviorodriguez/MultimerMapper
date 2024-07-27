
import pandas as pd
import igraph

# -------------------------------------------------------------------------------------
# --------------------------- Classification df import --------------------------------
# -------------------------------------------------------------------------------------

def read_classification_df(path: str = "cfg/interaction_classification.tsv"):
    classification_df = pd.read_csv(path, sep= "\t")
    return classification_df

classification_df = read_classification_df()

# -------------------------------------------------------------------------------------
# ------------------------------- Classify the edge -----------------------------------
# -------------------------------------------------------------------------------------σ



def classify_edge_dynamics(edge: tuple,
                           graph_2mers  : igraph.Graph,
                           graph_Nmers  : igraph.Graph,

                           sorted_edges_2mers_graph  : list[tuple], 
                           sorted_edges_Nmers_graph  : list[tuple],
                           sorted_edges_Comb_graph   : list[tuple],
                           tested_Nmers_edges_sorted : list[tuple],

                           classification_df: pd.DataFrame = classification_df):
    
    # 
    is_present_in_2mers = edge in sorted_edges_2mers_graph
    is_present_in_Nmers = edge in sorted_edges_Nmers_graph
    was_tested_in_Nmers = edge in tested_Nmers_edges_sorted

    e_dynamics = classification_df.query('')

    return e_dynamics

    # Shared by both graphs
    if edge in sorted_edges_2mers_graph and edge in sorted_edges_Nmers_graph:
        edge_colors.append(edge_color_both)

    # Edges only in 2-mers
    elif edge in edges_g1_sort and edge not in edges_g2_sort:
        
        # but not tested in N-mers
        if edge not in tested_Nmers_edges_sorted:
            edge_colors.append(edge_color3)
            dynamic_interactions = pd.concat([dynamic_interactions,
                                            pd.DataFrame({"protein1": [edge[0]],
                                                            "protein2": [edge[1]],
                                                            "only_in": ["2mers-but_not_tested_in_Nmers"]})
                                            ], ignore_index = True)
            
        # Edges only in 2-mers
        else:
            edge_colors.append(edge_color1)
            dynamic_interactions = pd.concat([dynamic_interactions,
                                            pd.DataFrame({"protein1": [edge[0]],
                                                            "protein2": [edge[1]],
                                                            "only_in": ["2mers"]})
                                            ], ignore_index = True)
                
    # Edges only in N-mers
    elif edge not in edges_g1_sort and edge in edges_g2_sort:
        edge_colors.append(edge_color2)
        dynamic_interactions = pd.concat([dynamic_interactions,
                                        pd.DataFrame({"protein1": [edge[0]],
                                                        "protein2": [edge[1]],
                                                        "only_in": ["Nmers"]})
                                        ], ignore_index = True)
    
    
    


# -------------------------------------------------------------------------------------
# -------------------- Getters based on dynamic classification ------------------------
# -------------------------------------------------------------------------------------

def get_edge_color(graph_edge: igraph.Edge, classification_df: pd.DataFrame):
    pass

def get_edge_linetype(graph_edge: igraph.Edge, classification_df: pd.DataFrame):
    pass

def get_edge_weight(graph_edge: igraph.Edge, classification_df: pd.DataFrame):
    pass


# Extract edge attributes. If they are not able, set them to a default value
# try:
#     edge_colors = graph.es["color"]
# except:
#     edge_colors = len(graph.get_edgelist()) * ["black"]
#     graph.es["color"] = len(graph.get_edgelist()) * ["black"]
# try:
#     graph.es["meaning"]
# except:
#     graph.es["meaning"] = len(graph.get_edgelist()) * ["Interactions"]
# try:
#     graph.es["N_mers_info"]
# except:
#     graph.es["N_mers_info"] = len(graph.get_edgelist()) * [""]
# try:
#     graph.es["2_mers_info"]
# except:
#     graph.es["2_mers_info"] = len(graph.get_edgelist()) * [""]


# # Modify the edge representation depending on the "meaning" >>>>>>>>>>>

# if ("Dynamic " in edge["meaning"] or "Indirect " in edge["meaning"]) and use_dot_dynamic_edges:
#     edge_linetype = "dot"
#     edge_weight = 0.5
    
# if edge["meaning"] == 'Static interaction' or edge["meaning"] == "Predominantly static interaction":
#     edge_weight = int(np.mean(list(edge["2_mers_data"]["N_models"]) + list(edge["N_mers_data"]["N_models"])) *\
#                         np.mean(list(edge["2_mers_data"]["ipTM"]) + list(edge["N_mers_data"]["ipTM"])) *\
#                         (1/ np.mean(list(edge["2_mers_data"]["min_PAE"]) + list(edge["N_mers_data"]["min_PAE"]))))
#     if edge_weight < 1:
#         edge_weight = 1
        
# elif "(appears in N-mers)" in edge["meaning"] or "Ambiguous Dynamic" in edge["meaning"]:
#     edge_weight = 1
    
# if edge["meaning"] == "Predominantly static interaction":
#     edge_linetype = "solid"

# # Modify the edge representation depending on the "meaning" <<<<<<<<<<<