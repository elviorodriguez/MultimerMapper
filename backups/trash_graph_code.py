
# ------------------------------------------------------------------------------
# ----------------------- These functions are crap! ----------------------------
# ------------------------------------------------------------------------------

# Functions to add meaning column to vertex and edges
def add_edges_meaning(graph, edge_color1='red', edge_color2='green', edge_color3 = 'orange', edge_color_both='black'):
    
    # Function to determine the meaning based on color
    def get_meaning(row):
        if row['color'] == edge_color_both:
            return 'Static interaction'
        elif row['color'] == edge_color1:
            return 'Dynamic interaction (disappears in N-mers)'
        elif row['color'] == edge_color2:
            return 'Dynamic interaction (appears in N-mers)'
        elif row['color'] == edge_color3:
            return 'Interaction dynamics not explored in N-mers'
        else:
            return 'unknown'
        
    edge_df = graph.get_edge_dataframe()
    # Apply the function to create the new 'meaning' column
    edge_df['meaning'] = edge_df.apply(get_meaning, axis=1)
    
    graph.es["meaning"] = edge_df['meaning']
    




                
            
def modify_ambiguous_Nmers_edges(graph, edge_color4, edge_color6, N_models_cutoff, fraction_cutoff = 0.5):
    for edge in graph.es:
        all_are_bigger = all(list(edge["N_mers_data"]["N_models"] >= N_models_cutoff))
        all_are_smaller = all(list(edge["N_mers_data"]["N_models"] < N_models_cutoff))
        if not (all_are_bigger or all_are_smaller):
            edge["meaning"] = "Ambiguous Dynamic (In some N-mers appear and in others disappear)"
            edge["color"] = edge_color4
            
            # Also check if there is litt
            total_models = len(list(edge["N_mers_data"]["N_models"]))
            models_that_surpass_cutoffs = sum(edge["N_mers_data"]["N_models"] >= N_models_cutoff)
            
            if models_that_surpass_cutoffs / total_models >= fraction_cutoff:
                edge["meaning"] = "Predominantly static interaction"
                edge["color"] = edge_color6
        
        
def modify_indirect_interaction_edges(graph, edge_color5, pdockq_indirect_interaction_cutoff = 0.23, remove_indirect_interactions=True):
    
    # To remove indirect interaction edges
    edges_to_remove = []
    
    for edge in graph.es:
        if edge["meaning"] == 'Dynamic interaction (appears in N-mers)' or edge["2_mers_info"] == "No rank surpass cutoff" or edge["2_mers_data"].query(f'N_models >= {N_models_cutoff}').empty:
            if np.mean(edge["N_mers_data"].query(f'N_models >= {N_models_cutoff}')["pDockQ"]) < pdockq_indirect_interaction_cutoff:
                edge["meaning"] = 'Indirect interaction'
                edge["color"] = edge_color5
                edges_to_remove.append(edge.index)
    
    if remove_indirect_interactions:
        # Remove edges from the graph
        graph.delete_edges(edges_to_remove)








# Add edges and its colors ------------------------------------------------

edges_dynamics = []
for edge in edges_gC_sort:
    # Shared by both graphs
    if edge in edges_g1_sort and edge in edges_g2_sort:
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
        
    # This if something happens
    else:
        if is_debug: print("And This???:", edge)
        raise ValueError("For some reason an edge that comes from the graphs to compare is not in either graphs...")






add_edges_meaning(graphC, edge_color1, edge_color2, edge_color3, edge_color_both)
modify_ambiguous_Nmers_edges(graphC, edge_color4, edge_color6, N_models_cutoff, fraction_cutoff=predominantly_static_cutoff)
modify_indirect_interaction_edges(graphC, edge_color5,
                                    pdockq_indirect_interaction_cutoff = pdockq_indirect_interaction_cutoff,
                                    remove_indirect_interactions = remove_indirect_interactions)




