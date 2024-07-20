from src.input_check import seq_input_from_fasta, extract_seqs_from_AF2_PDBs, merge_fasta_with_PDB_data, logger

# -----------------------------------------------------------------------------
# ---------------- Input checking functions and wrapper -----------------------
# -----------------------------------------------------------------------------

# Wrapper for previous functions
def parse_AF2_and_sequences(fasta_file_path, AF2_2mers, AF2_Nmers = None, use_names = True,
                            graph_resolution = 0.075, auto_domain_detection = True,
                            graph_resolution_preset = None, save_preset = False,
                            save_PAE_png = True, save_ref_structures = True,display_PAE_domains = False, show_structures = True,
                            display_PAE_domains_inline = True, save_domains_html = True, save_domains_tsv = True,
                            # 2-mers cutoffs
                            min_PAE_cutoff_2mers = 4.5, ipTM_cutoff_2mers = 0.4,
                            # N-mers cutoffs
                            min_PAE_cutoff_Nmers = 2, pDockQ_cutoff_Nmers = 0.15,
                            # General cutoff
                            N_models_cutoff = 3, pdockq_indirect_interaction_cutoff = 0.23,
                            # For RMSD calculations
                            domain_RMSD_plddt_cutoff = 60, trimming_RMSD_plddt_cutoff = 70, predominantly_static_cutoff = 0.6,
                            # You can customize the edges and vertex colors of combined graph
                            edge_color1='red', edge_color2='green', edge_color3 = 'orange', edge_color4 = 'purple', edge_color5 = "pink",
                            edge_color6='blue',  edge_color_both='black',
                            vertex_color1='red', vertex_color2='green', vertex_color3='orange', vertex_color_both='gray',
                            remove_indirect_interactions = True):
    
    # ###### Check input data
    # fasta_file_path
    
    prot_IDs, prot_names, prot_seqs, prot_lens, prot_N = seq_input_from_fasta(fasta_file_path, use_names = use_names)
    
    # Work with names?
    if use_names:
        # Switch IDs with names
        prot_IDs_backup = prot_IDs
        prot_IDs = prot_names
        prot_names = prot_IDs_backup
    
    all_pdb_data = extract_seqs_from_AF2_PDBs(AF2_2mers, AF2_Nmers)
    
    merge_fasta_with_PDB_data(all_pdb_data, prot_IDs, prot_seqs)
    
    sliced_PAE_and_pLDDTs = extract_AF2_metrics_from_JSON(all_pdb_data, fasta_file_path)
    
    domains_df = detect_domains(sliced_PAE_and_pLDDTs, fasta_file_path, graph_resolution = graph_resolution,
                                auto_domain_detection = auto_domain_detection,
                                graph_resolution_preset = graph_resolution_preset, save_preset = save_preset,
                                save_png_file = save_PAE_png, show_image = display_PAE_domains,
                                show_inline = display_PAE_domains_inline, show_structure = show_structures,
                                save_html = save_domains_html, save_tsv = save_domains_tsv)
        
    pairwise_2mers_df = generate_pairwise_2mers_df(all_pdb_data)
    
    pairwise_2mers_df_F3, unique_proteins = filter_non_interactions(pairwise_2mers_df,
                                                              min_PAE_cutoff = min_PAE_cutoff_2mers,
                                                              ipTM_cutoff = ipTM_cutoff_2mers,
                                                              N_models_cutoff = N_models_cutoff)
        
    graph_2mers = generate_full_graph_2mers(pairwise_2mers_df_F3, directory_path = "./2D_graphs")
    
    if AF2_Nmers != None:
        pairwise_Nmers_df = generate_pairwise_Nmers_df(all_pdb_data, is_debug = False)
        pairwise_Nmers_df_F3, unique_Nmers_proteins = filter_pairwise_Nmers_df(pairwise_Nmers_df,
                                                                               min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers,
                                                                               pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers,
                                                                               N_models_cutoff = N_models_cutoff)
        graph_Nmers = generate_full_graph_Nmers(pairwise_Nmers_df_F3)
        
        combined_graph, dynamic_proteins, dynamic_interactions =\
            compare_and_plot_graphs(graph_2mers, graph_Nmers, pairwise_2mers_df, pairwise_Nmers_df, domains_df, sliced_PAE_and_pLDDTs,
                                    # Prot_IDs and names to add them to the graph
                                    prot_IDs = prot_IDs, prot_names = prot_names,
                                    # 2-mers cutoffs
                                    min_PAE_cutoff_2mers = min_PAE_cutoff_2mers, ipTM_cutoff_2mers = ipTM_cutoff_2mers,
                                    # N-mers cutoffs
                                    min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers, pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers,
                                    # General cutoff
                                    N_models_cutoff = N_models_cutoff, 
                                    # For RMSD calculations
                                    domain_RMSD_plddt_cutoff = domain_RMSD_plddt_cutoff, trimming_RMSD_plddt_cutoff = trimming_RMSD_plddt_cutoff,
                                    # Style options
                                    edge_color1=edge_color1, edge_color2=edge_color2, edge_color3=edge_color3, edge_color4 = edge_color4,
                                    edge_color5 = edge_color5, edge_color6 = edge_color6, edge_color_both=edge_color_both,
                                    vertex_color1=vertex_color1, vertex_color2=vertex_color2, vertex_color3=vertex_color3, vertex_color_both=vertex_color_both,
                                    is_debug = False, pdockq_indirect_interaction_cutoff=pdockq_indirect_interaction_cutoff, predominantly_static_cutoff=predominantly_static_cutoff,
                                    remove_indirect_interactions=remove_indirect_interactions)

        
    
    fully_connected_subgraphs = find_sub_graphs(graph_2mers, directory_path = "./2D_graphs")
    
    fully_connected_subgraphs_pairwise_2mers_dfs = get_fully_connected_subgraphs_pairwise_2mers_dfs(pairwise_2mers_df_F3, fully_connected_subgraphs)
    
    
    # Save reference monomers?
    if save_ref_structures:
        
        # Create a folder named "PDB_ref_monomers" if it doesn't exist
        save_folder = "PDB_ref_monomers"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # Save each reference monomer chain
        for protein_ID in sliced_PAE_and_pLDDTs.keys():
            # Create a PDBIO instance
            pdbio = PDBIO()
                        # Set the structure to the Model
            pdbio.set_structure(sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"])
            # Save the Model to a PDB file
            output_pdb_file = save_folder + f"/{protein_ID}_ref.pdb"
            pdbio.save(output_pdb_file)
            
    
    # If AF2_Nmers were included, the returned object will be different
    if AF2_Nmers != None:
        return (all_pdb_data, sliced_PAE_and_pLDDTs, domains_df, #my_dataframe_with_combinations,
                # 2-mers data objects 
                pairwise_2mers_df, pairwise_2mers_df_F3, graph_2mers,
                fully_connected_subgraphs, fully_connected_subgraphs_pairwise_2mers_dfs,
                # N-mers data objects
                pairwise_Nmers_df, pairwise_Nmers_df_F3, graph_Nmers, combined_graph,
                dynamic_proteins, dynamic_interactions)
    

    
    return (all_pdb_data, sliced_PAE_and_pLDDTs, domains_df, #y_dataframe_with_combinations,
            # 2-mers data objects 
            pairwise_2mers_df, pairwise_2mers_df_F3, graph_2mers,
            fully_connected_subgraphs, fully_connected_subgraphs_pairwise_2mers_dfs)



# DEBUGGGGGGGGGGGGGGGGGGGGGGGG ------------------------------------------------

# prot_IDs, prot_names, prot_seqs, prot_len, prot_N, Q_values = seq_input_from_fasta(fasta_file_path, use_names = use_names)

# all_pdb_data = extract_seqs_from_AF2_PDBs(AF2_2mers, AF2_Nmers)

# merge_fasta_with_PDB_data(all_pdb_data, prot_IDs, prot_seqs, Q_values)

# sliced_PAE_and_pLDDTs = extract_AF2_metrics_from_JSON(all_pdb_data, fasta_file_path)

# # Debug plotting


# fig1 = plot_domains(protein_ID = "BDF6",
#               matrix_data = sliced_PAE_and_pLDDTs["BDF6"]["best_PAE_matrix"],
#               positions = sliced_PAE_and_pLDDTs["BDF6"]["ref_domain_clusters"][0],
#               colors = sliced_PAE_and_pLDDTs["BDF6"]["ref_domain_clusters"][1],
#               custom_title = None, out_folder = 'domains', save_plot = False, show_plot = False)

# fig2 = plot_domains(protein_ID = "BDF6",
#               matrix_data = sliced_PAE_and_pLDDTs["BDF6"]["best_PAE_matrix"],
#               positions = sliced_PAE_and_pLDDTs["BDF6"]["ref_domain_clusters"][0],
#               colors = sliced_PAE_and_pLDDTs["BDF6"]["ref_domain_clusters"][1],
#               custom_title = None, out_folder = 'domains', save_plot = False, show_plot = False)


# # combine_figures_and_plot(fig1, fig2, protein_ID = "BDF6", save_file = True, show_image = True)

# detect_domains(sliced_PAE_and_pLDDTs, fasta_file_path, graph_resolution = graph_resolution,
#                 auto_domain_detection = False, graph_resolution_preset = None, save_preset = True,
#                 save_png_file = True, show_image = False, show_structure = True, show_inline = True, save_html = True)

# DEBUGGGGGGGGGGGGGGGGGGGGGGGG ------------------------------------------------