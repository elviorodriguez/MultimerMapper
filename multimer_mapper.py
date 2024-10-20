
import os
import sys


try:
    # Get multimer_mapper installation path and current path
    mm_path = os.path.dirname(os.path.realpath(__file__))
    wd_path = os.getcwd()

    # Set the working dir as the path to MultimerMapper repo 
    os.chdir(mm_path)

    from cfg.default_settings import *
    from utils.logger_setup import configure_logger
    from src.input_check import seq_input_from_fasta, extract_seqs_from_AF2_PDBs, merge_fasta_with_PDB_data, check_graph_resolution_preset
    from src.metrics_extractor import extract_AF2_metrics_from_JSON, generate_pairwise_2mers_df, generate_pairwise_Nmers_df
    from src.detect_domains import detect_domains
    from src.ppi_detector import filter_non_int_2mers_df, filter_non_int_Nmers_df
    from src.ppi_graphs import generate_2mers_graph, generate_Nmers_graph, generate_combined_graph, igraph_to_plotly
    from src.coordinate_analyzer import generate_RMSF_pLDDT_cluster_and_RMSD_trajectories
    from utils.temp_files_manager import setup_temp_file
    from utils.combinations import suggest_combinations
    from src.contact_extractor import compute_pairwise_contacts, visualize_pair_matrices
    from src.analyze_multivalency import cluster_all_pairs, add_cluster_contribution_by_dataset
    from src.fallback import analyze_fallback
    from src.contact_graph import Residue, Surface, Protein, PPI, Network
    from src.stoichiometries import stoichiometric_space_exploration_pipeline

    # These are for interactive usage
    from traj.pairwise_rmsd_trajectories import generate_pairwise_domain_trajectories, generate_pairwise_domain_trajectory_in_context

    # Get back the working dir as the path to MultimerMapper repo 
    os.chdir(wd_path)

except ModuleNotFoundError as e:

    print()
    print(f"The following exception was encountered when loading MultimerMapper package:")
    print(f"   - Exception: {e}")
    print(f"   - Make sure you have installed and activated MultimerMapper environment")
    print()
    print( "To install it:")
    print( "   $ conda env create -f MultimerMapper/environment.yml")
    print()
    print( "To activate it:")
    print( "   $ conda activate MultimerMapper")
    print()

    sys.exit()

# Main MultimerMapper pipeline
def parse_AF2_and_sequences(
    
    # Input
    fasta_file: str, AF2_2mers: str,
    AF2_Nmers: str | None = None,
    out_path: str | None = None,
    manual_domains: str | None = None,
    
    # Options imported from cfg.default_settings or cfg.custom_settings
    use_names = use_names,
    graph_resolution = graph_resolution, auto_domain_detection = auto_domain_detection,
    graph_resolution_preset = graph_resolution_preset, save_preset = save_preset,
    save_PAE_png = save_PAE_png, save_ref_structures = save_ref_structures,
    display_PAE_domains = display_PAE_domains, show_monomer_structures = show_monomer_structures,
    display_PAE_domains_inline = display_PAE_domains_inline, save_domains_html = save_domains_html,
    save_domains_tsv = save_domains_tsv,
    show_PAE_along_backbone = show_PAE_along_backbone,
    display_contact_clusters = display_contact_clusters,
    save_contact_clusters = save_contact_clusters,

    # 2-mers cutoffs
    min_PAE_cutoff_2mers = min_PAE_cutoff_2mers, ipTM_cutoff_2mers = ipTM_cutoff_2mers,

    # N-mers cutoffs
    min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers, pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers,

    # General cutoff
    N_models_cutoff = N_models_cutoff,

    # Contacts cutoffs
    contact_distance_cutoff = contact_distance_cutoff,
    contact_PAE_cutoff      = contact_PAE_cutoff,
    contact_pLDDT_cutoff    = contact_pLDDT_cutoff,

    # For multivalency detection (contact clustering)
    multivalency_silhouette_threshold         = multivalency_silhouette_threshold,
    multivalency_contact_similarity_threshold = multivalency_contact_similarity_threshold,
    max_contact_clusters                      = max_contact_clusters,    
    contact_fraction_threshold                = contact_fraction_threshold,
    mc_threshold                              = mc_threshold,
    use_median                                = use_median,
    refinement_contact_similarity_threshold   = refinement_contact_similarity_threshold,
    refinement_cf_threshold                   = refinement_cf_threshold,

    # For RMSD calculations
    domain_RMSD_plddt_cutoff = domain_RMSD_plddt_cutoff,
    trimming_RMSD_plddt_cutoff = trimming_RMSD_plddt_cutoff,

    # Edges and vertex colors of combined graph (see cfg/default_settings module for their meaning)
    vertex_color1=vertex_color1, vertex_color2=vertex_color2, vertex_color3=vertex_color3, vertex_color_both=vertex_color_both,
    
    # Set it to "error" to reduce verbosity
    log_level: str = log_level, overwrite = overwrite):

    # --------------------------------------------------------------------------
    # -------------- Initialize the logger and initial setup -------------------
    # --------------------------------------------------------------------------
    
    logger = configure_logger(out_path, log_level = log_level)(__name__)

    # Create a tmp empty dir that erase itself when the program exits
    if AF2_Nmers is None:
        from utils.temp_files_manager import setup_temp_dir
        AF2_Nmers = setup_temp_dir(logger)

    # The default out_path will contain the basename of the fasta file
    if out_path is None:
        out_path = default_out_path + str(os.path.splitext(os.path.basename(fasta_file))[0])

    # Sometimes plotly leaves a tmp HTML file. This removes it at the end of the execution
    setup_temp_file(logger = logger, file_path= 'temp-plot.html')
    
    # --------------------------------------------------------------------------
    # -------------------- Input verification and merging ----------------------
    # --------------------------------------------------------------------------

    # FASTA file processing
    prot_IDs, prot_names, prot_seqs, prot_lens, prot_N = seq_input_from_fasta(
        fasta_file, use_names, logger = logger)

    # PDB files processing
    all_pdb_data = extract_seqs_from_AF2_PDBs(AF2_2mers, AF2_Nmers, logger = logger)

    # Combine data
    merge_fasta_with_PDB_data(all_pdb_data = all_pdb_data,
                              prot_IDs = prot_IDs,
                              prot_names = prot_names, 
                              prot_seqs = prot_seqs,
                              prot_lens = prot_lens,
                              prot_N = prot_N,
                              use_names = use_names,
                              logger = logger)
    
    # Verify the graph resolution preset
    if graph_resolution_preset is not None:
        graph_resolution_preset = check_graph_resolution_preset(graph_resolution_preset,
                                                                prot_IDs,
                                                                logger)
    
    # --------------------------------------------------------------------------
    # -------------------------- Metrics extraction ----------------------------
    # --------------------------------------------------------------------------

    # Extract AF2 metrics
    sliced_PAE_and_pLDDTs = extract_AF2_metrics_from_JSON(all_pdb_data, fasta_file, out_path, overwrite = overwrite, logger = logger)

    # Get pairwise data for 2-mers and N-mers
    pairwise_2mers_df = generate_pairwise_2mers_df(all_pdb_data, out_path = out_path, save_pairwise_data = save_pairwise_data, 
                                                overwrite = overwrite, logger = logger)
    pairwise_Nmers_df = generate_pairwise_Nmers_df(all_pdb_data, out_path = out_path, save_pairwise_data = save_pairwise_data, 
                                                overwrite = overwrite, logger = logger)
    
    # Save reference monomers?
    if save_ref_structures:

        logger.info("Saving PDB reference monomer structures...")

        from src.save_ref_pdbs import save_reference_pdbs

        save_reference_pdbs(sliced_PAE_and_pLDDTs = sliced_PAE_and_pLDDTs,
                            out_path = out_path,
                            overwrite = overwrite)    

    # --------------------------------------------------------------------------
    # --------------------------- Domain detection -----------------------------
    # --------------------------------------------------------------------------

    # Domain detection
    domains_df = detect_domains(
        sliced_PAE_and_pLDDTs, fasta_file, graph_resolution = graph_resolution,
        auto_domain_detection = auto_domain_detection,
        graph_resolution_preset = graph_resolution_preset, save_preset = save_preset,
        save_png_file = save_PAE_png, show_image = display_PAE_domains,
        show_inline = display_PAE_domains_inline, show_structure = show_monomer_structures,
        save_html = save_domains_html, save_tsv = save_domains_tsv,
        out_path = out_path, overwrite = True, log_level = log_level, manual_domains = manual_domains,
        show_PAE_along_backbone = show_PAE_along_backbone)
    
    # Progress
    logger.info(f"Resulting domains:\n{domains_df}")


    # --------------------------------------------------------------------------
    # ----------------------------- PPI detection ------------------------------
    # --------------------------------------------------------------------------

    # Progress
    logger.info("Detecting PPI using cutoffs...")

    # For 2-mers
    pairwise_2mers_df_F3, unique_2mers_proteins = filter_non_int_2mers_df(
        pairwise_2mers_df, 
        min_PAE_cutoff = min_PAE_cutoff_2mers,
        ipTM_cutoff = ipTM_cutoff_2mers,
        N_models_cutoff = N_models_cutoff)

    # For N-mers
    pairwise_Nmers_df_F3, unique_Nmers_proteins = filter_non_int_Nmers_df(
        pairwise_Nmers_df,
        min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers,
        pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers,
        N_models_cutoff = N_models_cutoff)
    
    # Progress
    logger.info("Resulting interactions:")
    logger.info(f"   - 2-mers proteins: {unique_2mers_proteins}")
    logger.info(f"   - 2-mers PPIs:\n{pairwise_2mers_df_F3}")
    logger.info(f"   - N-mers proteins: {unique_Nmers_proteins}")
    logger.info(f"   - N-mers PPIs:\n{pairwise_Nmers_df_F3}")

    multimer_mapper_output = {
        "prot_IDs": prot_IDs,
        "prot_names": prot_names,
        "prot_seqs": prot_seqs,
        "prot_lens": prot_lens,
        "prot_N": prot_N,
        "out_path": out_path,
        "all_pdb_data": all_pdb_data,
        "sliced_PAE_and_pLDDTs": sliced_PAE_and_pLDDTs,
        "domains_df": domains_df,
        "pairwise_2mers_df": pairwise_2mers_df,
        "pairwise_Nmers_df": pairwise_Nmers_df,
        "pairwise_2mers_df_F3": pairwise_2mers_df_F3,
        "pairwise_Nmers_df_F3": pairwise_Nmers_df_F3,
        "unique_2mers_proteins": unique_2mers_proteins,
        "unique_Nmers_proteins": unique_Nmers_proteins
    }

    # --------------------------------------------------------------------------
    # ------------------ Contacts detection and clustering ---------------------
    # --------------------------------------------------------------------------
        
    # Compute contacts and add it to output
    pairwise_contact_matrices = compute_pairwise_contacts(multimer_mapper_output,
                                                          out_path = out_path,
                                                          contact_distance_cutoff = contact_distance_cutoff,
                                                          contact_PAE_cutoff      = contact_PAE_cutoff,
                                                          contact_pLDDT_cutoff    = contact_pLDDT_cutoff,
                                                          )
    multimer_mapper_output["pairwise_contact_matrices"] = pairwise_contact_matrices

    # Cluster contacts (extract valency) and add it to output
    contacts_clusters = cluster_all_pairs(pairwise_contact_matrices, 
                                          multimer_mapper_output,
                                          contacts_clustering_method                = contacts_clustering_method,
                                          max_clusters                              = max_contact_clusters,
                                          silhouette_threshold                      = multivalency_silhouette_threshold,
                                          contact_similarity_threshold              = multivalency_contact_similarity_threshold,
                                          contact_fraction_threshold                = contact_fraction_threshold,
                                          mc_threshold                              = mc_threshold,
                                          use_median                                = use_median,
                                          refinement_contact_similarity_threshold   = refinement_contact_similarity_threshold,
                                          refinement_cf_threshold                   = refinement_cf_threshold,
                                          show_plot = display_contact_clusters,
                                          save_plot = save_contact_clusters,
                                          log_level = log_level)    
    multimer_mapper_output["contacts_clusters"] = contacts_clusters

    # Add cluster contribution by 2/N-mers dataset
    add_cluster_contribution_by_dataset(multimer_mapper_output)

    # --------------------------------------------------------------------------
    # ----------------------- Detect symmetry fallbacks ------------------------
    # --------------------------------------------------------------------------

    symmetry_fallbacks, symmetry_fallbacks_df = analyze_fallback(mm_output = multimer_mapper_output,
                                                                 low_fraction            = fallback_low_fraction,
                                                                 up_fraction             = fallback_up_fraction,
                                                                 save_figs               = save_fallback_plots,
                                                                 figsize                 = fallback_plot_figsize,
                                                                 dpi                     = fallback_plot_dpi,
                                                                 save_dataframes         = save_fallback_df,
                                                                 display_fallback_ranges = display_fallback_ranges,
                                                                 log_level               = log_level)
    
    # Add data to output
    multimer_mapper_output['symmetry_fallbacks']    = symmetry_fallbacks
    multimer_mapper_output['symmetry_fallbacks_df'] = symmetry_fallbacks_df

    # --------------------------------------------------------------------------
    # ----------------------- 2D PPI graph generation --------------------------
    # --------------------------------------------------------------------------

    # Progress
    logger.info("Converting PPIs to graphs...")

    # 2-mers PPI graph
    graph_2mers = generate_2mers_graph(pairwise_2mers_df_F3 = pairwise_2mers_df_F3,
                                        out_path = out_path,
                                        overwrite = True)
    
    # Debug
    logger.debug(f"Resulting 2-mers graph:\n{graph_2mers}")

    # N-mers PPI graph
    graph_Nmers = generate_Nmers_graph(pairwise_Nmers_df_F3 = pairwise_Nmers_df_F3,
                                        out_path = out_path,
                                        overwrite = True)
    
    # Add results to output dict
    multimer_mapper_output["graph_2mers"] = graph_2mers
    multimer_mapper_output["graph_Nmers"] = graph_Nmers

    # Debug
    logger.debug(f"Resulting N-mers graph:\n{graph_Nmers}")

    # Combined PPI graph
    combined_graph, dynamic_proteins, homooligomerization_states, multivalency_states = generate_combined_graph(
        
        # Input
        mm_output = multimer_mapper_output,
        
        # 2-mers cutoffs
        min_PAE_cutoff_2mers = min_PAE_cutoff_2mers, ipTM_cutoff_2mers = ipTM_cutoff_2mers,
        
        # N-mers cutoffs
        min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers, pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers,
        
        # General cutoffs
        N_models_cutoff = N_models_cutoff,

        # For RMSD calculations
        domain_RMSD_plddt_cutoff = domain_RMSD_plddt_cutoff,
        trimming_RMSD_plddt_cutoff = trimming_RMSD_plddt_cutoff,

        # Style options (see cfg/default_settings module for their meaning)
        vertex_color1=vertex_color1, vertex_color2=vertex_color2, vertex_color3=vertex_color3, vertex_color_both=vertex_color_both)
    
    # Debug
    logger.debug(f"Resulting combined graph:\n{combined_graph}")
    logger.debug(f"Dynamic proteins:\n{dynamic_proteins}")
    # logger.debug(f"Dynamic interactions:\n{dynamic_interactions}")

    # Add combined graph output
    multimer_mapper_output["combined_graph"]             = combined_graph
    multimer_mapper_output["dynamic_proteins"]           = dynamic_proteins
    multimer_mapper_output["homooligomerization_states"] = homooligomerization_states
    multimer_mapper_output["multivalency_states"]        = multivalency_states


    return multimer_mapper_output


def interactive_igraph_to_plotly(combined_graph,
                                 out_path: str,
                                 log_level = log_level,
                                 # Aspect of homooligomerization edges
                                 self_loop_orientation = self_loop_orientation,
                                 self_loop_size = self_loop_size,
                                 remove_interactions = remove_interactions_from_ppi_graph,
                                 layout_algorithm = ppi_graph_layout_algorithm,
                                 automatic_true = igraph_to_plotly_automatic_true):

    # Initialize the logger
    logger = configure_logger(out_path, log_level = log_level)(__name__)

    # Convert combined PPI graph to interactive plotly
    save_html = out_path + "/2D_graph.html"
    
    while True:

        logger.info('INITIALIZING: converting igraph combined graph to interactive PPI graph...')

        combined_graph_interactive = igraph_to_plotly(

            # Input
            graph = combined_graph,
            layout = layout_algorithm,

            # Aspect of homooligomerization edges
            self_loop_orientation = self_loop_orientation,
            self_loop_size = self_loop_size,

            # Keep background axis and grid? (not recommended)
            show_axis = show_axis, showgrid = showgrid,
            
            # Set protein names as bold?
            use_bold_protein_names = use_bold_protein_names,

            # Domain RMSDs bigger than this value will be highlighted in bold in the nodes hovertext
            add_bold_RMSD_cutoff = 5,

            # Save the plot as HTML to specific file
            save_html = save_html, 

            # Add cutoff values to the legends?
            add_cutoff_legend = add_cutoff_legend,

            # Discards interactions types from this list
            remove_interactions = remove_interactions,

            # Logger
            logger = logger
        )
        
        if automatic_true:
            logger.info("   Enjoy your interactive PPI graph!")
            return combined_graph_interactive
        
        logger.info('   Default layout generation algorithm is Fruchterman-Reingold ("fr")')
        logger.info('   This algorithm is stochastic. Try several layouts and save the one you like.')
        logger.info('   If you have changed it to a non stochastic algorithm, it will not be modified.')
        logger.info('   You can set automatic_true = True to skip the prompting.')

        while True:
            prompt = "   Do you like the plot? (y/n): "
            logger.info(prompt)
            user_input = input().strip().lower()
            logger.info(f'   Answer: {user_input}')
            
            if user_input == 'y':
                logger.info("   Great! Enjoy your interactive PPI graph.")
                return combined_graph_interactive
            elif user_input == 'n':
                logger.info("   Generating a new PPI graph layout and graph...")
                break
            else:
                logger.info("   Invalid input. Please enter 'y' or 'n'.")

def interactive_igraph_to_py3dmol(combined_graph, logger, automatic_true = False):

    # Create 3D network
    nw = Network(combined_graph, logger = logger)

    # Generate 3D network visualization
    while True:

        nw.generate_layout()
        nw.generate_py3dmol_plot(save_path = out_path + '/3D_graph_py3Dmol.html', show_plot = True)
        nw.generate_plotly_3d_plot(save_path = out_path + '/3D_graph_Plotly.html', show_plot = True)

        logger.info("Some 3D layout generation algorithms are stochastic:")
        logger.info("   - Do you like the plot? (y/n): ")
        user_input = input().strip().lower()

        if automatic_true:
            logger.info("   - Automatic True: Enjoy your interactive 3D plot!")
            break

        if user_input == "y":
            logger.info(f"   - User response: {user_input} -> Great! Enjoy your interactive 3D plot!")
            break
        elif user_input == "n":
            logger.info(f"   - User response: {user_input} -> OK. Here we go again!")
        else:
            logger.info(f"   - Unknown response: {user_input} -> Generating a new layout...")

    return nw


###############################################################################
# --------------------------------------------------------------------------- #
# ------------------------- Main: Command line call ------------------------- #
# --------------------------------------------------------------------------- #
###############################################################################

if __name__ == "__main__":

    import argparse

    # --------------------------------------------------------------------------
    # ----------------------- Usage function definition ------------------------
    # --------------------------------------------------------------------------
    
    parser = argparse.ArgumentParser(
        description='MultimerMapper pipeline for extracting and analyzing AF2-multimer landscapes.')
    
    parser.add_argument('fasta_file', type = str,
        help='Path to the input FASTA file')
    
    parser.add_argument('--AF_2mers', type = str, default= None,
        help='Path to the directory containing AF 2mers predictions')
    
    parser.add_argument('--AF_Nmers', type = str, default = None,
        help='Path to the directory containing AF Nmers predictions')
    
    parser.add_argument('--N_value', type = int, default = 4,
        help='Current N value (Only 2-mers => N=2 | 2+3-mers => N=3 | 2+3+4-mers => N=4 | ...). This is to suggest combinations.')
    
    parser.add_argument('--out_path', type = str, default = "mm_output",
        help='Output directory to store results')
    
    parser.add_argument('--manual_domains', type = str, default = None,
        help='Path to tsv file with manually defined domains (look at tests/EAF6_EPL1_PHD1/manual_domains.tsv for an example)')
    
    parser.add_argument('--use_IDs', action='store_true',
        help='Use protein IDs instead of names')
    
    parser.add_argument('--overwrite', action='store_true',
        help='If exists, overwrites the existent folder')
    
    parser.add_argument('--reduce_verbosity', action='store_true',
        help='Changes logging level from INFO to WARNING (only displays warnings and errors)')
    
    # --------------------------------------------------------------------------
    # --------------------- Command line arguments parsing ---------------------
    # --------------------------------------------------------------------------

    # Parse arguments
    args = parser.parse_args()

    # Depackage arguments
    use_names       = False if args.use_IDs else True
    fasta_file      = args.fasta_file
    AF_2mers        = args.AF_2mers
    AF_Nmers        = args.AF_Nmers
    out_path        = args.out_path
    overwrite       = args.overwrite
    manual_domains  = args.manual_domains
    N_value         = args.N_value

    # Verbosity level
    if args.reduce_verbosity:
        log_level = 'warn'
    else:
        log_level = 'info'
    
    # Initialize __main__ level logger
    logger = configure_logger(out_path = out_path, log_level = log_level)(__name__)
    
    # --------------------------------------------------------------------------
    # ------------------------- 2-mers initialization --------------------------
    # --------------------------------------------------------------------------
    
    if AF_2mers is None and AF_Nmers is None:

        from utils.combinations import initialize_multimer_mapper

        # Progress
        logger.info("No AF predictions passed: Initializing MultimerMapper 2-mers combinations suggestions...")

        # Generate suggestions
        suggestions = initialize_multimer_mapper(fasta_file, out_path, use_names, logger)

        # End MultimerMapper
        logger.info(f"FINISHED: 2-mers combinations suggestions can be found in {out_path}/combinations_suggestions")
        logger.info( "   1) Compute them using AF and store them on a single directory (uncompressed)")
        logger.info( "   2) Run MultimerMapper using --AF_2mers flag using this directory path as value")
        logger.info( "The result will be consider the first iteration (N=2). You must compute higher combinations to catch dynamic information.")
        sys.exit()

    # --------------------------------------------------------------------------
    # --------------------------- Pipeline execution ---------------------------
    # --------------------------------------------------------------------------

    # Run the main MultimerMapper pipeline
    mm_output = parse_AF2_and_sequences(fasta_file, AF_2mers, AF_Nmers, out_path,
                                        manual_domains = manual_domains,
                                        use_names = use_names,
                                        overwrite = overwrite,
                                        log_level = log_level)

    # Generate interactive 2D PPI graph
    combined_graph_interactive = interactive_igraph_to_plotly(mm_output["combined_graph"],
                                                              out_path = out_path,
                                                              log_level = log_level)
    
    # Generate RMSF, pLDDT clusters and RMSD trajectories
    mm_traj = generate_RMSF_pLDDT_cluster_and_RMSD_trajectories(mm_output = mm_output,
                                                                out_path = out_path,
                                                                log_level = log_level)
    
    
    
    # Generate suggested combinations files
    sug_combs = suggest_combinations(mm_output = mm_output,
                                     out_path = out_path,
                                     log_level = log_level,
                                     max_N = N_value + 1)
    
    # Create 3D network
    nw = interactive_igraph_to_py3dmol(mm_output['combined_graph'], logger = logger)
    
    # Explore the stoichiometric space of the complex
    logger.warning("Stoichiometric Space Exploration Algorithm is not available yet")
    logger.warning("   - We making sure this feature is fully functional before release")
    logger.warning("   - It will be available soon. Skipping...")
    # stoichiometric_space_exploration_pipeline(mm_output, log_level = log_level, open_plots = True)

    # Progress
    logger.info("MultimerMapper pipeline completed! Enjoy exploring your interactions!")
