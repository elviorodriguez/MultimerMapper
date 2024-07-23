
from cfg.default_settings import *
from utils.logger_setup import configure_logger
from src.input_check import seq_input_from_fasta, extract_seqs_from_AF2_PDBs, merge_fasta_with_PDB_data
from src.metrics_extractor import extract_AF2_metrics_from_JSON, generate_pairwise_2mers_df, generate_pairwise_Nmers_df
from src.detect_domains import detect_domains
from src.ppi_detector import filter_non_int_2mers_df, filter_non_int_Nmers_df
from src.ppi_graphs import generate_2mers_graph, generate_Nmers_graph, generate_combined_graph

# Main MultimerMapper pipeline
def parse_AF2_and_sequences(
    
    # Input
    fasta_file: str, AF2_2mers: str, AF2_Nmers: str, out_path: str,
    
    # Options imported from cfg.default_settings
    use_names = use_names,
    graph_resolution = graph_resolution, auto_domain_detection = auto_domain_detection,
    graph_resolution_preset = graph_resolution_preset, save_preset = save_preset,
    save_PAE_png = save_PAE_png, save_ref_structures = save_ref_structures,
    display_PAE_domains = display_PAE_domains, show_monomer_structures = show_monomer_structures,
    display_PAE_domains_inline = display_PAE_domains_inline, save_domains_html = save_domains_html,
      save_domains_tsv = save_domains_tsv,

    # 2-mers cutoffs
    min_PAE_cutoff_2mers = min_PAE_cutoff_2mers, ipTM_cutoff_2mers = ipTM_cutoff_2mers,

    # N-mers cutoffs
    min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers, pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers,

    # General cutoff
    N_models_cutoff = N_models_cutoff, pdockq_indirect_interaction_cutoff = pdockq_indirect_interaction_cutoff,

    # For RMSD calculations
    domain_RMSD_plddt_cutoff = domain_RMSD_plddt_cutoff,
    trimming_RMSD_plddt_cutoff = trimming_RMSD_plddt_cutoff,
    predominantly_static_cutoff = predominantly_static_cutoff,

    # Edges and vertex colors of combined graph (see cfg/default_settings module for their meaning)
    edge_color1 = edge_color1, edge_color2 = edge_color2, edge_color3 = edge_color3, edge_color4 = edge_color4,
    edge_color5 = edge_color5, edge_color6 = edge_color6, edge_color_both = edge_color_both,
    vertex_color1=vertex_color1, vertex_color2=vertex_color2, vertex_color3=vertex_color3, vertex_color_both=vertex_color_both,

    remove_indirect_interactions = remove_indirect_interactions,
    
    # Set it to "error" to reduce verbosity
    log_level: str = log_level):

    # --------------------------------------------------------------------------
    # ------------------------ Initialize the logger ---------------------------
    # --------------------------------------------------------------------------
    
    logger = configure_logger(out_path, log_level = log_level)
    
    # --------------------------------------------------------------------------
    # -------------------- Input verification and merging ----------------------
    # --------------------------------------------------------------------------

    # FASTA file processing
    prot_IDs, prot_names, prot_seqs, prot_lens, prot_N = seq_input_from_fasta(
        fasta_file, use_names, logger = logger)

    # PDB files processing
    all_pdb_data = extract_seqs_from_AF2_PDBs(AF2_2mers, AF2_Nmers, logger = logger)

    # Combine data
    merge_fasta_with_PDB_data(all_pdb_data, prot_IDs, prot_seqs, 
                                prot_seqs, prot_lens, prot_N, use_names, logger = logger)
    
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
        out_path = out_path, overwrite = True, logger = logger, manual_domains = manual_domains)
    
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
    logger.info("For N-mers:")
    logger.info(f"   - N-mers proteins: {unique_Nmers_proteins}")
    logger.info(f"   - N-mers PPIs:\n{pairwise_Nmers_df_F3}")


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

    # Debug
    logger.debug(f"Resulting 2-mers graph:\n{graph_Nmers}")

    # Combined PPI graph
    combined_graph, dynamic_proteins, dynamic_interactions = generate_combined_graph(
        
        # Input
        graph_2mers, graph_Nmers, 
        pairwise_2mers_df, pairwise_Nmers_df, 
        domains_df, sliced_PAE_and_pLDDTs,
        
        # Prot_IDs and names to add them to the graph
        prot_IDs = prot_IDs, prot_names = prot_names,
        
        # 2-mers cutoffs
        min_PAE_cutoff_2mers = min_PAE_cutoff_2mers, ipTM_cutoff_2mers = ipTM_cutoff_2mers,
        
        # N-mers cutoffs
        min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers, pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers,
        
        # General cutoffs
        N_models_cutoff = N_models_cutoff,
        pdockq_indirect_interaction_cutoff = pdockq_indirect_interaction_cutoff, 
        predominantly_static_cutoff = predominantly_static_cutoff,

        # For RMSD calculations
        domain_RMSD_plddt_cutoff = domain_RMSD_plddt_cutoff,
        trimming_RMSD_plddt_cutoff = trimming_RMSD_plddt_cutoff,

        # Style options (see cfg/default_settings module for their meaning)
        edge_color1 = edge_color1, edge_color2 = edge_color2, edge_color3 = edge_color3, edge_color4 = edge_color4,
        edge_color5 = edge_color5, edge_color6 = edge_color6, edge_color_both = edge_color_both,
        vertex_color1=vertex_color1, vertex_color2=vertex_color2, vertex_color3=vertex_color3, vertex_color_both=vertex_color_both,

        # Remove indirect interactions?
        remove_indirect_interactions = remove_indirect_interactions,
        
        # Is debug?
        is_debug = False)
    
    # Debug
    logger.debug(f"Resulting combined graph:\n{combined_graph}")
    logger.debug(f"Dynamic proteins:\n{dynamic_proteins}")
    logger.debug(f"Dynamic interactions:\n{dynamic_interactions}")

    # Save reference monomers?
    if save_ref_structures:

        logger.info("Saving PDB reference monomer structures...")

        from src.save_ref_pdbs import save_reference_pdbs

        save_reference_pdbs(sliced_PAE_and_pLDDTs = sliced_PAE_and_pLDDTs,
                            out_path = out_path,
                            overwrite = overwrite)


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
    
    parser.add_argument('AF2_2mers', type = str,
        help='Path to the directory containing AF2 2mers PDB files')
    
    parser.add_argument('AF2_Nmers', type = str, 
        help='Path to the directory containing AF2 Nmers PDB files (optional: pass an empty directory)')
    
    parser.add_argument('out_path', type = str,
        help='Output directory to store results')
    
    parser.add_argument('--use_names', action='store_true',
        help='Use protein names instead of IDs')
    
    # --------------------------------------------------------------------------
    # --------------------- Command line arguments parsing ---------------------
    # ------------------------------------------------------------------------

    # Parse arguments
    args = parser.parse_args()

    # Depackage arguments
    use_names = args.use_names
    fasta_file = args.fasta_file
    AF2_2mers = args.AF2_2mers
    AF2_Nmers = args.AF2_Nmers
    out_path = args.out_path


    # --------------------------------------------------------------------------
    # --------------------------- Pipeline execution ---------------------------
    # --------------------------------------------------------------------------

    parse_AF2_and_sequences(fasta_file, AF2_2mers, AF2_Nmers, out_path, use_names)