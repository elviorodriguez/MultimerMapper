

import os
import sys

from cfg.default_settings import *
from utils.logger_setup import configure_logger
from src.input_check import seq_input_from_fasta, extract_seqs_from_AF2_PDBs, merge_fasta_with_PDB_data, check_graph_resolution_preset
from src.metrics_extractor import extract_AF2_metrics_from_JSON, generate_pairwise_2mers_df, generate_pairwise_Nmers_df
from src.detect_domains import detect_domains
from src.ppi_detector import filter_non_int_2mers_df, filter_non_int_Nmers_df
from utils.temp_files_manager import setup_temp_file
from src.contact_extractor import compute_pairwise_contacts, remove_Nmers_without_enough_contacts

# Main MultimerMapper pipeline
def parse_raw_data(
    
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

    # Remove Nmers that do not have enough contacts from pairwise_Nmers_df_F3 df and their matrices
    pairwise_Nmers_df_F3, pairwise_contact_matrices = remove_Nmers_without_enough_contacts(multimer_mapper_output)
    multimer_mapper_output["pairwise_contact_matrices"] = pairwise_contact_matrices
    multimer_mapper_output['pairwise_Nmers_df_F3'] = pairwise_Nmers_df_F3

    return multimer_mapper_output