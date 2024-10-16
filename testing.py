#################################################################################
################ Python file for debugging MultimerMapper ########################
#################################################################################

import os
import pickle

from cfg.default_settings import *
from utils.logger_setup import configure_logger
from src.input_check import seq_input_from_fasta, extract_seqs_from_AF2_PDBs, merge_fasta_with_PDB_data
from src.metrics_extractor import extract_AF2_metrics_from_JSON, generate_pairwise_2mers_df, generate_pairwise_Nmers_df
from src.detect_domains import detect_domains
from src.ppi_detector import filter_non_int_2mers_df, filter_non_int_Nmers_df
from src.ppi_graphs import generate_2mers_graph, generate_Nmers_graph, generate_combined_graph, igraph_to_plotly


# Input/Output
fasta_file = "tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta"
AF2_2mers = "tests/EAF6_EPL1_PHD1/2-mers"
AF2_Nmers = "tests/EAF6_EPL1_PHD1/N-mers"
out_path = "tests/output"
overwrite = True

# Setup the logger:
logger = configure_logger(out_path)

# Check if the pickle file exists
pickle_dir = 'tests/pickles'
pickle_file = os.path.join(pickle_dir, 'variables.pkl')
os.makedirs(pickle_dir, exist_ok=True)
try:
    logger.info(f"Opening pickle file: {pickle_file}. This might take a while...")

    with open(pickle_file, 'rb') as f:
        variables = pickle.load(f)
    print(f"Loaded variables from pickle: {pickle_file}")
    
    # Unpack the variables
    prot_IDs = variables['prot_IDs']
    prot_names = variables['prot_names']
    prot_seqs = variables['prot_seqs']
    prot_lens = variables['prot_lens']
    prot_N = variables['prot_N']
    all_pdb_data = variables['all_pdb_data']
    sliced_PAE_and_pLDDTs = variables['sliced_PAE_and_pLDDTs']
    pairwise_2mers_df = variables['pairwise_2mers_df']
    pairwise_Nmers_df = variables['pairwise_Nmers_df']

    print(f"Unpacking variables from pickle complete")

except FileNotFoundError:
    
    print("Pickle file not found. Running the full protocol...")

    # --------------------------------------------------------------------------
    # -------------------- Input verification and merging ----------------------
    # --------------------------------------------------------------------------

    # FASTA file processing
    prot_IDs, prot_names, prot_seqs, prot_lens, prot_N = seq_input_from_fasta(
        fasta_file, use_names, logger = logger)

    # PDB files processing
    all_pdb_data = extract_seqs_from_AF2_PDBs(AF2_2mers, AF2_Nmers, logger = logger)
    # all_pdb_data = extract_seqs_from_AF2_PDBs(AF2_2mers, logger = logger)

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
    
    # Save all variables to a pickle file
    variables = {
        'prot_IDs': prot_IDs,
        'prot_names': prot_names,
        'prot_seqs': prot_seqs,
        'prot_lens': prot_lens,
        'prot_N': prot_N,
        'all_pdb_data': all_pdb_data,
        'sliced_PAE_and_pLDDTs': sliced_PAE_and_pLDDTs,
        'pairwise_2mers_df': pairwise_2mers_df,
        'pairwise_Nmers_df': pairwise_Nmers_df
    }

    with open(pickle_file, 'wb') as f:
        pickle.dump(variables, f)

    logger.warning(f"Saved variables to pickle: {pickle_file}")

# --------------------------------------------------------------------------
# --------------------------- Domain detection -----------------------------
# --------------------------------------------------------------------------

# Domain detection
display_PAE_domains = False
display_PAE_domains_inline = False
show_monomer_structures = False
auto_domain_detection = False
graph_resolution_preset = None
manual_domains = "tests/EAF6_EPL1_PHD1/manual_domains.tsv"
# manual_domains = None
domains_df = detect_domains(sliced_PAE_and_pLDDTs, fasta_file, graph_resolution = graph_resolution,
                            auto_domain_detection = auto_domain_detection,
                            graph_resolution_preset = graph_resolution_preset, save_preset = save_preset,
                            save_png_file = save_PAE_png, show_image = display_PAE_domains,
                            show_inline = display_PAE_domains_inline, show_structure = show_monomer_structures,
                            save_html = save_domains_html, save_tsv = save_domains_tsv,
                            out_path = out_path, overwrite = overwrite, logger = logger, manual_domains = manual_domains)

logger.info(f"Resulting domains:\n{domains_df}")

# --------------------------------------------------------------------------
# ----------------------------- PPI detection ------------------------------
# --------------------------------------------------------------------------

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

logger.info("Resulting interactions:")
logger.info(f"   - 2-mers proteins: {unique_2mers_proteins}")
logger.info(f"   - 2-mers PPIs:\n{pairwise_2mers_df_F3}")
logger.info("For N-mers:")
logger.info(f"   - N-mers proteins: {unique_Nmers_proteins}")
logger.info(f"   - N-mers PPIs:\n{pairwise_Nmers_df_F3}")

# --------------------------------------------------------------------------
# ----------------------- 2D PPI graph generation --------------------------
# --------------------------------------------------------------------------

logger.info("Converting PPIs to graphs...")

# 2-mers PPI graph
graph_2mers = generate_2mers_graph(pairwise_2mers_df_F3 = pairwise_2mers_df_F3,
                                    out_path = out_path,
                                    overwrite = overwrite)

logger.debug(f"Resulting 2-mers graph:\n{graph_2mers}")

# N-mers PPI graph
graph_Nmers = generate_Nmers_graph(pairwise_Nmers_df_F3 = pairwise_Nmers_df_F3,
                                    out_path = out_path,
                                    overwrite = overwrite)

logger.debug(f"Resulting 2-mers graph:\n{graph_Nmers}")


# Combined graph
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

# Convert combined PPI graph to interactive plotly
save_html = out_path + "/2D_graph.html"
while True:

    combined_graph_interactive = igraph_to_plotly(

        # Input
        graph = combined_graph, layout = None,

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
        add_cutoff_legend = add_cutoff_legend)
    
    logger.info('Default layout generation algorithm is Fruchterman-Reingold ("fr")')
    logger.info('This algorithm is stochastic. Try several layouts and save the one you like.')
    user_input = input("Do you like the plot? (y/n): ").strip().lower()
    
    if user_input == 'y':
        logger.info("Great! Enjoy your interactive PPI graph.")
        break
    elif user_input == 'n':
        logger.info("Generating a new PPI graph layout and graph...")
    else:
        logger.info("Invalid input. Please enter 'y' or 'n'.")


############  IMPORTS from main module ###########
# from Bio import SeqIO, PDB
# from Bio.PDB import Chain, Superimposer
# from Bio.SeqUtils import seq1
# from Bio.PDB.Polypeptide import protein_letters_3to1
# import igraph
# import plotly.graph_objects as go           # For plotly plotting
# from plotly.offline import plot             # To allow displaying plots
# from src.detect_domains import detect_domains, plot_backbone

