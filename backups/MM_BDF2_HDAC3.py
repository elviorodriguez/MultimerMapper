# -----------------------------------------------------------------------------
# ------------------------- Package general usage -----------------------------
# -----------------------------------------------------------------------------

# Set the working directory
import os
os.chdir("/home/elvio/Desktop/Assemblies/BDF2_HDAC3")

# Import the package
import sys
MultimerMapper_path = "/home/elvio/Desktop/Assemblies"
sys.path.insert(0, MultimerMapper_path)
import MultimerMapper as MM

# Import other packages
import pandas as pd
import numpy as np

# Set the maximum number of columns to display
pd.set_option('display.max_columns', 20)  # Set the limit to 10 columns
pd.set_option('display.max_rows', 10000)  # Set the limit to 10 columns
# Set the maximum width for displaying columns
pd.set_option('display.width', 100000)

# ---------------- Parameters setup and files locations -----------------------

# Specify the path to your FASTA file 
fasta_file_path = "BDF2_HDAC3_proteins.fasta"                # REPLACE HERE!!!!!

# List of folders to search for PDB files
AF2_2mers = "2-mers"
AF2_Nmers = "N-mers"

# If you want to work with names, set it to True (if not, IDs will be used)
use_names = True


############ For domain detection algorithm:

# General domain resolution value
graph_resolution = 0.075

# To set a different resolution for each protein, set it to False
# If True: graph_resolution will be used for all proteins
auto_domain_detection = True

# If you need to re-do all the pipeline, it is better to save a preset for domain
# detection. The first time you change the value of save_preset to True, which 
# saves the preset in "domains/" folder as "{fasta_file_name}-graph_resolution_preset.json".
save_preset = False
# The second and following times you run the pipeline, set save_preset to False
# and pass "domains/{fasta_file_name}-graph_resolution_preset.json" as value here.
# It will automatically charge the graph_resolution value you set the first time
# for each protein.
graph_resolution_preset = None        # Path to JSON graph resolution preset

# Cutoff to consider a domain disordered (domain_mean_pLDDT < cutoff => disordered)
domain_RMSD_plddt_cutoff = 60
trimming_RMSD_plddt_cutoff = 70


############ Interaction definitions Cutoffs (check cutoff table from paper)

# For 2mers
min_PAE_cutoff_2mers = 8.99
ipTM_cutoff_2mers = 0.24

# For Nmers
min_PAE_cutoff_Nmers = 8.99
pDockQ_cutoff_Nmers = 0.022   # As ipTM loses sense in N-mers, we use pDockQ with a low cutoff value

# General cutoffs
N_models_cutoff = 3          # At least this many models have to surpass both cutoffs to end up as interactions in 2mers and Nmers
pdockq_indirect_interaction_cutoff = 0.23
predominantly_static_cutoff = 0.5           # if 50% of the N-mer models show interaction => solid blue line edge

# Due to the fact that the Interactor/Non-interactor filter is mailty given by 
# the minimum PAE value for the protein pair, there are cases in which the 2-mer
# pair does not comes out as positive interactors, but at least one of the N-mers
# models comes out as positive. In this case, we face the possibility of:
#    1) The PAE value is low because they interact.
#    2) The PAE value is low because it is an indirect interaction mediated by
#       a third protein.
# To distinguish between these two cases, we use the standard pDockQ cutoff of
# 0.23 (lower imply them to be non-interactors). The cutoff is compared with
# the mean pDockQ of those pairs comming from models that surpassed the triple
# cutoff (PAE+pDockQ+N_models).

############ Color blind friendly palette (Paul Tol's + orange)

PT_palette = {
    "black"         : "#000000",
    "green"         : "#228833",
    "blue"          : "#4477AA",
    "cyan"          : "#66CCEE",
    "yellow"        : "#CCBB44",
    "purple"        : "#AA3377",
    "orange"        : "#e69f00",
    "deep orange"   : "#d55e00",
    "red"           : "#EE6677",
    "gray"          : "#bbbbbb",
    }

# You can call it like this
PT_palette["orange"]

# -------------------- Preprocess AF2-multimer data ---------------------------

# Depending on the size of your 2-mers and N-mers datasets, and your CPU this part may take from 1 to 20 minutes, or even more.
# But don't worry, progress will be shown in the console. 

# Execute data extractor
all_pdb_data, sliced_PAE_and_pLDDTs, domains_df, \
    pairwise_2mers_df, pairwise_2mers_df_F3, graph_2mers,\
        fully_connected_subgraphs, fully_connected_subgraphs_pairwise_2mers_dfs,\
            pairwise_Nmers_df, pairwise_Nmers_df_F3, graph_Nmers, combined_graph,\
                dynamic_proteins, dynamic_interactions = \
                    MM.parse_AF2_and_sequences(fasta_file_path, AF2_2mers, AF2_Nmers, use_names = True,
                                    graph_resolution = 0.075, auto_domain_detection = False,
                                    # Use previous preset?
                                    graph_resolution_preset = "./domains/BDF2_HDAC3_proteins-graph_resolution_preset.json", 
                                    # save_preset = True,
                                    save_PAE_png = True, display_PAE_domains = False, show_structures = True,
                                    display_PAE_domains_inline = True,
                                    save_domains_html = True, save_domains_tsv = True,
                                    # 2-mers cutoffs
                                    min_PAE_cutoff_2mers = min_PAE_cutoff_2mers, ipTM_cutoff_2mers = ipTM_cutoff_2mers,
                                    # N-mers cutoffs
                                    min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers, pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers,
                                    # General cutoffs
                                    N_models_cutoff = N_models_cutoff,
                                    pdockq_indirect_interaction_cutoff = pdockq_indirect_interaction_cutoff,
                                    predominantly_static_cutoff=predominantly_static_cutoff,
                                    # You can customize the edges and vertex colors for the combined_graph
                                    edge_color1=PT_palette["red"], edge_color2=PT_palette["green"], edge_color3=PT_palette["orange"],
                                    edge_color4=PT_palette["cyan"], edge_color5=PT_palette["yellow"], edge_color6=PT_palette["blue"],
                                    edge_color_both=PT_palette["black"],
                                    vertex_color1=PT_palette["red"], vertex_color2=PT_palette["green"], vertex_color3=PT_palette["orange"],
                                    vertex_color_both=PT_palette["gray"])
                    
####### Let's see the generated data

# Dictionary (Contains information about each protein comming from both 2mers and Nmers data)
sliced_PAE_and_pLDDTs.keys()                                # Protein IDs are keys 
protein0_ID = list(sliced_PAE_and_pLDDTs.keys())[0]         # Extract first protein ID
sliced_PAE_and_pLDDTs[protein0_ID].keys()                   # Info you can explore
sliced_PAE_and_pLDDTs[protein0_ID]["sequence"]              # Its sequence
sliced_PAE_and_pLDDTs[protein0_ID]["length"]                # Its length in aminoacids
sliced_PAE_and_pLDDTs[protein0_ID]["Q_value"]               # The Q_value in the input fasta file
sliced_PAE_and_pLDDTs[protein0_ID]["PDB_file"]              # Paths to all the PDB files that contain the protein
sliced_PAE_and_pLDDTs[protein0_ID]["pLDDTs"]                # Per residue pLDDT of the chain corresponding to the protein on each PDB file
sliced_PAE_and_pLDDTs[protein0_ID]["max_mean_pLDDT_index"]  # Index of the PDB file with highest mean pLDDT for the protein
sliced_PAE_and_pLDDTs[protein0_ID]["best_PAE_matrix"]       # Protein PAE matrix from chain with highest mean pLDDT (used to define domains)
sliced_PAE_and_pLDDTs[protein0_ID]["PDB_xyz"]               # Bio.PDB.Chain.Chain object containing the highest mean pLDDT chain for the protein

# The rest sub-keys are intermediate computations necessary for the program to excecute.

# Dataframes
domains_df                      # Detected domains information for each protein
pairwise_2mers_df.columns       # Contains protein pairs information for 2mers dataset
pairwise_2mers_df_F3            # Contains protein pairs that surpasses the cutoffs and their values (from 2mers dataset)
pairwise_Nmers_df.columns       # Contains protein pairs information for Nmers dataset
pairwise_Nmers_df_F3            # Contains protein pairs that surpasses the cutoffs and their values (from Nmers dataset)

# Graphs data
graph_2mers
graph_Nmers
combined_graph
fully_connected_subgraphs

# Sub-dataframes for each graph in fully_connected_subgraphs
fully_connected_subgraphs_pairwise_2mers_dfs


##### Let's take a better look at combined_graph

# Generate a layout (using all possible edges)
combined_graph_layout = combined_graph.layout("fr")

# Generate a layout using only the edges you want
combined_graph_layout = MM.generate_layout_for_combined_graph(
    combined_graph,
    # You can choose which type of interactions use to built the layout
    edge_attribute_value=['Static interaction',
                          'Ambiguous Dynamic (In some N-mers appear and in others disappear)',
                          'Predominantly static interaction'],
    layout_algorithm="fr")


# And then convert the plot 
combined_graph_interactive = MM.igraph_to_plotly(combined_graph, combined_graph_layout,
                            # This changes the orientation of circular edges (0: no change, 0.25: quarter turn, etc)
                            self_loop_orientation = 0.25, self_loop_size=3,
                            # Remove background and set protein names as bold
                            show_axis= False, showgrid= False, use_bold_protein_names= True,
                            # Domain RMSDs bigger than this value will be highlighted in bold in the nodes hovertext
                            add_bold_RMSD_cutoff = 5,
                            # # You can save the plot as HTML to share it, for example, with yourself via whatsapp and analyze it
                            # # using your cellphone' browser
                            save_html = "BDF2_SWR1_2D_graph.html", 
                            # It's highly recommended to add these labels to keep track cutoffs values that generated the graph
                            add_cutoff_legend = True)

# combined_graph is a graph made by the overlapping of the interactions detected in 2-mers dataset (represented in graph_2mers),
# and the ones detected in the N-mers dataset (graph_Nmers). It catches how the interactions change depending on the context.
# In other words, how the addition of other proteins to the predictions affect the interactions between each pair. For example,
# there are some proteins that causes some interactions to disapear (inhibitors). Others may cause the interaction to appear
# (activators). These predictions may be related to protein function.
# If you do not like the layout, you can re-run both layout and plotting code (only for layout algorithms that have intrinsic
# randomization, like "fr").


# Now, let's access programatically to some characteristics of combined_graph:

combined_graph["cutoffs_dict"]              # Keeps track cutoffs values used to generate the graph

# Vertices (nodes) data
combined_graph.get_vertex_dataframe() 
combined_graph.vs["name"]                   # Protein (nodes) names in the graph
combined_graph.vs["color"]                  # Nodes color
combined_graph.vs["meaning"]                # Meaning of the colors
combined_graph.vs["RMSD_df"][0]             # Domain RMSD against reference structure of first ([0]) node
combined_graph.vs["RMSD_df"][1]             # Domain RMSD against reference structure of second ([1]) node

# Edges data
combined_graph.get_edge_dataframe()
combined_graph.get_edgelist()               # Edges (source_vertex_idx , target_vertex_idx ) as tuples
combined_graph.es["name"]                   # Edges (source_vertex_name, target_vertex_name) as tuples
combined_graph.es["color"]                  # Edges color
combined_graph.es["meaning"]                # Meaning of the colors
combined_graph.es["2_mers_data"][0]         # df containing 2_mers data for first ([0]) edge
combined_graph.es["2_mers_data"][1]         # df containing 2_mers data for second ([1]) edge
combined_graph.es["N_mers_data"][0]         # df containing N_mers data for first ([0]) edge
combined_graph.es["N_mers_data"][1]         # df containing N_mers data for second ([1]) edge


# ---------------- Residue-residue contacts extraction ------------------------

# We have seen what was happening at the protein level. Now, let's dive deeper 
# and see what is happening at the residue level.

# Generate contacts dataframe for bigger subgraph (fully_connected_subgraphs_pairwise_2mers_dfs[0])
# NOTE: I think it is fixed now, but sometimes the bigger graph ends at index 1, not sure why.

contacts_2mers_df = MM.compute_contacts_from_pairwise_2mers_df(
    filtered_pairwise_2mers_df = fully_connected_subgraphs_pairwise_2mers_dfs[0],
    pairwise_2mers_df = pairwise_2mers_df, 
    sliced_PAE_and_pLDDTs = sliced_PAE_and_pLDDTs,
    # Cutoffs that define a contact between residue centroids
    contact_distance = 8.0, contact_PAE_cutoff = 2, contact_pLDDT_cutoff = 70)


contacts_Nmers_df = MM.compute_contacts_from_pairwise_Nmers_df(
    pairwise_Nmers_df = pairwise_Nmers_df, 
    filtered_pairwise_Nmers_df = pairwise_Nmers_df_F3, 
    sliced_PAE_and_pLDDTs = sliced_PAE_and_pLDDTs,
    # Cutoff parameters
    contact_distance_cutoff = 8.0, contact_PAE_cutoff = 2, contact_pLDDT_cutoff = 70)


contacts_2mers_df
contacts_Nmers_df

# ----------------- Protein creation and visualization ------------------------

# Initialize proteins dictionary
proteins_dict = MM.create_proteins_dict_from_contacts_2mers_df(contacts_2mers_df, sliced_PAE_and_pLDDTs, plot_proteins = False, print_proteins = False)

# Access dict data
proteins_dict                   # full dict
print(proteins_dict["BDF6"])    # Access one of the proteins

# Representation of a single protein node (not so speed efficient)
proteins_dict["BDF6"].plot_alone()

# --------------- Partners addittions and 2D network --------------------------

# Add partners to proteins
MM.add_partners_to_proteins_dict_from_contacts_2mers_df(proteins_dict, contacts_2mers_df)

# Let's see what happend to BDF6
proteins_dict["BDF6"].plot_alone()

# See general 2D network that contains BDF6 
proteins_dict["BDF6"].plot_fully_connected_protein_level_2D_graph(show_plot = True, return_graph= False,
                                                                  algorithm = "kk", save_png = None)


# ---------- Protein 3D manipulation and contacts network plotting ------------

# # List of algorithms available for protein 3D distribution and sugested scaling factors
# algorithm_list  = ["drl", "fr", "kk", "circle", "grid", "random"]
# scaling_factors = [200  , 100 , 100 , 100     , 120   , 100     ]

# Move the proteins in 3D space to allow propper network plotting
proteins_dict["BDF6"].set_fully_connected_network_3D_coordinates(algorithm = "drl", scaling_factor = 200)

# Plot network visualization (some algorithms have intrinsic randomization to
#                             produce protein 3D distributions. So,
#                             if you don't like the visualization, you can
#                             try again by resetting 3D coordinates of the
#                             proteins and plotting again)
fig, network_contacts_2mers_df, proteins_df, proteins_df_c, proteins_df_c2 =\
    proteins_dict["BDF6"].plot_fully_connected2(plddt_cutoff = 70,
                                                show_axis = False,
                                                visible_backbones = False,
                                                # Save as HTML file to share and open it in any web browser
                                                save_html = "BDF6_HAT1_3D_graph.html"
                                                )
    
    
    

# --------------- Repeat the same with the other sub-network ------------------

sub_graph_2 = fully_connected_subgraphs_pairwise_2mers_dfs[1]

contacts_2mers_df_2 = \
    MM.compute_contacts_from_pairwise_2mers_df(filtered_pairwise_2mers_df = sub_graph_2,
                                         pairwise_2mers_df = pairwise_2mers_df,
                                         sliced_PAE_and_pLDDTs = sliced_PAE_and_pLDDTs,
                                         # Cutoffs that define contacts between residue centroids
                                         contact_distance = 8.0,
                                         contact_PAE_cutoff = 3,
                                         contact_pLDDT_cutoff = 70)

# Initialize proteins dictionary
proteins_dict_2 = \
    MM.create_proteins_dict_from_contacts_2mers_df(contacts_2mers_df_2, sliced_PAE_and_pLDDTs, 
                                             plot_proteins = False, print_proteins = False)

MM.add_partners_to_proteins_dict_from_contacts_2mers_df(proteins_dict_2, contacts_2mers_df_2)

proteins_dict_2["YNG2"].set_fully_connected_network_3D_coordinates(algorithm = "drl", scaling_factor = 200)

fig_2, network_contacts_2mers_df_2, proteins_df_2, proteins_df_c_2, proteins_df_c2_2 =\
    proteins_dict_2["YNG2"].plot_fully_connected2(plddt_cutoff = 70,
                                                  show_axis = False,
                                                  visible_backbones = False)


# -----------------------------------------------------------------------------
# ------------------------- Package advance usage -----------------------------
# -----------------------------------------------------------------------------

# # MultimerMapper.Protein class keeps track of all the proteins created
# MM.Protein.protein_list
# MM.Protein.protein_list_IDs

# # Also, it assigns a tag to each protein and the last tag is stored in protein_tag class variable
# MM.Protein.protein_tag





# -----------------------------------------------------------------------------
# ---------------------------- TESTING FUNCTIONS ------------------------------
# -----------------------------------------------------------------------------
    
# import sys
# MultimerMapper_path = "C:/Users/elvio/OneDrive/1_CRUZI/07-AF2_Tryps_complexes_AWS/00_RESULTS/scripts/MultimerMapper"
# sys.path.insert(0, MultimerMapper_path)
# import MultimerMapper as MM


# prot_IDs, prot_names, prot_seqs, prot_len, prot_N, Q_values = MM.seq_input_from_fasta(fasta_file_path, use_names = use_names)

# # Work with names?
# if use_names:
#     # Switch IDs with names
#     prot_IDs_backup = prot_IDs
#     prot_IDs = prot_names
#     prot_names = prot_IDs_backup

# combined_graph, dynamic_proteins, dynamic_interactions =\
#     MM.compare_and_plot_graphs(graph_2mers, graph_Nmers, pairwise_2mers_df, pairwise_Nmers_df, domains_df, sliced_PAE_and_pLDDTs,
#                             # Prot_IDs and names to add them to the graph
#                             prot_IDs = prot_IDs, prot_names = prot_names,
#                             # 2-mers cutoffs
#                             min_PAE_cutoff_2mers = min_PAE_cutoff_2mers, ipTM_cutoff_2mers = ipTM_cutoff_2mers,
#                             # N-mers cutoffs
#                             min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers, pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers,
#                             # General cutoff
#                             N_models_cutoff = N_models_cutoff, 
#                             # For RMSD calculations
#                             domain_RMSD_plddt_cutoff = domain_RMSD_plddt_cutoff, trimming_RMSD_plddt_cutoff = trimming_RMSD_plddt_cutoff,
#                             # Style options
#                             is_debug = False)
    

# # Generate a layout
# combined_graph_layout = combined_graph.layout("fr")

# # And then convert the plot 
# combined_graph_interactive = MM.igraph_to_plotly(combined_graph, combined_graph_layout,
#                             # This changes the orientation of circular edges (0: no change, 0.25: quarter turn, etc)
#                             self_loop_orientation = 0.75, 
#                             # Remove background and set protein names as bold
#                             show_axis= False, showgrid= False, use_bold_protein_names= True, add_bold_RMSD_cutoff = 5,
#                             # You can save the plot as HTML to share it with yourself via whatsapp and analyze it
#                             # using your cellphone' browser
#                             save_html = "BDF6_HAT1_2D_graph.html", 
#                             # It's highly recommended to add these labels to keep track cutoffs values that generated the graph
#                             add_cutoff_legend = True)






# # Tests for Nmers contacts prediction -----------------------------------------

# import matplotlib.pyplot as plt
# from Bio.PDB import PDBIO
# import random as rd


# test_contacts_Nmers_df = MM.compute_contacts_from_pairwise_Nmers_df(
#     pairwise_Nmers_df, pairwise_Nmers_df_F3, sliced_PAE_and_pLDDTs,
#     # Cutoff parameters
#     contact_distance_cutoff = 8.0, contact_PAE_cutoff = 2, contact_pLDDT_cutoff = 70)

# i = rd.randint(0, len(test_contacts_Nmers_df))
# code = test_contacts_Nmers_df["chimera_code"][i]
# prot_a = str(test_contacts_Nmers_df["protein_ID_a"][i])
# prot_b = str(test_contacts_Nmers_df["protein_ID_b"][i])
# prots_in_mod = tuple([prot for prot in tuple(test_contacts_Nmers_df["proteins_in_model"][i])])

# pairwise_model = pairwise_Nmers_df.query(f'(protein1 == "{prot_a}" & protein2 == "{prot_b}" & rank == 1)').\
#     loc[pairwise_Nmers_df['proteins_in_model'].apply(lambda x: tuple(x) == prots_in_mod)]
# plt.matshow(pairwise_model["diagonal_sub_PAE"].values[0])
# structure = pairwise_model["model"].values[0]
# # Save model for debbugging
# pdbio = PDBIO()
# pdbio.set_structure(structure)
# pdbio.save("contacts_Nmers_test.pdb")
# print(code)










# ---------------------- Res-Res contacts classifier --------------------------


# Function to print progress
def print_progress_bar(current, total, text = "", progress_length = 40):
    '''
    Prints a progress bar:
        
        Progress{text}: [===--------------------------] 10.00%
        Progress{text}: [=============================] 100.00%
        Progress{text}: [=========--------------------] 33.00%
    
    Parameters:
    - current (float):
    - total (float): 
    - progress_length (int): length of the full progress bar.
    - text (str):
        
    Returns:
    - None    
    '''
    percent = current / total * 100
    progress = int(progress_length * current / total)
    progress_bar_template = "[{:<" + str(progress_length - 1) + "}]"
    
    if current >= total:
        progress_bar = progress_bar_template.format("=" * (progress_length - 1))
    else:
        progress_bar = progress_bar_template.format("=" * progress + "-" * (progress_length - progress - 1))
        
    print(f"Progress{text}: {progress_bar} {percent:.2f}%")


def get_contact_residues_set(contacts_2mers_df, contacts_Nmers_df = None, remove_proteins = None):
    '''
    Analyzes contacts_dfs and returns a list of detected contact residues. Each
    residue is defined by a tuple:
        
        (protein_ID, res, AA_one_letter_code, res_name, centroid_xyz_coords)
        
    Parameters:
    - contacts_2mers_df (pd.DataFrame): result of compute_contacts_from_pairwise_2mers_df function
    - contacts_2mers_df (pd.DataFrame): result of compute_contacts_from_pairwise_Nmers_df function.
        If you have not generated an N-mers dataset, you can parse only the 2-mers.
    - remove_proteins (list of str): list of proteins to remove from the final contact residue set.
        E.g: ["BDF6", "YEA2", "Tb927.6.1240"]
    
    Returns:
    - contact_residues_set (list of tuples): contains the set of residues detected
        to be in contact inside the datasets.
    '''
    if remove_proteins is not None:
        contacts_2mers_df = contacts_2mers_df.copy().query(f'protein_ID_a not in {remove_proteins}').query(f'protein_ID_b not in {remove_proteins}')
        contacts_Nmers_df = contacts_Nmers_df.copy().query(f'protein_ID_a not in {remove_proteins}').query(f'protein_ID_b not in {remove_proteins}')
    
    contact_residues_set = []
    
    # ---------------------------- 2-mers -------------------------------------
    
    # proteins A residues, 2-mers
    for prot, res, AA, res_name, xyz in zip(contacts_2mers_df["protein_ID_a"], contacts_2mers_df["res_a"], contacts_2mers_df["AA_a"], contacts_2mers_df["res_name_a"], contacts_2mers_df["xyz_a"]):
        contact_res = (prot, res, AA, res_name, tuple(xyz))
        if contact_res not in contact_residues_set: contact_residues_set.append(contact_res)
    
    # proteins B residues, 2-mers
    for prot, res, AA, res_name, xyz in zip(contacts_2mers_df["protein_ID_b"], contacts_2mers_df["res_b"], contacts_2mers_df["AA_b"], contacts_2mers_df["res_name_b"], contacts_2mers_df["xyz_b"]):
        contact_res = (prot, res, AA, res_name, tuple(xyz))
        if contact_res not in contact_residues_set: contact_residues_set.append(contact_res)
        
    # If Nmers data was provided
    if contacts_Nmers_df is not None:
    
        # ---------------------------- N-mers -------------------------------------
        # proteins A residues
        for prot, res, AA, res_name, xyz in zip(contacts_Nmers_df["protein_ID_a"], contacts_Nmers_df["res_a"], contacts_Nmers_df["AA_a"], contacts_Nmers_df["res_name_a"], contacts_Nmers_df["xyz_a"]):
            contact_res = (prot, res, AA, res_name, tuple(xyz))
            if contact_res not in contact_residues_set: contact_residues_set.append(contact_res)
            
        # proteins b residues, N-mers
        for prot, res, AA, res_name, xyz in zip(contacts_Nmers_df["protein_ID_b"], contacts_Nmers_df["res_b"], contacts_Nmers_df["AA_b"], contacts_Nmers_df["res_name_b"], contacts_Nmers_df["xyz_b"]):
            contact_res = (prot, res, AA, res_name, tuple(xyz))
            if contact_res not in contact_residues_set: contact_residues_set.append(contact_res)
        
    return sorted(contact_residues_set)


def get_contact_pairs_set(contacts_2mers_df, contacts_Nmers_df = None, remove_proteins = None):
    '''
    Analyzes contact DataFrames and returns a list of detected contact residue pairs. 
    Each residue pair is defined by a tuple of sorted tuples:
    
        ((protein_ID_a, res_a), (protein_ID_b, res_b))
    
    Parameters:
    - contacts_2mers_df (pd.DataFrame): Result of compute_contacts_from_pairwise_2mers_df function.
    - contacts_Nmers_df (pd.DataFrame): Result of compute_contacts_from_pairwise_Nmers_df function.
        If you have not generated an N-mers dataset, you can parse only the 2-mers.
    - remove_proteins (list of str): List of proteins to remove from the final contact residue pairs set.
        E.g.: ["BDF6", "YEA2", "Tb927.6.1240"]
    
    Returns:
    - contact_pairs_set (list of tuples): Contains the set of residue pairs detected to be in contact inside the datasets.
    '''
    
    
    if remove_proteins is not None:
        contacts_2mers_df = contacts_2mers_df.copy().query(f'protein_ID_a not in {remove_proteins}').query(f'protein_ID_b not in {remove_proteins}')
        contacts_Nmers_df = contacts_Nmers_df.copy().query(f'protein_ID_a not in {remove_proteins}').query(f'protein_ID_b not in {remove_proteins}')
    
    contact_pairs_set = []
        
    # ---------------------------- 2-mers -------------------------------------    
    for prot_a, res_a, prot_b, res_b in zip(contacts_2mers_df["protein_ID_a"], contacts_2mers_df["res_a"],
                                            contacts_2mers_df["protein_ID_b"], contacts_2mers_df["res_b"]):
        # Sort the pair
        contact_pair = tuple(sorted(((prot_a, res_a), (prot_b, res_b))))
        if contact_pair not in contact_pairs_set: contact_pairs_set.append(contact_pair)
        
    # ---------------------------- N-mers -------------------------------------    
    # If Nmers data was provided
    if contacts_Nmers_df is not None:    
        for prot_a, res_a, prot_b, res_b in zip(contacts_Nmers_df["protein_ID_a"], contacts_Nmers_df["res_a"],
                                                contacts_Nmers_df["protein_ID_b"], contacts_Nmers_df["res_b"]):
            # Sort the pair
            contact_pair = tuple(sorted(((prot_a, res_a), (prot_b, res_b))))
            if contact_pair not in contact_pairs_set: contact_pairs_set.append(contact_pair)
        
    return sorted(contact_pairs_set)


def get_residues_df(contacts_2mers_df, contacts_Nmers_df = None, remove_proteins = None):
    '''
    Analyzes contact DataFrames and returns a DataFrame containing information about detected contact residues.
    
    Parameters:
    - contacts_2mers_df (pd.DataFrame): Result of compute_contacts_from_pairwise_2mers_df function.
    - contacts_Nmers_df (pd.DataFrame): Result of compute_contacts_from_pairwise_Nmers_df function.
        If you have not generated an N-mers dataset, you can parse only the 2-mers.
    - remove_proteins (list of str): List of proteins to remove from the final contact residue pairs set.
        E.g.: ["BDF6", "YEA2", "Tb927.6.1240"]
    
    Returns:
    - residues_df (pd.DataFrame): Contains information about detected contact residues, including reference residue,
      contact partner, X-mer type (2-mer or N-mer), and the corresponding protein model(s) the partner residue comes.
    '''
    
    # Progress
    
    # If specific proteins need to be removed from the calculations    
    if remove_proteins is not None:
        contacts_2mers_df = contacts_2mers_df.copy().query(f'protein_ID_a not in {remove_proteins}').query(f'protein_ID_b not in {remove_proteins}')
        contacts_Nmers_df = contacts_Nmers_df.copy().query(f'protein_ID_a not in {remove_proteins}').query(f'protein_ID_b not in {remove_proteins}')
    
    # Extract the set of residues involved in contacts
    contact_residues_set = get_contact_residues_set(contacts_2mers_df = contacts_2mers_df,
                                                    contacts_Nmers_df = contacts_Nmers_df,
                                                    remove_proteins = remove_proteins)
    
    # Empty df to store results
    columns = ["ref_residue", "contact_partner", "Xmer", "model"]
    residues_df = pd.DataFrame(columns = columns)
    
    
    for res in contact_residues_set:
        
        # Reference residue
        ref_residue = res[0:2]
        
        # Contacts that involves the residue
        ref_contacts_2mers_df = contacts_2mers_df.copy().query(f'(protein_ID_a == "{ref_residue[0]}" & res_a == {ref_residue[1]}) | (protein_ID_b == "{ref_residue[0]}" & res_b == {ref_residue[1]})')
                
        # --------------------------- 2-mers ----------------------------------
        for i, row in ref_contacts_2mers_df.iterrows():
            row_residue_a = tuple((str(row["protein_ID_a"]), int(row["res_a"])))
            row_residue_b = tuple((str(row["protein_ID_b"]), int(row["res_b"])))
            
            if row_residue_a == ref_residue:
                partner_res_info = pd.DataFrame(
                    {"ref_residue": [ref_residue],
                     "contact_partner": [row_residue_b], 
                     "Xmer": ["2-mer"], 
                     "model": [(row["protein_ID_a"], row["protein_ID_b"])]
                     })
            
            elif row_residue_b == ref_residue:
                partner_res_info = pd.DataFrame(
                    {"ref_residue": [ref_residue],
                     "contact_partner": [row_residue_a], 
                     "Xmer": ["2-mer"], 
                     "model": [(row["protein_ID_a"], row["protein_ID_b"])]
                     })
                
            residues_df = pd.concat([residues_df, partner_res_info], ignore_index = True)
        
        # --------------------------- N-mers ----------------------------------
        # If Nmers data was provided
        if contacts_Nmers_df is not None:
            
            # Contacts that involves the residue
            ref_contacts_Nmers_df = contacts_Nmers_df.copy().query(f'(protein_ID_a == "{ref_residue[0]}" & res_a == {ref_residue[1]}) | (protein_ID_b == "{ref_residue[0]}" & res_b == {ref_residue[1]})')
                    
            
            for i, row in ref_contacts_Nmers_df.iterrows():
                row_residue_a = tuple((str(row["protein_ID_a"]), int(row["res_a"])))
                row_residue_b = tuple((str(row["protein_ID_b"]), int(row["res_b"])))
                
                if row_residue_a == ref_residue:
                    partner_res_info = pd.DataFrame(
                        {"ref_residue": [ref_residue],
                         "contact_partner": [row_residue_b], 
                         "Xmer": ["N-mer"], 
                         "model": [row["proteins_in_model"]]
                         })
                
                elif row_residue_b == ref_residue:
                    partner_res_info = pd.DataFrame(
                        {"ref_residue": [ref_residue],
                         "contact_partner": [row_residue_a], 
                         "Xmer": ["N-mer"], 
                         "model": [row["proteins_in_model"]]
                         })
                    
                residues_df = pd.concat([residues_df, partner_res_info], ignore_index = True)
                
    return residues_df
            


# def remove_duplicates_from_residues_df(residues_df, is_debug = False):
    
#     columns = ["ref_residue", "contact_partner", "Xmer", "model"]
#     no_dup_residues_df = pd.DataFrame(columns = columns)
    
#     # For progress bar
#     total_contacts = len(residues_df)
#     current_contact = 0
    
#     for i, res_pair_row in residues_df.iterrows():
#         sorted_res_pair = tuple(sorted((res_pair_row["ref_residue"], res_pair_row["contact_partner"])))
#         ref_res = sorted_res_pair[0]
#         part_res = sorted_res_pair[1]
        
#         is_duplicate = False
        
#         for i, result_row in no_dup_residues_df.iterrows():
#             if result_row["ref_residue"] == ref_res and result_row["contact_partner"] == part_res and\
#                 result_row["Xmer"] == res_pair_row["Xmer"] and result_row["model"] == res_pair_row["model"]:
#                     is_duplicate = True
#                     if is_debug: print("is_duplicate:", is_duplicate)
#                     break
        
#         if not is_duplicate:
       
#             contact_pair_info = pd.DataFrame(
#                 {"ref_residue": [ref_res],
#                  "contact_partner": [part_res],
#                  "Xmer": [res_pair_row["Xmer"]], 
#                  "model": [res_pair_row["model"]]
#                  })
            
#             no_dup_residues_df = pd.concat([no_dup_residues_df, contact_pair_info], ignore_index = True)
            
#         current_contact += 1
#         if current_contact % 50 == 0 or current_contact >= total_contacts:
#             print_progress_bar(current_contact, total_contacts, text = " (Contact Classifier 1)", progress_length = 40)
    
    
#     return no_dup_residues_df


# Get all models that were computed (each model is sorted)
models_2mers_set = set(sorted([ tuple(sorted((p1, p2)))  for p1, p2 in zip(pairwise_2mers_df["protein1"], pairwise_2mers_df["protein2"]) ]))
models_Nmers_set = set(sorted([ tuple(sorted(proteins_in_model)) for proteins_in_model in pairwise_Nmers_df["proteins_in_model"]]))
total_models_set = set(sorted(list(models_2mers_set) + list(models_Nmers_set)))

def get_models_that_include_proteins(protein_s, total_models_set):
    
    models_that_include_protein_s = []
    
    if type(protein_s) == str:
        for model in total_models_set:
            if protein in model:
                models_that_include_protein_s.append(model)
    
    if type(protein_s) == tuple or type(protein_s) == list or type(protein_s) == set:
        for model in total_models_set:
            if all(prot in model for prot in protein_s):
                models_that_include_protein_s.append(model)
                
    return models_that_include_protein_s

get_models_that_include_proteins(["EPL1", "BDF6"], total_models_set)




# contact_pairs_set = get_contact_pairs_set(contacts_2mers_df, contacts_Nmers_df, remove_proteins = None)
residues_df = get_residues_df(contacts_2mers_df, contacts_Nmers_df, remove_proteins = None)


# Filter rows where the first element of 'ref_residue' is equal to "EPL1"
filtered_df = residues_df[residues_df['ref_residue'].apply(lambda x: x[0] == 'EPL1')]

for i, residue_data in filtered_df.groupby("ref_residue"):
    
    protein = residue_data["ref_residue"].values[0][0]
    
    # Protein partners for residue
    partner_residues = set(residue_data["contact_partner"])
    partners_proteins = set([ part_res[0] for part_res in residue_data["contact_partner"]])
    partner_residues_num = len(partner_residues)
    partners_proteins_num = len(partners_proteins)

    
    print("")    
    print("######################################################################")
    print("")
    print("----------------------------------------------------------------------")
    print("")
    print(residue_data)
    print("")
    print("partner_residues:", partner_residues)
    print("   - partner_residues_num:", partner_residues_num)
    print("partners_proteins:", partners_proteins)
    print("   - partners_proteins_num:", partners_proteins_num)
    print("----------------------------------------------------------------------")
    

    
    
    for j, pair_data in residue_data.groupby("contact_partner"):
        
        partner = pair_data["contact_partner"].values[0][0]
        
        # In which dataset does the contact appear?
        is_in_2mers = "2-mer" in list(pair_data["Xmer"])
        is_in_Nmers = "N-mer" in list(pair_data["Xmer"])
        
        print("")
        print(pair_data)
        print("   - In 2-mers?:", is_in_2mers)
        print("   - In N-mers?:", is_in_Nmers)
        
        
        # # Now 'protein' and 'partner' are accessible directly without indexing
        # filtered_df2 = residues_df[residues_df['model'].apply(lambda x: protein in x)]
        # print(filtered_df2)
    
    
    
    
    
    
    