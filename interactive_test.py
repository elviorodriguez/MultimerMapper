# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import multimer_mapper as mm

pd.set_option( 'display.max_columns' , None )

remove_interactions = ("Indirect",)

# Hides domains plots
show_PAE_along_backbone = False

################################# Test 1 ######################################

# fasta_file = "tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta"
# AF2_2mers = "tests/EAF6_EPL1_PHD1/2-mers"
# AF2_Nmers = "tests/EAF6_EPL1_PHD1/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/MM_interactive_test"
# use_names = True 
# overwrite = True
# auto_domain_detection = True
# # graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# graph_resolution_preset = None

##############################################################################

# ################################# Test 2 ######################################

# fasta_file = "/home/elvio/Desktop/Assemblies/BDF2_HDAC3/SIN3_proteins.fasta"
# AF2_2mers = "/home/elvio/Desktop/Assemblies/BDF2_HDAC3/2-mers"
# AF2_Nmers = "/home/elvio/Desktop/Assemblies/BDF2_HDAC3/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/MM_BDF2_HDAC3"
# use_names = True
# overwrite = True
# auto_domain_detection = False
# # graph_resolution_preset = "/home/elvio/Desktop/MM_SIN3/graph_resolution_preset.json"
# graph_resolution_preset = None

# ################################# Test 3 ######################################
# T. brucei

# fasta_file = "/home/elvio/Desktop/Assemblies/NuA4/NuA4_proteins.fasta"
# AF2_2mers = "/home/elvio/Desktop/Assemblies/NuA4/2-mers"
# AF2_Nmers = "/home/elvio/Desktop/Assemblies/NuA4/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/Assemblies/NuA4/MM_NuA4"
# use_names = True 
# overwrite = True
# # graph_resolution_preset = "/home/elvio/Desktop/Assemblies/NuA4/graph_resolution_preset.json"
# auto_domain_detection = True
# graph_resolution_preset = None

# ###################### Test 4 (indirect interactions) #########################

# fasta_file = "tests/indirect_interactions/TINTIN.fasta"
# AF2_2mers = "tests/indirect_interactions/2-mers"
# AF2_Nmers = "tests/indirect_interactions/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/Assemblies/indirect_interaction_tests_N_mers"
# use_names = True 
# overwrite = True
# # graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# graph_resolution_preset = None
# auto_domain_detection = True

# ################################ Test 5 (SIN3) ################################

# fasta_file = "/home/elvio/Desktop/Assemblies/SIN3/SIN3_proteins.fasta"
# AF2_2mers = "/home/elvio/Desktop/Assemblies/SIN3/2-mers"
# AF2_Nmers = "/home/elvio/Desktop/Assemblies/SIN3/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/Assemblies/SIN3/MM_output"
# use_names = True 
# overwrite = True
# auto_domain_detection = False
# graph_resolution_preset = "/home/elvio/Desktop/Assemblies/SIN3/graph_resolution_preset.json"
# # graph_resolution_preset = None

# ###############################################################################

######################## Test 6 (multivalency detection) ######################

# fasta_file = "/home/elvio/Desktop/heteromultimeric_states_benchmark/to_test_9EMC/proteins_mm.fasta"
# AF2_2mers = "/home/elvio/Desktop/heteromultimeric_states_benchmark/to_test_9EMC/2-mers"
# AF2_Nmers = "/home/elvio/Desktop/heteromultimeric_states_benchmark/to_test_9EMC/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/heteromultimeric_states_benchmark/to_test_9EMC/MM_metrics_profiles_test"
# use_names = True
# overwrite = True
# # graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# auto_domain_detection = True
# graph_resolution_preset = None

###############################################################################

################### Test 6' (multivalency detection RuvBL) ####################

# fasta_file = "tests/multivalency_test/RuvBL_proteins.fasta"
# AF2_2mers = "tests/multivalency_test/2-mers"
# AF2_Nmers = "tests/multivalency_test/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/RuvBL_test"
# use_names = True
# overwrite = True
# # graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# auto_domain_detection = True
# graph_resolution_preset = None

###############################################################################

################### Test 6' (multivalency detection actin) ####################

# fasta_file = "/home/elvio/Desktop/heteromultimers_benchmark/actin/proteins_mm.fasta"
# AF2_2mers = "/home/elvio/Desktop/heteromultimers_benchmark/actin/AF2_2mers"
# AF2_Nmers = "/home/elvio/Desktop/heteromultimers_benchmark/actin/AF2_Nmers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/heteromultimers_benchmark/actin/MM_out_Nmers"
# use_names = False
# overwrite = True
# # graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# auto_domain_detection = True
# graph_resolution_preset = None

###############################################################################

####################### Test 6'' (multivalency PEX13/31) ######################

# fasta_file = "/home/elvio/Desktop/heteromultimeric_states_benchmark/3MZL_problematic/proteins_mm.fasta"
# AF2_2mers = "/home/elvio/Desktop/heteromultimeric_states_benchmark/3MZL_problematic/2-mers"
# AF2_Nmers = "/home/elvio/Desktop/heteromultimeric_states_benchmark/3MZL_problematic/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/heteromultimeric_states_benchmark/3MZL_problematic/MM_metrics_profiles_test"
# use_names = True
# overwrite = True
# # graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# auto_domain_detection = True
# graph_resolution_preset = None

###############################################################################

##################### Test 6''' (multivalency ATPase) ######################

# fasta_file = "/home/elvio/Desktop/heteromultimeric_states_benchmark/converged_5FIL/proteins_mm.fasta"
# AF2_2mers = "/home/elvio/Desktop/heteromultimeric_states_benchmark/converged_5FIL/2-mers"
# AF2_Nmers = "/home/elvio/Desktop/heteromultimeric_states_benchmark/converged_5FIL/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/heteromultimeric_states_benchmark/converged_5FIL/MM_metrics_profiles_test"
# use_names = True
# overwrite = True
# # graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# auto_domain_detection = True
# graph_resolution_preset = None

###############################################################################

################ Test 6'''' (multivalency homo-3-mers) #######################

# fasta_file = "/home/elvio/Desktop/multivalency_benchmark/input_fasta_files/multivalency_homooligomers.fasta"
# AF2_2mers = "/home/elvio/Desktop/multivalency_benchmark/multivalency_test_AF_2mers/homo2mers/"
# AF2_Nmers = "/home/elvio/Desktop/multivalency_benchmark/multivalency_test_AF_Nmers/3mers/homo3mers/"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/multivalency_benchmark/MM_output_homooligomers_test"
# use_names = False
# overwrite = True
# # graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# auto_domain_detection = True
# graph_resolution_preset = None

###############################################################################

################ Test 6'''' (multivalency hetero-3-mers) ######################

# fasta_file = "/home/elvio/Desktop/multivalency_benchmark/input_fasta_files/multivalency_heterooligomers.fasta"
# AF2_2mers = "/home/elvio/Desktop/multivalency_benchmark/multivalency_test_AF_2mers/hetero2mers/"
# AF2_Nmers = "/home/elvio/Desktop/multivalency_benchmark/multivalency_test_AF_Nmers/3mers/hetero3mers/"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/multivalency_benchmark/MM_output_heterooligomers_test"
# use_names = False
# overwrite = True
# remove_interactions = ("Indirect", "No 2-mers Data")
# # graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# auto_domain_detection = True
# graph_resolution_preset = None

###############################################################################


################ Test 6''''' (multivalency 3-mers) ############################

# fasta_file = "/home/elvio/Desktop/multivalency_benchmark/input_fasta_files/multivalency_3mers.fasta"
# AF2_2mers = "/home/elvio/Desktop/multivalency_benchmark/multivalency_test_AF_2mers"
# AF2_Nmers = "/home/elvio/Desktop/multivalency_benchmark/multivalency_test_AF_Nmers/3mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/multivalency_benchmark/MM_output_3mers_test"
# use_names = True
# overwrite = True
# remove_interactions = ("Indirect", "No 2-mers Data")
# # graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# auto_domain_detection = True
# graph_resolution_preset = None

###############################################################################

####################### Test 7 (multivalency homodimers) ######################

# fasta_file = "/home/elvio/Desktop/homomultimers_benchmark/proteins.fasta"
# AF2_2mers = "/home/elvio/Desktop/homomultimers_benchmark/AF2_2mers"
# AF2_Nmers = "/home/elvio/Desktop/homomultimers_benchmark/AF2_3-4-5-6-7-8-9-10mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/homomultimers_benchmark/multimers_Nstate_2to10mers"
# use_names = False
# overwrite = True
# # graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# auto_domain_detection = True
# graph_resolution_preset = None

###############################################################################

##################### Test 8 (Format multivalency states) #####################

# fasta_file = "/home/elvio/Desktop/test_mult/EAF6-DNTL-INGL-YEA2-EPL1_proteins.fasta"
# AF2_2mers = "/home/elvio/Desktop/test_mult/2-mers"
# AF2_Nmers = "/home/elvio/Desktop/test_mult/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/test_mult/mm_out"
# use_names = True 
# overwrite = True
# # graph_resolution_preset = "/home/elvio/Desktop/Assemblies/NuA4/graph_resolution_preset.json"
# auto_domain_detection = True
# graph_resolution_preset = None

###############################################################################

# ###################### Test 9 (TbNuA4 - No piccolo) #########################

fasta_file = "/home/elvio/Desktop/Assemblies/NuA4/no_piccolo_for_FLAP/NuA4_no_piccolo.fasta"
AF2_2mers = "/home/elvio/Desktop/Assemblies/NuA4/no_piccolo_for_FLAP/2-mers"
AF2_Nmers = "/home/elvio/Desktop/Assemblies/NuA4/no_piccolo_for_FLAP/N-mers"
# AF2_Nmers = None
out_path = "/home/elvio/Desktop/Assemblies/NuA4/no_piccolo_for_FLAP/mm_out_testing_layout_algorithm"
use_names = True 
overwrite = True
# graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
graph_resolution_preset = None
auto_domain_detection = True


###############################################################################
############################### MM main run ###################################
###############################################################################

# import multimer_mapper as mm

# Setup the root logger with desired level
log_level = 'info'
logger = mm.configure_logger(out_path = out_path, log_level = log_level, clear_root_handlers = True)(__name__)


# Run the main MultimerMapper pipeline
import multimer_mapper as mm
mm_output = mm.parse_AF2_and_sequences(fasta_file,
                                       AF2_2mers,
                                       AF2_Nmers,
                                       out_path,
                                       use_names = use_names,
                                       overwrite = overwrite,
                                       auto_domain_detection = auto_domain_detection,
                                       graph_resolution_preset = graph_resolution_preset,
                                       show_PAE_along_backbone = False)

# Generate interactive graph
# import multimer_mapper as mm
# combined_graph, dynamic_proteins, homooligomerization_states, multivalency_states = mm.generate_combined_graph(mm_output)
combined_graph_interactive = mm.interactive_igraph_to_plotly(
    mm_output["combined_graph"], out_path = out_path,
    layout_algorithm = 'fr',    
    
    # You can remove specific interaction types from the graph
    # "No 2-mers Data"
    # remove_interactions = ("Indirect", "No 2-mers Data"),
    remove_interactions = remove_interactions,
    self_loop_size = 4,
    
    # Answer y automatically
    automatic_true = True)



# mm_output['contacts_clusters'][list(mm_output['contacts_clusters'].keys())[1]][0].keys()
# ['models', 'representative', 'average_matrix', 'x_lab', 'y_lab', 'x_dom', 'y_dom', 'was_tested_in_2mers', 'was_tested_in_Nmers', 'average_2mers_matrix', 'average_Nmers_matrix', 'cluster_n']

# mm_output['contacts_clusters'][list(mm_output['contacts_clusters'].keys())[1]][0]['representative']
# mm_output['contacts_clusters'][list(mm_output['contacts_clusters'].keys())[1]][1]['representative']

# mm_output['pairwise_2mers_df'].head()
# mm_output['pairwise_Nmers_df'].head()["model"][134]


# # Explore the stoichiometric space
# import multimer_mapper as mm
# best_stoichiometry, paths = mm.stoichiometric_space_exploration_pipeline(mm_output)

# from src.stoichiometries import stoichiometric_space_exploration_pipeline
# best_stoichiometry, paths = stoichiometric_space_exploration_pipeline(mm_output)

# Create 3D network, generate a layout and create py3Dmol/Plotly visualizations
# DEFAULT FINE GRAIN CONFIG
fine_grain_layout_cfg = {
    "algorithm": "residue_optimized",
    "iterations": 200,
    "min_contact_distance": 50,
    "max_contact_distance": 60,
    "contact_force_strength": 10.0,
    "repulsion_strength": 10.0,
    "global_repulsion_strength": 5,
    "torque_strength": 100.0,
    "initial_step_size": 0.5,
    "final_step_size": 0.005,
    "min_interprotein_distance": 200.0,  # Minimum distance between protein centers
    "surface_alignment_strength": 25,   # Strength of surface face-to-face alignment
    "line_separation_strength": 10.0,     # Strength of contact line separation
    "n_contacts_sample": 5
}

nw = mm.Network(mm_output['combined_graph'], logger = logger)
nw.generate_layout_fine_grain(**fine_grain_layout_cfg)
nw.generate_interactive_3d_plot(save_path = out_path + '/graphs/3D_graph_py3Dmol.html', show_plot=True)
nw.generate_plotly_3d_plot(save_path = out_path + '/graphs/3D_graph_plotly.html', show_plot=True)

# Generate RMSF, pLDDT clusters & RMSD trajectories considering models as monomers
# import multimer_mapper as mm
mm_monomers_traj = mm.generate_RMSF_pLDDT_cluster_and_RMSD_trajectories(
    mm_output = mm_output, out_path = out_path)

# Get suggested combinations
suggested_combinations = mm.suggest_combinations(mm_output = mm_output, 
                                                 # To ommit saving, change to None
                                                 out_path = out_path)

# Create the final report
mm.create_report(out_path)

###############################################################################
############################### Access output #################################
###############################################################################

mm_output.keys()
sorted_tuple_pair = ('RuvBL1', 'RuvBL2')
mm_output['contacts_clusters'][sorted_tuple_pair].keys()
mm_output['contacts_clusters'][sorted_tuple_pair][1]
model_k = mm_output['contacts_clusters'][sorted_tuple_pair][1]['models'][0]
model_k[0]

combined_graph = mm_output["combined_graph"]


###############################################################################
############################# Advanced features ###############################
###############################################################################

# For visualization of contacts data
mm.visualize_pair_matrices(mm_output,
                           pair=None,
                           matrix_types=['is_contact', 'PAE', 'min_pLDDT', 'distance'], 
                           # Combine all models into one single average matrix?
                           combine_models=False,
                           # Max number of models to display
                           max_models=5,
                           # Can be set to 'auto'
                           aspect_ratio = 'equal')

# Generate RMSD trajectories for pairs of interacting protein domains
mm_pairwise_domain_traj = mm.generate_pairwise_domain_trajectories(
    # Pair of domains to get the trajectory
    P1_ID = 'EAF6', P1_dom = 2, 
    P2_ID = 'EAF6', P2_dom = 2,
    mm_output = mm_output, out_path = out_path,
    
    # Configuration of the trajectory -----------------------------------
    
    # One of ['domains_mean_plddt', 'domains_CM_dist', 'domains_pdockq'] 
    reference_metric = 'domains_pdockq',
    # One of [max, min]
    ref_metric_method = max,
    # True or False
    reversed_trajectory = False)

# Generates the same trajectory, but with other domains as context
mm.generate_pairwise_domain_trajectory_in_context(mm_pairwise_domain_traj,
                                                  mm_output,
                                                  out_path,
                                                  P3_ID = "EPL1", P3_dom = 4,
                                                  sort_by= 'RMSD')


###############################################################################
############################### For developers ################################
###############################################################################


# Access protein contact surface 
for prot in nw.get_proteins():
    # surf = prot.get_surface()
    surf = prot.surface
    
    # Residues involved in interactions
    print(surf.get_interacting_residues())
    
    # Each distinct surface
    print(surf.get_surfaces())
    
    # Residues clasified by group ( A: not shared,
    #                               B: Co-occupied by differt contact clusters with the same protein)
    #                               C: Co-occupied by differt proteins)
    print(surf.get_residues_by_group())



###############################################################################
######################### TESTS: Contacts clustering ##########################
###############################################################################

# Contact Cluster graphing
from src.analyze_multivalency import cluster_all_pairs
pairwise_contacts = mm_output['pairwise_contact_matrices']
results = cluster_all_pairs(
    mm_contacts                             = pairwise_contacts,
    mm_output                               = mm_output,
    contacts_clustering_method              = "mc_threshold",
    contact_fraction_threshold              = 0.1,
    mc_threshold                            = 10.0,
    use_median                              = True,
    refinement_contact_similarity_threshold = 0.5,
    refinement_cf_threshold                 = 0.5)

###############################################################################
################## TESTS: For residue-residue contacts graph ##################
###############################################################################


combined_graph = mm_output['combined_graph']

for p, prot_ID in enumerate(combined_graph.vs['name']):
    print(combined_graph.vs[p]['name'] == prot_ID)
    
# Vertices attributes
['name', 'color', 'meaning', 'IDs', 'domains_df', 'RMSD_df']
combined_graph.vertex_attributes()
combined_graph.vs['name']
combined_graph.vs['IDs']
combined_graph.vs['meaning']
combined_graph.vs[0]['ref_PDB_chain']                           # <---------------- PDB.Model.Model


# Edges attributes
['N_mers_data', 'N_mers_info', '2_mers_data', '2_mers_info',
 'homooligomerization_states', 'dynamics', 'name']
combined_graph.edge_attributes()
combined_graph.es[2]['name']
combined_graph.es['dynamics']
combined_graph.es['homooligomerization_states']
combined_graph.es[60]['valency']['models']                       # <---------------- dict
combined_graph.es['multivalency_states']


combined_graph.es[60]['name']
combined_graph.es[60]['homooligomerization_states']
# valency = combined_graph.es[2]['valency']
# from src.contact_graph import add_contact_classification_matrix
# add_contact_classification_matrix(combined_graph)
# contact_classification_example = valency['contact_classification_matrix']
# plt.imshow(contact_classification_example, cmap = 'tab10')


from src.contact_graph import Network
nw = Network(mm_output['combined_graph'], logger = logger)
nw.generate_layout()
nw.generate_py3dmol_plot(save_path = out_path + '/3D_graph.html')

###############################################################################

# Examples
tuple_pair = tuple(sorted(['EAF6', 'EAF6']))
tuple_pair = tuple(sorted(['EAF6', 'EPL1']))
tuple_pair = tuple(sorted(['EAF6', 'PHD1']))
cluster_n = 0

# ----------------------- Contact clusters dict -------------------------------

# Each interacting pairs have contact clusters / contact maps
mm_output['contacts_clusters'].keys()

# Number of clusters for the speficif protein 
len(mm_output['contacts_clusters'][tuple_pair].keys())

# Models apporting to cluster 0 and their average matrix
mm_output['contacts_clusters'][tuple_pair][cluster_n].keys()
mm_output['contacts_clusters'][tuple_pair][cluster_n]['models']
mm_output['contacts_clusters'][tuple_pair][cluster_n]['average_matrix']
mm_output['contacts_clusters'][tuple_pair][cluster_n]['average_2mers_matrix']
mm_output['contacts_clusters'][tuple_pair][cluster_n]['average_Nmers_matrix']
mm_output['contacts_clusters'][tuple_pair][cluster_n]['was_tested_in_2mers']
mm_output['contacts_clusters'][tuple_pair][cluster_n]['was_tested_in_Nmers']


from src.analyze_multivalency import print_contact_clusters_number
print_contact_clusters_number(mm_output)

mm_output['contacts_clusters'][tuple_pair][0]


###############################################################################
###################### TESTS: To generate metrics plots #######################
###############################################################################

# Create a dataframe to store data
stability_df = pd.DataFrame()

# For each model
mm_output.keys()

# Extract msiPAE, miPAE, pDockQ, pTM, ipTM
mm_output['pairwise_2mers_df'].columns
mm_output['pairwise_Nmers_df'].columns
mm_output['pairwise_2mers_df_F3'].columns
mm_output['pairwise_Nmers_df_F3'].columns

# ----------------- Extract aiPAE, miPAE

# Protein pairs that interact (e.g: {('RuvBL1', 'RuvBL1'), ('RuvBL1', 'RuvBL2'), ('RuvBL2', 'RuvBL2')})
int_pairs = {tuple(sorted(e["name"])) for e in mm_output['combined_graph'].es}

# These keys are the pairs of proteins that have matrixes. Example: dict_keys([('RuvBL1', 'RuvBL1'), ('RuvBL1', 'RuvBL2'), ('RuvBL2', 'RuvBL2')])
pairs = list(mm_output['pairwise_contact_matrices'].keys())

# These keys represent the decomposed sub-models in a pairwise fashion of bigger (or equally, eg 2 proteins) models
# Example: dict_keys([(('RuvBL1', 'RuvBL2'), ('A', 'B'), 1), (('RuvBL1', 'RuvBL2'), ('A', 'B'), 2), (('RuvBL1', 'RuvBL2'), ('A', 'B'), 3), (('RuvBL1', 'RuvBL2'), ('A', 'B'), 4), (('RuvBL1', 'RuvBL2'), ('A', 'B'), 5), (('RuvBL1', 'RuvBL2', 'RuvBL2'), ('A', 'B'), 1), (('RuvBL1', 'RuvBL2', 'RuvBL2'), ('A', 'B'), 2), ...])
# Each key is a tuple with the information of each chain pair and its matching model: [(tuple with proteins in the model: combination of proteins, corresponding chain IDs: e.g. ("A","B") or ("B","C") etc., the rank of the model: 1 to 5), ...] The combination of proteins and the rank identifies the model per se
# I. e., it tells which chains the matixes correspond and which model (proteins in the model + rank)
sub_models = list(mm_output['pairwise_contact_matrices'][pairs[1]].keys())

# These are the sub-matrixes available for each decomposed pair: dict_keys(['PAE', 'min_pLDDT', 'distance', 'is_contact'])
mm_output['pairwise_contact_matrices'][pairs[1]][sub_models[5]].keys()

# How to extract the miPAE (e.g: 4.26)
np.min(mm_output['pairwise_contact_matrices'][pairs[1]][sub_models[5]]['PAE'])

# How to compute aiPAE (eg: 16.011199544046406)
np.mean(mm_output['pairwise_contact_matrices'][pairs[1]][sub_models[5]]['PAE'])

# How to know if they interact (eg: True)
np.sum(mm_output['pairwise_contact_matrices'][pairs[1]][sub_models[5]]['is_contact']) >= 5

# To know if the pair of chains is the same of the protein pair under analysis during the iteration,
# the chain ids (which can be converted to indexes, eg: A=0, B=1, etc.) can be used to match the protein
# index in the "tuple with proteins in the model: combination of proteins"

# ----------------- Extract pLDDTs

# Protein pairs that interact (e.g: {('RuvBL1', 'RuvBL1'), ('RuvBL1', 'RuvBL2'), ('RuvBL2', 'RuvBL2')})
int_pairs = {tuple(sorted(e["name"])) for e in mm_output['combined_graph'].es}

# Example: dict_keys(['RuvBL1', 'RuvBL2'])
prots = list(mm_monomers_traj.keys())

# dict_keys(['rmsd_values', 'rmsf_values', 'rmsf_values_per_domain', 'b_factors', 'b_factor_clusters', 'chain_types', 'model_info', 'rmsd_trajectory_file'])
variables = list(mm_monomers_traj[prot].keys())

# list of tuples with the information of each chain: [(chain ID: e.g.: "A" or "B" or "C" etc., tuple with proteins in the model: combination of proteins, rank: 1 to 5), ...] The combination of proteins and the rank identifies the model per se
# I. e., it tells which chain it correspond and which model (proteins in the model + rank)
mm_monomers_traj[prot][variables[6]]

# list of arrays with per residue pLDDT (each index matches their corresponding model info )
mm_monomers_traj[prot][variables[3]]





# Get pLDDTs
[[np.mean([a.bfactor for a in ch.get_atoms() if a.id == "CA"]) for ch in mod.get_chains()] for mod in mm_output['pairwise_2mers_df']['model']]
[[np.mean([a.bfactor for a in ch.get_atoms() if a.id == "CA"]) for ch in mod.get_chains()] for mod in mm_output['pairwise_Nmers_df']['model']]


###############################################################################
################### TESTS: To find the best stoichiometries ###################
###############################################################################




# Weight
def get_edge_weight(graph_edge, classification_df: pd.DataFrame, default_edge_weight = 0.5):

    edge_dynamics = graph_edge["dynamics"]
    edge_width_is_variable = classification_df.query(f'Classification == "{edge_dynamics}"')["Variable_Edge_width"].iloc[0]
    

    if edge_width_is_variable:
        edge_weight_2mer_iptm = np.mean(list(graph_edge["2_mers_data"]["ipTM"]))
        edge_weight_PAE = 1/ np.mean(list(graph_edge["2_mers_data"]["min_PAE"]) + list(graph_edge["N_mers_data"]["min_PAE"]))
        edge_weight = edge_weight_2mer_iptm * edge_weight_PAE * 10
        
    
    # Use mean number of models that surpass the cutoff and 1/mean(miPAE) to construct a weight
    # edge_weight_Nmers = int(np.mean(list(graph_edge["2_mers_data"]["N_models"]) + list(graph_edge["N_mers_data"]["N_models"])))
    # edge_weight_PAE = int(1/ np.mean(list(graph_edge["2_mers_data"]["min_PAE"]) + list(graph_edge["N_mers_data"]["min_PAE"])))
    # edge_weight = edge_weight_Nmers * edge_weight_PAE

        # Limit to reasonable values
        if edge_weight < 1:
            return 1
        elif edge_weight > 8:
            return 8
        return edge_weight

    # If it has fixed length
    else:
        return default_edge_weight


from src.interpret_dynamics import read_classification_df
for e in combined_graph.es:
    print(e['valency']['cluster_n'], e['name'])
    print(f'   - Dynamics: {e["dynamics"]}')
    print(f'   - Weight: {get_edge_weight(e, read_classification_df())}')







# ###############################################################################
# ####################### To find a better stability check ######################
# ###############################################################################

# from src.analyze_multivalency import get_expanded_Nmers_df_for_pair, add_chain_information_to_df

# def get_set_of_chains_in_model(model_pairwise_df: pd.DataFrame) -> set:
    
#     chains_set = set()
    
#     for i, row in model_pairwise_df.iterrows():
#         model_chains = list(row['model'].get_chains())
#         chain_ID1 = model_chains[0].get_id()
#         chain_ID2 = model_chains[1].get_id()
        
#         chains_set.add(chain_ID1)
#         chains_set.add(chain_ID2)
    
#     return chains_set

# def does_all_have_at_least_one_interactor(model_pairwise_df: pd.DataFrame,
#                                           min_PAE_cutoff_Nmers: int | float,
#                                           pDockQ_cutoff_Nmers: int | float,
#                                           N_models_cutoff: int) -> bool:
    

#     for i, row in model_pairwise_df.iterrows():
        
#         for chain_ID in get_set_of_chains_in_model(model_pairwise_df):
            
#             # Variable to count the number of times the chains surpass the cutoffs
#             models_in_which_chain_surpass_cutoff = 0
            
#             # Count one by one
#             for rank in range(1, 6):
#                 chain_df = (model_pairwise_df
#                                  .query('rank == @rank')
#                                  .query('chain_ID1 == @chain_ID | chain_ID2 == @chain_ID')
#                             )
#                 for c, chain_pair in chain_df.iterrows():
#                     if chain_pair["min_PAE"] <= min_PAE_cutoff_Nmers and chain_pair["pDockQ"] >= pDockQ_cutoff_Nmers:
#                         models_in_which_chain_surpass_cutoff += 1
#                         break
                
#             if not models_in_which_chain_surpass_cutoff >= N_models_cutoff:
#                 return False
#     return True

# # Multivalent pair to test and its data
# pair = ("RuvBL1", "RuvBL2")
# expanded_Nmers_for_pair_df: pd.DataFrame        = get_expanded_Nmers_df_for_pair(pair, mm_output)
# expanded_Nmers_for_pair_models: set[tuple[str]] = set(expanded_Nmers_for_pair_df['proteins_in_model'])

# # I want to use the contacts stored in here instead
# mm_output['pairwise_contact_matrices'][pair].keys()
# # For example, the couple of chains B and C for the rank 3 model of the prediction of the 2(RuvBL1)/1(RuvBL2)
# mm_output['pairwise_contact_matrices'][pair][(('RuvBL1', 'RuvBL1', 'RuvBL2'), ('B', 'C'), 3)]
# # This is how the number of contacts for this particular prediction, couple of chains and rank can be counted
# mm_output['pairwise_contact_matrices'][pair][(('RuvBL1', 'RuvBL1', 'RuvBL2'), ('B', 'C'), 3)]['is_contact'].sum()

# # Create empty variable to store which states are stable
# multivalent_pairs: list[tuple[str]] = [pair]
# multivalency_states: dict = {pair: {} for pair in multivalent_pairs}

# # For each expanded Nmer
# for model in list(expanded_Nmers_for_pair_models):
    
#     # Separate only data for the current expanded heteromeric state and add chain info
#     model_pairwise_df: pd.DataFrame = expanded_Nmers_for_pair_df.query('proteins_in_model == @model')
#     add_chain_information_to_df(model_pairwise_df)
    
#     # Make the verification
#     all_have_at_least_one_interactor: bool = does_all_have_at_least_one_interactor(
#                                                 model_pairwise_df,
#                                                 mm.min_PAE_cutoff_Nmers,
#                                                 mm.pDockQ_cutoff_Nmers,
#                                                 mm.N_models_cutoff)
    
#     # Add if it surpass cutoff to N_states
#     multivalency_states[pair][tuple(sorted(model))] = all_have_at_least_one_interactor








# import pandas as pd
# import networkx as nx
# from typing import Dict, Set, Tuple, List, Union, Any

# def does_nmer_is_fully_connected_network(
#         model_pairwise_df: pd.DataFrame,
#         mm_output: Dict,
#         pair: Tuple[str, str],
#         Nmers_contacts_cutoff: int = 3,
#         N_models_cutoff: int = 1) -> bool:
#     """
#     Check if all subunits form a fully connected network using contacts.
    
#     Args:
#         model_pairwise_df (pd.DataFrame): DataFrame containing pairwise interactions.
#         mm_output (Dict): Dictionary containing contact matrices.
#         pair (Tuple[str, str]): The protein pair being analyzed.
#         Nmers_contacts_cutoff (int, optional): Minimum number of contacts to consider interaction. Defaults to 3.
#         N_models_cutoff (int, optional): Minimum number of ranks that need to be fully connected. Defaults to 1.
    
#     Returns:
#         bool: True if network is fully connected in at least N_models_cutoff ranks, False otherwise.
#     """
#     # Get all unique chains in this model
#     all_chains = get_set_of_chains_in_model(model_pairwise_df)
    
#     # Get the proteins_in_model from the first row (should be the same for all rows)
#     if model_pairwise_df.empty:
#         return False
#     proteins_in_model = model_pairwise_df.iloc[0]['proteins_in_model']
    
#     # Track how many ranks have fully connected networks
#     ranks_with_fully_connected_network = 0
    
#     # For each rank (1-5)
#     for rank in range(1, 6):
#         # Create a graph for this rank
#         G = nx.Graph()
#         # Add all chains as nodes
#         G.add_nodes_from(all_chains)
        
#         # For each pair of chains
#         for chain1 in all_chains:
#             for chain2 in all_chains:
#                 if chain1 >= chain2:  # Skip self-connections and avoid double counting
#                     continue
                
#                 # Try to find contact data for this chain pair in this rank
#                 chain_pair = (chain1, chain2)
#                 try:
#                     contacts = mm_output['pairwise_contact_matrices'][pair][(proteins_in_model, chain_pair, rank)]
#                     num_contacts = contacts['is_contact'].sum()
                    
#                     # If contacts exceed threshold, add edge to graph
#                     if num_contacts >= Nmers_contacts_cutoff:
#                         G.add_edge(chain1, chain2)
#                 except KeyError:
#                     # This chain pair might not exist in the contact matrices
#                     pass
        
#         # Check if graph is connected (all nodes can reach all other nodes)
#         if len(all_chains) > 0 and nx.is_connected(G):
#             ranks_with_fully_connected_network += 1
    
#     # Return True if enough ranks have fully connected networks
#     return ranks_with_fully_connected_network >= N_models_cutoff

# def get_set_of_chains_in_model(model_pairwise_df: pd.DataFrame) -> set:
#     """
#     Extract all unique chain IDs from the model_pairwise_df.
    
#     Args:
#         model_pairwise_df (pd.DataFrame): DataFrame containing pairwise interactions.
    
#     Returns:
#         set: Set of all unique chain IDs.
#     """
#     chains_set = set()
    
#     for i, row in model_pairwise_df.iterrows():
#         model_chains = list(row['model'].get_chains())
#         chain_ID1 = model_chains[0].get_id()
#         chain_ID2 = model_chains[1].get_id()
        
#         chains_set.add(chain_ID1)
#         chains_set.add(chain_ID2)
    
#     return chains_set

# # Example usage (similar to your current implementation):
# # Multivalent pair to test and its data
# pair = ("RuvBL1", "RuvBL2")
# expanded_Nmers_for_pair_df = get_expanded_Nmers_df_for_pair(pair, mm_output)
# expanded_Nmers_for_pair_models = set(expanded_Nmers_for_pair_df['proteins_in_model'])

# # Create empty variable to store which states are stable
# multivalent_pairs = [pair]
# multivalency_states = {pair: {} for pair in multivalent_pairs}

# # For each expanded Nmer
# for model in list(expanded_Nmers_for_pair_models):
    
#     # Separate only data for the current expanded heteromeric state and add chain info
#     model_pairwise_df = expanded_Nmers_for_pair_df.query('proteins_in_model == @model')
#     add_chain_information_to_df(model_pairwise_df)
    
#     # Make the verification using the new function
#     fully_connected_network = is_fully_connected_network(
#                                 model_pairwise_df,
#                                 mm_output,
#                                 pair,
#                                 Nmers_contacts_cutoff=3,
#                                 N_models_cutoff=mm.N_models_cutoff-1)
    
#     # Add if it surpass cutoff to N_states
#     multivalency_states[pair][tuple(sorted(model))] = fully_connected_network
