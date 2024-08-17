# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import multimer_mapper as mm

pd.set_option( 'display.max_columns' , None )


################################# Test 1 ######################################

fasta_file = "tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta"
AF2_2mers = "tests/EAF6_EPL1_PHD1/2-mers"
AF2_Nmers = "tests/EAF6_EPL1_PHD1/N-mers"
# AF2_Nmers = None
out_path = "/home/elvio/Desktop/MM_interactive_test"
use_names = True 
overwrite = True
auto_domain_detection = False
graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# graph_resolution_preset = None

##############################################################################

# ################################# Test 2 ######################################

# fasta_file = "/home/elvio/Desktop/Assemblies/BDF2_HDAC3/SIN3_proteins.fasta"
# AF2_2mers = "/home/elvio/Desktop/Assemblies/BDF2_HDAC3/2-mers"
# AF2_Nmers = "/home/elvio/Desktop/Assemblies/BDF2_HDAC3/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/MM_SIN3"
# use_names = True 
# overwrite = True
# graph_resolution_preset = "/home/elvio/Desktop/MM_SIN3/graph_resolution_preset.json"
# # graph_resolution_preset = None

# ################################# Test 3 ######################################

# fasta_file = "/home/elvio/Desktop/Assemblies/NuA4/NuA4_proteins.fasta"
# AF2_2mers = "/home/elvio/Desktop/Assemblies/NuA4/2-mers"
# AF2_Nmers = "/home/elvio/Desktop/Assemblies/NuA4/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/Assemblies/NuA4/MM_NuA4"
# use_names = True 
# overwrite = True
# graph_resolution_preset = "/home/elvio/Desktop/Assemblies/NuA4/MM_NuA4/graph_resolution_preset.json"
# # graph_resolution_preset = None

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

# ################################ Test 5 (SIN3) ################################

# fasta_file = "/home/elvio/Desktop/Assemblies/SIN3/SIN3_proteins.fasta"
# AF2_2mers = "/home/elvio/Desktop/Assemblies/SIN3/2-mers"
# AF2_Nmers = "/home/elvio/Desktop/Assemblies/SIN3/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/Assemblies/SIN3/MM_output"
# use_names = True 
# overwrite = True
# # graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# graph_resolution_preset = None

# ###############################################################################

######################## Test 6 (multivalency detection) ######################

# fasta_file = "tests/multivalency_test/RuvBL_proteins.fasta"
# AF2_2mers = "tests/multivalency_test/2-mers"
# AF2_Nmers = "tests/multivalency_test/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/MM_multivalency_test"
# use_names = True 
# overwrite = True
# # graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# auto_domain_detection = True
# graph_resolution_preset = None

###############################################################################

###############################################################################
############################### MM main run ###################################
###############################################################################

# Setup the root logger with desired level
log_level = 'info'
logger = mm.configure_logger(out_path = out_path, log_level = log_level, clear_root_handlers = True)(__name__)

# Run the main MultimerMapper pipeline
mm_output = mm.parse_AF2_and_sequences(fasta_file,
                                       AF2_2mers,
                                       AF2_Nmers,
                                       out_path,
                                       use_names = use_names,
                                       overwrite = overwrite,
                                       auto_domain_detection = auto_domain_detection,
                                       graph_resolution_preset = graph_resolution_preset)

# Generate interactive graph
combined_graph_interactive = mm.interactive_igraph_to_plotly(
    mm_output["combined_graph"], out_path = out_path,
    layout_algorithm = 'kk',    
    
    # You can remove specific interaction types from the graph
    remove_interactions = ("Indirect",),
    
    # Answer y automatically
    automatic_true = True)

# Get suggested combinations
suggested_combinations = mm.suggest_combinations(mm_output = mm_output, 
                                                 # To ommit saving, change to None
                                                 out_path = out_path)

# Generate RMSF, pLDDT clusters & RMSD trajectories considering models as monomers
mm_monomers_traj = mm.generate_RMSF_pLDDT_cluster_and_RMSD_trajectories(
    mm_output = mm_output, out_path = out_path)


###############################################################################
############################### Access output #################################
###############################################################################

mm_output.keys()


###############################################################################
############################# Advanced features ###############################
###############################################################################

# For visualization of contact clusters data
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
################################## TESTS ######################################
###############################################################################

tuple_pair = ('EAF6', 'EAF6')
cluster_n = 0

# ----------------------- Contact clusters dict -------------------------------

# Each interacting pairs have contact clusters / contact maps
mm_output['contacts_clusters'].keys()

# Number of clusters for the speficif protein 
len(mm_output['contacts_clusters'][tuple_pair].keys())

# Models apporting to cluster 0 and their average matrix
mm_output['contacts_clusters'][tuple_pair][cluster_n]['models']
mm_output['contacts_clusters'][tuple_pair][cluster_n]['average_matrix']

# I need to add this information
mm_output['contacts_clusters'][tuple_pair][cluster_n]['average_2mers_matrix']
mm_output['contacts_clusters'][tuple_pair][cluster_n]['average_Nmers_matrix']

IDs_2mers_models = [ model_id for model_id in mm_output['contacts_clusters'][tuple_pair][cluster_n]['models'] if len(model_id[0]) == 2 ]
IDs_Nmers_models = [ model_id for model_id in mm_output['contacts_clusters'][tuple_pair][cluster_n]['models'] if len(model_id[0])  > 2 ]


# ----------------------- Contact clusters dict -------------------------------

# Available pairs with contacts
mm_output['pairwise_contact_matrices'].keys()

# Available models for the pair
mm_output['pairwise_contact_matrices'][('EAF6', 'EAF6')].keys()

# For each model, we have ['PAE', 'min_pLDDT', 'distance', 'is_contact'] matrices
mm_output['pairwise_contact_matrices'][('EAF6', 'EAF6')][(('EAF6', 'EAF6'), ('A', 'B'), 1)].keys()

def print_contact_clusters_number(mm_output):
    
    # Everything together
    for pair in mm_output['contacts_clusters'].keys():
  
        # Extract the Nº of contact clusters for the pair (valency)
        clusters = len(mm_output['contacts_clusters'][('EAF6', 'EAF6')].keys())
        print(f'Pair {pair} interact through {clusters} modes')

print_contact_clusters_number(mm_output)