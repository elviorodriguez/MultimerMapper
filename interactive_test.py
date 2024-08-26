# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import multimer_mapper as mm

pd.set_option( 'display.max_columns' , None )


################################# Test 1 ######################################

# fasta_file = "tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta"
# AF2_2mers = "tests/EAF6_EPL1_PHD1/2-mers"
# AF2_Nmers = "tests/EAF6_EPL1_PHD1/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/MM_interactive_test"
# use_names = True 
# overwrite = True
# auto_domain_detection = False
# graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# # graph_resolution_preset = None

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

####################### Test 7 (multivalency homodimers) ######################

fasta_file = "/home/elvio/Desktop/homomultimers_benchmark/proteins.fasta"
AF2_2mers = "/home/elvio/Desktop/homomultimers_benchmark/AF2_2mers"
AF2_Nmers = "/home/elvio/Desktop/homomultimers_benchmark/AF2_3-4-5-6-7-8-9-10mers"
# AF2_Nmers = None
out_path = "/home/elvio/Desktop/homomultimers_benchmark/multimers_Nstate_2to10mers"
use_names = False
overwrite = True
# graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
auto_domain_detection = True
graph_resolution_preset = None

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

import multimer_mapper as mm
# Generate interactive graph
combined_graph_interactive = mm.interactive_igraph_to_plotly(
    mm_output["combined_graph"], out_path = out_path,
    layout_algorithm = 'fr',    
    
    # You can remove specific interaction types from the graph
    
    remove_interactions = ("Indirect", "No 2-mers Data"),
    self_loop_size = 5,
    
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
sorted_tuple_pair = ('RuvBL1', 'RuvBL2')
mm_output['contacts_clusters'][sorted_tuple_pair].keys()


mm_output["combined_graph"].es.attributes()
mm_output["combined_graph"].es[0]['valency']['cluster_n']


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
################################## TESTS ######################################
###############################################################################

# Contact Cluster graphing
from src.analyze_multivalency import  cluster_all_pairs
results = cluster_all_pairs(
    mm_contacts = mm_output['pairwise_contact_matrices'],
    mm_output   = mm_output,
    contact_fraction_threshold = 0.5)

















combined_graph = mm_output['combined_graph']

for p, prot_ID in enumerate(combined_graph.vs['name']):
    print(combined_graph.vs[p]['name'] == prot_ID)
    



# I have to add 'ref_PDB_model'
['name', 'color', 'meaning', 'IDs', 'domains_df', 'RMSD_df']
combined_graph.vertex_attributes()
combined_graph.vs['name']
combined_graph.vs['IDs']
combined_graph.vs['meaning']
combined_graph.vs[0]['ref_PDB_chain']               # <---------------- PDB.Model.Model


# I have to add 'valency'
['N_mers_data', 'N_mers_info', '2_mers_data', '2_mers_info',
 'homooligomerization_states', 'dynamics', 'name']
combined_graph.edge_attributes()
combined_graph.es['name']
combined_graph.es['dynamics']
combined_graph.es['homooligomerization_states']
combined_graph.es['valency']                        # <---------------- dict


# To search ref pdb
protein_ID = 'EAF6'
mm_output['sliced_PAE_and_pLDDTs'][protein_ID].keys()
mm_output['sliced_PAE_and_pLDDTs'][protein_ID]['PDB_xyz']
mm_output['sliced_PAE_and_pLDDTs'][protein_ID]['PDB_xyz']


###############################################################################
################### For residue-residue contacts graph ########################
###############################################################################

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

    
    




