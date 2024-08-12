# -*- coding: utf-8 -*-

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
graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# graph_resolution_preset = None

###############################################################################

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

###############################################################################
############################### MM main run ###################################
###############################################################################

import multimer_mapper as mm
log_level = 'info'
logger = mm.configure_logger(out_path = out_path, log_level = log_level, clear_root_handlers = True)(__name__)

# Run the main MultimerMapper pipeline
mm_output = mm.parse_AF2_and_sequences(fasta_file,
                                       AF2_2mers,
                                       AF2_Nmers,
                                       out_path,
                                       use_names = use_names,
                                       overwrite = overwrite,
                                       graph_resolution_preset = graph_resolution_preset)

# Generate interactive graph
combined_graph_interactive = mm.interactive_igraph_to_plotly(
    mm_output["combined_graph"], out_path = out_path,
    
    # You can remove specific interaction types from the graph
    remove_interactions = ("Indirect",))

# Get suggested combinations
suggested_combinations = mm.suggest_combinations(mm_output = mm_output, 
                                                 # To ommit saving, change to None
                                                 out_path = out_path)

# Generate RMSF, pLDDT clusters & RMSD trajectories considering models as monomers
mm_monomers_traj = mm.generate_RMSF_pLDDT_cluster_and_RMSD_trajectories(
    mm_output = mm_output, out_path = out_path)

# Contacts extraction
import multimer_mapper as mm
mm_contacts = mm.compute_contacts(mm_output, out_path)


###############################################################################
############################# Advanced features ###############################
###############################################################################

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


import matplotlib.pyplot as plt

domains_df = mm_output['domains_df']


# 2-mers contacts ---------------------------------
ID_a = mm_output['prot_IDs'][0]    # EPL1
ID_b = mm_output['prot_IDs'][1]    # EAF6
L_a  = mm_output['prot_lens'][0]
L_b  = mm_output['prot_lens'][1]
domains_a = domains_df[domains_df['Protein_ID'] == ID_a]
domains_b = domains_df[domains_df['Protein_ID'] == ID_b]
ab_contacts_df = mm_contacts['contacts_2mers_df'].query(
    f'protein_ID_a == "{ID_a}" & protein_ID_b == "{ID_b}"')
r_a  = ab_contacts_df["res_a"]
r_b  = ab_contacts_df["res_b"]

# Create a scatter plot
plt.scatter(r_a, r_b, s = 1)

# Set axis limits
plt.xlim(0, L_a - 1)  # Set x-axis limits from 0 to 6
plt.ylim(0, L_b - 1)  # Set y-axis limits from 5 to 40

# Add vertical lines for domains of protein A
for _, row in domains_a.iterrows():
    plt.axvline(x=row['Start'] - 1, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(x=row['End'], color='black', linestyle='--', linewidth=0.5)

# Add horizontal lines for domains of protein B
for _, row in domains_b.iterrows():
    plt.axhline(y=row['Start'] - 1, color='black', linestyle='--', linewidth=0.5)
    plt.axhline(y=row['End'], color='black', linestyle='--', linewidth=0.5)

# Add titles and labels
plt.title(f'{ID_a} vs {ID_b} contacts (2-mers)')
plt.xlabel(f'{ID_a}')
plt.ylabel(f'{ID_b}')

# Show the plot
plt.show()


# N-mers contacts ---------------------------------
ID_a = mm_output['prot_IDs'][0]    # EPL1
ID_b = mm_output['prot_IDs'][1]    # EAF6
L_a  = mm_output['prot_lens'][0]
L_b  = mm_output['prot_lens'][1]
ab_contacts_df = mm_contacts['contacts_Nmers_df'].query(
    f'protein_ID_a == "{ID_b}" & protein_ID_b == "{ID_a}"')
r_a  = ab_contacts_df["res_a"]
r_b  = ab_contacts_df["res_b"]

# Create a scatter plot
plt.scatter(r_b, r_a, s = 1)

# Set axis limits
plt.xlim(0, L_a - 1)  # Set x-axis limits from 0 to 6
plt.ylim(0, L_b - 1)  # Set y-axis limits from 5 to 40

# Add vertical lines for domains of protein A
for _, row in domains_a.iterrows():
    plt.axvline(x=row['Start'] - 1, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(x=row['End'], color='black', linestyle='--', linewidth=0.5)

# Add horizontal lines for domains of protein B
for _, row in domains_b.iterrows():
    plt.axhline(y=row['Start'] - 1, color='black', linestyle='--', linewidth=0.5)
    plt.axhline(y=row['End'], color='black', linestyle='--', linewidth=0.5)

# Add titles and labels
plt.title(f'{ID_a} vs {ID_b} contacts (N-mers)')
plt.xlabel(f'{ID_a}')
plt.ylabel(f'{ID_b}')

# Show the plot
plt.show()
