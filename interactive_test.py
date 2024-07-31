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

################################# Test 2 ######################################

fasta_file = "/home/elvio/Desktop/Assemblies/BDF2_HDAC3/SIN3_proteins.fasta"
AF2_2mers = "/home/elvio/Desktop/Assemblies/BDF2_HDAC3/2-mers"
AF2_Nmers = "/home/elvio/Desktop/Assemblies/BDF2_HDAC3/N-mers"
# AF2_Nmers = None
out_path = "/home/elvio/Desktop/MM_SIN3"
use_names = True 
overwrite = True
graph_resolution_preset = "/home/elvio/Desktop/MM_SIN3/graph_resolution_preset.json"
# graph_resolution_preset = None

################################# Test 3 ######################################

fasta_file = "/home/elvio/Desktop/Assemblies/NuA4/NuA4_proteins.fasta"
AF2_2mers = "/home/elvio/Desktop/Assemblies/NuA4/2-mers"
AF2_Nmers = "/home/elvio/Desktop/Assemblies/NuA4/N-mers"
# AF2_Nmers = None
out_path = "/home/elvio/Desktop/Assemblies/NuA4/MM_NuA4"
use_names = True 
overwrite = True
graph_resolution_preset = "/home/elvio/Desktop/Assemblies/NuA4/MM_NuA4/graph_resolution_preset.json"
# graph_resolution_preset = None

###################### Test 4 (indirect interactions) #########################

fasta_file = "tests/indirect_interactions/TINTIN.fasta"
AF2_2mers = "tests/indirect_interactions/2-mers"
AF2_Nmers = "tests/indirect_interactions/N-mers"
# AF2_Nmers = None
out_path = "/home/elvio/Desktop/Assemblies/indirect_interaction_tests_N_mers"
use_names = True 
overwrite = True
# graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
graph_resolution_preset = None

################################ Test 5 (SIN3) ################################

fasta_file = "/home/elvio/Desktop/Assemblies/SIN3/SIN3_proteins.fasta"
AF2_2mers = "/home/elvio/Desktop/Assemblies/SIN3/2-mers"
AF2_Nmers = "/home/elvio/Desktop/Assemblies/SIN3/N-mers"
# AF2_Nmers = None
out_path = "/home/elvio/Desktop/Assemblies/SIN3/MM_output"
use_names = True 
overwrite = True
# graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
graph_resolution_preset = None

###############################################################################

###############################################################################
############################### MM main run ###################################
###############################################################################

logger = mm.configure_logger(out_path = out_path, log_level = "debug")(__name__)

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
    mm_output["combined_graph"], out_path = out_path)

# Generate RMSF, pLDDT clusters and RMSD trajectories
mm_traj = mm.generate_RMSF_pLDDT_cluster_and_RMSD_trajectories(
    mm_output = mm_output, out_path = out_path)

###############################################################################
########################## Testing RMSD traject ###############################
###############################################################################

