# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import multimer_mapper as mm

pd.set_option( 'display.max_columns' , None )


################### Test 6' (multivalency detection RuvBL) ####################

fasta_file = "/home/elvio/Desktop/homomultimers_benchmark/fallback_homoolig_test/proteins.fasta"
AF2_2mers = "/home/elvio/Desktop/homomultimers_benchmark/fallback_homoolig_test/AF2_2mers"
AF2_Nmers = "/home/elvio/Desktop/homomultimers_benchmark/fallback_homoolig_test/AF2_Nmers"
AF2_Nmers = None
out_path = "//home/elvio/Desktop/homomultimers_benchmark/fallback_homoolig_test/mm_out2"
use_names = True
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

# Generate interactive graph
# import multimer_mapper as mm
# combined_graph, dynamic_proteins, homooligomerization_states, multivalency_states = mm.generate_combined_graph(mm_output)
combined_graph_interactive = mm.interactive_igraph_to_plotly(
    mm_output['combined_graph'], out_path = out_path,
    layout_algorithm = 'kk',    
    
    # You can remove specific interaction types from the graph
    # "No 2-mers Data"
    remove_interactions = ("Indirect", "No 2-mers Data"),
    self_loop_size = 4,
    
    # Answer y automatically
    automatic_true = True)

# # Explore the stoichiometric space
# import multimer_mapper as mm
# best_stoichiometry, paths = mm.stoichiometric_space_exploration_pipeline(mm_output)

# from src.stoichiometries import stoichiometric_space_exploration_pipeline
# best_stoichiometry, paths = stoichiometric_space_exploration_pipeline(mm_output)

# Create 3D network, generate a layout and create py3Dmol/Plotly visualizations
nw = mm.Network(mm_output['combined_graph'], logger = logger)
nw.generate_layout()
nw.generate_py3dmol_plot(save_path = out_path + '/3D_graph_py3Dmol.html', show_plot=True)
nw.generate_plotly_3d_plot(save_path = out_path + '/3D_graph_plotly.html', show_plot=True)

# Get suggested combinations
suggested_combinations = mm.suggest_combinations(mm_output = mm_output, 
                                                 # To ommit saving, change to None
                                                 out_path = out_path)

# Generate RMSF, pLDDT clusters & RMSD trajectories considering models as monomers
mm_monomers_traj = mm.generate_RMSF_pLDDT_cluster_and_RMSD_trajectories(
    mm_output = mm_output, out_path = out_path)

