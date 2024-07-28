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

logger = mm.configure_logger(out_path=out_path)

###############################################################################

################################# Test 2 ######################################

fasta_file = "/home/elvio/Desktop/Assemblies/BDF2_HDAC3/SIN3_proteins.fasta"
AF2_2mers = "/home/elvio/Desktop/Assemblies/BDF2_HDAC3/2-mers"
AF2_Nmers = "/home/elvio/Desktop/Assemblies/BDF2_HDAC3/N-mers"
AF2_Nmers = None
out_path = "/home/elvio/Desktop/MM_SIN3"
use_names = True 
overwrite = True
graph_resolution_preset = "/home/elvio/Desktop/MM_SIN3/graph_resolution_preset.json"
# graph_resolution_preset = None

# logger = mm.configure_logger(out_path=out_path)

################################# Test 3 ######################################

fasta_file = "tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta"
AF2_2mers = "tests/EAF6_EPL1_PHD1/2-mers"
AF2_Nmers = "tests/EAF6_EPL1_PHD1/N-mers"
# AF2_Nmers = None
out_path = "/home/elvio/Desktop/MM_interactive_test"
use_names = True 
overwrite = True
graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
graph_resolution_preset = None

# logger = mm.configure_logger(out_path=out_path)

###############################################################################

# Run the main MultimerMapper pipeline
mm_output = mm.parse_AF2_and_sequences(fasta_file,
                                       AF2_2mers, 
                                       AF2_Nmers,
                                       out_path,
                                       use_names = use_names,
                                       overwrite = overwrite,
                                       graph_resolution_preset = graph_resolution_preset)


combined_graph_interactive = mm.interactive_igraph_to_plotly(
    mm_output["combined_graph"], out_path = out_path)






