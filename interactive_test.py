# -*- coding: utf-8 -*-

import pandas as pd
import multimer_mapper as mm

pd.set_option( 'display.max_columns' , None )

###############################################################################


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

# Run the main MultimerMapper pipeline
mm_output = mm.parse_AF2_and_sequences(fasta_file,
                                       AF2_2mers, 
                                       AF2_Nmers,
                                       out_path,
                                       use_names = use_names,
                                       overwrite = overwrite,
                                       graph_resolution_preset = graph_resolution_preset)


import multimer_mapper as mm
combined_graph_interactive = mm.interactive_igraph_to_plotly(
    mm_output["combined_graph"], out_path = out_path)

from src.interpret_dynamics import classification_df

classification_df




