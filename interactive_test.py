# -*- coding: utf-8 -*-

import pandas as pd
import multimer_mapper as mm

fasta_file = "tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta"
AF2_2mers = "tests/EAF6_EPL1_PHD1/2-mers"
AF2_Nmers = "tests/EAF6_EPL1_PHD1/N-mers"
# AF2_Nmers = None
out_path = "/home/elvio/Desktop/MM_interactive_test"
use_names = True 
overwrite = True
graph_resolution_preset = "/home/elvio/Desktop/MM_interactive_test/domains/graph_resolution_preset.json"

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


# Data structure of pairwise_Nmers_df
for i, row in mm_output["pairwise_Nmers_df"].iterrows():
    print("")
    print(f"Row: {i}")
    print(row)

# Get proteins that dimerize (mm_output["pairwise_2mers_df_F3"]):
def get_proteins_that_homodimerize(pairwise_2mers_df_F3: pd.DataFrame):
    '''
    Analyzes pairwise_2mers_df_F3 and return the set of proteins that
    forms homodimers.

    Parameters
    ----------
    pairwise_2mers_df_F3 : pd.DataFrame
        Dataframe of interacting proteins from 2-mers.

    Returns
    -------
    homodim_prots : set
        Set of proteins that homodimerize.

    '''
    
    homodim_prots: set = set()
    
    for i, row in pairwise_2mers_df_F3.iterrows():
        
        protein1 = str(row["protein1"])
        protein2 = str(row["protein2"])
               
        if (protein1 == protein2) and (protein1 not in homodim_prots):
            homodim_prots.add(protein1)
            
    return homodim_prots


get_proteins_that_homodimerize(mm_output["pairwise_2mers_df_F3"])


def get_path_of_homooligomers(
        pairwise_2mers_df_F3: pd.DataFrame,
        all_pdb_data: dict):
    
    homodim_prots: set = get_proteins_that_homodimerize(pairwise_2mers_df_F3)
    
    for prot in homodim_prots:
        
        for prediction in all_pdb_data.values():
            
            for chain, values in prediction.items():
                
                # there are items that are not chains
                try:
                    values['sequence']
                    print(chain)
                    print(values)
                except KeyError:
                    continue

get_path_of_homooligomers(pairwise_2mers_df_F3 = mm_output["pairwise_2mers_df_F3"],
                          all_pdb_data = mm_output["all_pdb_data"])
            
        
    

'''
full_PAE_matrices
full_PDB_models
pairwise_data
min_diagonal_PAE
'''
    
    
    
    
    
    
    
    