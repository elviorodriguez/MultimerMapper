
import os
import pandas as pd
import numpy as np
import logging
from Bio import PDB
from typing import Literal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

from utils.logger_setup import configure_logger



# def get_2mers_models_that_contains_pair(P1: str, P2: str, ):

#     # List of monomers
#     pair_chains_from_2mers   : list[PDB.Chain.Chain] = []
#     is_chain                 : list[str]             = []
#     is_model                 : list[tuple]           = []
#     is_rank                  : list[int]             = []
    
#     # Get 2-mers chains
#     for i, row in pairwise_2mers_df.iterrows():
#         protein1 = row['protein1']
#         protein2 = row['protein2']
        
#         # Extract the chains if there is a match
#         if protein1 == protein_ID or protein2 == protein_ID:
#             model_chains = [chain for chain in row['model'].get_chains()]
#             model = (row['protein1'], row['protein2'])
#             rank = row['rank']
        
#         # Extract chain A
#         if protein1 == protein_ID:
#             monomer_chains_from_2mers.append(model_chains[0])
#             is_chain.append("A")
#             is_model.append(model)
#             is_rank.append(rank)
        
#         # Extract chain B
#         if protein2 == protein_ID:
#             monomer_chains_from_2mers.append(model_chains[1])
#             is_chain.append("B")
#             is_model.append(model)
#             is_rank.append(rank)

#     # List of attributes of the models, to get specific info
#     monomer_chains_from_2mers_dict: dict = {
#         "pair_chains": pair_chains_from_2mers,
#         "are_chains": is_chain,
#         "is_model": is_model,
#         "is_rank": is_rank
#     }