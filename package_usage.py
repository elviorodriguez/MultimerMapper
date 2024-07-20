


# # -----------------------------------------------------------------------------
# # ----------------------------- Package usage ---------------------------------
# # -----------------------------------------------------------------------------


# # ---------------- Parameters setup and files locations -----------------------

# # Specify the path to your FASTA file 
# fasta_file_path = "BDF6-HAT1_proteins.fasta"                # REPLACE HERE!!!!!!!!

# # List of folders to search for PDB files
# AF2_2mers = "../../AF2_results/BDF6_HAT1_2-mers/AF2"        # REPLACE HERE!!!!!
# AF2_Nmers = "../../AF2_results/BDF6_HAT1_3-4-5mers/AF2"     # REPLACE HERE!!!!!

# # If you want to work with names, set it to True
# use_names = True

# # For domain detection
# graph_resolution = 0.075    # REPLACE HERE!!! to optimize for the proteins
# pae_power = 1
# pae_cutoff = 5
# # To set a different resolution for each protein, set it to False (NOT IMPLEMENTED YET)
# # If True: graph_resolution will be used for all proteins
# auto_domain_detection = True
# graph_resolution_preset = None        # Path to JSON graph resolution preset
# save_preset = False                   


# # Interaction definitions Cutoffs
# min_PAE_cutoff = 4.5
# ipTM_cutoff = 0.4
# N_models_cutoff = 3   # At least this many models have to surpass both cutoffs



# # -------------------- Preprocess AF2-multimer data ---------------------------

# # Execute data extractor
# all_pdb_data, sliced_PAE_and_pLDDTs, domains_df, my_dataframe_with_combinations, pairwise_2mers_df,\
#     pairwise_2mers_df_F3, unique_proteins, pairwise_Nmers_df, graph, fully_connected_subgraphs,\
#         fully_connected_subgraphs_pairwise_2mers_df = \
#             parse_AF2_and_sequences(fasta_file_path, AF2_2mers, AF2_Nmers, use_names = True,
#                             graph_resolution = 0.075, auto_domain_detection = False,
#                             # Use previous preset?
#                             graph_resolution_preset = "./domains/BDF6-HAT1_proteins-graph_resolution_preset.json", 
#                             save_preset = False,
#                             save_PAE_png = True, display_PAE_domains = False, show_structures = True,
#                             display_PAE_domains_inline = True,
#                             save_domains_html = True, save_domains_tsv = True)

# # Let's see the generated data
# sliced_PAE_and_pLDDTs
# domains_df
# pairwise_2mers_df.columns
# pairwise_2mers_df_F3.columns
# graph
# fully_connected_subgraphs
# fully_connected_subgraphs_pairwise_2mers_df

# # Generate contacts dataframe for bigger subgraph (fully_connected_subgraphs_pairwise_2mers_df[0])
# # NOTE: sometimes it ends at index 1, not sure why
# contacts_2mers_df = compute_contacts_from_pairwise_2mers_df(filtered_pairwise_2mers_df = fully_connected_subgraphs_pairwise_2mers_df[0],
#                                   pairwise_2mers_df = pairwise_2mers_df,
#                                   sliced_PAE_and_pLDDTs = sliced_PAE_and_pLDDTs,
#                                   is_Nmer = False)


# # ----------------- Protein creation and visualization ------------------------

# # Initialize proteins dictionary
# proteins_dict = create_proteins_dict_from_contacts_2mers_df(contacts_2mers_df, sliced_PAE_and_pLDDTs, plot_proteins = False, print_proteins = False)

# # # Access dict data
# # proteins_dict                   # full dict
# # print(proteins_dict["BDF6"])    # Access one of the proteins

# # # Representation of a single protein node (not so speed efficient)
# # proteins_dict["BDF6"].plot_alone()

# # ------------------------ partners addittions --------------------------------

# # Add partners to proteins
# add_partners_to_proteins_dict_from_contacts_2mers_df(proteins_dict, contacts_2mers_df)


# # ---------- Protein 3D manipulation and contacts network plotting ------------

# # # List of algorithms available for protein 3D distribution and sugested scaling factors
# # algorithm_list  = ["drl", "fr", "kk", "circle", "grid", "random"]
# # scaling_factors = [200  , 100 , 100 , 100     , 120   , 100     ]

# # Move the proteins in 3D space to allow propper network plotting
# proteins_dict["BDF6"].set_fully_connected_network_3D_coordinates(algorithm = "drl", scaling_factor = 200)

# # Plot network visualization (some algorithms have intrinsic randomization. So,
# #                             if you don't like the visualization, you can
# #                             try again by resetting 3D coordinats of the
# #                             proteins befo)
# fig, network_contacts_2mers_df, proteins_df, proteins_df_c, proteins_df_c2 =\
#     proteins_dict["BDF6"].plot_fully_connected2(plddt_cutoff = 70,
#                                                 show_axis = False,
#                                                 visible_backbones = False)












# # Tests 
# chain_test = next(pairwise_2mers_df['model'][0].get_chains())
# atom_test = next(chain_test.get_atoms())
# atom_test.get_coord()
# atom_test.set_coord()

