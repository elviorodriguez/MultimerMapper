
#############################################################################
# ------------------------------------------------------------------------- #
# -------------------------- Custom settings ------------------------------ #
# ------------------------------------------------------------------------- #
#############################################################################

'''
If you want to change any MultimerMapper configuration, do it from here.

Simply copy the configuration you want to change from default_settings, paste
it here and change its value. The default setting will be overwritten. Save
the file, restart your ipython console and reimport MultimerMapper (or run 
again the pipeline from the command line after saving).
'''

# # Example custom cfg. Uncomment below line to reduce verbosity
# log_level: str = "warn"

# show_PAE_along_backbone
# show_PAE_along_backbone = False

# remove_interactions_from_ppi_graph
# remove_interactions_from_ppi_graph = ("Indirect", "No 2-mers Data")
# remove_interactions_from_ppi_graph = ("Indirect", )

# auto_domain_detection
# auto_domain_detection = True

# self_loop_size (radius)
self_loop_size = 2


# multivalency_silhouette_threshold         = 0.25
# multivalency_contact_similarity_threshold = 0.1


# General cutoffs ---------- (AF3) -----------
N_models_cutoff = 4
PAE_cutoff = 10
contact_PAE_cutoff = PAE_cutoff
min_PAE_cutoff_2mers = PAE_cutoff
min_PAE_cutoff_Nmers = PAE_cutoff
N_models_cutoff_conv_soft = N_models_cutoff
miPAE_cutoff_conv_soft = PAE_cutoff

# # Custom analysis
# contact_clustering_config = {
#     'distance_metric': 'mixed_jmc',
#     'clustering_method': 'hierarchical',
#     'linkage_method': 'average',
#     'validation_metric': 'silhouette',
#     'quality_weight': True,
#     'silhouette_improvement': 0.0,
#     'max_extra_clusters': 3,
#     'min_extra_clusters': 2,
#     'overlap_structural_contribution': 1,
#     'overlap_use_contact_region_only': False,
#     'use_median': False
# }


available_layout = ['weighted_fr', 'fr', 'kk', 'circle', 'drl', 'lgl', 'random', 'rt', 'rt_circular', ]
ppi_graph_layout_algorithm = available_layout[2]

# For 3D visualization
# layout_3d_iterations = 1000



##########################################
# Starting graph_resolution value for PAE clustering
# graph_resolution: float | int = 0.5

# If True, uses same graph_resolution value for all proteins
# If False, you can select interactively 
# auto_domain_detection: bool = True