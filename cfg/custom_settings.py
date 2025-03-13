
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
remove_interactions_from_ppi_graph = ("Indirect", )

# auto_domain_detection
# auto_domain_detection = True

# self_loop_size (radius)
# self_loop_size = 4


# multivalency_silhouette_threshold         = 0.25
# multivalency_contact_similarity_threshold = 0.1

# General cutoff
# N_models_cutoff = 4


# ----------- Conflictive cutoffs -----------

# This is OK?
Nmers_contacts_cutoff = 5
contact_distance_cutoff: float | int = 8.0
contact_PAE_cutoff     : float | int = 13
contact_pLDDT_cutoff   : float | int = 50

# # This adds more spurious interactions?
# Nmers_contacts_cutoff = 3
# contact_distance_cutoff: float | int = 8.0
# contact_PAE_cutoff     : float | int = 15
# contact_pLDDT_cutoff   : float | int = 45

# # Changing this to 4 causes the "No 2-mers Data" bug?
# N_models_cutoff = 4

# 
available_layout = ['fr', 'kk', 'circle', 'drl', 'lgl', 'random', 'rt', 'rt_circular', ]
ppi_graph_layout_algorithm = available_layout[0]

# For 3D visualization
layout_3d_iterations = 1000