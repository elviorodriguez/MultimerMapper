
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
show_PAE_along_backbone = False

# remove_interactions_from_ppi_graph
remove_interactions_from_ppi_graph = ("Indirect", "No 2-mers Data")

# auto_domain_detection
auto_domain_detection = True

# self_loop_size (radius)
self_loop_size = 6