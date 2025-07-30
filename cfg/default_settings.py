
# ----------------------------------------------------------------------------
# ------------------------- General Configurations ---------------------------
# ----------------------------------------------------------------------------

# For using names as node labels (if false, IDs will be used)
use_names: bool = True

# For overwriting everything in output folder
overwrite: bool = False

# Logging level ("notset", "debug", "info", "warn", "error", "critical")
# Increase it to "warn", "error" or "critical" to reduce verbosity
log_level = "debug"

# If you do not provide an out_path
default_out_path = "MM_output_"

# ----------------------------------------------------------------------------
# --------------------------- For metric extraction  -------------------------
# ----------------------------------------------------------------------------

# Save the extracted interaction metrics as tsv file?
save_pairwise_data = True

# ----------------------------------------------------------------------------
# --------------------------- For domain detection ---------------------------
# ----------------------------------------------------------------------------

# Starting graph_resolution value for PAE clustering
graph_resolution: float | int = 0.075

# If True, uses same graph_resolution value for all proteins
# If False, you can select interactively 
auto_domain_detection: bool = False

# If you have saved previously a preset, add its path in this option (Look in <out_path>/domains dir)
# If not, set it as None. All modification must be done in custom_settings.py.
graph_resolution_preset: str | None = None

# First time running the pipeline? Leave it as True (saves the preset in <out_path>/domains dir)
save_preset: bool = True

# Saves the monomeric PAE matrix (as png) of each protein with detected domains (<out_path>/domains dir)
save_PAE_png: bool = True

# If True, opens the monomer PAE matrix (as png) of each protein with detected domains (S.O. image viewer)
display_PAE_domains: bool = False

# If True, opens the monomer PAE matrix (as png) of each protein with detected domains (In-line of the console)
# Use this when you run the pipeline interactively using ipython (eg jupyter notebook)
display_PAE_domains_inline: bool = False

# If True, opens the monomer structure of each protein with detected domains as HTML (Browser)
show_monomer_structures: bool = False

# To display the PAE and backbone colored by detected domains
show_PAE_along_backbone: bool = True

# If True, saves the monomer structure of each protein with detected domains as HTML (<out_path>/domains dir)
save_domains_html: bool = True

# When you know the exact span of the domains of your proteins, set this as the path to the manual_domains.tsv file
# Look for a sample file in tests
manual_domains: str | None = None

# If True, saves the detected domains table (<out_path>/domains dir)
save_domains_tsv: bool = True

# Define a common color list for the discrete integer values in clusters
DOMAIN_COLORS = ['red', 'green', 'blue', 'purple', 'yellow', 'orange', 'brown', 'pink', 'cyan', 'lime', 'gray', 'olive']

# ----------------------------------------------------------------------------
# --------------------------- For PPI detection ------------------------------
# ----------------------------------------------------------------------------

# General cutoff
N_models_cutoff = 5

# Cutoffs for 2-mers (Sens = 61.5%, FPR = 0.05)
min_PAE_cutoff_2mers = 13
ipTM_cutoff_2mers = 0.0     # Do not change
  
# Cutoffs for N-mers (Sens = 61.5%, FPR = 0.05)
min_PAE_cutoff_Nmers = 13
pDockQ_cutoff_Nmers = 0.0   # Do not change

# For Nmers
Nmers_contacts_cutoff = 3

# To classify PPI dynamics just using N_models cutoff set to False
# If true, the cluster N-mers variation will be used
use_cluster_aware_Nmers_variation = True

# ----------------------------------------------------------------------------
# ---------------------- For Symmetry Fallback Analysis ----------------------
# ----------------------------------------------------------------------------

fallback_low_fraction: float        = 0.5
fallback_up_fraction: float         = 0.5
save_fallback_plots: bool           = True
fallback_plot_figsize: tuple[int]   = (5,5)
fallback_plot_dpi: int              = 200
save_fallback_df: bool              = True
display_fallback_ranges: bool       = True


# ----------------------------------------------------------------------------
# -------------------------- For RMSD calculations ---------------------------
# ----------------------------------------------------------------------------

# Cutoff to consider a domain disordered (domain_mean_pLDDT < cutoff => disordered)
domain_RMSD_plddt_cutoff = 60
trimming_RMSD_plddt_cutoff = 70

# ----------------------------------------------------------------------------
# ------------- For PPI classification and interactive PPI graph -------------
# ----------------------------------------------------------------------------

# Color blind friendly palette (Paul Tol's + orange)
PT_palette = {
    "black"         : "#000000",
    "green"         : "#228833",
    "blue"          : "#4477AA",
    "cyan"          : "#66CCEE",
    "yellow"        : "#CCBB44",
    "purple"        : "#AA3377",
    "orange"        : "#e69f00",
    "deep orange"   : "#d55e00",
    "red"           : "#EE6677",
    "gray"          : "#bbbbbb",
    }

# # You can call it like this: PT_palette["orange"]

# # Default interactive combined graph colors
# edge_color1=PT_palette["red"]           # 
# edge_color2=PT_palette["green"]         # 
# edge_color3=PT_palette["orange"]        # 
# edge_color4=PT_palette["cyan"]          # 
# edge_color5=PT_palette["yellow"]        # 
# edge_color6=PT_palette["blue"]          # 
# edge_color_both=PT_palette["black"]     # 

# Colors for protein dynamics
vertex_color1     = PT_palette["red"]       # Negative Dynamic Protein
vertex_color2     = PT_palette["green"]     # Positive Dynamic Protein
vertex_color3     = PT_palette["orange"]    # No N-mers Data Protein
vertex_color_both = PT_palette["gray"]      # Static Protein

# Which interactions do not take int account in interactive combined graph?
remove_interactions_from_ppi_graph = ("Indirect",)      # (indirect: mediated by a 3rd protein)

# ----------------- For edges (PPIs) width -----------------------------------

edge_default_weight = 0.5
edge_scaling_factor = 5
edge_min_weight = 1
edge_max_weight = 4
edge_midpoint_PAE = 2
edge_weight_sigmoidal_sharpness = 0.1


# ----- For homooligomerization edges (self-loops) ---------------------------
# 0: up, 0.25: left, 0.5: down, 0.75: right
self_loop_orientation: float = 0.0
# Size of the homooligomerization edge circle
self_loop_size: float = 4
# ----------------------------------------------------------------------------

# Remove background and set protein names as bold
show_axis = False
showgrid = False
use_bold_protein_names = True

# Domain RMSDs bigger than this value will be highlighted in bold in the nodes hovertext
add_bold_RMSD_cutoff = 5

# You can save the plot as HTML to share it, for example, with yourself via whatsapp and analyze it
# using your cellphone' browser
save_html = "2D_PPI_graph.html"

# Add cutoffs labels to keep track cutoffs values that generated the graph
# Highly recommended if you are experimenting with cutoffs in other organisms
add_cutoff_legend = False

# Layout generation algorithm ------------------------------------------------

# Have a look at https://igraph.org/python/tutorial/0.9.8/tutorial.html
# for more info about layout algorithms
available_layout = ['weighted_fr', 'fr', 'kk', 'circle', 'drl', 'lgl', 'random', 'rt', 'rt_circular']
ppi_graph_layout_algorithm = available_layout[0]

# Fraction of the weight that N-mers contribute
weighted_fr_Nmers_contribution = 9/10

# If True, the first generated interactive PPI graph will be saved, without asking
igraph_to_plotly_automatic_true = False

# ----------------------------------------------------------------------------
# ------------------------- For coordinate analysis --------------------------
# ----------------------------------------------------------------------------

# Save reference structures of each protein as PDB file? (<out_path>/PDB_ref_monomers dir)
save_ref_structures = True

# ----------------------------------------------------------------------------
# -------------------------- For contact detection ---------------------------
# ----------------------------------------------------------------------------

contact_distance_cutoff: float | int = 8.0
contact_PAE_cutoff     : float | int = 13
contact_pLDDT_cutoff   : float | int = 0

# Save/display the contact clusters with PCA plots?
display_contact_clusters = False
save_contact_clusters    = True

# ----------------------------------------------------------------------------
# ------------------ Enhanced Multivalency Detection Method ------------------
# ----------------------------------------------------------------------------

# Select method using index
method_index = 0

# Detection method
multivalency_detection_metric = ["fraction_of_multivalent_chains", "max_valency"][method_index]

# Detection threshold (Best values)
multivalency_metric_threshold = [0.25, 2][method_index]


# ----------------------------------------------------------------------------
# -------------- Enhanced Contact Matrix Clustering Config -------------------
# ----------------------------------------------------------------------------

# Use the enhanced method?
use_enhanced_matrix_clustering = True

# Custom analysis
contact_clustering_config = {
    'distance_metric': 'closeness',
    'clustering_method': 'hierarchical',
    'linkage_method': 'average',
    'validation_metric': 'silhouette',
    'quality_weight': True,
    'silhouette_improvement': 0.2,
    'max_extra_clusters': 3,
    'min_extra_clusters': 2,
    'overlap_structural_contribution': 1,
    'overlap_use_contact_region_only': False,
    'use_median': False
}

# ----------------------------------------------------------------------------
# ------------------------- For convergency detection ------------------------
# ----------------------------------------------------------------------------

# Recommended: contact_network
Nmer_stability_method = ["pae", "contact_network"][1]

# Use the same cutoff as for other Nmers? 
Nmers_contacts_cutoff_convergency = Nmers_contacts_cutoff

# Dynamic softening function
use_dynamic_conv_soft_func = True

# ------------------------------ Dynamic method ------------------------------

# With FPR = 0.01
# miPAE_cutoff_conv_soft_list = [10.22, 5.61, 3.98, 2.04, 1.86]

# With FPR = 0.05
miPAE_cutoff_conv_soft_list = [13.0, 10.5, 7.20, 4.50, 3.00]

# Start and end
dynamic_conv_start = 5
dynamic_conv_end   = 2


# ------------------------------ Static method -------------------------------

# Soften the N_models cutoff?
softening_index = 3             # <--- Change this (higher is more soft)
N_models_cutoff_conv_soft   = [5   , 4   , 3   , 2   , 1   ][softening_index]
miPAE_cutoff_conv_soft      = [13.0, 10.5, 7.20, 4.50, 3.00][softening_index]

# ----------------------------------------------------------------------------
# ------------- For Residue-Residue Contact (RRC) visualizations -------------
# ----------------------------------------------------------------------------

use_coarse_grain_method = False

# --------------------------- Coarse grain method ----------------------------
layout_3d_iterations = 10000

# ---------------------------- Fine grain method -----------------------------

# DEFAULT FINE GRAIN CONFIG
fine_grain_layout_cfg = {
    "algorithm": "residue_optimized",
    "iterations": 50,
    # Controls the inter-surface distance
    "min_contact_distance": 50,
    "max_contact_distance": 60,
    # Repulsive and attractive forces
    "contact_force_strength": 10.0,
    "repulsion_strength": 10.0,
    # Controls the separation of center of masses
    "global_repulsion_strength": 5,
    # Minimum distance between protein centers of mass to exert force
    "min_interprotein_distance": 200.0,
    # Generates torque to orient protein surfaces to face each other
    "torque_strength": 100.0,
    # Decreases the strength of the forces over iterations
    "initial_step_size": 0.5,
    "final_step_size": 0.005,
    # Strength of surface face-to-face alignment
    "surface_alignment_strength": 25,
    # Strength of contact line separation ("untwisting" force strength)
    "line_separation_strength": 10.0,
    # Nº of contact residues to sample at each iteration (reduces computation time)
    "n_contacts_sample": 5  # None -> No sampling
}

# # Darker colors for domains
# DOMAIN_COLORS_RRC = [
#     '#8B0000',  # Dark Red
#     '#006400',  # Dark Green
#     '#00008B',  # Dark Blue
#     '#4B0082',  # Indigo
#     '#8B4513',  # Saddle Brown
#     '#2F4F4F',  # Dark Slate Gray
#     '#800080',  # Purple
#     '#3C1414',  # Dark Sienna
#     '#1C1C1C',  # Very Dark Gray
#     '#004D40',  # Dark Teal
#     '#3E2723',  # Dark Brown
#     '#1A237E',  # Dark Navy
# ]

# # Lighter colors for domains
# DOMAIN_COLORS_RRC = [
# "#d50000",
# "#00e676",
# "#2979ff",
# "#ffea00",
# "#c6ff00",
# "#ff9100",
# "#d500f9",
# "#00b0ff",
# "#1de9b6",
# "#76ff03",
# "#ffc400",
# "#ff3d00",
# "#f50057",
# "#651fff",
# "#00e5ff",
# "#3d5afe",
# ]


# Lighter colors for surfaces
SURFACE_COLORS_RRC = [
    '#FF6347',  # Tomato
    '#98FB98',  # Pale Green
    '#87CEFA',  # Light Sky Blue
    '#DDA0DD',  # Plum
    '#F0E68C',  # Khaki
    '#FFA07A',  # Light Salmon
    '#F4A460',  # Sandy Brown
    '#FFB6C1',  # Light Pink
    '#E0FFFF',  # Light Cyan
    '#7FFFD4',  # Aquamarine
    '#D3D3D3',  # Light Gray
    '#FAFAD2',  # Light Goldenrod Yellow
]

# Color palette for network representation
default_color_palette = {
    "Green":          ["#e8f5e9", "#c8e6c9", "#a5d6a7", "#81c784", "#66bb6a", "#4caf50", "#43a047", "#388e3c", "#2e7d32", "#b9f6ca", "#69f0ae", "#00e676", "#4caf50", "#00c853", "#1b5e20"],
    "Red":            ["#ffebee", "#ffcdd2", "#ef9a9a", "#e57373", "#ef5350", "#f44336", "#e53935", "#d32f2f", "#c62828", "#ff8a80", "#ff5252", "#d50000", "#f44336", "#ff1744", "#b71c1c"],
    "Blue":           ["#e3f2fd", "#bbdefb", "#90caf9", "#64b5f6", "#42a5f5", "#2196f3", "#1e88e5", "#1976d2", "#1565c0", "#82b1ff", "#448aff", "#2979ff", "#2962ff", "#2196f3", "#0d47a1"],
    "Light_Green":    ["#f1f8e9", "#dcedc8", "#c5e1a5", "#aed581", "#9ccc65", "#8bc34a", "#7cb342", "#689f38", "#558b2f", "#ccff90", "#b2ff59", "#76ff03", "#64dd17", "#8bc34a", "#33691e"],
    "Purple":         ["#f3e5f5", "#e1bee7", "#ce93d8", "#ba68c8", "#ab47bc", "#9c27b0", "#8e24aa", "#7b1fa2", "#6a1b9a", "#ea80fc", "#e040fb", "#d500f9", "#aa00ff", "#9c27b0", "#4a148c"],
    "Orange":         ["#fff3e0", "#ffe0b2", "#ffcc80", "#ffb74d", "#ffa726", "#ff9800", "#fb8c00", "#f57c00", "#ef6c00", "#ffd180", "#ffab40", "#ff9100", "#ff6d00", "#ff9800", "#e65100"],
    "Yellow":         ["#fffde7", "#fff9c4", "#fff59d", "#fff176", "#ffee58", "#ffeb3b", "#fdd835", "#fbc02d", "#f9a825", "#ffff8d", "#ffff00", "#ffea00", "#ffd600", "#ffeb3b", "#f57f17"],
    "Light_Blue":     ["#e1f5fe", "#b3e5fc", "#81d4fa", "#4fc3f7", "#29b6f6", "#03a9f4", "#039be5", "#0288d1", "#0277bd", "#80d8ff", "#40c4ff", "#00b0ff", "#0091ea", "#03a9f4", "#01579b"],
    "Teal":           ["#e0f2f1", "#b2dfdb", "#80cbc4", "#4db6ac", "#26a69a", "#009688", "#00897b", "#00796b", "#00695c", "#a7ffeb", "#64ffda", "#1de9b6", "#00bfa5", "#009688", "#004d40"],
    "Lime":           ["#f9fbe7", "#f0f4c3", "#e6ee9c", "#dce775", "#d4e157", "#cddc39", "#c0ca33", "#afb42b", "#9e9d24", "#f4ff81", "#eeff41", "#c6ff00", "#aeea00", "#cddc39", "#827717"],
    "Amber":          ["#fff8e1", "#ffecb3", "#ffe082", "#ffd54f", "#ffca28", "#ffc107", "#ffb300", "#ffa000", "#ff8f00", "#ffe57f", "#ffd740", "#ffc400", "#ffab00", "#ffc107", "#ff6f00"],
    "Deep_Orange":    ["#fbe9e7", "#ffccbc", "#ffab91", "#ff8a65", "#ff7043", "#ff5722", "#f4511e", "#e64a19", "#d84315", "#ff9e80", "#ff6e40", "#ff3d00", "#dd2c00", "#ff5722", "#bf360c"],
    "Pink":           ["#fce4ec", "#f8bbd0", "#f48fb1", "#f06292", "#ec407a", "#e91e63", "#d81b60", "#c2185b", "#ad1457", "#ff80ab", "#ff4081", "#f50057", "#c51162", "#e91e63", "#880e4f"],
    "Deep_Purple":    ["#ede7f6", "#d1c4e9", "#b39ddb", "#9575cd", "#7e57c2", "#673ab7", "#5e35b1", "#512da8", "#4527a0", "#b388ff", "#7c4dff", "#651fff", "#6200ea", "#673ab7", "#311b92"],
    "Cyan":           ["#e0f7fa", "#b2ebf2", "#80deea", "#4dd0e1", "#26c6da", "#00bcd4", "#00acc1", "#0097a7", "#00838f", "#84ffff", "#18ffff", "#00e5ff", "#00b8d4", "#00bcd4", "#006064"],
    "Indigo":         ["#e8eaf6", "#c5cae9", "#9fa8da", "#7986cb", "#5c6bc0", "#3f51b5", "#3949ab", "#303f9f", "#283593", "#8c9eff", "#536dfe", "#3d5afe", "#304ffe", "#3f51b5", "#1a237e"],
}

# Domain colors derived from default_color_palette
DOMAIN_COLORS_RRC = [v[-5] for v in default_color_palette.values()] * 10

# ----------------------------------------------------------------------------
# -------------------- EXPERIMENTAL SECTION (do not touch) -------------------
# ----------------------------------------------------------------------------

# Contact Clustering Matrix Method -------------------------------------------
# default is "contact_fraction_comparison" (MCFT), the rest are experimental
contacts_clustering_method = ["contact_similarity_matrix",
                             "agglomerative_clustering",
                             "contact_fraction_comparison",
                             "mc_threshold"][3]
mc_threshold = 10
use_median = True

# Refine using contact similarity (experimental)
refine_contact_clusters = False

# For Agglomerative Hierarchical Clustering + Silhouette (experimental) ------
# Silhouette threshold to surpass in order to consider multivalent interaction
multivalency_silhouette_threshold: float = 0.3

# For Agglomerative Hierarchical Clustering + Silhouette (experimental) ------

# Contacts must be less similar than this to be considered separate clusters (0 to 1)
multivalency_contact_similarity_threshold: float = 0.7
# Maximum valency to test (max_contact_clusters modes of interactions)
max_contact_clusters: int = 5

# For MCFT (Merging by Contact Fraction Threshold) - Experimental ------------

# To optimize
contact_fraction_threshold: float               = 0.1
refinement_contact_similarity_threshold: float  = 0.5
refinement_cf_threshold: float                  = 0.5


# ----------------------------------------------------------------------------
#------------------------ To modify default settings -------------------------
# ----------------------------------------------------------------------------

# If you want to experiment with custom settings (e.g. changing cutoff values),
# you can do it by overwriting the above variables in cfg/custom_settings.py.
from cfg.custom_settings import *