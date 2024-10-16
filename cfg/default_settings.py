
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
N_models_cutoff = 3

# Cutoffs for 2-mers (Sens = 57.4%, FPR = 0.05)
min_PAE_cutoff_2mers = 8.99
ipTM_cutoff_2mers = 0.240
  
# Cutoffs for N-mers (Sens = 57.4%, FPR = 0.05)
min_PAE_cutoff_Nmers = 8.99
pDockQ_cutoff_Nmers = 0.022

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

vertex_color1     = PT_palette["red"]
vertex_color2     = PT_palette["green"]
vertex_color3     = PT_palette["orange"]
vertex_color_both = PT_palette["gray"]

# Which interactions do not take int account in interactive combined graph?
remove_interactions_from_ppi_graph = ("Indirect",)      # (indirect: mediated by a 3rd protein)

# ----- For homooligomerization edges (self-loops) ---------------------------
# 0: up, 0.25: left, 0.5: down, 0.75: right
self_loop_orientation: float = 0.0
# Size of the homooligomerization edge circle
self_loop_size: float = 3
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
available_layout = ['fr', 'kk', 'circle', 'drl', 'lgl', 'random', 'rt', 'rt_circular', ]
ppi_graph_layout_algorithm = available_layout[0]

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
contact_PAE_cutoff     : float | int = 9
contact_pLDDT_cutoff   : float | int = 50

# ----------------------------------------------------------------------------
# -------------------------- For contact detection ---------------------------
# ----------------------------------------------------------------------------

# Contact Clustering Matrix Method -------------------------------------------
# default is "contact_fraction_comparison" (MCFT), the rest are experimental
contacts_clustering_method = ["contact_similarity_matrix",
                             "agglomerative_clustering",
                             "contact_fraction_comparison",
                             "mc_threshold"][3]
mc_threshold = 10
use_median = True

# For Agglomerative Hierarchical Clustering + Silhouette (experimental) ------
# Silhouette threshold to surpass in order to consider multivalent interaction
multivalency_silhouette_threshold: float = 0.3

# For Agglomerative Hierarchical Clustering + Silhouette (experimental) ------

# Contacts must be less similar than this to be considered separate clusters (0 to 1)
multivalency_contact_similarity_threshold: float = 0.7
# Maximum valency to test (max_contact_clusters modes of interactions)
max_contact_clusters: int = 5

# For MCFT (Merging by Contact Fraction Threshold) - DEFAULT -----------------

# To optimize
contact_fraction_threshold: float               = 0.1
refinement_contact_similarity_threshold: float  = 0.5
refinement_cf_threshold: float                  = 0.5

# Save/display the contact clusters with PCA plots?
display_contact_clusters = False
save_contact_clusters    = True

# ----------------------------------------------------------------------------
# ------------- For Residue-Residue Contact (RRC) visualizations -------------
# ----------------------------------------------------------------------------

# Darker colors for domains
DOMAIN_COLORS_RRC = [
    '#8B0000',  # Dark Red
    '#006400',  # Dark Green
    '#00008B',  # Dark Blue
    '#4B0082',  # Indigo
    '#8B4513',  # Saddle Brown
    '#2F4F4F',  # Dark Slate Gray
    '#800080',  # Purple
    '#3C1414',  # Dark Sienna
    '#1C1C1C',  # Very Dark Gray
    '#004D40',  # Dark Teal
    '#3E2723',  # Dark Brown
    '#1A237E',  # Dark Navy
]

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

# ----------------------------------------------------------------------------
#------------------------ To modify default settings -------------------------
# ----------------------------------------------------------------------------

# If you want to experiment with custom settings (e.g. changing cutoff values),
# you can do it by overwriting the above variables in cfg/custom_settings.py.
from cfg.custom_settings import *