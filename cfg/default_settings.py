
# For using names as node labels (if false, IDs will be used)
use_names: bool = True

# For overwritting everything in output folder
overwrite: bool = False

# ----------------------------------------------------------------------------
# --------------------------- For metric extraction  -------------------------
# ----------------------------------------------------------------------------

# Save the extracted interaction metris as tsv file?
save_pairwise_data = True

# ----------------------------------------------------------------------------
# --------------------------- For domain detection ---------------------------
# ----------------------------------------------------------------------------

# Starting graph_resolution value for PAE clustering
graph_resolution: float | int = 0.075

# If True, uses same graph_resolution value for all proteins
# If False, you can select interactevily 
auto_domain_detection: bool = True

# If you have saved previously a preset, add its path here (Look in <out_path>/domains dir)
# If not, set it as None
graph_resolution_preset: str | None = None

# First time running the pipeline? Leave it as True (saves the preset in <out_path>/domains dir)
save_preset: bool = True

# Saves the monomeric PAE matrix (as png) of each protein with detected domains (<out_path>/domains dir)
save_PAE_png: bool = True

# If True, opens the monomer PAE matrix (as png) of each protein with detected domains (S.O. image viewer)
display_PAE_domains: bool = True # Use this when you run the pipeline from command line

# If True, opens the monomer PAE matrix (as png) of each protein with detected domains (In-line of the console)
# Use this when you run the pipeline interactively using ipython (eg jupyter notebook)
display_PAE_domains_inline: bool = True

# If True, opens the monomer structure of each protein with detected domains as HTML (Browser)
show_monomer_structures: bool = True

# If True, saves the monomer structure of each protein with detected domains as HTML (<out_path>/domains dir)
save_domains_html: bool = True

# When you know the exact span of the domains of your proteins, set this as the path to the manual_domains.tsv file
# Look for a sample file in tests
manual_domains: str | None = None

# If True, saves the detected domains table (<out_path>/domains dir)
save_domains_tsv: bool = True


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
# -------------------------- For RMSD calculations ---------------------------
# ----------------------------------------------------------------------------

# Cutoff to consider a domain disordered (domain_mean_pLDDT < cutoff => disordered)
domain_RMSD_plddt_cutoff = 60
trimming_RMSD_plddt_cutoff = 70


# ----------------------------------------------------------------------------
# -------------------------- For PPI classification --------------------------
# ----------------------------------------------------------------------------

# Standard pDockQ value from Bryant et al (Nature 2022)
pdockq_indirect_interaction_cutoff = 0.23

# if 50% of the N-mer models show interaction => solid blue line edge
predominantly_static_cutoff = 0.5

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

# Default interactive combined graph colors
edge_color1=PT_palette["red"]           # 
edge_color2=PT_palette["green"]         # 
edge_color3=PT_palette["orange"]        # 
edge_color4=PT_palette["cyan"]          # 
edge_color5=PT_palette["yellow"]        # 
edge_color6=PT_palette["blue"]          # 
edge_color_both=PT_palette["black"]     # 

vertex_color1=PT_palette["red"]         # 
vertex_color2=PT_palette["green"]       # 
vertex_color3=PT_palette["orange"]      # 
vertex_color_both=PT_palette["gray"]    # 

# Remove indirect interactions from combined graph? (mediated by a 3rd protein)
remove_indirect_interactions = True


#

# Save reference structures of each protein as PDB file? (<out_path>/PDB_ref_monomers dir)
save_ref_structures = True