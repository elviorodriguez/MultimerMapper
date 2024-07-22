
# For using names as node labels (if false, IDs will be used)
use_names: bool = True

# For overwritting everything in output folder
overwrite: bool = False

# ----------------------------------------------------------------------------
# --------------------------- For domain detection ---------------------------
# ----------------------------------------------------------------------------

# Starting graph_resolution value for PAE clustering
graph_resolution: float | int = 0.075

# If True, uses same graph_resolution value for all proteins
# If False, you can select interactevily 
auto_domain_detection: bool = True

# If you have saved previously a preset, add its path here (Look in <out_path>/domains folder)
# If not, set it as None
graph_resolution_preset: str | None = None

# First time running the pipeline? Leave it as True (saves the preset in <out_path>/domains folder)
save_preset: bool = True

# Saves the monomeric PAE matrix (as png) of each protein with detected domains (<out_path>/domains folder)
save_PAE_png: bool = True

# If True, opens the monomer PAE matrix (as png) of each protein with detected domains (S.O. image viewer)
display_PAE_domains: bool = True

# If True, opens the monomer PAE matrix (as png) of each protein with detected domains (In-line of the console)
display_PAE_domains_inline: bool = True

# If True, opens the monomer structure of each protein with detected domains as HTML (Browser)
show_monomer_structures: bool = True

# If True, saves the monomer structure of each protein with detected domains as HTML (<out_path>/domains folder)
save_domains_html: bool = True

# If True, saves the detected domains table (./domains folder)
save_domains_tsv: bool = True


