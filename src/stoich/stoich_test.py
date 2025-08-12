
import numpy as np
import pandas as pd
    
from src.stoich.stoich_space_exploration import initialize_stoich_dict, add_xyz_coord_to_stoich_dict, plot_stoich_space

stoich_dict = initialize_stoich_dict(mm_output)

stoich_dict = add_xyz_coord_to_stoich_dict(stoich_dict)

plot_stoich_space(stoich_dict, "/home/elvio/Desktop/stoichiometric_space.html")