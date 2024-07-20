
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import plotly.graph_objects as go           # For plotly ploting
from plotly.offline import plot             # To allow displaying plots

# -----------------------------------------------------------------------------
# Split proteins into domain using pae_to_domains.py --------------------------
# -----------------------------------------------------------------------------

# Function from pae_to_domains.py
def domains_from_pae_matrix_igraph(pae_matrix, pae_power=1, pae_cutoff=5, graph_resolution=1):
    '''
    Takes a predicted aligned error (PAE) matrix representing the predicted error in distances between each 
    pair of residues in a model, and uses a graph-based community clustering algorithm to partition the model
    into approximately rigid groups.

    Arguments:

        * pae_matrix: a (n_residues x n_residues) numpy array. Diagonal elements should be set to some non-zero
          value to avoid divide-by-zero warnings
        * pae_power (optional, default=1): each edge in the graph will be weighted proportional to (1/pae**pae_power)
        * pae_cutoff (optional, default=5): graph edges will only be created for residue pairs with pae<pae_cutoff
        * graph_resolution (optional, default=1): regulates how aggressively the clustering algorithm is. Smaller values
          lead to larger clusters. Value should be larger than zero, and values larger than 5 are unlikely to be useful.

    Returns: a series of lists, where each list contains the indices of residues belonging to one cluster.
    '''
    try:
        import igraph
    except ImportError:
        print('ERROR: This method requires python-igraph to be installed. Please install it using "pip install python-igraph" '
            'in a Python >=3.6 environment and try again.')
        import sys
        sys.exit()
    import numpy as np
    weights = 1/pae_matrix**pae_power

    g = igraph.Graph()
    size = weights.shape[0]
    g.add_vertices(range(size))
    edges = np.argwhere(pae_matrix < pae_cutoff)
    sel_weights = weights[edges.T[0], edges.T[1]]
    g.add_edges(edges)
    g.es['weight']=sel_weights

    vc = g.community_leiden(weights='weight', resolution=graph_resolution/100, n_iterations=-1)
    membership = np.array(vc.membership)
    from collections import defaultdict
    clusters = defaultdict(list)
    for i, c in enumerate(membership):
        clusters[c].append(i)
    clusters = list(sorted(clusters.values(), key=lambda l:(len(l)), reverse=True))
    return clusters


def reformat_clusters(domain_clusters):
    '''
    Reformats the output of domains_from_pae_matrix_igraph to make it easier
    to plot and further processing.

    Parameters
    ----------
    domain_clusters : list of lists.
        Clusters generated with domains_from_pae_matrix_igraph.

    Returns
    -------
    reformat_domain_clusters
        A list of list with the resiudes positions in index 0 and the cluster
        assignment in index 1: [[residues], [clusters]]

    '''
    # Lists to store reformatted clusters
    resid_list = []
    clust_list = []
    
    # Process one cluster at a time
    for i, cluster in enumerate(domain_clusters):
        
        for residue in cluster:
            resid_list.append(residue)
            clust_list.append(i)
    
    # Combine lists into pairs
    combined_lists = list(zip(resid_list, clust_list))

    # Sort the pairs based on the values in the first list
    sorted_pairs = sorted(combined_lists, key=lambda x: x[0])
    
    # Unpack the sorted pairs into separate lists
    resid_list, clust_list = zip(*sorted_pairs)
    
    return [resid_list, clust_list]


def plot_domains(protein_ID, matrix_data, positions, colors, custom_title = None, out_folder = 'domains', save_plot = True, show_plot = True):

    # Define a diverging colormap for the matrix
    matrix_cmap = 'coolwarm'

    # Define a custom colormap for the discrete integer values in clusters
    cluster_cmap = ListedColormap(['red', 'green', 'blue', 'purple', 'yellow', 'orange', 'brown', 'pink', 'cyan', 'lime', 'gray', 'olive'])

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the matrix using matshow with the diverging colormap
    cax = ax.matshow(matrix_data, cmap=matrix_cmap)

    # Add a colorbar for the matrix
    # cbar = fig.colorbar(cax)
    fig.colorbar(cax)

    # Normalize the cluster values to match the colormap range
    norm = Normalize(vmin=min(colors), vmax=max(colors))

    # Scatter plot on top of the matrix with the custom colormap for clusters
    # scatter = ax.scatter(positions, positions, c=colors, cmap=cluster_cmap, s=100, norm=norm)
    ax.scatter(positions, positions, c=colors, cmap=cluster_cmap, s=100, norm=norm)

    # Get unique cluster values
    unique_clusters = np.unique(colors)

    # Create a legend by associating normalized cluster values with corresponding colors
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=cluster_cmap(norm(c)),
                                 markersize=10,
                                 label=f'Domain {c}') for c in unique_clusters]

    # Add legend
    ax.legend(handles=legend_handles, title='Domains', loc='upper right')

    # Set labels and title
    plt.xlabel('Positions')
    plt.ylabel('Positions')
    plt.title(f"{custom_title}")

    if save_plot == True:
        # Create a folder named "domains" if it doesn't exist
        save_folder = out_folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    
        # Save the plot
        plt.savefig(os.path.join(save_folder, f"{protein_ID}_domains_plot.png"))

    # Show the plot
    if show_plot: plt.show()
    
    if show_plot == False:
      # Turn interactive mode back on to display plots later
        plt.close()
        
    return fig


def combine_figures_and_plot(fig1, fig2, protein_ID = None, save_png_file = False, show_image = False, show_inline = True):
    '''
    Generates a single figure with fig1 and fig2 side by side.
    
    Parameters:
        - fig1 (matplotlib.figure.Figure): figure to be plotted at the left.
        - fig2 (matplotlib.figure.Figure): figure to be plotted at the right.
        - save_file (bool): If true, saves a file in a directory called "domains", with
            the name protein_ID-domains_plot.png.
        - show_image (bool): If True, displays the image with your default image viewer.
        - show_inline (bool): If True, displays the image in the plot pane (or in the console).
        
    Returns:
        None
    '''
    
    from PIL import Image
    from io import BytesIO
    from IPython.display import display

    # Create BytesIO objects to hold the image data in memory
    image1_bytesio = BytesIO()
    image2_bytesio = BytesIO()
    
    # Save each figure to the BytesIO object
    fig1.savefig(image1_bytesio, format='png')
    fig2.savefig(image2_bytesio, format='png')
    
    # Rewind the BytesIO objects to the beginning
    image1_bytesio.seek(0)
    image2_bytesio.seek(0)
    
    # Open the images using PIL from the BytesIO objects
    image1 = Image.open(image1_bytesio)
    image2 = Image.open(image2_bytesio)
    
    # Get the size of the images
    width, height = image1.size
    
    # Create a new image with double the width for side-by-side display
    combined_image = Image.new('RGB', (width * 2, height))
    
    # Paste the images into the combined image
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (width, 0))
    
    # Show the combined image
    if show_image: combined_image.show()
    if show_inline: display(combined_image)

    # Save the combined image to a file?
    if save_png_file:
        
        if protein_ID == None:
            raise ValueError("protein_ID not provided. Required for saving domains plot file.")
        
        # Create a folder named "domains" if it doesn't exist
        save_folder = "domains"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    
        # Save figure
        combined_image.save(os.path.join(save_folder, f"{protein_ID}-domains_plot.png"))


# Convert clusters of loops into the wrapper cluster domain
def remove_loop_clusters(domain_clusters):
    result = domain_clusters.copy()
    i = 0

    while i < len(domain_clusters):
        current_number = domain_clusters[i]
        tandem_start = i
        tandem_end = i

        # Find the boundaries of the current tandem
        while tandem_end < len(domain_clusters) - 1 and domain_clusters[tandem_end + 1] == current_number:
            tandem_end += 1

        # Check if the current number is different from its neighbors and surrounded by equal numbers
        if tandem_start > 0 and tandem_end < len(domain_clusters) - 1:
            left_neighbor = domain_clusters[tandem_start - 1]
            right_neighbor = domain_clusters[tandem_end + 1]

            if current_number != left_neighbor and current_number != right_neighbor and left_neighbor == right_neighbor:
                # Find the majority number in the surroundings
                majority_number = left_neighbor if domain_clusters.count(left_neighbor) > domain_clusters.count(right_neighbor) else right_neighbor

                # Replace the numbers within the tandem with the majority number
                for j in range(tandem_start, tandem_end + 1):
                    result[j] = majority_number

        # Move to the next tandem
        i = tandem_end + 1

    return result

# For semi-auto domain defining
def plot_backbone(protein_chain, domains, protein_ID = "", legend_position = dict(x=1.02, y=0.5), showgrid = True, margin=dict(l=0, r=0, b=0, t=0), show_axis = False, show_structure = False, save_html = False, return_fig = False, is_for_network = False):
    
    # Protein CM
    protein_CM = list(protein_chain.center_of_mass())
    
    # Create a 3D scatter plot
    fig = go.Figure()
    
    domain_colors = ['red', 'green', 'blue', 'purple', 'yellow', 'orange', 'brown', 'pink', 'cyan', 'lime', 'gray', 'olive'] * 10
    
    pLDDT_colors = ["darkblue", "lightblue", "yellow", "orange", "red"]
    
    CA_x = []
    CA_y = []
    CA_z = []
    res_color = []
    res_name = []
    res_plddt_color = []
            
    for R, residue in enumerate(protein_chain.get_residues()):
       CA_x.append(residue["CA"].get_coord()[0])
       CA_y.append(residue["CA"].get_coord()[1])
       CA_z.append(residue["CA"].get_coord()[2])
       res_color.append(domain_colors[domains[R]])
       res_name.append(residue.get_resname() + str(R + 1))
       plddt = residue["CA"].bfactor
       if plddt >= 90:
           res_plddt_color.append(pLDDT_colors[0])
       elif plddt >= 70:
           res_plddt_color.append(pLDDT_colors[1])
       elif plddt >= 50:
           res_plddt_color.append(pLDDT_colors[2])
       elif plddt >= 40:
           res_plddt_color.append(pLDDT_colors[3])
       elif plddt < 40:
           res_plddt_color.append(pLDDT_colors[4])

    # pLDDT per residue trace
    fig.add_trace(go.Scatter3d(
        x=CA_x,
        y=CA_y,
        z=CA_z,
        mode='lines',
        line=dict(
            color = res_plddt_color,
            width = 20,
            dash = 'solid'
        ),
        # opacity = 0,
        name = f"{protein_ID} pLDDT",
        showlegend = True,
        hovertext = res_name
    ))
    
    # Domain trace
    fig.add_trace(go.Scatter3d(
        x=CA_x,
        y=CA_y,
        z=CA_z,
        mode='lines',
        line=dict(
            color = res_color,
            width = 20,
            dash = 'solid'
        ),
        # opacity = 0,
        name = f"{protein_ID} domains",
        showlegend = True,
        hovertext = res_name
    ))
    
    
    if not is_for_network:
        # Protein name trace
        fig.add_trace(go.Scatter3d(
            x = (protein_CM[0],),
            y = (protein_CM[1],),
            z = (protein_CM[2] + 40,),
            text = protein_ID,
            mode = 'text',
            textposition = 'top center',
            textfont = dict(size = 40, color = "black"),
            name = "Protein ID",
            showlegend = True            
        ))
    
        # N-ter name trace
        fig.add_trace(go.Scatter3d(
            x = (CA_x[0],),
            y = (CA_y[0],),
            z = (CA_z[0],),
            text = "N",
            mode = 'text',
            textposition = 'top center',
            textfont = dict(size = 20, color = "black"),
            name = "N-ter",
            showlegend = True            
        ))
        
        # C-ter name trace
        fig.add_trace(go.Scatter3d(
            x = (CA_x[-1],),
            y = (CA_y[-1],),
            z = (CA_z[-1],),
            text = "C",
            mode = 'text',
            textposition = 'top center',
            textfont = dict(size = 20, color = "black"),
            name = "C-ter",
            showlegend = True            
        ))
    
        # Custom layout    
        fig.update_layout(
            title=f" Domains and pLDDT: {protein_ID}",
            legend=legend_position,
            scene=dict(
                # Show grid?
                xaxis=dict(showgrid=showgrid), yaxis=dict(showgrid=showgrid), zaxis=dict(showgrid=showgrid),
                # Show axis?
                xaxis_visible = show_axis, yaxis_visible = show_axis, zaxis_visible = show_axis,
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode="data", 
                # Allow free rotation along all axis
                dragmode="orbit",
            ),
            # Adjust layout margins
            margin=margin
        )
    
    if show_structure: plot(fig)
    
    if save_html:
        save_folder = "domains"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    
        # Save figure
        fig.write_html(os.path.join(save_folder, f"{protein_ID}-domains_plot.html"))
        
    if return_fig: return fig
    
    
    


# Working function
def detect_domains(sliced_PAE_and_pLDDTs, fasta_file_path, graph_resolution = 0.075, pae_power = 1, pae_cutoff = 5,
                   auto_domain_detection = True, graph_resolution_preset = None, save_preset = False,
                   save_png_file = True, show_image = False, show_structure = True, show_inline = True,
                   save_html = True, save_tsv = True):
    
    '''Modifies sliced_PAE_and_pLDDTs to add domain information. Generates the 
    following sub-keys for each protein_ID key:
        
        domain_clusters:
        ref_domain_clusters:
        no_loops_domain_clusters:
            
    Parameters:
    - sliced_PAE_and_pLDDTs (dict):
    - fasta_file_path (str):
    - graph_resolution (int):
    - auto_domain_detection (bool): set to False if you want to do semi-automatic
    - graph_resolution_preset (str): path to graph_resolution_preset.json file
    - save_preset (bool): set to True if you want to store the graph_resolution
        preset of each protein for later use.
    '''    
    
    # Progress
    print("")
    print("INITIALIZING: (Semi)Automatic domain detection algorithm...")
    print("")
    
    # If you want to save the domains definitions as a preset, this will be saved as JSON
    if save_preset: graph_resolution_for_preset = {}
    
    # If you have a preset, load it and use it
    if graph_resolution_preset != None:
        with open(graph_resolution_preset, 'r') as json_file:
            graph_resolution_preset = json.load(json_file)

    # Make a backup for later
    general_graph_resolution = graph_resolution
    
    # Detect domains for one protein at a time
    for P, protein_ID in enumerate(sliced_PAE_and_pLDDTs.keys()):
        
        # Flag used fon semi-automatic domain detection
        you_like = False
        
        # Return graph resolution to the general one
        graph_resolution = general_graph_resolution
        
        # If you have a preset
        if graph_resolution_preset != None:
            graph_resolution = graph_resolution_preset[protein_ID]
        
        while not you_like:
    
    ######### Compute domain clusters for all the best PAE matrices
        
            # Compute it with a resolution on 0.5
            domain_clusters = domains_from_pae_matrix_igraph(
                sliced_PAE_and_pLDDTs[protein_ID]['best_PAE_matrix'],
                pae_power, pae_cutoff, graph_resolution)
            
            # Save on dictionary
            sliced_PAE_and_pLDDTs[protein_ID]["domain_clusters"] = sorted(domain_clusters)
        
    ######### Reformat the domain clusters to make the plotting easier (for each protein)
            
            # Do reformatting
            ref_domain_clusters = reformat_clusters(sliced_PAE_and_pLDDTs[protein_ID]["domain_clusters"])
            
            # Save to dictionary
            sliced_PAE_and_pLDDTs[protein_ID]["ref_domain_clusters"] = ref_domain_clusters
        
    ######### Save plots of domain clusters for all the proteins
            matrix_data = sliced_PAE_and_pLDDTs[protein_ID]["best_PAE_matrix"]
            positions = sliced_PAE_and_pLDDTs[protein_ID]["ref_domain_clusters"][0]
            domain_clusters = sliced_PAE_and_pLDDTs[protein_ID]["ref_domain_clusters"][1]
            
            # plot before loop removal
            plot_before = plot_domains(protein_ID, matrix_data, positions, domain_clusters,
                         custom_title = "Before Loop Removal", out_folder= "domains_no_modification",
                         save_plot = False, show_plot = False)
        
            
    ######### Convert clusters of loops into the wrapper cluster domain and replot
            matrix_data = sliced_PAE_and_pLDDTs[protein_ID]["best_PAE_matrix"]
            positions = sliced_PAE_and_pLDDTs[protein_ID]["ref_domain_clusters"][0]
            domain_clusters = list(sliced_PAE_and_pLDDTs[protein_ID]["ref_domain_clusters"][1])
            
            no_loops_domain_clusters = remove_loop_clusters(domain_clusters)
            
            # Save to dictionary
            sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"] = [positions, tuple(no_loops_domain_clusters)]
            
            # Plot after loop removal
            plot_after = plot_domains(protein_ID, matrix_data, positions, no_loops_domain_clusters,
                         custom_title = "After Loop Removal", out_folder= "domains_no_loops",
                         save_plot = False, show_plot = False)
            
            
            # If the dataset was already converted to domains
            if graph_resolution_preset != None:
                you_like = True
                
            # If you want to do semi-auto domain detection
            elif not auto_domain_detection:
                
                # Create a single figure with both domain definitions subplots
                combine_figures_and_plot(plot_before, plot_after, protein_ID = protein_ID, save_png_file = save_png_file,
                                         show_image = show_image, show_inline = show_inline)
                
                # Plot the protein
                plot_backbone(protein_chain = sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"],
                              domains = sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"][1],
                              protein_ID = protein_ID, show_structure = show_structure, save_html = save_html)
                
                # Ask user if the detected domain distribution is OK
                user_input = input(f"Do you like the resulting domains for {protein_ID}? (y or n) - ")
                if user_input == "y":
                    print("   - Saving domain definition.")
                    you_like = True
                    
                    # Save it if you need to run again your pipeline 
                    if save_preset: graph_resolution_for_preset[protein_ID] = graph_resolution
                    
                elif user_input == "n":
                    while True:
                        try:
                            print(f"   - Current graph_resolution is: {graph_resolution}")
                            graph_resolution = float(input("   - Set a new graph_resolution value (int/float): "))
                            break  # Break out of the loop if conversion to float succeeds
                        except ValueError:
                            print("   - Invalid input. Please enter a valid float/int.")
                else: print("Unknown command: Try again.")
                
            else:
                you_like = True
                
                # Create a single figure with both domain definitions subplots
                combine_figures_and_plot(plot_before, plot_after, protein_ID = protein_ID, save_png_file = save_png_file,
                                         show_image = show_image, show_inline = show_inline)
                
                # Plot the protein
                plot_backbone(protein_chain = sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"],
                              domains = sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"][1],
                              protein_ID = protein_ID, show_structure = show_structure, save_html = save_html)
                
                
                # Save it if you need to run again your pipeline 
                if save_preset: graph_resolution_for_preset[protein_ID] = graph_resolution
                
    
    # save_preset is the path to the JSON file
    if save_preset:
        # Create a folder named "domains" if it doesn't exist
        save_folder = "domains"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        preset_out_JSON = save_folder + "/" + os.path.splitext(fasta_file_path)[0] + "-graph_resolution_preset.json"
        with open(preset_out_JSON, 'w') as json_file:
            json.dump(graph_resolution_for_preset, json_file)
    
    # Create domains_df -------------------------------------------------------
    
    # Helper fx
    def find_min_max_indices(lst, value):
        indices = [i for i, x in enumerate(lst) if x == value]
        if not indices:
            # The value is not in the list
            return None, None
        min_index = min(indices)
        max_index = max(indices)
        return min_index, max_index
    
    # Initialize df
    domains_columns = ["Protein_ID", "Domain", "Start", "End", "Mean_pLDDT"]
    domains_df = pd.DataFrame(columns = domains_columns)
    
    
    # Define domains and add them to domains_df
    for P, protein_ID in enumerate(sliced_PAE_and_pLDDTs.keys()):
        protein_domains = set(sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"][1])
        protein_residues = list(sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"].get_residues())
        for domain in protein_domains:
            start, end = find_min_max_indices(sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"][1], domain)
            domain_residues = protein_residues[start:end]
            domain_residues_plddts = [list(res.get_atoms())[0].get_bfactor() for res in domain_residues]
            domain_mean_plddt = np.mean(domain_residues_plddts)
            domain_row = pd.DataFrame(
                {"Protein_ID": [protein_ID],
                 # Save them starting at 1 (not zero)
                 "Domain": [domain + 1], 
                 "Start": [start + 1],
                 "End": [end + 1],
                 "Mean_pLDDT": [round(domain_mean_plddt, 1)]
                 })
            domains_df = pd.concat([domains_df, domain_row], ignore_index = True)
    
    # Convert domain, start and end values to int (and mean_plddt to float)
    domains_df['Domain'] = domains_df['Domain'].astype(int)
    domains_df['Start'] = domains_df['Start'].astype(int)
    domains_df['End'] = domains_df['End'].astype(int)
    domains_df['Mean_pLDDT'] = domains_df['Mean_pLDDT'].astype(float)
    
    if save_tsv:
        # Create a folder named "domains" if it doesn't exist
        save_folder = "domains"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        tsv_file_path = save_folder + "/" + os.path.splitext(fasta_file_path)[0] + "-domains.tsv"
        domains_df.to_csv(tsv_file_path, sep='\t', index=False)
        
    return domains_df