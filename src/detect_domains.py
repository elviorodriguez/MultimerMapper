
import os
import json
import numpy as np
import pandas as pd
import igraph
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import plotly.graph_objects as go           # For plotly plotting
from plotly.offline import plot             # To allow displaying plots
from logging import Logger
from io import BytesIO
import base64
import webbrowser
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
from copy import deepcopy

from utils.logger_setup import configure_logger

# -----------------------------------------------------------------------------
# Split proteins into domain using pae_to_domains.py --------------------------
# -----------------------------------------------------------------------------

# Function from pae_to_domains.py
def domains_from_pae_matrix_igraph(pae_matrix,
                                   pae_power: float | int = 1,
                                   pae_cutoff: float | int = 5,
                                   graph_resolution: float | int = 1,
                                   logger: Logger | None = None):
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
    if logger is None:
        logger = configure_logger()(__name__)

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
    clusters = defaultdict(list)
    for i, c in enumerate(membership):
        clusters[c].append(i)
    clusters = list(sorted(clusters.values(), key=lambda l:(len(l)), reverse=True))
    return clusters


def reformat_clusters(domain_clusters: list):
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
        A list of list with the residues positions in index 0 and the cluster
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


def plot_domains(protein_ID, matrix_data, positions, colors, custom_title=None, 
                 out_folder='domains', save_plot=True, show_plot=True):

    # Define a diverging colormap for the matrix
    matrix_cmap = 'coolwarm'

    # Define a common color list for the discrete integer values in clusters
    DOMAIN_COLORS = ['red', 'green', 'blue', 'purple', 'yellow', 'orange', 'brown', 'pink', 'cyan', 'lime', 'gray', 'olive']

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the matrix using matshow with the diverging colormap
    cax = ax.matshow(matrix_data, cmap=matrix_cmap)

    # Add a colorbar for the matrix
    fig.colorbar(cax)

    # Map the colors to the DOMAIN_COLORS list
    mapped_colors = [DOMAIN_COLORS[color % len(DOMAIN_COLORS)] for color in colors]

    # Scatter plot on top of the matrix with the mapped colors
    ax.scatter(positions, positions, c=mapped_colors, s=100)

    # Get unique cluster values
    unique_clusters = np.unique(colors)

    # Create a legend by associating cluster values with corresponding colors
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor = DOMAIN_COLORS[c % len(DOMAIN_COLORS)],
                                 markersize = 10,
                                 label=f'Domain {c}') for c in unique_clusters]

    # Add legend
    ax.legend(handles=legend_handles, title='Domains', loc='upper right')

    # Set labels and title
    plt.xlabel('Positions')
    plt.ylabel('Positions')
    plt.title(f"{custom_title}")

    if save_plot:
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        plt.savefig(os.path.join(out_folder, f"{protein_ID}_domains_plot.png"))

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig

def combine_figures_and_plot(fig1, fig2, out_path: str = ".", protein_ID = None, save_png_file = False, show_image = False, show_inline = True,
                             return_combined: bool = False):
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

    # Create a new image with double the width for side-by-side display and extra space for title
    combined_image = Image.new('RGB', (width * 2, height + 40), color='white')

    # Paste the images into the combined image
    combined_image.paste(image1, (0, 40))
    combined_image.paste(image2, (width, 40))    

    # Get font, create a drawing object and draw the title
    draw = ImageDraw.Draw(combined_image)
    title_font = ImageFont.load_default(size = 80)
    title_text = f"{protein_ID}" if protein_ID else "Protein: Unknown"
    text_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (width * 2 - text_width) / 2
    text_y = 10
    draw.text((text_x, text_y), title_text, font=title_font, fill='black')

 
    # Show the combined image
    if show_image: combined_image.show()        # (with the default image viewer of the S.O.)
    if show_inline: display(combined_image)     # (in-line in ipython console)

    # Save the combined image to a file?
    if save_png_file:
        
        if protein_ID == None:
            raise ValueError("protein_ID not provided. Required for saving domains plot file.")
        
        # Create a folder named "domains" if it doesn't exist
        save_folder = out_path + "/domains"
        os.makedirs(save_folder, exist_ok = True)
    
        # Save figure
        combined_image.save(os.path.join(save_folder, f"{protein_ID}-domains_plot.png"))
    
    if return_combined:
        # Convert the combined image to a base64 string
        combined_image_bytesio = BytesIO()
        combined_image.save(combined_image_bytesio, format='png')
        combined_image_bytesio.seek(0)
        img_base64 = base64.b64encode(combined_image_bytesio.read()).decode('utf-8')
    else:
        # Convert the combined image to a base64 string
        image2_bytesio = BytesIO()
        image2.save(image2_bytesio, format='png')
        image2_bytesio.seek(0)
        img_base64 = base64.b64encode(image2_bytesio.read()).decode('utf-8')
    
    return img_base64


# Convert clusters of loops into the wrapper cluster domain
def remove_loop_clusters(domain_clusters: list, logger: Logger | None = None):

    result = deepcopy(domain_clusters)
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
    
    # Function to solve any conflict with the domains
    def convert_clusters(original_list):
        converted_list = []
        current_cluster = -1
        previous_number = None
        
        for number in original_list:
            if number != previous_number:
                current_cluster += 1
                previous_number = number
            converted_list.append(current_cluster)
        
        return converted_list

    return convert_clusters(result)

# For semi-auto domain defining
def plot_backbone(protein_chain, domains, protein_ID = "", legend_position = dict(x=1.02, y=0.5), 
                  showgrid = True, margin=dict(l=0, r=0, b=0, t=0), show_axis = False, show_structure = False, 
                  save_html = False, return_fig = False, is_for_network = False, out_path: str = ".",
                  img_base64 = None):
    
    # Protein CM
    protein_CM = list(protein_chain.center_of_mass())
    
    # Create a 3D scatter plot
    fig = go.Figure()
    
    domain_colors = ['red', 'green', 'blue', 'purple', 'yellow', 'orange', 'brown', 'pink', 'cyan', 'lime', 'gray', 'olive'] * 20
    
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
        save_folder = os.path.join(out_path, "domains")
        os.makedirs(save_folder, exist_ok=True)

        # Save figure
        html_path = os.path.join(save_folder, f"{protein_ID}-domains_plot.html")
        fig.write_html(html_path)

        # Embed the 2D image in the HTML
        if img_base64:
            with open(html_path, 'r+') as f:
                
                html_content = f.read()
                
                # Add CSS for side-by-side layout with proportional sizes
                style = '''
                <style>
                .container {
                    display: flex;
                    justify-content: center; /* Horizontally center the contents */
                    align-items: center; /* Vertically center the contents */
                    height: 100vh; /* Set the container height to the viewport height */
                }
                .left-panel {
                    flex: 1;
                    padding: 10px;
                }
                .right-panel {
                    flex: 2;
                    padding: 10px;
                }
                .left-panel img {
                    width: 100%;
                    height: auto;
                }
                </style>
                '''
                
                # Add HTML structure for side-by-side layout
                html_structure = f'''
                <div class="container">
                    <div class="left-panel">
                        <img src="data:image/png;base64,{img_base64}" alt="Combined Image">
                    </div>
                    <div class="right-panel">
                        <!-- Interactive plot will be here -->
                '''
                
                # Insert the style and structure in the HTML content
                html_content = html_content.replace('<head>', f'<head>{style}')
                html_content = html_content.replace('<body>', f'<body>{html_structure}')
                
                # Close the right panel div after the plot
                html_content = html_content.replace('</body>', '</div></div></body>')
                
                f.seek(0)
                f.write(html_content)
                f.truncate()
        
        return html_path

    if return_fig: return fig

def convert_manual_domains_df_to_clusters(sliced_PAE_and_pLDDTs: dict, manual_domains_df: pd.DataFrame, logger: Logger | None = None):
    '''
    Modifies sliced_PAE_and_pLDDTs to contain "domain_clusters" and "ref_domain_clusters" information
    for each protein_ID.

    Returns:
        None
    '''
    if logger is None:
        logger = configure_logger()(__name__)

    for P, protein_ID in enumerate(sliced_PAE_and_pLDDTs.keys()):

        logger.info(f"  - Converting manual domains of {protein_ID} to clusters")
        
        # Assign domains clusters and ref domain cluster using manual_domains_df
        protein_domains_df = manual_domains_df.loc[manual_domains_df['Protein_ID'] == protein_ID]

        sliced_PAE_and_pLDDTs[protein_ID]["domain_clusters"] = [ [int(domain_row["Domain"])] * int(domain_row["End"] - domain_row["Start"] + 1) for i, domain_row in protein_domains_df.iterrows()]
        positions = list(range(0, sliced_PAE_and_pLDDTs[protein_ID]["length"]))
        domain_clusters = [item for sublist in sliced_PAE_and_pLDDTs[protein_ID]["domain_clusters"] for item in sublist]
        no_loops_domain_clusters = remove_loop_clusters(domain_clusters, logger)
        sliced_PAE_and_pLDDTs[protein_ID]["ref_domain_clusters"] = [positions, domain_clusters]
        sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"] = [positions, tuple(no_loops_domain_clusters)]
        

# Working function
def detect_domains(sliced_PAE_and_pLDDTs: dict, fasta_file_path: str, out_path: str,
                   graph_resolution: float | int = 0.075, pae_power: float | int  = 1, pae_cutoff: float | int  = 5,
                   auto_domain_detection: bool = True, graph_resolution_preset: bool | None = None, save_preset: bool  = True,
                   save_png_file: bool  = True, show_image: bool  = False, show_structure: bool  = False, show_inline: bool  = False,
                   save_html: bool  = True, save_tsv: bool  = True, overwrite: bool = False, manual_domains: str | None = None,
                   logger: Logger | None = None, show_PAE_along_backbone: bool = True):
    
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

    if logger is None:
        logger = configure_logger()(__name__)

    # Progress
    logger.info("")
    logger.info("INITIALIZING: (Semi)Automatic domain detection algorithm...")
    logger.info("")
    
    # If you want to save the domains definitions as a preset, this will be saved as JSON
    if save_preset: graph_resolution_for_preset = {}
    
    # If you have a preset, load it and use it
    if graph_resolution_preset is not None:
        with open(graph_resolution_preset, 'r') as json_file:
            graph_resolution_preset = json.load(json_file)
        
        # Some input verification
        if auto_domain_detection:
            logger.warning( '"auto_domain_detection = True" was passed together with "graph_resolution_preset"')
            logger.warning(f'   - graph_resolution_preset : {graph_resolution_preset}')
            logger.warning( '(Semi)Automatic domain definitions will be skipped')
            logger.warning( 'Make sure that is what you want')

    # Make a backup for later
    general_graph_resolution = graph_resolution

    # Create output folder
    if save_preset or save_tsv or save_png_file or save_html:
        save_folder = out_path + "/domains"
        os.makedirs(save_folder, exist_ok = overwrite)

    # If you want to manually set the domains (look at the sample file)
    use_manual = False
    if manual_domains is not None:
        use_manual = True
        logger.info(f"Domain assigned manually using manual_domains file: {manual_domains}")
        manual_domains_df = pd.read_csv(manual_domains, sep='\t')
        convert_manual_domains_df_to_clusters(sliced_PAE_and_pLDDTs, manual_domains_df, logger)

        # Some input verification
        if graph_resolution_preset is not None:
            logger.warning(f'"graph_resolution_preset" was passed together with "manual_domains"')
            logger.warning(f'   - graph_resolution_preset: {graph_resolution_preset}')
            logger.warning(f'   - manual_domains         : {manual_domains}')
            logger.warning( 'Domains definitions with "graph_resolution_preset" will be skipped')
            logger.warning( 'Make sure that is what you want')
        if auto_domain_detection:
            logger.warning(f'"auto_domain_detection = True" was passed together with "manual_domains"')
            logger.warning(f'   - manual_domains         : {manual_domains}')
            logger.warning( 'Automatic domain definitions will be skipped')
            logger.warning( 'Make sure that is what you want')
    
    # Detect domains for one protein at a time
    for P, protein_ID in enumerate(sliced_PAE_and_pLDDTs.keys()):
        
        # If manual domains was set, skip domain detection
        if use_manual:
            break

        # Flag used fon semi-automatic domain detection
        you_like = False
        
        # Return graph resolution to the general one
        graph_resolution = general_graph_resolution
        
        # If you have a preset
        if graph_resolution_preset is not None:
            try:
                graph_resolution = graph_resolution_preset[protein_ID]
            except Exception as e:
                logger.error(f'There is something wrong with the graph_resolution_preset: {graph_resolution_preset}')
                logger.error(f'The following exception was encountered: {e}')
                logger.error(f'"protein_ID" during exception: {protein_ID}')
                logger.warn(f'Continuing with semi-automatic domain detection')
                graph_resolution_preset = None
        
        while not you_like:
    
     ######## Compute domain clusters for all the best PAE matrices
        
            # Compute it with a resolution on 0.5
            domain_clusters = domains_from_pae_matrix_igraph(
                                sliced_PAE_and_pLDDTs[protein_ID]['best_PAE_matrix'],
                                pae_power, pae_cutoff, graph_resolution, logger = logger)
            
            # Save on dictionary
            sliced_PAE_and_pLDDTs[protein_ID]["domain_clusters"] = sorted(domain_clusters)
        
     ######## Reformat the domain clusters to make the plotting easier (for each protein)
            
            # Do reformatting
            ref_domain_clusters = reformat_clusters(sliced_PAE_and_pLDDTs[protein_ID]["domain_clusters"])
            
            # Save to dictionary
            sliced_PAE_and_pLDDTs[protein_ID]["ref_domain_clusters"] = ref_domain_clusters
        
     ######## Save plots of domain clusters for all the proteins
            matrix_data = sliced_PAE_and_pLDDTs[protein_ID]["best_PAE_matrix"]
            positions = sliced_PAE_and_pLDDTs[protein_ID]["ref_domain_clusters"][0]
            domain_clusters = sliced_PAE_and_pLDDTs[protein_ID]["ref_domain_clusters"][1]
            
            # plot before loop removal
            plot_before = plot_domains(protein_ID, matrix_data, positions, domain_clusters,
                            custom_title = "Before Loop Removal", out_folder = None,
                            save_plot = False, show_plot = False)
        
            
     ######## Convert clusters of loops into the wrapper cluster domain and replot
            matrix_data = sliced_PAE_and_pLDDTs[protein_ID]["best_PAE_matrix"]
            positions = sliced_PAE_and_pLDDTs[protein_ID]["ref_domain_clusters"][0]
            domain_clusters = list(sliced_PAE_and_pLDDTs[protein_ID]["ref_domain_clusters"][1])
            
            no_loops_domain_clusters = remove_loop_clusters(domain_clusters, logger = logger)
            
            # Save to dictionary
            sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"] = [positions, tuple(no_loops_domain_clusters)]
            
            # Plot after loop removal
            plot_after = plot_domains(protein_ID, matrix_data, positions, no_loops_domain_clusters,
                         custom_title = "After Loop Removal", out_folder = None,
                         save_plot = False, show_plot = False)
            
            
            # If the dataset was already converted to domains
            if graph_resolution_preset is not None:
                you_like = True
                
            # If you want to do semi-auto domain detection
            elif not auto_domain_detection:
                
                # Create a single figure with both domain definitions subplots
                comb_img_base64 = combine_figures_and_plot(
                    plot_before, plot_after, protein_ID = protein_ID, save_png_file = save_png_file,
                    show_image = show_image, show_inline = show_inline, out_path = out_path)
                
                # Plot the protein
                html_path = plot_backbone(
                    protein_chain = sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"],
                    domains = no_loops_domain_clusters,
                    protein_ID = protein_ID, show_structure = show_structure, save_html = save_html,
                    out_path = out_path, img_base64 = comb_img_base64)
                
                # Open HTML file with PAE and backbone colored by detected domains
                if show_PAE_along_backbone:
                    webbrowser.open(f"{html_path}")
                
                # Ask user if the detected domain distribution is OK
                user_input = input(f"Do you like the resulting domains for {protein_ID}? (y or n) - ")
                if user_input == "y":
                    logger.info("   - Saving domain definition.")
                    you_like = True
                    
                    # Save it if you need to run again your pipeline 
                    if save_preset: graph_resolution_for_preset[protein_ID] = graph_resolution
                    
                elif user_input == "n":
                    while True:
                        try:
                            logger.info(f"   - Current graph_resolution is: {graph_resolution}")
                            graph_resolution = float(input("   - Set a new graph_resolution value (int/float): "))
                            break  # Break out of the loop if conversion to float succeeds
                        except ValueError:
                            logger.info("   - Invalid input. Please enter a valid float/int.")
                else: logger.info("Unknown command: Try again.")
                
            else:
                you_like = True
                
                # Create a single figure with both domain definitions subplots
                comb_img_base64 = combine_figures_and_plot(
                    plot_before, plot_after, protein_ID = protein_ID, save_png_file = save_png_file,
                    show_image = show_image, show_inline = show_inline, out_path = out_path)
                
                # Plot the protein
                html_path = plot_backbone(
                    protein_chain = sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"],
                    domains = sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"][1],
                    protein_ID = protein_ID, show_structure = show_structure, save_html = save_html,
                    out_path = out_path, img_base64 = comb_img_base64)
                
                # Open HTML file with PAE and backbone colored by detected domains
                if show_PAE_along_backbone:
                    webbrowser.open(f"{html_path}")
                
                
                # Save it if you need to run again your pipeline 
                if save_preset: graph_resolution_for_preset[protein_ID] = graph_resolution
    
    # save_preset is the path to the JSON file
    if save_preset:
        # Create a folder named "domains" if it doesn't exist
        preset_out_JSON = save_folder + "/graph_resolution_preset.json"
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
        save_folder = out_path + "/domains"
        tsv_file_path = save_folder + "/domains.tsv"
        domains_df.to_csv(tsv_file_path, sep='\t', index=False)
        
    return domains_df