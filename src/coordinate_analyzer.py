
from asyncio import subprocess
from logging import Logger
import pandas as pd
import numpy as np
from Bio.PDB import Chain, Superimposer
from Bio.PDB.Polypeptide import protein_letters_3to1
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
# from plotly.subplots import make_subplots
import base64
from collections import Counter
import os

from traj.partners_density import analyze_protein_distribution, save_results_to_csv, plot_distributions, html_interactive_metadata
from utils.logger_setup import configure_logger


def add_domain_RMSD_against_reference(graph, domains_df, sliced_PAE_and_pLDDTs,
                                        pairwise_2mers_df, pairwise_Nmers_df,
                                        domain_RMSD_plddt_cutoff,
                                        trimming_RMSD_plddt_cutoff,
                                        logger: Logger | None = None):
    
    hydrogens = ('H', 'H1', 'H2', 'H3', 'HA', 'HA2', 'HA3', 'HB', 'HB1', 'HB2', 
                    'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE2', 'HE3', 'HZ1', 'HZ2', 
                    'HZ3', 'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23', 'HZ', 'HD1',
                    'HE1', 'HD11', 'HD12', 'HD13', 'HG', 'HG1', 'HD21', 'HD22', 'HD23',
                    'NH1', 'NH2', 'HE', 'HH11', 'HH12', 'HH21', 'HH22', 'HE21', 'HE22',
                    'HE2', 'HH', 'HH2')
    
    if logger is None:
        logger = configure_logger()(__name__)
    
    def create_model_chain_from_residues(residue_list, model_id=0, chain_id='A'):

        # Create a Biopython Chain
        chain = Chain.Chain(chain_id)

        # Add atoms to the chain
        for residue in residue_list:
            chain.add(residue)
            
        return chain

    def calculate_rmsd(chain1, chain2, trimming_RMSD_plddt_cutoff):
        # Make sure both chains have the same number of atoms
        if len(chain1) != len(chain2):
            raise ValueError("Both chains must have the same number of atoms.")

        # Initialize the Superimposer
        superimposer = Superimposer()

        # Extract atom objects from the chains (remove H atoms)
        atoms1 = [atom for atom in list(chain1.get_atoms()) if atom.id not in hydrogens]
        atoms2 = [atom for atom in list(chain2.get_atoms()) if atom.id not in hydrogens]
        
        # Check equal length
        if len(atoms1) != len(atoms2):
            raise ValueError("Something went wrong after H removal: len(atoms1) != len(atoms2)")
        
        # Get indexes with lower than trimming_RMSD_plddt_cutoff atoms in the reference 
        indices_to_remove = [i for i, atom in enumerate(atoms1) if atom.bfactor is not None and atom.bfactor < trimming_RMSD_plddt_cutoff]
        
        # Remove the atoms
        for i in sorted(indices_to_remove, reverse=True):
            del atoms1[i]
            del atoms2[i]
            
        # Check equal length after removal
        if len(atoms1) != len(atoms2):
            raise ValueError("Something went wrong after less than pLDDT_cutoff atoms removal: len(atoms1) != len(atoms2)")

        # Set the atoms to the Superimposer
        superimposer.set_atoms(atoms1, atoms2)

        # Calculate RMSD
        rmsd = superimposer.rms

        return rmsd
    
    def get_graph_protein_pairs(graph):
        graph_pairs = []
        
        for edge in graph.es:
            prot1 = edge.source_vertex["name"]
            prot2 = edge.target_vertex["name"]
            
            graph_pairs.append((prot1,prot2))
            graph_pairs.append((prot2,prot1))
            
        return graph_pairs
    
    logger.info("Computing domain RMSD against reference and adding it to combined graph.")
    
    # Get all pairs in the graph
    graph_pairs = get_graph_protein_pairs(graph)
    
    # Work protein by protein
    for vertex in graph.vs:
        
        protein_ID = vertex["name"]
        ref_structure = sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"]
        ref_residues = list(ref_structure.get_residues())
        
        # Add sub_domains_df to vertex
        vertex["domains_df"] = domains_df.query(f'Protein_ID == "{protein_ID}"').filter(["Domain", "Start", "End", "Mean_pLDDT"])
        
        # Initialize dataframes to store RMSD
        columns = ["Domain","Model","Chain", "Mean_pLDDT", "RMSD"]
        vertex["RMSD_df"] = pd.DataFrame(columns = columns)
        
        logger.info(f"   - Computing RMSD for {protein_ID}...")
        
        # Work domain by domain
        for D, domain in domains_df.query(f'Protein_ID == "{protein_ID}"').iterrows():
            
            
            # Do not compute RMSD for disordered domains
            if domain["Mean_pLDDT"] < domain_RMSD_plddt_cutoff:
                continue
            
            # Start and end indexes for the domain
            start = domain["Start"] - 1
            end = domain["End"] - 1
            domain_num = domain["Domain"]
            
            # Create a reference chain for the domain (comparisons are made against it)
            ref_domain_chain = create_model_chain_from_residues(ref_residues[start:end])
            
            # Compute RMSD for 2-mers models that are part of interactions (use only rank 1)
            for M, model in pairwise_2mers_df.query(f'(protein1 == "{protein_ID}" | protein2 == "{protein_ID}") & rank == 1').iterrows():
                
                prot1 = str(model["protein1"])
                prot2 = str(model["protein2"])
                
                model_proteins = (prot1, prot2)
                
                # If the model does not represents an interaction, jump to the next one
                if (prot1, prot2) not in graph_pairs:
                    continue
                
                # Work chain by chain in the model
                for query_chain in model["model"].get_chains():
                    query_chain_ID = query_chain.id
                    query_chain_seq = "".join([protein_letters_3to1[res.get_resname()] for res in query_chain.get_residues()])
                    
                    # Compute RMSD only if sequence match
                    if query_chain_seq == sliced_PAE_and_pLDDTs[protein_ID]["sequence"]:
                        
                        query_domain_residues = list(query_chain.get_residues())
                        query_domain_chain = create_model_chain_from_residues(query_domain_residues[start:end])
                        query_domain_mean_pLDDT = np.mean([list(res.get_atoms())[0].get_bfactor() for res in query_domain_chain.get_residues()])
                        query_domain_RMSD = calculate_rmsd(ref_domain_chain, query_domain_chain, domain_RMSD_plddt_cutoff)
                        
                        query_domain_RMSD_data = pd.DataFrame({
                            "Domain": [domain_num],
                            "Model": [model_proteins],
                            "Chain": [query_chain_ID],
                            "Mean_pLDDT": [round(query_domain_mean_pLDDT, 1)],
                            "RMSD": [round(query_domain_RMSD, 2)] 
                            })
                        
                        vertex["RMSD_df"] = pd.concat([vertex["RMSD_df"], query_domain_RMSD_data], ignore_index = True)
            
            
            # Compute RMSD for N-mers models that are part of interactions (use only rank 1)
            for M, model in pairwise_Nmers_df.query(f'(protein1 == "{protein_ID}" | protein2 == "{protein_ID}") & rank == 1').iterrows():
                
                prot1 = model["protein1"]
                prot2 = model["protein2"]
                
                model_proteins = tuple(model["proteins_in_model"])
                
                # If the model does not represents an interaction, jump to the next one
                if (prot1, prot2) not in graph_pairs:
                    continue
                
                # Work chain by chain in the model
                for query_chain in model["model"].get_chains():
                    query_chain_ID = query_chain.id
                    query_chain_seq = "".join([protein_letters_3to1[res.get_resname()] for res in query_chain.get_residues()])
                    
                    # Compute RMSD only if sequence match
                    if query_chain_seq == sliced_PAE_and_pLDDTs[protein_ID]["sequence"]:
                        
                        query_domain_residues = list(query_chain.get_residues())
                        query_domain_chain = create_model_chain_from_residues(query_domain_residues[start:end])
                        query_domain_mean_pLDDT = np.mean([list(res.get_atoms())[0].get_bfactor() for res in query_domain_chain.get_residues()])
                        query_domain_RMSD = calculate_rmsd(ref_domain_chain, query_domain_chain, domain_RMSD_plddt_cutoff)
                        
                        query_domain_RMSD_data = pd.DataFrame({
                            "Domain": [domain_num],
                            "Model": [model_proteins],
                            "Chain": [query_chain_ID],
                            "Mean_pLDDT": [round(query_domain_mean_pLDDT, 1)],
                            "RMSD": [round(query_domain_RMSD, 2)]
                            })
                        
                        vertex["RMSD_df"] = pd.concat([vertex["RMSD_df"], query_domain_RMSD_data], ignore_index = True)

    # remove duplicates
    for vertex in graph.vs:
        vertex["RMSD_df"] = vertex["RMSD_df"].drop_duplicates().reset_index(drop = True)




# -----------------------------------------------------------------------------
# --------------------------- RMSD trajectories -------------------------------
# -----------------------------------------------------------------------------

import os
import logging
from Bio import PDB
from typing import Literal
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

from utils.logger_setup import configure_logger
from utils.pdb_utils import get_domain_atoms, ChainSelect, DomainSelect


def get_monomers_models_from_pairwise_2mers(protein_ID: str, protein_seq: str,
                                            pairwise_2mers_df: pd.DataFrame):
    """
    Extract monomer chains from pairwise 2-mer models that match the given protein ID.

    Args:
        protein_ID (str): The ID of the protein to match.
        protein_seq (str): The sequence of the protein (not used in the function, but kept for consistency).
        pairwise_2mers_df (pd.DataFrame): DataFrame containing pairwise 2-mer model information.

    Returns:
        dict: A dictionary containing lists of monomer chains and their attributes:
            - 'monomer_chains': List of PDB.Chain.Chain objects
            - 'is_chain': List of chain identifiers ('A' or 'B')
            - 'is_model': List of model identifiers (empty in this function)
            - 'is_rank': List of rank identifiers (empty in this function)
    """
    # List of monomers
    monomer_chains_from_2mers: list[PDB.Chain.Chain] = []
    is_chain                 : list[str]             = []
    is_model                 : list[tuple]           = []
    is_rank                  : list[int]             = []
    
    # Get 2-mers chains
    for i, row in pairwise_2mers_df.iterrows():
        protein1 = row['protein1']
        protein2 = row['protein2']
        
        # Extract the chains if there is a match
        if protein1 == protein_ID or protein2 == protein_ID:
            model_chains = [chain for chain in row['model'].get_chains()]
            model = (row['protein1'], row['protein2'])
            rank = row['rank']
        
        # Extract chain A
        if protein1 == protein_ID:
            monomer_chains_from_2mers.append(model_chains[0])
            is_chain.append("A")
            is_model.append(model)
            is_rank.append(rank)
        
        # Extract chain B
        if protein2 == protein_ID:
            monomer_chains_from_2mers.append(model_chains[1])
            is_chain.append("B")
            is_model.append(model)
            is_rank.append(rank)
            
    # List of attributes of the models, to get specific info
    monomer_chains_from_2mers_dict: dict = {
        "monomer_chains": monomer_chains_from_2mers,
        "is_chain": is_chain,
        "is_model": is_model,
        "is_rank": is_rank
    }

    return monomer_chains_from_2mers_dict

# Returns the path keys that correspond to Nmers
def get_Nmers_paths_in_all_pdb_data(all_pdb_data: dict):
    """
    Identify and return the paths in the all_pdb_data dictionary that correspond to N-mers.

    Args:
        all_pdb_data (dict): A dictionary containing PDB data structures.

    Returns:
        list[str]: A list of path keys that correspond to N-mers (structures with more than 2 chains).
    """
    
    N_mer_paths: list[str] = []
    
    for path in all_pdb_data.keys():
        
        path_chain_IDs = [k for k in all_pdb_data[path].keys() if len(k) == 1]
        
        if len(path_chain_IDs) > 2:
            N_mer_paths.append(path)
    
    return N_mer_paths
            
        
    

def get_monomers_models_from_all_pdb_data(protein_ID: str, protein_seq: str,
                                          all_pdb_data: pd.DataFrame):
    """
    Extract monomer chains from N-mer models in all_pdb_data that match the given protein ID.

    Args:
        protein_ID (str): The ID of the protein to match.
        protein_seq (str): The sequence of the protein (not used in the function, but kept for consistency).
        all_pdb_data (dict): A dictionary containing PDB data structures.

    Returns:
        dict: A dictionary containing lists of monomer chains and their attributes:
            - 'monomer_chains': List of PDB.Chain.Chain objects
            - 'is_chain': List of chain identifiers
            - 'is_model': List of model identifiers (tuples of protein IDs)
            - 'is_rank': List of rank identifiers
    """
    # Get the keys that correspond to Nmer predictions
    N_mer_path_keys: list[str] = get_Nmers_paths_in_all_pdb_data(all_pdb_data)
    
    # List of monomers
    monomer_chains_from_Nmers: list[PDB.Chain.Chain] = []
    is_chain                 : list[str]             = []
    is_model                 : list[tuple]           = []
    is_rank                  : list[int]             = []
    
    # For each Nmer prediction path
    for path_key in N_mer_path_keys:
        
        # Extract Chain IDs and protein ID of each chain
        prediction_chain_IDs = [k for k in all_pdb_data[path_key].keys() if len(k) == 1]
        prediction_protein_IDs = [all_pdb_data[path_key][chain_ID]["protein_ID"] for chain_ID in prediction_chain_IDs]
        
        if protein_ID not in prediction_protein_IDs:
            continue
        
        # Get the indices that match the query protein_ID and get only matching chains
        matching_chain_IDs_indexes = [i for i, s in enumerate(prediction_protein_IDs) if s == protein_ID]
                        
        for rank in sorted(all_pdb_data[path_key]['full_PDB_models'].keys()):
            
            # print(rank)
            
            # # Get all the model chains
            model_chains = list(all_pdb_data[path_key]['full_PDB_models'][rank].get_chains())
            
            # And keep only those that match the protein ID
            matching_model_chains     = [model_chains[i] for i in matching_chain_IDs_indexes]
            matching_model_chains_IDs = [prediction_chain_IDs[i] for i in matching_chain_IDs_indexes]
            
            # print(matching_model_chains)
            
            monomer_chains_from_Nmers.extend(matching_model_chains)
            is_chain.extend(matching_model_chains_IDs)
            is_model.extend([tuple(prediction_protein_IDs)] * len(matching_chain_IDs_indexes))
            is_rank.extend([rank] * len(matching_chain_IDs_indexes))


    # List of attributes of the models, to get specific info
    monomer_chains_from_Nmers_dict: dict = {
        "monomer_chains": monomer_chains_from_Nmers,
        "is_chain": is_chain,
        "is_model": is_model,
        "is_rank": is_rank
    }
    
    return monomer_chains_from_Nmers_dict
    
def calculate_weighted_rmsd(coords1, coords2, weights):
    """
    Calculate the weighted RMSD between two sets of coordinates.
    
    Args:
        coords1 (np.array): First set of coordinates (N x 3).
        coords2 (np.array): Second set of coordinates (N x 3).
        weights (np.array): Weights for each atom (N).
        
    Returns:
        float: Weighted RMSD value.
    """
    diff = coords1 - coords2
    weighted_diff_sq = weights[:, np.newaxis] * (diff ** 2)
    return np.sqrt(np.sum(weighted_diff_sq) / np.sum(weights))


def determine_num_clusters(data_matrix, method='silhouette', logger: Logger | None = None):

    if method == 'silhouette':

        silhouette_scores = []
        max_clusters = min(len(data_matrix), 10)

        for k in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init = "auto")
                cluster_labels = kmeans.fit_predict(data_matrix)
                silhouette_avg = silhouette_score(data_matrix, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            except Exception as e:
                logger.warn(f"   k-means pLDDT clustering with {k} clusters caused an error: {str(e)}")
                logger.warn(f"   Assigning silhouette score of 0 for k={k}")
                silhouette_scores.append(0)

        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

        return optimal_clusters

    else:
        raise ValueError("Unsupported method. Use 'silhouette'.")

def cluster_curves(curve_lists, domains_df: pd.DataFrame | None = None,
                   num_clusters=None, method='interactive',
                   y_label="Mean cluster pLDDT", x_label="Position", 
                   protein_ID="Protein", filename="cluster_plot.png",
                   plot_width_factor: float | int = 0.05,
                   plot_height: float | int = 10,
                   fontsize: int = 30,
                   show_plot: bool = False,
                   logger: Logger | None = None):
    """
    Clusters curves based on similarity.

    Args:
        curve_lists (List[np.ndarray]): List of NumPy arrays, each representing a curve.
        num_clusters (int, optional): Number of clusters. If None, it will be determined automatically.
        method (str, optional): Method to determine the number of clusters ('interactive', 'silhouette', 'gap').
        y_label (str): Label for the y-axis.
        x_label (str): Label for the x-axis.
        protein_ID (str): Title for the plot.
        filename (str): Filename to save the plot.

    Returns:
        Tuple[List[int], List[np.ndarray], List[np.ndarray]]: A tuple containing:
            - Cluster assignments for each curve
            - Mean curves for each cluster
            - Standard deviation curves for each cluster
    """
    if logger is None:
        logger = configure_logger()(__name__)

    data_matrix = np.vstack(curve_lists)

    if num_clusters is None:
        if method == 'interactive':
            distortions = []
            max_clusters = min(len(curve_lists), 10)
            for k in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init = "auto")
                kmeans.fit(data_matrix)
                distortions.append(kmeans.inertia_)

            plt.plot(range(1, max_clusters + 1), distortions, marker='o')
            plt.xlabel('Number of clusters')
            plt.ylabel('Distortion (within-cluster sum of squares)')
            plt.title('Elbow Method for Optimal K')
            plt.show()

            num_clusters = int(input("Enter the number of clusters (K): "))

        elif method in ['silhouette']:
            num_clusters = determine_num_clusters(data_matrix, method, logger = logger)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init = 'auto')
    cluster_assignments = kmeans.fit_predict(data_matrix)

    # Calculate the mean and standard deviation for each cluster
    mean_curves = []
    std_curves = []
    for i in range(num_clusters):
        cluster_curves = data_matrix[cluster_assignments == i]
        mean_curve = np.mean(cluster_curves, axis=0)
        std_curve = np.std(cluster_curves, axis=0)
        mean_curves.append(mean_curve)
        std_curves.append(std_curve)

    # Initialize figure
    width = plot_width_factor * len(data_matrix[0])
    if width > 40:
        width = 40 # limit max width
    elif width < 10:
        width = 10 # limit min width
    plt.figure(figsize=(width, plot_height))
    
    # Add vertical lines for domain boundaries
    if domains_df is not None:
        matching_domains = domains_df[domains_df['Protein_ID'] == protein_ID]
        plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
        for _, row in matching_domains.iterrows():
            plt.axvline(x=row['End'], color='black', linestyle='--', linewidth=2)
            plt.text((row['Start'] + row['End'])/2,
                     data_matrix.max(),
                     f"Dom{row['Domain']}",
                     horizontalalignment='center',
                     verticalalignment='bottom',
                     fontsize = fontsize / 2)
    
    # Plotting the mean curves with standard deviation
    for i, (mean_curve, std_curve) in enumerate(zip(mean_curves, std_curves)):
        plt.plot(mean_curve, label=f'Cluster {i+1}', linewidth = 4)
        plt.fill_between(range(len(mean_curve)), mean_curve - std_curve, mean_curve + std_curve, alpha=0.3)
    
        
    # Adjust font size for each part
    title_fontsize = fontsize * (14/10)
    labels_fontsize = fontsize * (12/10)
    ticks_fontsize =  fontsize * (10/10)
    
    # Apply Style
    plt.title(protein_ID, fontsize = title_fontsize)
    plt.xlabel(x_label, fontsize = labels_fontsize)
    plt.ylabel(y_label, fontsize = labels_fontsize)
    plt.legend(fontsize = ticks_fontsize)
    plt.yticks(fontsize = ticks_fontsize)
    plt.xticks(fontsize = ticks_fontsize)
    plt.grid(True)
    
    plt.savefig(filename, bbox_inches='tight')
    if show_plot:
        plt.show()
    
    plt.close()

    return cluster_assignments, mean_curves, std_curves

def calculate_rmsf(all_coords):
    """
    Calculates Root Mean Square Fluctuation (RMSF)
    
    Args:
    all_coords (np.ndarray): Coordinates of models, shape (num_models, num_residues, 3)
    
    Returns:
    np.ndarray: RMSF values for each residue
    """
    # Compute mean position for each residue across all models
    mean_coords = np.mean(all_coords, axis=0)
    
    # Compute per-residue RMSF
    rmsf_values = np.sqrt(np.mean((all_coords - mean_coords)**2, axis=0))
    rmsf_values = np.mean(rmsf_values, axis=1)  # Average RMSF per residue
    
    return rmsf_values


def plot_rmsf(rmsf_values, domains_df,
              y_label="RMSF (Å)", x_label="Residue", 
              protein_ID="Protein", filename="RMSF.png",
              plot_width_factor: float | int = 0.05,
              plot_height: float | int = 10,
              fontsize: int = 30,
              show_plot: bool = False,
              domain_start: int|None = None):
    
    # Initialize figure
    width = plot_width_factor * len(rmsf_values)
    if width > 40:
        width = 40 # limit max width
    elif width < 10:
        width = 10 # limit min width
    plt.figure(figsize=(width, plot_height))
    
    # Add vertical lines for domain boundaries
    if domains_df is not None:
        matching_domains = domains_df[domains_df['Protein_ID'] == protein_ID]
        plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
        for _, row in matching_domains.iterrows():
            plt.axvline(x=row['End'], color='black', linestyle='--', linewidth=2)
            plt.text((row['Start'] + row['End'])/2,
                     rmsf_values.max(),
                     f"Dom{row['Domain']}",
                     horizontalalignment='center',
                     verticalalignment='bottom',
                     fontsize = fontsize / 2)
    
    # Plotting the mean curves with standard deviation
    plt.plot(rmsf_values, linewidth = 4)
    
    # Adjust font size for each part
    title_fontsize = fontsize * (14/10)
    labels_fontsize = fontsize * (12/10)
    ticks_fontsize =  fontsize * (10/10)
    
    # Apply Style
    plt.title(protein_ID, fontsize = title_fontsize)
    plt.xlabel(x_label, fontsize = labels_fontsize)
    plt.ylabel(y_label, fontsize = labels_fontsize)
    plt.yticks(fontsize = ticks_fontsize)
    plt.xticks(fontsize = ticks_fontsize)
    plt.grid(True)

    # Adjust the x-axis labels to reflect domain_start
    if domain_start is not None:
        ax = plt.gca()  # Get the current axis
        ax.set_xticks(ax.get_xticks())  # Retain default tick positions
        ax.set_xticklabels(domain_start + ax.get_xticks().astype(int))  # Adjust the labels by domain_start

    plt.savefig(filename, bbox_inches='tight')
    if show_plot:
        plt.show()
        
    plt.close()

def calculate_radius_of_gyration(coords):
    """
    Calculate the radius of gyration for a set of coordinates.
    
    Args:
        coords (numpy.ndarray): Array of shape (n_atoms, 3) containing the x, y, z coordinates of atoms.
    Returns:
        float: The radius of gyration.
    """
    # Calculate the center of mass
    center_of_mass = np.mean(coords, axis=0)
    
    # Calculate the squared distances from the center of mass
    squared_distances = np.sum((coords - center_of_mass) ** 2, axis=1)
    
    # Calculate the mean squared distance
    mean_squared_distance = np.mean(squared_distances)
    
    # Calculate the radius of gyration
    radius_of_gyration = np.sqrt(mean_squared_distance)
    
    return radius_of_gyration

def compute_rog_for_all_chains(all_coords):
    """
    Compute the radius of gyration for each set of chain coordinates.
    
    Args:
    all_coords (numpy.ndarray): Array of shape (n_chains, n_atoms, 3) containing coordinates for multiple chains.
    
    Returns:
    numpy.ndarray: Array of radius of gyration values for each chain.
    """
    rog_values = np.array([calculate_radius_of_gyration(chain_coords) for chain_coords in all_coords])
    return rog_values


def plot_traj_metadata(metadata_list, metadata_type: str, protein_trajectory_folder: str, protein_ID: str, filename_suffix: str):

    plot_filename = os.path.join(protein_trajectory_folder, f'{protein_ID}_{filename_suffix}_traj_{metadata_type}.png')

    plt.figure()
    plt.plot(range(1, len(metadata_list) + 1 ), metadata_list)
    plt.title(f'{protein_ID} {filename_suffix} {metadata_type}', fontsize = 18)
    plt.xlabel("Trajectory Model (Nº)", fontsize = 16)
    plt.ylabel(f"{metadata_type}", fontsize = 16)

    plt.savefig(plot_filename)

    plt.close()


def plot_traj_heatmap(metadata_list, sorted_indices: list[int], metadata_type_y: str, metadata_type_color: str, 
                      protein_trajectory_folder: str, protein_ID: str, filename_suffix: str,
                      domain_start = None, domain_end = None):
    
    plot_filename = os.path.join(protein_trajectory_folder, f'{protein_ID}_{filename_suffix}_traj_{metadata_type_color}_heatmap.png')
    
    # Convert metadata_list to a NumPy array for processing
    metadata_array = np.array([metadata_list[idx] for idx in sorted_indices])
    
    # Ensure shape is (n_models, n_residues)
    if metadata_array.ndim != 2:
        raise ValueError("metadata_list must be a 2D array-like structure with shape (n_models, n_residues)")
    
    n_models, n_residues = metadata_array.shape
    
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    if metadata_type_color == "pLDDT":
        # Define the custom colormap for pLDDT
        from matplotlib.colors import LinearSegmentedColormap
        pLDDT_colors = ["#FF0000", "#FFA500", "#FFFF00", "#ADD8E6", "#00008B"]
        pLDDT_boundaries = [0, 40, 60, 80, 100]
        
        cmap = LinearSegmentedColormap.from_list("pLDDT_cmap", list(zip(np.array(pLDDT_boundaries) / 100, pLDDT_colors)))
        vmin, vmax = 0, 100  # pLDDT values range from 0 to 100
    else:
        cmap = "viridis"
        vmin, vmax = metadata_array.min(), metadata_array.max()
    
    # Plot heatmap
    ax = sns.heatmap(metadata_array.T, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={'label': metadata_type_color})
    
    # Remove the background grid
    ax.grid(False)

    # Labels and title
    ax.set_xlabel("Trajectory Model (Nº)", fontsize=16)
    ax.set_ylabel(metadata_type_y, fontsize=16)
    ax.set_title(f"{protein_ID} - {metadata_type_color} Heatmap", fontsize=18)
    
    # Ensure y-axis starts from residue 1 (or domain start residue)
    if metadata_type_y == "Residue":
        if domain_start is None:
            domain_start = 1
        if domain_end is None:
            domain_end = n_residues
        
        # Displace the axis to reflect the domain's start
        ax.set_yticks(ax.get_yticks())  # Get the default tick positions
        ax.set_yticklabels(domain_start + ax.get_yticks().astype(int))  # Adjust labels to reflect the domain

    # Save plot
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
  

# Helper fx to save trajectory file and metadata
def save_trajectory(sorted_indices, protein_ID, filename_suffix, protein_trajectory_folder,
                    
                    RMSDs: list, pLDDTs: np.array, mean_pLDDTs: list, ROGs: np.array,

                    rot_matrixes: list, tran_vectors: list,

                    all_chains, all_chain_types, all_chain_info, domain_start=None, domain_end=None):
    
    trajectory_file = os.path.join(protein_trajectory_folder, f'{protein_ID}_{filename_suffix}_traj.pdb')
    
    df_cols = ['Traj_N', 'Type', 'Is_chain', 'Rank', 'Model', 'RMSD', 'pLDDT', 'ROG']
    trajectory_df = pd.DataFrame(columns=df_cols)
    
    io = PDB.PDBIO()
    with open(trajectory_file, 'w') as f1:
        for i, idx in enumerate(sorted_indices):
            chain = all_chains[idx]
            chain_type = all_chain_types[idx]
            chain_info = all_chain_info[idx]
            model_name = f"MODEL_{i+1}_{chain_type}_{chain_info[0]}_{'-'.join(map(str, chain_info[1]))}_{chain_info[2]}"
            
            model_data = pd.DataFrame({
                "Traj_N"  : [i+1], 
                "Type"    : [chain_type],
                "Is_chain": [chain_info[0]],
                "Rank"    : [chain_info[2]],
                "Model"   : ['__vs__'.join(map(str, chain_info[1]))],
                "RMSD"    : [RMSDs[idx]],
                "pLDDT"   : [mean_pLDDTs[idx]],
                "ROG"     : [ROGs[idx]]
                })
            trajectory_df = pd.concat([trajectory_df, model_data], ignore_index=True)

            ### APPLY ROTATION AND TRANSLATION TO ALL ATOMS ###
            rotation_matrix = rot_matrixes[idx]
            translation_vector = tran_vectors[idx]
            
            # Apply transformation at the chain level
            chain.transform(rotation_matrix, translation_vector)

            # Save transformed chain            
            io.set_structure(chain.parent)
            if domain_start is not None and domain_end is not None:
                io.save(f1, select=DomainSelect(chain, domain_start, domain_end), write_end=False)
            else:
                io.save(f1, select=ChainSelect(chain), write_end=False)
            f1.write(f"ENDMDL\nTITLE     {model_name}\n")

            ### REVERT BACK TO ORIGINAL POSITIONS ###
            inverse_rotation = rotation_matrix.T
            inverse_translation = -np.dot(inverse_rotation, translation_vector)
            chain.transform(inverse_rotation, inverse_translation)
    
    # Generate some plots
    plot_traj_metadata(metadata_list = trajectory_df['RMSD'],
                       metadata_type = "RMSD",
                       protein_trajectory_folder = protein_trajectory_folder,
                       protein_ID = protein_ID, filename_suffix = filename_suffix)
    plot_traj_metadata(metadata_list = trajectory_df['pLDDT'],
                       metadata_type = "Mean pLDDT",
                       protein_trajectory_folder = protein_trajectory_folder,
                       protein_ID = protein_ID, filename_suffix = filename_suffix)
    plot_traj_metadata(metadata_list = trajectory_df['ROG'],
                       metadata_type = "ROG",
                       protein_trajectory_folder = protein_trajectory_folder,
                       protein_ID = protein_ID, filename_suffix = filename_suffix)
    plot_traj_heatmap(metadata_list = pLDDTs, sorted_indices = sorted_indices,
                      metadata_type_y = "Residue",
                      metadata_type_color = "pLDDT",
                      domain_start = domain_start, domain_end = domain_start,
                      protein_trajectory_folder = protein_trajectory_folder,
                      protein_ID = protein_ID, filename_suffix = filename_suffix)
    
    trajectory_df_file = os.path.join(protein_trajectory_folder, f'{protein_ID}_{filename_suffix}_traj.tsv')
    trajectory_df.to_csv(trajectory_df_file, sep="\t", index=False)
    
    return trajectory_file


def generate_protein_enrichment(proteins_in_models,
                                b_factor_clusters,
                                protein_ID,
                                plddt_clusters_folder,
                                mode: Literal["no_query_repeats",
                                              "single_query_repeat",
                                              "multiple_query_repeats"] = "single_query_repeat",
                                logger = None, out_path = "."):
    
    default_mode = "single_query_repeat"
    if logger is None:
        logger = configure_logger(out_path)(__name__)
    
    # Create a dictionary to store protein counts for each cluster
    cluster_protein_counts = {}
    for cluster, model_proteins in zip(b_factor_clusters, proteins_in_models):
        if cluster not in cluster_protein_counts:
            cluster_protein_counts[cluster] = Counter()

        # Split the model string and count proteins, excluding the query protein
        times_query_is_present = len([p for p in model_proteins if p == protein_ID]) 
        is_query_repeated = times_query_is_present > 1
        proteins = [p for p in model_proteins if p != protein_ID]
        
        # Add the query if it is repeated?
        if is_query_repeated:
            if mode == "do_not_repeat_query":
                pass
            elif mode == "single_query_repeat":
                proteins.append(protein_ID)
            elif mode == "multiple_query_repeats":
                repeated_query = [protein_ID] * times_query_is_present
                proteins.extend(repeated_query)
            else:
                repeated_query = [protein_ID] * times_query_is_present
                proteins.extend(repeated_query)
                mode = default_mode
                
        cluster_protein_counts[cluster].update(proteins)
        
    # Create a folder with the protein ID enrichments
    protein_IDs_cloud = os.path.join(plddt_clusters_folder, "protein_IDs_clouds")
    os.makedirs(protein_IDs_cloud, exist_ok=True)
    
    # Generate word clouds for each cluster
    for cluster, protein_counts in cluster_protein_counts.items():
        if not protein_counts:  # Skip if no proteins in the cluster
            continue

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(protein_counts)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Protein Enrichment in Cluster {cluster+1}')
        plt.tight_layout(pad=0)
        
        # Save the word cloud
        wordcloud_filename = os.path.join(protein_IDs_cloud, f'{protein_ID}_cluster_{cluster+1}_protein_clouds.png')
        plt.savefig(wordcloud_filename)
        plt.close()

    # Create a new column with representative protein names
    representative_proteins = []
    for cluster, model in zip(b_factor_clusters, proteins_in_models):
        proteins = [p for p in model if p != protein_ID]
        if proteins:
            most_common = cluster_protein_counts[cluster].most_common(1)[0][0]
            representative_proteins.append(most_common)
        else:
            representative_proteins.append('')

    return representative_proteins

########### Interactive pLDDT clusters #########
def create_interactive_plddt_visualization(
    protein_ID: str,
    mean_curves: list,
    std_curves: list,
    b_factor_clusters: list,
    domains_df: pd.DataFrame = None,
    plddt_clust_df: pd.DataFrame = None,
    plddt_clusters_folder: str = None,
    y_label: str = "Mean cluster pLDDT",
    fontsize: int = 30,
    show_plot: bool = False,
    logger: Logger | None = None):
    """
    Creates an interactive visualization combining pLDDT clusters and protein enrichment wordclouds
    using Plotly for the cluster plot and HTML for the layout.
    
    Args:
        protein_ID (str): ID of the protein being analyzed
        mean_curves (list): List of mean curves for each cluster
        std_curves (list): List of standard deviation curves for each cluster
        b_factor_clusters (list): Cluster assignments for each model
        domains_df (pd.DataFrame, optional): DataFrame with domain information
        plddt_clust_df (pd.DataFrame, optional): ["Cluster","Type","Is_chain","Rank","Model","Representative_Partner"]
        plddt_clusters_folder (str, optional): Folder where wordcloud images are stored
        y_label (str): Label for the y-axis
        fontsize (int): Base font size for plot elements
        show_plot (bool): Whether to show the plot in browser
        logger (Logger, optional): Logger object for logging information
        
    Returns:
        str: Path to the saved HTML file
    """
    
    if logger is None:
        logger = configure_logger()(__name__)
    
    # Count the number of models in each cluster
    cluster_counts = Counter(b_factor_clusters)
    
    # Number of clusters
    num_clusters = len(set(b_factor_clusters))
    
    # Get image paths for each cluster
    wordcloud_images = []
    for cluster in range(num_clusters):
        wordcloud_path = os.path.join(
            plddt_clusters_folder, 
            "protein_IDs_clouds", 
            f"{protein_ID}_cluster_{cluster+1}_protein_clouds.png"
        )
        wordcloud_images.append(wordcloud_path)
    
    # Read and encode images
    encoded_images = []
    for img_path in wordcloud_images:
        try:
            with open(img_path, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode()
                encoded_images.append(encoded_image)
        except FileNotFoundError:
            logger.warn(f"Image file not found: {img_path}")
            encoded_images.append("")
    
    # Create figure for pLDDT plot
    fig = go.Figure()
    
    # Colors for clusters
    colors = px.colors.qualitative.D3 if hasattr(px.colors.qualitative, 'D3') else px.colors.qualitative.Plotly
    
    # Create a mapping of unique clusters to indices
    unique_clusters = sorted(set(b_factor_clusters))
    cluster_to_index = {cluster: i for i, cluster in enumerate(unique_clusters)}
    
    # Add traces for each cluster
    for cluster in unique_clusters:
        i = cluster_to_index[cluster]
        cluster_number = cluster + 1
        color = colors[i % len(colors)]
        x_values = list(range(1, len(mean_curves[i]) + 1))
        cluster_size = cluster_counts[cluster]

        # Extract the most common representative partner
        repr_partner_list = plddt_clust_df.query(f'Cluster == {cluster_number}')['Representative_Partner'].dropna().tolist()
        if repr_partner_list:  # Ensure the list is not empty
            repr_partner, repr_partner_frequency = Counter(repr_partner_list).most_common(1)[0]
            repr_partner_frequency = repr_partner_frequency/cluster_size*100
        else:
            repr_partner, repr_partner_frequency = None, 0  # Handle empty cases

        # Generate the hovertext
        customdata = list(std_curves[i])

        # Add the trace for each cluster
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=mean_curves[i],
                line=dict(color=color, width=4),
                name=f'Cluster {cluster_number}',
                hovertemplate=(
                    f"<b>Cluster {cluster_number}</b><br>" +
                    f"Size: {cluster_size} models<br>" +
                    "Residue: %{x}<br>" +
                    "Mean pLDDT: %{y:.2f}<br>" +
                    "Std Dev: ±%{customdata:.2f}<br>" +
                    f"Representative Partner:<br>" +
                    f"   - Partner: <b>{repr_partner}</b><br>" +
                    f"   - Freq.: {repr_partner_frequency:.1f}%<br>" +
                    "<extra></extra>"
                ),
                customdata=customdata,
                legendgroup=f'cluster_{cluster_number}',
                showlegend=True,
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_values + x_values[::-1],
                y=list(mean_curves[i] + std_curves[i]) + list(mean_curves[i] - std_curves[i])[::-1],
                fill='toself',
                fillcolor=color,
                opacity=0.3,
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                legendgroup=f'cluster_{cluster_number}',
                showlegend=False,
            )
        )
    
    if domains_df is not None:
        matching_domains = domains_df[domains_df['Protein_ID'] == protein_ID]
        fig.add_vline(x=1, line=dict(color="black", width=2, dash="dash"))
        for _, row in matching_domains.iterrows():
            end_position = row['End'] + 1
            fig.add_vline(x=end_position, line=dict(color="black", width=2, dash="dash"))
            mid_point = (row['Start'] + row['End'] + 1) / 2
            y_max = max([max(curve) for curve in mean_curves])
            fig.add_annotation(
                x=mid_point,
                y=y_max,
                text=f"Dom{row['Domain']}",
                showarrow=False,
                font=dict(size=fontsize/2),
                yshift=10,
            )
    
    fig.update_layout(
        title=dict(
            text=f"{protein_ID}",
            font=dict(size=fontsize * 1.4),
            x=0.5,
        ),
        legend=dict(
            font=dict(size=fontsize * 0.8),
            borderwidth=2,
            bordercolor="black",
            bgcolor="rgba(255, 255, 255, 0.7)",
            xanchor="auto",
            yanchor="auto",
        ),
        height=700,
        margin=dict(l=50, r=50, t=100, b=50),
    )
    
    fig.update_xaxes(
        title=dict(text="Residue", font=dict(size=fontsize * 1.2)),
        tickfont=dict(size=fontsize),
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
    )
    
    fig.update_yaxes(
        title=dict(text=y_label, font=dict(size=fontsize * 1.2)),
        tickfont=dict(size=fontsize),
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
    )

    # # Ensure mode bar is correctly positioned
    # config = {
    #     "displayModeBar": True,
    #     "modeBarButtonsToAdd": [],
    #     "displaylogo": True
    # }
    
    # Convert the Plotly figure to HTML
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')#, config=config)
    
    # Create HTML with split layout
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{protein_ID} - pLDDT Clusters and Protein Enrichment</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                display: flex;
                flex-wrap: wrap;
                width: 100%;
            }}
            .plot-container {{
                width: 66%;
                min-width: 600px;
            }}
            .wordcloud-container {{
                width: 33%;
                min-width: 300px;
                padding: 0 20px;
                box-sizing: border-box;
                text-align: center;
            }}
            .wordcloud-title {{
                font-size: {fontsize}px;
                margin-bottom: 15px;
                font-weight: bold;
            }}
            .wordcloud-image {{
                max-width: 100%;
                height: auto;
                margin: 20px 0;
                max-height: 500px;
            }}
            .button-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 8px;
                margin-bottom: 20px;
            }}
            .cluster-button {{
                width: 40px;
                height: 40px;
                font-size: 18px;
                font-weight: bold;
                background-color: #f0f0f0;
                border: 2px solid #999;
                border-radius: 4px;
                cursor: pointer;
                transition: background-color 0.3s;
            }}
            .cluster-button:hover {{
                background-color: #ddd;
            }}
            .cluster-button.active {{
                background-color: #4CAF50;
                color: white;
                border-color: #2E7D32;
            }}
            @media (max-width: 1100px) {{
                .plot-container, .wordcloud-container {{
                    width: 100%;
                    padding: 0;
                }}
            }}
        </style>
        <script>
            function showWordcloud(clusterNum) {{
                // Hide all wordclouds
                var wordclouds = document.getElementsByClassName('wordcloud-image');
                for (var i = 0; i < wordclouds.length; i++) {{
                    wordclouds[i].style.display = 'none';
                }}
                
                // Show the selected wordcloud
                var selectedWordcloud = document.getElementById('wordcloud-' + clusterNum);
                if (selectedWordcloud) {{
                    selectedWordcloud.style.display = 'block';
                }}
                
                // Update buttons
                var buttons = document.getElementsByClassName('cluster-button');
                for (var i = 0; i < buttons.length; i++) {{
                    buttons[i].classList.remove('active');
                }}
                
                var selectedButton = document.getElementById('button-' + clusterNum);
                if (selectedButton) {{
                    selectedButton.classList.add('active');
                }}
            }}
            
            // Initialize with the first cluster selected
            window.onload = function() {{
                showWordcloud(1);
            }};
        </script>
    </head>
    <body>
        <div class="container">
            <div class="plot-container">
                {plot_html}
            </div>
            <div class="wordcloud-container">
                <div class="wordcloud-title"><br><br><br><br><br>Protein Enrichment for Cluster:</div>
                <div class="button-container">
    """
    
    # Add buttons for each cluster
    for i in range(1, num_clusters + 1):
        html_content += f"""
                    <button id="button-{i}" class="cluster-button" onclick="showWordcloud({i})">{i}</button>
        """
    
    html_content += """
                </div>
    """
    
    # Add wordcloud images
    for i, encoded_image in enumerate(encoded_images, 1):
        if encoded_image:
            html_content += f"""
                <img id="wordcloud-{i}" class="wordcloud-image" src="data:image/png;base64,{encoded_image}" style="display: none;">
            """
    
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save the HTML file
    output_filename = os.path.join(plddt_clusters_folder, f"{protein_ID}-interactive_plddt_clusters.html")
    with open(output_filename, 'w') as f:
        f.write(html_content)
    
    logger.info(f"   - Interactive pLDDT visualization saved to {output_filename}")
    
    # Open in browser if requested
    if show_plot:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(output_filename))
    
    return output_filename

################################


# Helper functions for RMSD trajectories
def get_domain_mean_plddt(chain, domain_start, domain_end):
    """Calculate mean pLDDT for a specific domain in a chain."""
    domain_atoms = get_domain_atoms(chain, domain_start, domain_end)
    return np.mean([atom.bfactor for atom in domain_atoms])

def get_best_reference_domain(chains, domain_start, domain_end):
    """Find the domain with highest mean pLDDT to use as reference."""
    mean_plddts = [get_domain_mean_plddt(chain, domain_start, domain_end) 
                   for chain in chains]
    best_idx = np.argmax(mean_plddts)
    best_chain = chains[best_idx]
    return get_domain_atoms(best_chain, domain_start, domain_end)


def protein_RMSD_trajectory(protein_ID: str, protein_seq: str,
                            pairwise_2mers_df: pd.DataFrame,
                            sliced_PAE_and_pLDDTs: dict,
                            all_pdb_data: dict, 
                            domains_df: pd.DataFrame,
                            out_path: str = ".",
                            point_of_ref: Literal["lowest_plddt",
                                                  "highest_plddt"] = "highest_plddt",
                            plddt_clustering_method = 'silhouette',
                            repeated_protein_IDs_cloud_mode = "single_query_repeat",
                            generate_interactive_viz: bool = True,
                            logger: logging.Logger | None = None, log_level: str = "info"):
    """
    Calculate the RMSD trajectory, RMSF, and B-factor clustering for a protein across different models.

    Args:
        protein_ID (str): The ID of the protein to analyze.
        protein_seq (str): The sequence of the protein.
        pairwise_2mers_df (pd.DataFrame): DataFrame containing pairwise 2-mer model information.
        sliced_PAE_and_pLDDTs (dict): Dictionary containing PAE and pLDDT information.
        all_pdb_data (dict): A dictionary containing PDB data structures.
        out_path (str): output path of MM project
        point_of_ref (Literal["lowest_plddt", "highest_plddt"]): The reference point for RMSD calculation.
        plddt_clustering_method (str): Method for clustering pLDDT curves.
        repeated_protein_IDs_cloud_mode (str): Mode for handling repeated proteins in enrichment.
        generate_interactive_viz (bool): Whether to generate interactive visualization.
        logger (logging.Logger, optional): Logger for logging information.
        log_level (str): Logging level.

    Returns:
        dict: A dictionary containing RMSD values, RMSF values, B-factor clusters, and related information.
    """

    # Initialize the logger
    if logger is None:
        logger = configure_logger(out_path, log_level = log_level)(__name__)
        
    # Progress
    logger.info(f"Starting coordinate analysis of individual {protein_ID} models:")
    
    # Get the reference chain model
    if point_of_ref == "highest_plddt":
        ref_model: PDB.Chain.Chain = sliced_PAE_and_pLDDTs[protein_ID]['PDB_xyz']        
    
    # Progress
    logger.info(f"   - Isolating {protein_ID} models...")
    
    # Get the list of matching chain models with the protein from 2-mers
    monomer_chains_from_2mers: dict = get_monomers_models_from_pairwise_2mers(
        protein_ID=protein_ID, protein_seq=protein_seq,
        pairwise_2mers_df=pairwise_2mers_df)
    
    # Get the list of matching chain models with the protein from N-mers
    monomer_chains_from_Nmers = get_monomers_models_from_all_pdb_data(
        protein_ID=protein_ID, protein_seq=protein_seq,
        all_pdb_data=all_pdb_data)
    
    # Combine the chain models from both lists
    all_chains = monomer_chains_from_2mers['monomer_chains'] + monomer_chains_from_Nmers['monomer_chains']
    all_chain_types = ['2-mer'] * len(monomer_chains_from_2mers['monomer_chains']) + ['N-mer'] * len(monomer_chains_from_Nmers['monomer_chains'])
    all_chain_info = list(zip(monomer_chains_from_2mers['is_chain'] + monomer_chains_from_Nmers['is_chain'],
                              monomer_chains_from_2mers['is_model'] + monomer_chains_from_Nmers['is_model'],
                              monomer_chains_from_2mers['is_rank'] + monomer_chains_from_Nmers['is_rank']))
    
    # Initialize lists to store values
    rmsd_values = []
    all_coords = []
    b_factors = []
    rmsf_values = []
    rmsf_values_per_domain = []
    monomer_rot_matrixes = []
    monomer_tran_vectors = []

    # Initialize superimposer
    super_imposer = PDB.Superimposer()
    
    # Get alpha carbon atoms and coordinates for reference
    ref_atoms = [atom for atom in ref_model.get_atoms() if atom.name == 'CA']
    ref_L = len(ref_atoms)
    
    # Superimpose chains to the reference
    for chain in all_chains:
        
        # Get alpha carbon atoms for current chain
        chain_atoms = [atom for atom in chain.get_atoms() if atom.name == 'CA']
    
        # Ensure both have the same number of atoms
        chain_L = len(chain_atoms)
        if chain_L != ref_L:
            logger.error(f"   - Found chain with different length than the reference during trajectories construction of {protein_ID}")
            logger.error( "   - Trimming the longer chain to avoid program crashing during RMSD calculations.")
            logger.error( "   - Results may be unreliable under these circumstances.")
        min_length = min(ref_L, chain_L)
        ref_atoms = ref_atoms[:min_length]
        chain_atoms = chain_atoms[:min_length]
    
        # Calculate standard RMSD
        super_imposer.set_atoms(ref_atoms, chain_atoms)
        rmsd_values.append(super_imposer.rms)
                
        # Get coordinates and pLDDT values
        chain_coords = np.array([atom.coord for atom in chain_atoms])
        plddt_values = np.array([atom.bfactor for atom in chain_atoms])

        # Apply rotation and translation to the coordinates
        rotation_matrix, translation_vector = super_imposer.rotran
        transformed_coords = np.dot(chain_coords, rotation_matrix.T) + translation_vector

        # Save rotation matrix and translation vector
        monomer_rot_matrixes.append(rotation_matrix)
        monomer_tran_vectors.append(translation_vector)

        # Store coordinates and pLDDT values
        all_coords.append(transformed_coords)
        b_factors.append(plddt_values)

    # Convert coordinates to np.array
    all_coords = np.array(all_coords)
    
    # Create dir and subdir to store pLDDT clusters
    plddt_clusters_folder = os.path.join(out_path, "plddt_clusters")
    os.makedirs(plddt_clusters_folder, exist_ok=True)
    plddt_clusters_folder = os.path.join(plddt_clusters_folder, protein_ID)
    os.makedirs(plddt_clusters_folder, exist_ok=True)

    # -------------------------------------------------

    # Progress
    logger.info(f"   - Computing pLDDT clusters for {protein_ID}...")
    
    # Generate automatic pLDDT (b-factors) clusters using the silhouette method
    plddt_clusters_filename = os.path.join(plddt_clusters_folder, protein_ID + "-pLDDT_clusters.png")
    b_factor_clusters, mean_curves, std_curves = cluster_curves(
        b_factors,
        domains_df = domains_df,
        method = plddt_clustering_method,
        y_label = "Mean cluster pLDDT",
        x_label = "Position",
        protein_ID = protein_ID,
        filename = plddt_clusters_filename,
        show_plot = False,
        logger = logger
    )
    
    # Save pLDDT clusters metadata
    plddt_clust_df_cols = ['Cluster', 'Type', 'Is_chain', 'Rank', 'Model', 'Representative_Partner']
    plddt_clust_df = pd.DataFrame(columns = plddt_clust_df_cols)
    plddt_clusters_metadata_filename = os.path.join(
        plddt_clusters_folder, f'{protein_ID}-pLDDT_clusters_metadata.tsv')
    proteins_in_models = [model[1] for model in all_chain_info]
    representative_partners = generate_protein_enrichment(
        proteins_in_models = proteins_in_models,
        b_factor_clusters = b_factor_clusters,
        protein_ID = protein_ID,
        plddt_clusters_folder = plddt_clusters_folder,
        out_path = out_path,
        mode = repeated_protein_IDs_cloud_mode
    )
    
    for i, cluster in enumerate(b_factor_clusters):
        chain_type = all_chain_types[i]
        chain_info = all_chain_info[i]
        plddt_model_data =  pd.DataFrame({
            "Cluster"                : [cluster + 1], 
            "Type"                   : [chain_type],
            "Is_chain"               : [chain_info[0]],
            "Rank"                   : [chain_info[2]],
            "Model"                  : ['__vs__'.join(map(str, chain_info[1]))],
            "Representative_Partner" : [representative_partners[i]]
        })
        plddt_clust_df = pd.concat([plddt_clust_df, plddt_model_data], ignore_index=True)
    plddt_clust_df.sort_values('Cluster').to_csv(plddt_clusters_metadata_filename, sep = "\t", index = False)
    
    # Create interactive visualization if requested
    if generate_interactive_viz:
        logger.info(f"   - Creating interactive pLDDT visualization for {protein_ID}...")
        
        interactive_viz_path = create_interactive_plddt_visualization(
            protein_ID=protein_ID,
            mean_curves=mean_curves,
            std_curves=std_curves,
            b_factor_clusters=b_factor_clusters,
            domains_df=domains_df,
            plddt_clust_df=plddt_clust_df,
            plddt_clusters_folder=plddt_clusters_folder,
            y_label="Mean cluster pLDDT",
            show_plot=False,
            logger=logger
        )
        
        logger.info(f"   - Interactive visualization saved to: {interactive_viz_path}")

    # Create PDB trajectory dirs (one for each protein)
    trajectory_folder = os.path.join(out_path, 'monomer_trajectories')
    protein_trajectory_folder = os.path.join(trajectory_folder, protein_ID)
    os.makedirs(protein_trajectory_folder, exist_ok=True)

    # -------------------------------------------------------------------------
    # -------------------- Compute per domain trajectories --------------------
    # -------------------------------------------------------------------------

    # Check number of domains
    protein_domains = domains_df[domains_df['Protein_ID'] == protein_ID]
    perform_domain_analysis = len(protein_domains) > 1
    
    if perform_domain_analysis:
        logger.info(f"   - Multiple domains detected for {protein_ID}, performing domain analysis...")

        # Initialize superimposer
        super_imposer = PDB.Superimposer()
        
        # Process each domain
        for _, domain in protein_domains.iterrows():
            
            # Progress
            logger.info(f"   - Generating RMSD trajectories for domain {domain['Domain']}...")
            
            domain_start, domain_end = domain['Start'], domain['End']
            domain_name = f"{protein_ID}_domain_{domain['Domain']}"

            # Get the best reference domain based on pLDDT
            domain_ref_atoms = get_best_reference_domain(all_chains, domain_start, domain_end)

            # Initialize lists
            domain_rmsd_values  = []
            domain_all_coords   = []
            domain_b_factors    = []
            domain_rot_matrixes = []
            domain_tran_vectors = []

            for chain in all_chains:
                domain_chain_atoms = get_domain_atoms(chain, domain_start, domain_end)

                # Calculate standard RMSD for the domain
                super_imposer.set_atoms(domain_ref_atoms, domain_chain_atoms)
                domain_rmsd_values.append(super_imposer.rms)

                # Get coordinates and pLDDT values
                domain_chain_coords = np.array([atom.coord for atom in domain_chain_atoms])
                domain_plddt_values = np.array([atom.bfactor for atom in domain_chain_atoms])

                # Apply rotation and translation to the coordinates
                rotation_matrix, translation_vector = super_imposer.rotran
                transformed_coords = np.dot(domain_chain_coords, rotation_matrix.T) + translation_vector

                # Save rotation matrix and translation vector
                domain_rot_matrixes.append(rotation_matrix)
                domain_tran_vectors.append(translation_vector)

                # Store coordinates and pLDDT values
                domain_all_coords.append(transformed_coords)
                domain_b_factors.append(domain_plddt_values)

            # Save domain trajectories
            domain_trajectory_folder = os.path.join(protein_trajectory_folder, domain_name)
            os.makedirs(domain_trajectory_folder, exist_ok=True)
            
            # Sort by RMSD values and keep track of the indices
            domain_RMSD_traj_indices = np.argsort(domain_rmsd_values)

            # Compute some extra data (ROG and mean pLDDT per model)
            domain_mean_pLDDTs = [np.mean(model_plddt_values) for model_plddt_values in domain_b_factors]
            domain_ROGs = compute_rog_for_all_chains(domain_all_coords)

            # Calculate RMSF for domain and save it -----------------
            dom_rmsf_values = calculate_rmsf(domain_all_coords)
            dom_rmsf_plot_filename = os.path.join(domain_trajectory_folder, domain_name + "_RMSF.png")
            plot_rmsf(dom_rmsf_values,
                      domains_df = None,
                      protein_ID = f'{protein_ID} Domain {str(domain["Domain"])}',
                      filename = dom_rmsf_plot_filename,
                      domain_start = domain["Start"])
            # Save RMSF to CSV file
            dom_RMSF_filename = os.path.join(domain_trajectory_folder, f'{protein_ID}_RMSF.csv')
            with open(dom_RMSF_filename, 'w') as f:
                f.write(f'{protein_ID}: {str(list(dom_rmsf_values))}\n')
            # Append data
            rmsf_values_per_domain = np.concatenate((rmsf_values_per_domain, dom_rmsf_values), axis = 0)
            # ------------------------------------------------------------
    

            # RMSD traj
            save_trajectory(sorted_indices = domain_RMSD_traj_indices,
                            protein_ID=domain_name,
                            filename_suffix=f'RMSD',
                            protein_trajectory_folder=domain_trajectory_folder,

                            RMSDs = domain_rmsd_values,
                            pLDDTs = domain_b_factors,
                            mean_pLDDTs = domain_mean_pLDDTs,
                            ROGs = domain_ROGs,

                            rot_matrixes = domain_rot_matrixes,
                            tran_vectors = domain_tran_vectors,

                            all_chains=all_chains,
                            all_chain_types=all_chain_types,
                            all_chain_info=all_chain_info,
                            domain_start=domain_start,
                            domain_end=domain_end)
                        
            # Run distribution analysis if domain trajectory folder exists
            if os.path.exists(domain_trajectory_folder):
                
                # Find all RMSD trajectory TSV files (ends with RMSD_traj.tsv)
                rmsd_traj_files = [f for f in os.listdir(domain_trajectory_folder) 
                                 if f.endswith('RMSD_traj.tsv')]

                # Parse each RMSD file                              
                for rmsd_traj_file in rmsd_traj_files:

                    # Get the prefix (everything before _RMSD_traj.tsv)
                    tsv_prefix = rmsd_traj_file.replace('_RMSD_traj.tsv', '')
                    
                    # Full path to the TSV file
                    tsv_path = os.path.join(domain_trajectory_folder, rmsd_traj_file)
                    
                    # Read the trajectory file
                    domain_rmsd_traj_df = pd.read_csv(tsv_path, sep='\t')
                    
                    # Run the analysis functions directly
                    windows = [5, 10, 15, 20]
                    domain_rmsd_traj_results, rolling_data = analyze_protein_distribution(
                        domain_rmsd_traj_df,
                        target_protein = protein_ID,
                        windows = windows)
                    
                    # Save results to CSV using the same prefix
                    output_csv = os.path.join(domain_trajectory_folder, 
                                            f'{tsv_prefix}_BPD_results.csv')
                    domain_distribution_results_df = save_results_to_csv(domain_rmsd_traj_results, output_csv)
                    
                    # Create a subdirectory for plots using the same prefix
                    plots_dir = os.path.join(domain_trajectory_folder, f'{tsv_prefix}_BPD_plots')
                    os.makedirs(plots_dir, exist_ok=True)
                    
                    # Generate plots
                    plot_distributions(rolling_data, plots_dir, soft=True, noise_scale=0.01, target_protein=f'{protein_ID}-Dom{domain["Domain"]}')

                    # Run html
                    html_interactive_metadata(
                        protein_ID=protein_ID, 
                        protein_trajectory_folder=domain_trajectory_folder,
                        traj_df=domain_rmsd_traj_df,
                        domain_start=domain['Start'],

                        sorted_indexes=domain_RMSD_traj_indices,
                        rmsd_values=domain_rmsd_values,
                        rmsf_values=dom_rmsf_values,
                        mean_plddts=domain_mean_pLDDTs, 
                        per_res_plddts = domain_b_factors,
                        rog_values = domain_ROGs,
                        bpd_values = rolling_data,

                        domain_number=str(domain["Domain"])
                    )

    else:
        logger.info(f"   - Single domain detected for {protein_ID}, skipping domain analysis...")
    
    # -------------------------------------------------------------------------
    # ------------------- Compute whole protein trajectory --------------------
    # -------------------------------------------------------------------------
    
    # Output path
    monomer_trajectory_folder = os.path.join(protein_trajectory_folder, f"{protein_ID}_monomer")
    os.makedirs(monomer_trajectory_folder, exist_ok=True)
 
    # Sort models by RMSD (RMSD trajectories)
    monomer_RMSD_traj_indices = np.argsort(rmsd_values)

    # Compute some extra data (ROG and mean pLDDT per model)
    monomer_mean_pLDDTs = [np.mean(model_per_res_plddt) for model_per_res_plddt in b_factors]
    monomer_ROGs = compute_rog_for_all_chains(all_coords)

    # Calculate RMSF for full length and save it -----------------
    rmsf_values = calculate_rmsf(all_coords)
    rmsf_plot_filename = os.path.join(monomer_trajectory_folder, protein_ID + "_monomer_RMSF.png")
    plot_rmsf(rmsf_values, domains_df = domains_df,
              protein_ID = protein_ID,
              filename = rmsf_plot_filename)
    # Save RMSF to CSV file
    RMSF_filename = os.path.join(monomer_trajectory_folder, f'{protein_ID}_monomer_RMSF.csv')
    with open(RMSF_filename, 'w') as f:
        f.write(f'{protein_ID}: {str(list(rmsf_values))}\n')
    # ------------------------------------------------------------
    
    # Save RMSD traj and metadata
    logger.info(f"   - Generating RMSD trajectory files for whole {protein_ID}...")
    rmsd_traj_file = save_trajectory(sorted_indices = monomer_RMSD_traj_indices,
                                     protein_ID = protein_ID,
                                     filename_suffix = "monomer_RMSD",
                                     protein_trajectory_folder = monomer_trajectory_folder,

                                     RMSDs = rmsd_values,
                                     pLDDTs = b_factors,
                                     mean_pLDDTs = monomer_mean_pLDDTs,
                                     ROGs = monomer_ROGs,

                                     rot_matrixes = monomer_rot_matrixes,
                                     tran_vectors = monomer_tran_vectors,

                                     all_chains = all_chains,
                                     all_chain_types = all_chain_types,
                                     all_chain_info = all_chain_info)

    # Run distribution analysis if monomer trajectory folder exists
    if os.path.exists(monomer_trajectory_folder):
        
        # Find all RMSD trajectory TSV files (ends with RMSD_traj.tsv)
        rmsd_traj_files = [f for f in os.listdir(monomer_trajectory_folder) 
                            if f.endswith('RMSD_traj.tsv')]
        # Parse each RMSD file                                
        for rmsd_traj_file in rmsd_traj_files:

            # Get the prefix (everything before _RMSD_traj.tsv)
            tsv_prefix = rmsd_traj_file.replace('_RMSD_traj.tsv', '')
            
            # Full path to the TSV file
            tsv_path = os.path.join(monomer_trajectory_folder, rmsd_traj_file)
            
            # Read the trajectory file
            monomer_rmsd_traj_df = pd.read_csv(tsv_path, sep='\t')
            
            # Run the analysis functions directly
            windows = [5, 10, 15, 20]
            monomer_distribution_results, rolling_data = analyze_protein_distribution(
                monomer_rmsd_traj_df,
                target_protein = protein_ID,
                windows = windows)
            
            # Save results to CSV using the same prefix
            output_csv = os.path.join(monomer_trajectory_folder, 
                                    f'{tsv_prefix}_BPD_results.csv')
            monomer_distribution_results_df = save_results_to_csv(monomer_distribution_results, output_csv)
            
            # Create a subdirectory for plots using the same prefix
            plots_dir = os.path.join(monomer_trajectory_folder, f'{tsv_prefix}_BPD_plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Generate plots
            plot_distributions(rolling_data, plots_dir, soft=True, noise_scale=0.01, target_protein=protein_ID)

            html_interactive_metadata(
                        protein_ID=protein_ID, 
                        protein_trajectory_folder=monomer_trajectory_folder,
                        sorted_indexes=monomer_RMSD_traj_indices,
                        traj_df=monomer_rmsd_traj_df,

                        domain_start=1,
                        rmsd_values=rmsd_values,
                        rmsf_values=rmsf_values,
                        mean_plddts=monomer_mean_pLDDTs, 
                        per_res_plddts = b_factors,
                        rog_values = monomer_ROGs,
                        bpd_values = rolling_data
                    )
    
    # Fill per domain rmsf_values with whole protein
    if not perform_domain_analysis:
        rmsf_values_per_domain = rmsf_values
        
    # Prepare the results
    results = {
        'rmsd_values': rmsd_values,
        'rmsf_values': rmsf_values.tolist(),
        'rmsf_values_per_domain': rmsf_values_per_domain.tolist(),
        'b_factors': b_factors,
        'b_factor_clusters': b_factor_clusters.tolist(),
        'chain_types': all_chain_types,
        'model_info': all_chain_info,
        'rmsd_trajectory_file': rmsd_traj_file
    }
    
    return results


def generate_RMSF_pLDDT_cluster_and_RMSD_trajectories(
        mm_output, out_path,
        plddt_clustering_method: Literal["interactive", 'silhouette', None] = 'silhouette',
        repeated_protein_IDs_cloud_mode: Literal["no_query_repeats",
                                                 "single_query_repeat",
                                                 "multiple_query_repeats"]  = "single_query_repeat",
        log_level: str = 'info',
        logger: logging.Logger | None = None):
    
    
    mm_trajectories = {}
    
    # Debug
    for protein_index, _ in enumerate(mm_output['prot_IDs']):
        protein_ID = mm_output['prot_IDs'][protein_index]
        protein_seq = mm_output['prot_seqs'][protein_index]
        
        # Debug
        protein_traj = protein_RMSD_trajectory(protein_ID = protein_ID, protein_seq = protein_seq,
                                pairwise_2mers_df = mm_output['pairwise_2mers_df'],
                                sliced_PAE_and_pLDDTs = mm_output['sliced_PAE_and_pLDDTs'],
                                all_pdb_data = mm_output['all_pdb_data'],
                                domains_df = mm_output['domains_df'],
                                out_path = out_path,
                                point_of_ref = "highest_plddt",
                                plddt_clustering_method = plddt_clustering_method,
                                repeated_protein_IDs_cloud_mode = repeated_protein_IDs_cloud_mode,
                                logger = logger)
        
        mm_trajectories[protein_ID] = protein_traj
    
    return mm_trajectories
