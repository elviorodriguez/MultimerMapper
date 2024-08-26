
import os
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from scipy import stats
import matplotlib.pyplot as plt
from logging import Logger
from sklearn.decomposition import PCA
from scipy import optimize

from devs.mm_getters import get_protein_homooligomeric_models
from utils.logger_setup import configure_logger


def fit_circle(x, y):
    def calc_R(xc, yc):
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = np.mean(x), np.mean(y)
    center, _ = optimize.leastsq(f_2, center_estimate)
    
    xc, yc = center
    Ri = calc_R(xc, yc)
    R = Ri.mean()
    
    return xc, yc, R

def compute_fb_statistics(models_df, rank_filter=None):
    # Initialize a list to store the results
    results = []

    # Create a PDBParser instance
    parser = PDBParser(QUIET=True)

    # Filter models by rank if specified
    if rank_filter is not None:
        models_df = models_df[models_df['rank'] == rank_filter]

    for _, row in models_df.iterrows():
        pdb_model = row['pdb_model']
        
        # Parse the model if it's a string (assumed to be a file path)
        if isinstance(pdb_model, str):
            structure = parser.get_structure('structure', pdb_model)
            model = structure[0]  # Assuming there's only one model
        else:
            model = pdb_model  # Already a model object
        
        # Calculate the mean center of mass for all chains
        all_chain_coords = [atom.coord for chain in model.get_chains() for atom in chain.get_atoms()]
        all_chain_coords = np.array(all_chain_coords)
        mean_center_of_mass = np.mean(all_chain_coords, axis=0)
        
        # Calculate the mean center of mass for each chain
        chain_centers = {}
        for chain in model.get_chains():
            chain_coords = [atom.coord for atom in chain.get_atoms()]
            chain_coords = np.array(chain_coords)
            chain_center = np.mean(chain_coords, axis=0)
            chain_centers[chain.id] = chain_center
        
        # Calculate the distances from each chain center of mass to the general center of mass
        distances = []
        for chain_id, chain_center in chain_centers.items():
            distance = np.linalg.norm(mean_center_of_mass - chain_center)
            distances.append(distance)
        
        # Calculate statistics of the distances
        distances = np.array(distances)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        # Perform PCA on the chain centers
        chain_centers_array = np.array(list(chain_centers.values()))
        n_components = min(3, len(chain_centers_array))
        
        if n_components < 2:
            # If there's only one chain center, we can't do PCA
            projected_distances = distances
            mean_projected_distance = mean_distance
            std_projected_distance = std_distance
        
        else:
            pca = PCA(n_components=n_components)
            pca.fit(chain_centers_array)

            # Project chain centers onto the plane of the two principal components
            # If n_components is 2, this will project onto a line
            projected_centers = pca.transform(chain_centers_array)[:, :2]

            # Calculate the distances from each projected chain center to the origin (0, 0)
            projected_distances = np.linalg.norm(projected_centers, axis=1)
            
            # Fit a circle to the projected points
            xc, yc, R = fit_circle(projected_centers[:, 0], projected_centers[:, 1])

            # Calculate statistics of the projected distances
            mean_projected_distance = np.mean(projected_distances)
            std_projected_distance = np.std(projected_distances)
            fitted_radius = R
            
        
        # Add results to the list
        results.append({
            'N': row['N'],
            'rank': row['rank'],
            'mean_center_of_mass': mean_center_of_mass.tolist(),
            'chain_centers': chain_centers,
            'distances': distances.tolist(),
            'mean_distance': mean_distance,
            'std_distance': std_distance,
            'projected_distances': projected_distances.tolist(),
            'mean_projected_distance': mean_projected_distance,
            'std_projected_distance': std_projected_distance,
            'fitted_radius': R
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df


def plot_fb_statistics(stats_df, prot_ID = "", display_distribution = False,
                       figsize = (5, 5), dpi = 300, show_plot = False,
                       method = ['distance', 'projected_distance', 'fitted_radius'][1]):
    
    if method == 'distance':
        estimator    = 'mean_distance'
        measurements = 'distances'
        deviation    = 'std_distance'
        title        = f'{prot_ID} estimated radius'
        ylabel       = 'Chain CM distance to model CM (Å)'
    elif method == 'projected_distance':
        estimator    = 'mean_projected_distance'
        measurements = 'projected_distances'
        deviation    = 'std_projected_distance'
        title        = f'{prot_ID} estimated radius (projection)'
        ylabel       = 'Projected chain CM distance to model CM (Å)'
    elif method == 'fitted_radius':
        estimator    = 'fitted_radius'
        measurements = 'projected_distances'
        deviation    = 'std_projected_distance'
        title        = f'{prot_ID} fitted radius (projection)'
        ylabel       = 'Fitted Circle Radius (Å)'

    # Plot Mean and Standard Deviation
    fig = plt.figure(figsize=figsize, dpi = dpi)

    # Colors for the fallback ranges (a dim rainbow-like sequence)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(stats_df)))

    # Fill fallback ranges with transparent colors
    for i in range(len(stats_df)):
        N = stats_df.loc[i, 'N']
        lower = stats_df.loc[i, 'lower_fallback']
        upper = stats_df.loc[i, 'upper_fallback']
        
        plt.fill_between([-4, 30], lower, upper,
                         color=colors[i], alpha=0.2,
                         edgecolor='none')  # Adjust alpha for transparency

    plt.plot(stats_df['N'], stats_df[estimator], 
             color='black', linestyle='-', alpha=0.5)
    
    if display_distribution:
        dist_list = []
        ns_list   = []
        for n, dists in zip(stats_df['N'], stats_df[measurements]):
            for d in dists:
                dist_list.append(d)
                ns_list.append(n + np.random.uniform(-0.05, 0.05))
        plt.scatter(ns_list, dist_list, 
                    color="#228833",
                    marker = 'x')
    plt.errorbar(stats_df['N'], stats_df[estimator],
                 yerr = stats_df[deviation], 
                 fmt='o', capsize=5,
                 label='Mean Distance ± SD',
                 color="#4477AA", ecolor='black', elinewidth=2, alpha=0.7)
    plt.xlabel('N')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc = 'best')
    plt.grid(True)

    # Set x and y limits to fit the data points, excluding fill_between areas
    plt.xlim(stats_df['N'].min() - 0.5, stats_df['N'].max() + 0.5)
    plt.ylim(stats_df[estimator].min() - stats_df[deviation].max(),
                stats_df[estimator].max() + stats_df[deviation].max())


    if show_plot:
        plt.show()

    return fig


def add_fallback_ranges(stats_df, low_fraction = 0.25, up_fraction = 0.75):
    # Initialize lists to store lower and upper boundaries
    lower_boundaries = []
    upper_boundaries = []

    for i in range(len(stats_df)):
        mean_distance = stats_df.loc[i, 'mean_distance']
        
        # Calculate lower boundary
        if i > 0:
            prev_mean_distance = stats_df.loc[i - 1, 'mean_distance']
            lower_diff = mean_distance - prev_mean_distance
            lower_boundary = mean_distance - low_fraction * lower_diff
            if lower_boundary < max(upper_boundaries):
                lower_boundary = max(upper_boundaries)
        else:
            lower_boundary = 0  # For the first row, use the mean_distance itself

        # Calculate upper boundary
        if i < len(stats_df) - 1:
            next_mean_distance = stats_df.loc[i + 1, 'mean_distance']
            upper_diff = next_mean_distance - mean_distance
            upper_boundary = mean_distance + up_fraction * upper_diff
            try:
                if upper_boundary < max(upper_boundaries):
                    upper_boundary = max(upper_boundaries)
            except ValueError:
                pass
        else:
            upper_boundary = mean_distance  # For the last row, use the mean_distance itself

        lower_boundaries.append(lower_boundary)
        upper_boundaries.append(upper_boundary)
    
    upper_boundaries[-1] = 1000

    # Add the computed ranges to the DataFrame
    stats_df['lower_fallback'] = lower_boundaries
    stats_df['upper_fallback'] = upper_boundaries

    return stats_df

def detect_fallback(stats_df, drop_threshold=0.2):
    """
    Detect the point of fallback based on a significant drop in mean_distance.
    
    :param stats_df: DataFrame containing the statistics
    :param drop_threshold: The threshold for considering a drop significant (default 0.2 or 20%)
    :return: The N value where fallback occurs, or None if no fallback is detected
    """
    mean_distances = stats_df['mean_distance'].values
    N_values = stats_df['N'].values
    
    for i in range(1, len(mean_distances)):
        if mean_distances[i] < mean_distances[i-1] * (1 - drop_threshold):
            return N_values[i]
    
    return None


def identify_fallback_target(stats_df, fallback_N, confidence_level=0.95):
    """
    Identify which previous model the fallback corresponds to.
    
    :param stats_df: DataFrame containing the statistics
    :param fallback_N: The N value where fallback occurs
    :param confidence_level: Confidence level for interval calculation (default 0.95 for 95% CI)
    :return: The N value of the model it falls back to, or the closest if no exact match
    """
    fallback_index = stats_df.index[stats_df['N'] == fallback_N].tolist()[0]
    fallback_row = stats_df.iloc[fallback_index]
    
    metrics = ['mean_distance', 'mean_projected_distance']
    
    def calculate_ci(row, metric):
        
        if metric == "mean_distance":
            std_column = 'std_distance'
        elif metric == 'mean_projected_distance':
            std_column = 'std_projected_distance'
        
        mean = row[metric]
        std = row[std_column]
        n = row['N']  # Using N as sample size, adjust if needed
        ci = stats.t.interval(confidence_level, df=n-1, loc=mean, scale=std/np.sqrt(n))
        return ci
    
    fallback_cis = {metric: calculate_ci(fallback_row, metric) for metric in metrics}
    
    best_match = None
    smallest_diff = float('inf')
    
    for i in range(fallback_index):
        row = stats_df.iloc[i]
        matches = []
        
        for metric in metrics:
            if fallback_cis[metric][0] <= row[metric] <= fallback_cis[metric][1]:
                matches.append(True)
            else:
                matches.append(False)
        
        if any(matches):
            return row['N']
        
        # If no exact match, track the closest
        diff = min(abs(row[metric] - fallback_cis[metric][0]) for metric in metrics)
        if diff < smallest_diff:
            smallest_diff = diff
            best_match = row['N']
    
    return best_match


def interpret_fallback(stats_df, drop_threshold=0.2, confidence_level=0.95):
    """
    Analyze fallback in the data.
    
    :param stats_df: DataFrame containing the statistics
    :param drop_threshold: Threshold for fallback detection
    :param confidence_level: Confidence level for interval calculation
    :return: A dictionary with fallback analysis results
    """
    fallback_N = detect_fallback(stats_df, drop_threshold)
    
    if fallback_N is None:
        return {"fallback_detected": False}
    
    fallback_target = identify_fallback_target(stats_df, fallback_N, confidence_level)
    
    return {
        "fallback_detected": True,
        "fallback_N": fallback_N,
        "fallback_target": fallback_target
    }

def analyze_fallback(mm_output, save_figs = True, figsize = (5, 5), dpi = 200, logger: Logger | None = None):
    
    if logger is None:
        logger = configure_logger(mm_output['out_path'])(__name__)

    logger.info("INITIALIZING: Analyze homooligomeric symmetry fallbacks...")

    # Unpack data
    prot_IDs_list = mm_output['prot_IDs']

    if save_figs:
        # Create a directory for saving plots
        output_dir = os.path.join(mm_output['out_path'], 'fallback_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
    # Query ID
    for prot_ID in prot_IDs_list:
            
        homooligomeric_models_df = get_protein_homooligomeric_models(mm_output, prot_ID)
        
        # Only work with Homo-N-mers > 2
        if not max(homooligomeric_models_df['N']) > 2:
            continue

        logger.info("")
        logger.info(f"Analyzing fallback on homooligomer: {prot_ID}")
            
        stats_df = compute_fb_statistics(homooligomeric_models_df, rank_filter = 1)
        stats_df = add_fallback_ranges(stats_df)

        # Analyze fallback
        fallback_result = interpret_fallback(stats_df)
        
        if fallback_result["fallback_detected"]:
            logger.info(f"   Fallback detected for {prot_ID}:")
            logger.info(f"   Fallback occurs at N = {fallback_result['fallback_N']}")
            logger.info(f"   Falls back to model with N = {fallback_result['fallback_target']}")
            logger.info(f"   Users should examine the model with N = {fallback_result['fallback_N']}")
        else:
            logger.info(f"   No fallback detected for {prot_ID}")

        fig = plot_fb_statistics(stats_df,
                                    prot_ID,
                                    display_distribution = False,
                                    figsize = figsize,
                                    dpi = dpi)
        
        if save_figs:
            out_file = os.path.join(output_dir, f'{prot_ID}_fallback.png')
            fig.savefig(out_file, dpi=dpi)
            logger.info(f"   Plot saved to {out_file}")

    logger.info("")
    logger.info(f"FINISHED: Analyze homooligomeric symmetry fallbacks")














