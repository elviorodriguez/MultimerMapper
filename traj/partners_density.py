import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from scipy.interpolate import make_interp_spline  # For smooth curves

def analyze_protein_distribution(df, target_protein, windows=[5, 10, 15, 20]):
    """
    Analyze protein distribution with special handling for the target protein.
    
    Args:
        df: DataFrame with the trajectory data
        target_protein: str, the specific protein being studied
        windows: list of window sizes for analysis
    """
    # Extract unique proteins from the Model column
    all_proteins = set()
    for model in df['Model'].str.split('__vs__'):
        all_proteins.update(model)
    
    # Create a dictionary to store protein occurrences by trajectory
    protein_occurrences = defaultdict(list)
    
    # For each protein, create a binary list of its presence in each trajectory
    for protein in all_proteins:
        for idx in df.index:
            model_proteins = df.loc[idx, 'Model'].split('__vs__')
            
            if protein == target_protein:
                # For the target protein, count only if it appears twice (potential self-interaction)
                count = model_proteins.count(protein)
                protein_occurrences[protein].append(1 if count > 1 else 0)
            else:
                # For other proteins, count normal presence
                protein_occurrences[protein].append(1 if protein in model_proteins else 0)
    
    
    # Convert to DataFrame for easier analysis
    occurrence_df = pd.DataFrame(protein_occurrences)
    
    # Analysis for different window sizes
    results = {}
    rolling_data = {}  # Store rolling averages for plotting
    
    for window in windows:
        window_results = {}
        window_rolling = {}
        
        for protein in all_proteins:
            # Calculate rolling average of occurrence
            rolling_avg = pd.Series(occurrence_df[protein]).rolling(window=window).mean()
            window_results[protein] = {
                'max_density': rolling_avg.max(),
                'min_density': rolling_avg.min(),
                'mean_density': rolling_avg.mean(),
                'peak_positions': list(np.where(rolling_avg == rolling_avg.max())[0] + 1),
                'total_occurrences': occurrence_df[protein].sum()
            }
            window_rolling[protein] = rolling_avg
            
        results[window] = window_results
        rolling_data[window] = window_rolling
    
    return results, rolling_data

def save_results_to_csv(results, output_path):
    # Create a list to store all rows
    rows = []
    
    for window_size, window_data in results.items():
        for protein, metrics in window_data.items():
            row = {
                'Window_Size': window_size,
                'Protein': protein,
                'Total_Occurrences': metrics['total_occurrences'],
                'Max_Density': metrics['max_density'],
                'Mean_Density': metrics['mean_density'],
                'Min_Density': metrics['min_density'],
                'Peak_Positions': ','.join(map(str, metrics['peak_positions']))
            }
            rows.append(row)
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_path, index=False)
    return results_df

def plot_distributions(rolling_data, output_dir, soft=False, noise_scale=0.01, target_protein = ""):

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set figure style
    plt.rcParams['figure.figsize'] = [15, 8]
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.facecolor'] = '#f0f0f0'
    
    # Create a plot for each window size
    for window_size, protein_data in rolling_data.items():
        fig, ax = plt.subplots()
        
        # Generate a color palette with enough colors
        n_colors = len(protein_data)
        colors = sns.color_palette("husl", n_colors)
        
        # Plot each protein's distribution
        for (protein, distribution), color in zip(protein_data.items(), colors):
            x = distribution.index + 1
            y = distribution.values
            
            # Add small noise to y-values
            noise = np.random.normal(0, noise_scale, size=len(y))
            y = y + noise

            # Remove NaN values for interpolation
            mask = ~np.isnan(y)
            x, y = x[mask], y[mask]

            if soft:
                # Create a smoother line using cubic splines
                if len(x) > 3:  # Ensure enough points for cubic spline
                    spline = make_interp_spline(x, y, k=3)
                    x_new = np.linspace(x.min(), x.max(), 500)
                    y_new = spline(x_new)
                    ax.plot(x_new, y_new, label=protein, color=color, alpha=0.7, linewidth=2)
                else:
                    # Fallback to straight lines if not enough points
                    ax.plot(x, y, label=protein, color=color, alpha=0.7, linewidth=2)
            else:
                # Regular straight-line plot
                ax.plot(x, y, label=protein, color=color, alpha=0.7, linewidth=2)

        # Add reference line at y=1
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        ax.set_title(f'{target_protein} RMSD Trajectory Partners Distribution Bias (Window Size: {window_size})', 
                    pad=20, fontsize=14)
        ax.set_xlabel('Trajectory Number', fontsize=12)
        ax.set_ylabel('Partner Frequency (Bias)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / f'distribution_window_{window_size}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze protein distribution in trajectories')
    parser.add_argument('input_file', type=str, help='Input TSV file path')
    parser.add_argument('--output_dir', type=str, default='protein_analysis',
                      help='Output directory for results (default: protein_analysis)')
    parser.add_argument('--windows', type=int, nargs='+', default=[5, 10, 15, 20],
                      help='Window sizes for analysis (default: 5 10)')
    parser.add_argument('--noise', type=float, default=0.01,
                      help='Add small noise to curves to separate them (default: 0.01)')
    parser.add_argument('--soft', action='store_true', 
                      help='Use smooth curves for plotting (default: straight lines)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read input file
    print(f"Reading input file: {args.input_file}")
    df = pd.read_csv(args.input_file, sep='\t')
    
    # Run analysis
    print("Analyzing protein distributions...")
    results, rolling_data = analyze_protein_distribution(df, windows=args.windows)
    
    # Save results to CSV
    output_csv = output_dir / 'protein_distribution_results.csv'
    print(f"Saving results to: {output_csv}")
    results_df = save_results_to_csv(results, output_csv)
    
    # Create plots
    print("Generating distribution plots...")
    plot_distributions(rolling_data, output_dir, soft=args.soft, noise_scale = args.noise)
    
    print("\nAnalysis complete! Output files:")
    print(f"- Results table: {output_csv}")
    print(f"- Distribution plots: {output_dir}/distribution_window_*.png")

if __name__ == "__main__":
    main()
