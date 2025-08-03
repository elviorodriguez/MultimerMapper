
import pandas as pd
import ast
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import cycle

############################## Helper functions ###############################

def evaluate_clustering_methods(df, true_label_col, metadata_cols):
    """
    Evaluate clustering methods by computing various performance metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing true labels and predictions from different methods
    true_label_col : str
        Name of the column containing true labels
    metadata_cols : list
        List of column names that are metadata (to be excluded from evaluation)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with evaluation metrics for each method
    """
    
    # Get method columns (exclude true label and metadata columns)
    method_cols = [col for col in df.columns 
                   if col not in metadata_cols + [true_label_col]]
    
    results = []
    
    for method in method_cols:
        # Skip methods with all NaN values
        if df[method].isna().all():
            continue
            
        # Get valid (non-NaN) predictions
        valid_mask = ~df[method].isna()
        y_true = df.loc[valid_mask, true_label_col]
        y_pred = df.loc[valid_mask, method]
        
        if len(y_true) == 0:
            continue
        
        # Parse method components
        method_parts = method.split('_')
        distance = method_parts[0] if len(method_parts) > 0 else 'Unknown'
        clustering = method_parts[1] if len(method_parts) > 1 else 'Unknown'
        linkage = method_parts[2] if len(method_parts) > 2 else 'Unknown'
        validation = method_parts[3] if len(method_parts) > 3 else 'Unknown'
        
        # Compute classification metrics (treating as multi-class)
        try:
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        except:
            precision = recall = accuracy = f1 = np.nan
        
        # Compute regression-like metrics (since we're predicting counts)
        try:
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
        except:
            mae = mse = rmse = np.nan
        
        # Compute correlation metrics
        try:
            pearson_corr, pearson_p = pearsonr(y_true, y_pred)
            spearman_corr, spearman_p = spearmanr(y_true, y_pred)
        except:
            pearson_corr = pearson_p = spearman_corr = spearman_p = np.nan
        
        # Compute exact match accuracy (percentage of perfect predictions)
        exact_match_accuracy = (y_true == y_pred).mean()
        
        # Compute tolerance-based accuracy (within ±1 of true value)
        tolerance_1_accuracy = (np.abs(y_true - y_pred) <= 1).mean()
        
        # Count of valid predictions
        n_valid_predictions = len(y_true)
        n_total_predictions = len(df)
        coverage = n_valid_predictions / n_total_predictions
        
        # Store results
        results.append({
            'Method': method,
            'Distance': distance,
            'Clustering': clustering,
            'Linkage': linkage,
            'Validation': validation,
            'Precision': precision,
            'Recall': recall,
            'Accuracy': accuracy,
            'F1_Score': f1,
            'Exact_Match_Accuracy': exact_match_accuracy,
            'Tolerance_1_Accuracy': tolerance_1_accuracy,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'Pearson_Correlation': pearson_corr,
            'Pearson_p_value': pearson_p,
            'Spearman_Correlation': spearman_corr,
            'Spearman_p_value': spearman_p,
            'Coverage': coverage,
            'N_Valid_Predictions': n_valid_predictions,
            'N_Total_Samples': n_total_predictions
        })
    
    # Convert to DataFrame and sort by accuracy
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Exact_Match_Accuracy', ascending=False)
    
    return results_df


# You can also create a summary function to get the best methods for each metric
def get_best_methods(results_df, top_n=5):
    """
    Get the top N methods for key metrics.
    """
    metrics_of_interest = ['Exact_Match_Accuracy', 'Tolerance_1_Accuracy', 'MAE', 'RMSE', 'Pearson_Correlation']
    
    best_methods = {}
    for metric in metrics_of_interest:
        if metric in ['MAE', 'RMSE']:  # Lower is better
            best = results_df.nsmallest(top_n, metric)[['Method', metric]]
        else:  # Higher is better
            best = results_df.nlargest(top_n, metric)[['Method', metric]]
        best_methods[metric] = best
    
    return best_methods


# Function to create a summary report
def create_summary_report(results_df):
    """
    Create a summary report of method performance.
    """
    print("=== CLUSTERING METHOD EVALUATION SUMMARY ===\n")
    
    print(f"Total methods evaluated: {len(results_df)}")
    print(f"Methods with >90% exact match accuracy: {(results_df['Exact_Match_Accuracy'] > 0.9).sum()}")
    print(f"Methods with >95% tolerance-1 accuracy: {(results_df['Tolerance_1_Accuracy'] > 0.95).sum()}")
    
    print(f"\nBest overall method (exact match): {results_df.iloc[0]['Method']}")
    print(f"Best exact match accuracy: {results_df.iloc[0]['Exact_Match_Accuracy']:.3f}")
    
    print(f"\nMethods with lowest MAE:")
    print(results_df.nsmallest(3, 'MAE')[['Method', 'MAE']])
    
    print(f"\nMethods with highest correlation:")
    print(results_df.nlargest(3, 'Pearson_Correlation')[['Method', 'Pearson_Correlation']])
    
    return None


def visualize_clustering_methods(
    df,
    subplot_rows="Linkage",
    subplot_cols="Validation", 
    x_axis="Recall",
    y_axis="Precision",
    point_size="Accuracy",
    point_color="Distance",
    point_shape="Clustering",
    colors=None,
    shapes=None,
    min_size=5,
    max_size=25,
    opacity=0.7,
    width=1400,
    height=900,
    title="Clustering Methods Performance Comparison",
    filename="clustering_methods_visualization.html",
    show_plot=True,
    x_range=None,
    y_range=None,
    fullscreen=True,
    add_range_lines=False,
    grid_divisions=5
):
    """
    Create an advanced visualization of clustering method performance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe from evaluate_clustering_methods
    subplot_rows : str
        Column name for subplot rows (default: "Linkage")
    subplot_cols : str  
        Column name for subplot columns (default: "Validation")
    x_axis : str
        Column name for x-axis (default: "Recall")
    y_axis : str
        Column name for y-axis (default: "Precision")
    point_size : str
        Column name for point size (default: "Accuracy")
    point_color : str
        Column name for point color (default: "Distance")
    point_shape : str
        Column name for point shape (default: "Clustering")
    colors : list, optional
        Custom colors for categories
    shapes : list, optional
        Custom shapes for categories
    min_size : float
        Minimum point size (default: 5)
    max_size : float
        Maximum point size (default: 25)
    opacity : float
        Point opacity 0-1 (default: 0.7)
    width : int
        Plot width in pixels (default: 1400)
    height : int
        Plot height in pixels (default: 900)
    title : str
        Main plot title
    filename : str
        Output HTML filename
    show_plot : bool
        Whether to display the plot (default: True)
    x_range : list, optional
        [min, max] values for x-axis (default: None, auto-scale)
    y_range : list, optional
        [min, max] values for y-axis (default: None, auto-scale)
    fullscreen : bool
        Whether to make plot fullscreen in HTML (default: True)
    add_range_lines : bool
        Whether to add lines at x/y min/max when ranges are specified (default: False)
    grid_divisions : int
        Number of grid divisions between min and max values (default: 5)
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure object
    """
    
    # Remove rows with NaN values in required columns
    required_cols = [subplot_rows, subplot_cols, x_axis, y_axis, point_size, point_color, point_shape]
    df_clean = df.dropna(subset=required_cols)
    
    if len(df_clean) == 0:
        raise ValueError("No valid data after removing NaN values")
    
    # Get unique categories for subplots
    row_categories = sorted(df_clean[subplot_rows].unique())
    col_categories = sorted(df_clean[subplot_cols].unique())
    
    # Get unique categories for color and shape
    color_categories = sorted(df_clean[point_color].unique())
    shape_categories = sorted(df_clean[point_shape].unique())
    
    # Define default colors and shapes
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    if shapes is None:
        shapes = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 
                 'triangle-down', 'triangle-left', 'triangle-right', 'pentagon']
    
    # Create color and shape mappings
    color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(color_categories)}
    shape_map = {cat: shapes[i % len(shapes)] for i, cat in enumerate(shape_categories)}
    
    # Normalize size values
    size_min = df_clean[point_size].min()
    size_max = df_clean[point_size].max()
    size_range = size_max - size_min if size_max != size_min else 1
    
    # Create subplots
    n_rows = len(row_categories)
    n_cols = len(col_categories)
    
    # Create subplot titles (empty, we'll add custom annotations)
    subplot_titles = [""] * (n_rows * n_cols)
    
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.08
    )
    
    # Track which combinations we've added to the legend
    added_to_legend = {"color": set(), "shape": set(), "size": False}
    
    # Add traces for each subplot
    for i, row_cat in enumerate(row_categories):
        for j, col_cat in enumerate(col_categories):
            # Filter data for this subplot
            subset = df_clean[
                (df_clean[subplot_rows] == row_cat) & 
                (df_clean[subplot_cols] == col_cat)
            ]
            
            if len(subset) == 0:
                continue
            
            # Group by color and shape to create separate traces
            for color_cat in color_categories:
                for shape_cat in shape_categories:
                    data_subset = subset[
                        (subset[point_color] == color_cat) & 
                        (subset[point_shape] == shape_cat)
                    ]
                    
                    if len(data_subset) == 0:
                        continue
                    
                    # Normalize sizes
                    sizes = ((data_subset[point_size] - size_min) / size_range * 
                            (max_size - min_size) + min_size)
                    
                    # Create hover text with all metadata
                    hover_text = []
                    for idx, row in data_subset.iterrows():
                        text_parts = [f"Method: {row['Method']}"]
                        # Add all available columns as metadata
                        for col in df_clean.columns:
                            if col != 'Method' and not pd.isna(row[col]):
                                if isinstance(row[col], float):
                                    text_parts.append(f"{col}: {row[col]:.3f}")
                                else:
                                    text_parts.append(f"{col}: {row[col]}")
                        hover_text.append("<br>".join(text_parts))
                    
                    # Determine if this combination should show in legend
                    show_in_legend = (color_cat not in added_to_legend["color"] or 
                                    shape_cat not in added_to_legend["shape"])
                    
                    # Create trace
                    trace = go.Scatter(
                        x=data_subset[x_axis],
                        y=data_subset[y_axis],
                        mode='markers',
                        marker=dict(
                            size=sizes,
                            color=color_map[color_cat],
                            symbol=shape_map[shape_cat],
                            opacity=opacity,
                            line=dict(width=1, color='white')
                        ),
                        name=f"{color_cat} ({shape_cat})",
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                        showlegend=show_in_legend,
                        legendgroup=f"{color_cat}_{shape_cat}"
                    )
                    
                    fig.add_trace(trace, row=i+1, col=j+1)
                    
                    # Mark as added to legend
                    if show_in_legend:
                        added_to_legend["color"].add(color_cat)
                        added_to_legend["shape"].add(shape_cat)
    
    # Add size legend traces (invisible points for size reference)
    if not added_to_legend["size"]:
        size_values = [size_min, (size_min + size_max) / 2, size_max]
        size_sizes = [min_size, (min_size + max_size) / 2, max_size]
        
        for k, (val, size) in enumerate(zip(size_values, size_sizes)):
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(
                    size=size, 
                    color='rgba(128,128,128,0.7)', 
                    line=dict(width=1, color='white')
                ),
                showlegend=True,
                legendgroup="size_legend",
                legendgrouptitle=dict(text=f"<b>{point_size}</b>"),
                name=f"{val:.2f}"
            ))
        added_to_legend["size"] = True
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        width=width,
        height=height,
        showlegend=True,
        plot_bgcolor='white',
        font=dict(size=10),
        margin=dict(l=100, r=300, t=80, b=80),  # Increased right margin for legend
        legend=dict(
            x=1.02,
            y=1,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=10),
            groupclick="toggleitem"
        )
    )
    
    # Add custom annotations for row and column labels
    for i, row_cat in enumerate(row_categories):
        fig.add_annotation(
            text=f"<b>{str(row_cat)}</b>",
            x=1.01,
            y=(n_rows - i - 0.5) / n_rows,
            xref="paper",
            yref="paper",
            textangle=90,  # Rotated 180 degrees
            showarrow=False,
            font=dict(size=14, color="black"),
            xanchor="center",
            yanchor="middle"
        )
    
    for j, col_cat in enumerate(col_categories):
        fig.add_annotation(
            text=f"<b>{str(col_cat)}</b>",
            x=(j + 0.5) / n_cols,
            y=1.04,
            xref="paper", 
            yref="paper",
            showarrow=False,
            font=dict(size=14, color="black"),
            xanchor="center",
            yanchor="bottom"
        )
    
    # Add single x-axis and y-axis labels with better positioning
    fig.add_annotation(
        text=f"<b>{x_axis}</b>",
        x=0.5,
        y=-0.06,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color="black"),
        xanchor="center",
        yanchor="top"
    )
    
    # Y-axis label rotated 180 degrees
    fig.add_annotation(
        text=f"<b>{y_axis}</b>",
        x=-0.06,
        y=0.5,
        xref="paper",
        yref="paper",
        textangle=180,  # Rotated 180 degrees instead of 90
        showarrow=False,
        font=dict(size=16, color="black"),
        xanchor="center",
        yanchor="middle"
    )
    
    # Update axes for all subplots with custom grid
    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            # Calculate grid tick values
            if x_range:
                x_tick_vals = [x_range[0] + k * (x_range[1] - x_range[0]) / grid_divisions 
                              for k in range(grid_divisions + 1)]
            else:
                x_tick_vals = None
                
            if y_range:
                y_tick_vals = [y_range[0] + k * (y_range[1] - y_range[0]) / grid_divisions 
                              for k in range(grid_divisions + 1)]
            else:
                y_tick_vals = None
            
            fig.update_xaxes(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='lightgray',
                range=x_range if x_range else None,
                tickvals=x_tick_vals,
                row=i, col=j
            )
            fig.update_yaxes(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='lightgray',
                range=y_range if y_range else None,
                tickvals=y_tick_vals,
                row=i, col=j
            )
            
            # Add range lines if requested
            if add_range_lines and x_range:
                # Vertical lines at x_min and x_max
                fig.add_vline(
                    x=x_range[0], 
                    line_dash="solid", 
                    line_color="black", 
                    line_width=1,
                    row=i, col=j
                )
                fig.add_vline(
                    x=x_range[1], 
                    line_dash="solid", 
                    line_color="black", 
                    line_width=1,
                    row=i, col=j
                )
            
            if add_range_lines and y_range:
                # Horizontal lines at y_min and y_max
                fig.add_hline(
                    y=y_range[0], 
                    line_dash="solid", 
                    line_color="black", 
                    line_width=1,
                    row=i, col=j
                )
                fig.add_hline(
                    y=y_range[1], 
                    line_dash="solid", 
                    line_color="black", 
                    line_width=1,
                    row=i, col=j
                )
    
    # Create HTML with fullscreen option
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
        'responsive': fullscreen
    }
    
    # Save as HTML with custom styling for fullscreen
    if fullscreen:
        html_string = fig.to_html(
            include_plotlyjs=True,
            config=config,
            div_id="plotly-div"
        )
        
        # Add custom CSS for fullscreen
        custom_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <title>{title}</title>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    height: 100vh;
                    overflow: hidden;
                }}
                #plotly-div {{
                    width: 100vw !important;
                    height: 100vh !important;
                }}
            </style>
        </head>
        <body>
            {html_string.split('<body>')[1].split('</body>')[0]}
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(custom_html)
    else:
        fig.write_html(filename, config=config)
    
    if show_plot:
        fig.show()
    
    return fig

###############################################################################

# --------------------------------- Load data ---------------------------------

# Paths
working_dir = "/home/elvio/Desktop/multivalency_benchmark"
out_path = working_dir + "/matrix_clustering_benchmark_results"
benchmark_df_file =  out_path + '/valencies_by_method.tsv'

# Read benchmark data and convert string to actual tuples
benchmark_df = pd.read_csv(benchmark_df_file, sep="\t")
benchmark_df['sorted_tuple_names'] = benchmark_df['sorted_tuple_names'].apply(ast.literal_eval)


# ------------------------------ Analyze the data -----------------------------

# Example usage with your specific data:
metadata_cols = ['type', 'sorted_tuple_names']
true_label_col = 'true_val'

# Run the evaluation (assuming your DataFrame is called benchmark_df)
evaluation_results = evaluate_clustering_methods(benchmark_df, true_label_col, metadata_cols)

# Display results
print("Top 10 methods by exact match accuracy:")
print(evaluation_results[['Method', 'Distance', 'Clustering', 'Linkage', 'Validation', 
                         'Exact_Match_Accuracy', 'Tolerance_1_Accuracy', 'MAE', 'Coverage']].head(10))

# Create summary report
create_summary_report(evaluation_results)

# Get best methods for each metric
best_methods = get_best_methods(evaluation_results)
print("\n=== BEST METHODS BY METRIC ===")
for metric, methods in best_methods.items():
    print(f"\nTop methods for {metric}:")
    print(methods)

# --------------------------- Some plots of the data  -------------------------


# fig = visualize_clustering_methods(
#     evaluation_results,
#     # subplot_rows="Linkage",
#     # subplot_cols="Validation", 
#     # x_axis="Recall",
#     # y_axis="Precision",
#     point_size="Accuracy",  # Using exact match accuracy instead
#     # point_color="Distance",
#     # point_shape="Clustering",
#     min_size=8,
#     max_size=30,
#     opacity=0.8,
#     title="",
#     filename="/home/elvio/Desktop/clustering_performance_analysis.html",
#     x_range=[0, 1],  # Limit x-axis from 0 to 1
#     y_range=[0, 1],  # Limit y-axis from 0 to 1
#     fullscreen=True  # Make plot occupy full HTML page
# )

fig = visualize_clustering_methods(
    evaluation_results,
    point_size="Accuracy",
    x_range=[0, 1],
    y_range=[0, 1],
    add_range_lines=True,  # Add boundary lines
    grid_divisions=10,     # More grid lines
    filename="/home/elvio/Desktop/clustering_performance_analysis.html",
    fullscreen=True
)