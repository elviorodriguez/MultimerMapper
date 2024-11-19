
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

################################################################################################################
############################################# Preprocessing data ###############################################
################################################################################################################

def calculate_precision_recall(df):
    """Calculate accuracy, precision and recall (macro-average) based on true clusters and predicted clusters.
    Statistics formulas taken from https://www.evidentlyai.com/classification-metrics/multi-class-metrics"""
    threshold_list  = []
    accuracy_list   = []
    
    # # One for each class (true possible interaction modes)
    # precision_dict  = {}
    # recall_dict     = {}
    
    precision_list  = []
    recall_list     = []

    for threshold in df['threshold'].unique():
        subset = df[df['threshold'] == threshold]

        # True labels: `true_clusters`, Predicted labels: `predicted_clusters`
        true_labels = subset['true_clusters']
        predicted_labels = subset['predicted_clusters']
        
        # Accuracy
        acc = sum([a == b for a,b in zip(true_labels, predicted_labels)]) / len(true_labels)
        
        # Precision and Recall
        pres = []
        rec = []
        for interaction_mode in set(true_labels):
            
            # Compute TP, FP and FN
            TP_mode = sum([t == p and p == interaction_mode for t, p in zip(true_labels, predicted_labels)])
            FP_mode = sum([t != interaction_mode and p == interaction_mode for t, p in zip(true_labels, predicted_labels)])
            FN_mode = sum([t == interaction_mode and p != interaction_mode for t, p in zip(true_labels, predicted_labels)])
            

            # Compute Precision and Recall for the interaction mode
            try:
                precision_mode = TP_mode / (TP_mode + FP_mode)
            except ZeroDivisionError:
                precision_mode = 0
            try:
                recall_mode = TP_mode / (TP_mode + FN_mode)
            except ZeroDivisionError:
                recall_mode = 0
            
            pres.append(precision_mode)
            rec.append(recall_mode)
            
            # # Initialize lists
            # try:
            #     precision_dict[interaction_mode]
            #     recall_dict[interaction_mode]
            # except:
            #     precision_dict[interaction_mode] = []
            #     recall_dict[interaction_mode]    = []
            
            
            # precision_dict[interaction_mode].append()
            # recall_dict[interaction_mode].append()            
        
        # Compute macro-average precision and recall
        mean_precision = np.mean(pres)
        mean_recall = np.mean(rec)
        
        # append data
        threshold_list.append(threshold)
        accuracy_list.append(acc)
        precision_list.append(mean_precision)
        recall_list.append(mean_recall)

    return threshold_list, accuracy_list, precision_list, recall_list

def process_files(filepaths):
    """Process each CSV file and calculate precision and recall for different metrics and thresholds."""
    results = []

    # Iterate through each file and calculate precision and recall
    for filepath in filepaths:
        # Read the CSV file
        df = pd.read_csv(filepath)
        file_basename = os.path.basename(filepath)

        # Extract the metric from the file name (e.g., cf, iou, mc, medc)
        metric = file_basename.split('_')[0]

        # Calculate precision and recall at each threshold
        threshold_list, accuracy_list, precision_list, recall_list = calculate_precision_recall(df)

        # Store results in a list
        for threshold, precision, recall, accuracy in zip(threshold_list, precision_list, recall_list, accuracy_list):
            results.append({
                'metric': metric,
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy
                
            })

    # Convert the results into a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def normalize_threshold_by_metric(results_df):
    # Normalize the threshold column within each metric group
    results_df['normalized_threshold'] = results_df.groupby('metric')['threshold'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    return results_df


################################################################################################################
########################################## Preprocessing data 2 ################################################
################################################################################################################

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def read_csvs(filepaths):
    # Initialize an empty list to store each DataFrame
    df_list = []

    # Iterate through each file and read the CSV files
    for filepath in filepaths:
        # Read the CSV file
        df = pd.read_csv(filepath)

        # Get the metric name from the file path
        metric_name = os.path.basename(filepath).split('_')[0]

        # Add a new column for the metric name
        df['metric'] = metric_name

        # Append the DataFrame to the list
        df_list.append(df)

    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(df_list, ignore_index=True)

    return merged_df

def calculate_precision_recall_accuracy(df, threshold):
    """Calculate precision, recall, and accuracy using proper label alignment
    
    Args:
        df: DataFrame with true_clusters and predicted_clusters columns
        threshold: Current threshold value
    
    Returns:
        precision, recall, accuracy (macro averages for precision/recall)
    """
    subset = df[df['threshold'] == threshold]
    
    # Convert true and predicted clusters to zero-based consecutive integers
    le = LabelEncoder()
    true_labels = le.fit_transform(subset['true_clusters'])
    
    # Fit a new encoder for predicted labels to handle potentially different label sets
    pred_labels = LabelEncoder().fit_transform(subset['predicted_clusters'])
    
    # Calculate precision and recall using sklearn's implementation
    precision, recall, _, _ = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        average='macro',  # Use macro averaging
        zero_division=0   # Handle cases with zero predictions for a class
    )
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    
    return precision, recall, accuracy

def evaluate_clustering_metrics(df):
    """Evaluate clustering results with proper metric calculation
    
    Args:
        df: DataFrame with columns [protein1, protein2, threshold, true_clusters, 
            predicted_clusters, metric]
    
    Returns:
        DataFrame with precision-recall-accuracy values for each threshold and metric
    """
    results = []
    
    # Process each metric separately
    for metric in df['metric'].unique():
        metric_df = df[df['metric'] == metric]
        
        # Calculate P-R-A for each threshold
        for threshold in sorted(metric_df['threshold'].unique()):
            precision, recall, accuracy = calculate_precision_recall_accuracy(metric_df, threshold)
            
            results.append({
                'metric': metric,
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy
            })
    
    return pd.DataFrame(results)

# Example usage:
# results_df = evaluate_clustering_metrics(merged_df)
# 
# To see the results:
# print(results_df.sort_values(['metric', 'threshold']))

# Example usage:
# results_df = evaluate_clustering_metrics(merged_df)


################################################################################################################
############################################# Plotting functions ###############################################
################################################################################################################

def plot_precision_recall(results_df):
    # Set the plot style using seaborn
    sns.set(style="whitegrid")

    # Create a color palette for different metrics
    metrics = results_df['metric'].unique()
    palette = sns.color_palette("Set1", n_colors=len(metrics))

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Loop over each metric and plot Precision vs Recall
    for i, metric in enumerate(metrics):
        metric_data = results_df[results_df['metric'] == metric]
        plt.plot(metric_data['recall'], metric_data['precision'], label=metric, marker='o', color=palette[i])

    # Add labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall for Different Metrics')
    
    # Add a legend
    plt.legend(title='Metric')

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_accuracy_vs_threshold(results_df_input):
    
    # Copy df and compute normalized Threshold
    results_df = results_df_input.copy()
    results_df = normalize_threshold_by_metric(results_df)
    
    # Set the plot style using seaborn
    sns.set(style="whitegrid")

    # Create a color palette for different metrics
    metrics = results_df['metric'].unique()
    palette = sns.color_palette("Set2", n_colors=len(metrics))

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Loop over each metric and plot Accuracy vs Threshold
    for i, metric in enumerate(metrics):
        metric_data = results_df[results_df['metric'] == metric]
        plt.plot(metric_data['normalized_threshold'], metric_data['accuracy'], label=metric, marker='o', color=palette[i])

    # Add labels and title
    plt.xlabel('Normalized Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Threshold for Different Metrics')
    
    # Add a legend
    plt.legend(title='Metric')

    # Show the plot
    plt.tight_layout()
    plt.show()
    
def plot_accuracy_heatmap(results_df):
    # Pivot the DataFrame for a heatmap
    heatmap_df = results_df.pivot(index='metric', columns='threshold', values='accuracy')

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_df, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Accuracy'})

    # Add labels and title
    plt.xlabel('Threshold')
    plt.ylabel('Metric')
    plt.title('Accuracy Heatmap for Different Metrics and Thresholds')

    # Show the plot
    plt.tight_layout()
    plt.show()
    
def plot_accuracy_boxplot(results_df):
    # Set the plot style using seaborn
    sns.set(style="whitegrid")

    # Create the box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='metric', y='accuracy', data=results_df, palette="Set3")

    # Add labels and title
    plt.xlabel('Metric')
    plt.ylabel('Accuracy')
    plt.title('Distribution of Accuracy Across Different Metrics')

    # Show the plot
    plt.tight_layout()
    plt.show()
    

def plot_precision_recall_accuracy_static(results_df, out_dir = "."):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6.5, 6.5), dpi = 100)

    # Define the mapping for custom column names
    metric_mapping = {
        'mc': 'MC',
        'medc': 'MedC',
        'cf': 'CF',
        'iou': 'IoU'
    }
    df_copy = results_df.copy()
    df_copy['metric'] = df_copy['metric'].map(metric_mapping)
    df_copy = df_copy.rename(columns={
        'metric': 'Metric',
        'accuracy': 'Accuracy'
    })
    df_copy = df_copy.drop_duplicates(subset=['Metric', 'recall', 'precision', 'Accuracy'])

    # Create the scatter plot with accuracy as the size of the points
    sns.scatterplot(
        x='recall', y='precision', size='Accuracy', hue='Metric', style='Metric',
        data=df_copy, palette="Dark2", sizes=(20, 200), legend='brief',
        alpha = 0.7
    )

    plt.xlabel('Macro-Average Recall')
    plt.ylabel('Macro-Average Precision')
    plt.title('Precision vs Recall with Accuracy as Size')

    
    # Ensure the aspect ratio is 1:1 by setting the same limits for x and y axes
    max_limit = max(df_copy['recall'].max() + 0.05 , df_copy['precision'].max())
    min_limit = min(df_copy['recall'].min() - 0.02, df_copy['precision'].min())
    plt.xlim(min_limit, max_limit)
    plt.ylim(min_limit, max_limit)

    plt.tight_layout()
    plt.savefig(out_dir + "/precision_recall_accuracy_plot.png")
    plt.show()



def plot_precision_recall_accuracy_interactive(results_df_original, out_dir = "."):
    
    # Define the mapping for custom metric names
    metric_mapping = {
        'mc': 'MC',
        'medc': 'MedC',
        'cf': 'CF',
        'iou': 'IoU'
    }
    
    # Apply the mapping to the metric column
    results_df = results_df_original.copy()
    results_df['metric'] = results_df['metric'].map(metric_mapping)

    # Create the scatter plot with accuracy as the size of the points
    fig = px.scatter(
        results_df, x='recall', y='precision', size='accuracy', color='metric', symbol='metric',
        hover_data=['metric', 'threshold', 'precision', 'recall', 'accuracy'],
        title='Precision vs Recall with Accuracy as Size'
    )

    fig.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        legend_title_text='Metric',
        xaxis=dict(scaleanchor='y', scaleratio=1),
        yaxis=dict(scaleanchor='x', scaleratio=1)
    )
    
    # Save the plot as an HTML file
    fig.write_html(out_dir + "/precision_recall_accuracy_plot.html")

    # Show the plot
    fig.show()

