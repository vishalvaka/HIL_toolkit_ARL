import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_aggregate_distances(folder_path, base_name="distances"):
    # Find all the relevant CSV files in the folder
    existing_files = os.listdir(folder_path)
    base_pattern = re.compile(rf"{base_name}(\d+)\.csv")
    
    distances_list = []

    for file in existing_files:
        match = base_pattern.match(file)
        if match:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, delim_whitespace=True, header=None)
            distances_list.append(df[0].values)

    # Find the maximum length of the distances arrays
    max_length = max(len(distances) for distances in distances_list)

    # Create a DataFrame to aggregate the distances
    aggregated_df = pd.DataFrame(index=range(max_length))

    for i, distances in enumerate(distances_list):
        aggregated_df[i] = pd.Series(distances)

    # Calculate mean and standard deviation for each iteration
    mean_distances = aggregated_df.mean(axis=1)
    std_distances = aggregated_df.std(axis=1)

    return mean_distances, std_distances

def plot_mean_and_std(mean_distances_rgpe, std_distances_rgpe, mean_distances_gp, std_distances_gp):
    iterations_rgpe = range(len(mean_distances_rgpe))
    iterations_gp = range(len(mean_distances_gp))

    plt.figure(figsize=(12, 6))
    
    # Plot RGPE
    plt.plot(iterations_rgpe, mean_distances_rgpe, label='RGPE', color='red')
    plt.errorbar(iterations_rgpe, mean_distances_rgpe, yerr=std_distances_rgpe, fmt='o', ecolor='red', alpha=0.5, label='RGPE Std Dev')
    
    # Plot GP
    plt.plot(iterations_gp, mean_distances_gp, label='GP', color='blue')
    plt.errorbar(iterations_gp, mean_distances_gp, yerr=std_distances_gp, fmt='o', ecolor='blue', alpha=0.5, label='GP Std Dev')
    
    plt.title('Mean Distance at Each Iteration with Standard Deviation for Schaffer with high noise')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Load and aggregate distances for both RGPE and GP
folder_path_rgpe = "models/distances/RGPE"
folder_path_gp = "models/distances/GP"

mean_distances_rgpe, std_distances_rgpe = load_and_aggregate_distances(folder_path_rgpe)
mean_distances_gp, std_distances_gp = load_and_aggregate_distances(folder_path_gp)

# Plot the results
plot_mean_and_std(mean_distances_rgpe, std_distances_rgpe, mean_distances_gp, std_distances_gp)
