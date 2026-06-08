"""
Performance plotter

Quick utility script to visualize the metrics saved during training or testing.
It reads the CSV dumped by the simulation and generates a 3-panel plot 
showing throughput, waiting time, and congestion over time.

Usage:
    python plot_metrics.py [path_to_csv]
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_metrics(csv_path: str):
    """
    Reads the simulation metrics and plots them in a single figure.
    The subplots share the X-axis so we can correlate events across different metrics.

    Args:
        csv_path (str): Path to the metrics CSV file.
    """
    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}")
        return

    # Load the raw data
    df = pd.read_csv(csv_path)
    
    # Create a 3-row, 1-column layout. 
    # sharex=True is crucial here so zooming/panning the timeline syncs across all three subplots.
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Throughput (Arrived vehicles)
    # Tracking how many vehicles successfully exit the network. Higher is better.
    axes[0].plot(df['step'], df['throughput'], color='blue', label='Total Arrived')
    axes[0].set_ylabel('Vehicles Arrived')
    axes[0].set_title('Throughput over time')
    axes[0].grid(True)
    
    # Plot 2: Waiting Time
    # The mean accumulated waiting time across all vehicles. We want this curve to drop.
    axes[1].plot(df['step'], df['mean_waiting_time'], color='red', label='Mean Waiting Time')
    axes[1].set_ylabel('Waiting Time (s)')
    axes[1].set_title('Average Waiting Time')
    axes[1].grid(True)
    
    # Plot 3: Congestion
    # Visualizes the ratio of stopped vehicles vs total active vehicles.
    axes[2].plot(df['step'], df['congestion_index'], color='orange', label='Congestion Index')
    axes[2].set_xlabel('Simulation Step')
    axes[2].set_ylabel('Congestion Index')
    axes[2].set_title('Network Congestion')
    axes[2].grid(True)
    
    # Prevent axis labels and titles from overlapping
    plt.tight_layout()
    
    # Save a high-res copy for documentation/reports before popping the window
    plt.savefig("analyse_resultats.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # Grab the file from the command line if provided, otherwise fallback to the default test file
    file_to_plot = sys.argv[1] if len(sys.argv) > 1 else "configs/four_way_1int_1lanes/results_congested.csv"
    plot_metrics(file_to_plot)