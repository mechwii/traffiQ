import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_metrics(csv_path: str):
    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Throughput (Arrived vehicles)
    axes[0].plot(df['step'], df['throughput'], color='blue', label='Total Arrived')
    axes[0].set_ylabel('Vehicles Arrived')
    axes[0].set_title('Throughput over time')
    axes[0].grid(True)
    
    # Plot 2: Waiting Time
    axes[1].plot(df['step'], df['mean_waiting_time'], color='red', label='Mean Waiting Time')
    axes[1].set_ylabel('Waiting Time (s)')
    axes[1].set_title('Average Waiting Time')
    axes[1].grid(True)
    
    # Plot 3: Congestion
    axes[2].plot(df['step'], df['congestion_index'], color='orange', label='Congestion Index')
    axes[2].set_xlabel('Simulation Step')
    axes[2].set_ylabel('Congestion Index')
    axes[2].set_title('Network Congestion')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("analyse_resultats.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    file_to_plot = sys.argv[1] if len(sys.argv) > 1 else "configs/four_way_1int_1lanes/results_congested.csv"
    plot_metrics(file_to_plot)