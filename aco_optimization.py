# Run chosen algorithm for all graphs
import numpy as np
import os
import time
import csv
import glob
import matplotlib.pyplot as plt
import pandas as pd


def aco_find_parameters():
    from aco import run
    results_folder = 'data/results/'
    os.makedirs(results_folder, exist_ok=True)
    
    # Get all graph files in the data folder
    data_folder = 'data/sparse/'
    graph_files = sorted(glob.glob(os.path.join(data_folder, '*.mat')))

    output_files = {
        'ants': os.path.join(results_folder, 'ants.csv'),
        'decay': os.path.join(results_folder, 'decay.csv')
    }
    fieldnames = ['num_nodes', 'num_ants', 'pheromone_decay', 'runtime_ms']

    # Initialize CSV files with headers if they don't exist
    for file_path in output_files.values():
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    for graph_file in graph_files:

        # Get number of nodes for this graph
        with open(graph_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('//')]
            num_nodes = len(lines)
        del lines

        # Define hyperparameter ranges
        ratio_ants = [i*5 / 100 for i in range(1, 11)]  # 5% to 50% of nodes
        pheromone_decay_list = [i/10 for i in range(1, 10)]  # 10% to 90% pheromone decay
        no_ants_median = max(1, int(np.median(ratio_ants) * num_nodes))

        # 1. Evaluate number of ants
        with open(output_files['ants'], 'a', newline='') as csvfile: 
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  

            for ratio in ratio_ants:
                no_ants = max(1, int(num_nodes * ratio))
                result_times = []
                
                for _ in range(10):  # Run 10 times for more stable results
                    start_time = time.time()
                    coloring = run(
                        input_path=graph_file, 
                        no_ants=no_ants, 
                        pheromone_decay=np.median(pheromone_decay_list)
                    )
                    end_time = time.time()
                    runtime_ms = (end_time - start_time) * 1000
                    result_times.append(runtime_ms)
                
                row_data = {
                    'num_nodes': num_nodes,
                    'num_ants': ratio,
                    'pheromone_decay': np.median(pheromone_decay_list),
                    'runtime_ms': np.round(np.mean(result_times), 3),
                }
                writer.writerow(row_data)
                print(f"  Wrote result: {row_data}")
        
        # 2. Evaluate pheromone decay
        with open(output_files['decay'], 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            for pheromone_decay in pheromone_decay_list:
                result_times = []
                
                for _ in range(10):
                    start_time = time.time()
                    coloring = run(
                        input_path=graph_file, 
                        no_ants=no_ants_median, 
                        pheromone_decay=pheromone_decay
                    )
                    end_time = time.time()
                    runtime_ms = (end_time - start_time) * 1000
                    result_times.append(runtime_ms)
                
                writer.writerow({
                    'num_nodes': num_nodes,
                    'num_ants': no_ants_median,
                    'pheromone_decay': pheromone_decay,
                    'runtime_ms': np.mean(result_times),
                })    
    print(f"Results saved to {results_folder}")
    return



    results_folder = 'data/results/'
    output_files = {
        'ants': os.path.join(results_folder, 'ants.csv'),
        'decay': os.path.join(results_folder, 'decay.csv')
    }
    
    for key, file_path in output_files.items():
        df = pd.read_csv(file_path)
        
        plt.figure(figsize=(10, 6))
        
        # Get sorted unique node counts
        node_counts = sorted(df['num_nodes'].unique())
        
        # Create a color gradient from light to dark blue
        cmap = plt.cm.Blues
        color_norm = plt.Normalize(min(node_counts), max(node_counts))
        
        # Plot lines with color gradient
        for num_nodes in node_counts:
            subset = df[df['num_nodes'] == num_nodes]
            x_values = subset['num_ants' if key == 'ants' else 'ratio_elites' if key == 'elites' else 'pheromone_decay']
            y_values = subset['runtime_ms']
            
            # Calculate color based on node count (darker for more nodes)
            color = cmap(color_norm(num_nodes))
            
            plt.plot(x_values, y_values, marker='o', 
                     color=color, 
                     label=f'Nodes: {num_nodes}')
        
        plt.xlabel('Ant Ratio' if key == 'ants' else 'Pheromone Decay')
        plt.ylabel('Runtime (ms)')
        
        # Sort legend by node count
        handles, labels = plt.gca().get_legend_handles_labels()
        # Extract node numbers from labels for sorting
        label_nums = [int(label.split(': ')[1]) for label in labels]
        # Sort handles and labels based on node numbers
        sorted_pairs = sorted(zip(label_nums, handles, labels))
        # Unpack sorted pairs
        _, sorted_handles, sorted_labels = zip(*sorted_pairs)
        
        plt.legend(sorted_handles, sorted_labels)
        plt.grid()
        
        # Add colorbar to show the gradient scale
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, label='Number of Nodes')
        
        plt.title(f'Runtime vs {"Ant Ratio" if key == "ants" else "Pheromone Decay"}')
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f'{key}_results.png'))
        plt.close()
    
    print("Plots saved to", results_folder)

def plot_param_results():
    results_folder = 'data/results/'
    output_files = {
        'ants': os.path.join(results_folder, 'ants.csv'),
        'decay': os.path.join(results_folder, 'decay.csv')
    }
    
    for key, file_path in output_files.items():
        df = pd.read_csv(file_path)
        
        # Create figure and get the axes explicitly
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get sorted unique node counts
        node_counts = sorted(df['num_nodes'].unique())
        
        # Create a color gradient from light to dark blue
        cmap = plt.cm.Blues
        color_norm = plt.Normalize(min(node_counts), max(node_counts))
        
        # Plot lines with color gradient
        for num_nodes in node_counts:
            subset = df[df['num_nodes'] == num_nodes]
            x_values = subset['num_ants' if key == 'ants' else 'ratio_elites' if key == 'elites' else 'pheromone_decay']
            y_values = subset['runtime_ms']
            
            # Calculate color based on node count (darker for more nodes)
            color = cmap(color_norm(num_nodes))
            
            # Plot without adding to legend
            ax.plot(x_values, y_values, marker='o', color=color)
        
        ax.set_xlabel('Ant Ratio' if key == 'ants' else 'Pheromone Decay')
        ax.set_ylabel('Runtime (ms)')
        ax.grid()
        
        # Add colorbar to show the gradient scale - now with specified axes
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label='Number of Nodes')
        
        ax.set_title(f'Runtime vs {"Ant Ratio" if key == "ants" else "Pheromone Decay"}')
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f'{key}_results.png'))
        plt.close()
    
    print("Plots saved to", results_folder)

