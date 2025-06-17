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

def plot_overall_results(min_nodes=None, max_nodes=None):
    """
    Plot runtime comparison of algorithms with option to limit the node range.
    Uses consistent colors for algorithms across different plots.
    
    Parameters:
    min_nodes (int, optional): Minimum number of nodes to include in the plot
    max_nodes (int, optional): Maximum number of nodes to include in the plot
    """
    results_folder = 'data/results/'
    result_file = 'data/results/results_runtime.csv'
    
    # Define consistent colors for each algorithm
    algorithm_colors = {
        'ILP': 'tab:blue',
        'Ant Colony': 'tab:orange',
        'Algorithm': 'tab:green',
        # Add more algorithms and colors as needed
    }
    
    # Read the data
    df = pd.read_csv(result_file)
    
    # Filter by node range if specified
    if min_nodes is not None:
        df = df[df['nodes'] >= min_nodes]
    if max_nodes is not None:
        df = df[df['nodes'] <= max_nodes]
    
    if df.empty:
        print(f"No data available in the range of nodes: {min_nodes} to {max_nodes}")
        return
    
    # Create main comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot each algorithm, but skip those with all zeros
    for algorithm in df.columns[1:]:  # Skip nodes column
        # Skip algorithms with all zeros in this range
        if (df[algorithm] == 0).all():
            print(f"Skipping '{algorithm}' as all values are 0 in the selected range")
            continue
        
        # Use the predefined color if available, otherwise use the default color cycle
        color = algorithm_colors.get(algorithm, None)
        plt.plot(df['nodes'], df[algorithm], marker='o', label=algorithm, 
                 color=color)
    
    # Set title with node range information
    if min_nodes is not None or max_nodes is not None:
        node_range = ""
        if min_nodes is not None:
            node_range += f"{min_nodes}"
        else:
            node_range += "min"
        
        node_range += " to "
        
        if max_nodes is not None:
            node_range += f"{max_nodes}"
        else:
            node_range += "max"
            
        plt.title(f'Algorithm Runtime Comparison by Graph Size (Nodes: {node_range})')
    else:
        plt.title('Algorithm Runtime Comparison by Graph Size')
    
    # Set x-axis to start at the minimum node count
    plt.xlim(left=df['nodes'].min())
    
    plt.xlabel('Number of Nodes')
    plt.ylabel('Runtime (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save with node range in filename if specified
    if min_nodes is not None or max_nodes is not None:
        min_str = str(min_nodes) if min_nodes is not None else "min"
        max_str = str(max_nodes) if max_nodes is not None else "max"
        filename = f'results_runtime_{min_str}_to_{max_str}.png'
    else:
        filename = 'results_runtime.png'
    
    plt.savefig(os.path.join(results_folder, filename))
    plt.close()
    
    print(f"Plot saved to {os.path.join(results_folder, filename)}")
    return


def find_optimal_parameters():
    """
    Find optimal ACO parameters based on lowest mean runtime across all graph sizes.
    Returns a dict with the best ant ratio and pheromone decay values.
    """
    results_folder = 'data/results/'
    ants_df = pd.read_csv(os.path.join(results_folder, 'ants.csv'))
    decay_df = pd.read_csv(os.path.join(results_folder, 'decay.csv'))
    
    # Group by ant ratio and calculate mean runtime across all graph sizes
    ants_mean_runtime = ants_df.groupby('num_ants')['runtime_ms'].mean()
    best_ant_ratio = ants_mean_runtime.idxmin()
    
    # Group by decay value and calculate mean runtime across all graph sizes
    decay_mean_runtime = decay_df.groupby('pheromone_decay')['runtime_ms'].mean()
    best_decay = decay_mean_runtime.idxmin()
    
    print(f"=== Optimal ACO Parameters (Simplest Criterion) ===")
    print(f"Best ant ratio: {best_ant_ratio:.2f} × number of nodes")
    print(f"Best pheromone decay: {best_decay:.1f}")
       
    return {
        'ant_ratio': best_ant_ratio,
        'decay': best_decay
    }

def benchmark_ant_colony():
    """
    Run the Ant Colony Optimizer on each graph in data/generated five times 
    and update results_runtime.csv with the mean runtime.
    """
    from aco import run
    import os
    import glob
    import time
    import pandas as pd
    import numpy as np
    
    # Load existing results file
    results_folder = 'data/results/'
    results_file = os.path.join(results_folder, 'results_runtime.csv')
    df = pd.read_csv(results_file)
    
    # Get optimal parameters
    optimal_params = find_optimal_parameters()
    ant_ratio = optimal_params['ant_ratio']
    decay = optimal_params['decay']
    
    print(f"Running ACO with optimal parameters: ant_ratio={ant_ratio}, decay={decay}")
    
    # Get all graph files
    data_folder = 'data/generated/'
    graph_files = sorted(glob.glob(os.path.join(data_folder, '*.txt')))
    print(f"Found {len(graph_files)} graph files in {data_folder}")
    
    # Process each graph file
    for graph_file in graph_files:
        # Extract node count from filename or first line
        with open(graph_file, 'r') as f:
            first_line = f.readline().strip()
            while first_line.startswith('//'):
                first_line = f.readline().strip()
            num_nodes = int(first_line)
        
        if num_nodes > 150: continue
        
        # Check if this node count exists in our results file
        if num_nodes in df['nodes'].values:
            print(f"Processing graph with {num_nodes} nodes")
            
            # Calculate number of ants based on optimal ratio
            num_ants = max(1, int(num_nodes * ant_ratio))
            
            # Run 5 times and collect runtimes
            runtimes = []
            for i in range(5):
                start_time = time.time()
                coloring = run(
                    input_path=graph_file,
                    no_ants=num_ants,
                    pheromone_decay=decay
                )
                end_time = time.time()
                runtime = end_time - start_time
                runtimes.append(np.round(runtime, 5))
                print(f"  Run {i+1}/5: {runtime:.6f} seconds")
            
            # Calculate mean runtime
            mean_runtime = np.mean(runtimes)
            print(f"  Mean runtime: {mean_runtime:.6f} seconds")
            
            # Update the dataframe
            idx = df.index[df['nodes'] == num_nodes].tolist()[0]
            df.at[idx, 'Ant Colony'] = mean_runtime
        else:
            print(f"Skipping graph with {num_nodes} nodes (not in results file)")
    
    # Save updated results
    df.to_csv(results_file, index=False)
    print(f"Updated results saved to {results_file}")
    
    # Generate updated plot
    plot_overall_results()
    print("Updated comparison plot generated")
    
    return df