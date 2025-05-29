# Run chosen algorithm for all graphs
import numpy as np
import os
import time
import csv
import importlib.util
import glob


def aco_grid_search():
    from aco import run
    output_csv_path = f'data/results/aco_results.csv'

    # Get all graph files in the data folder
    data_folder = 'data/graphs/'
    graph_files = sorted(glob.glob(os.path.join(data_folder, '*.mat')))


    results = []
    # Create CSV file and write header
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['num_nodes', 'runtime_ms', 'num_ants', 'ratio_elites', 'decay']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for graph_file in graph_files:

            # Hyperparameters for this graph
            with open(graph_file, 'r') as f: num_nodes = len(f.readlines())
            no_ants_list = [int(i*num_nodes / 10) for i in range(1, 11)]  # 10% to 100% of nodes
            ratio_elites_list = [5*i/100 for i in range(1, 11)]  # 5 to 50% of ants are elites
            pheromone_decay_list = [i/10 for i in range(1, 10)]  # 10% to 90% pheromone decay
            
            for no_ants in no_ants_list:
                for ratio_elites in ratio_elites_list:
                    for pheromone_decay in pheromone_decay_list:  
                        result_times = [] 
                        for _ in range(10):        
                            # Run the algorithm and measure time
                            start_time = time.time()
                            coloring = run(input_path=graph_file, no_ants=no_ants, ratio_elites=ratio_elites, pheromone_decay=pheromone_decay)  
                            end_time = time.time()
                            runtime_ms = (end_time - start_time) * 1000
                            result_times.append(runtime_ms)
                            
                        num_nodes = len(coloring)                       
                        # Store the result
                        result = {
                            'num_nodes': num_nodes,
                            'runtime_ms': np.mean(result_times),
                            'num_ants': no_ants,
                            'ratio_elites': ratio_elites,
                            'decay': pheromone_decay
                        }
                        results.append(result)
                        
                        # Write to CSV
                        writer.writerow(result)
                
    print(f"Results saved to {output_csv_path}")
    return 
