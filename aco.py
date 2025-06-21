import argparse
from pathlib import Path
import random
import numpy as np
import time


class Graph(object):
    def __init__(self, input_path, decay):
        self.decay_parameter = decay
        self.adjacency_list = {}

        if input_path.endswith('.txt'): # input is an adjacency list
            with open(input_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('//')]
            
            self.num_nodes = int(lines[0])
            self.pheromones = np.ones((self.num_nodes, self.num_nodes), dtype=int) # pheromone[i][j] := desirability for nodes i, j to have same color

            
            # Initialize adjacency list
            for i in range(self.num_nodes): self.adjacency_list[i] = set()        
            
            # Process edges starting from line 2 (index 2)
            for i in range(2, len(lines)):
                parts = lines[i].split()
                if len(parts) >= 2:  # Ensure we have at least source and target
                    u, v = int(parts[0]), int(parts[1])
                    
                    # Add edge to adjacency list
                    self.adjacency_list[u].add(v)
                    self.adjacency_list[v].add(u)
                    
                    # Set pheromone for the edge
                    self.pheromones[u][v] = 0
                    self.pheromones[v][u] = 0    

        elif input_path.endswith('.mat'): # process adjacency matrix
            with open(input_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('//')]
            
            # Create matrix from the lines
            adjacency_matrix = []
            for line in lines:
                row = [int(val) for val in line.split()]
                adjacency_matrix.append(row)
            
            # Convert to numpy array for easier manipulation
            adjacency_matrix = np.array(adjacency_matrix)
            
            # Get number of nodes
            self.num_nodes = adjacency_matrix.shape[0]
            
            # Initialize adjacency list
            for i in range(self.num_nodes):
                self.adjacency_list[i] = set()
            
            # Fill adjacency list from matrix
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if adjacency_matrix[i][j] == 1 and i != j:  # if there's an edge (and not a self-loop)
                        self.adjacency_list[i].add(j)
            
            # Initialize pheromones
            self.pheromones = 1 - adjacency_matrix  # pheromone[i][j] := desirability for nodes i, j to have same color
        
        np.fill_diagonal(self.pheromones, 0)  # no self-loops
                    
    
    def decay(self):
        self.pheromones = np.round(self.pheromones * self.decay_parameter, 3)


class Ant(object):
    def __init__(self, graph):
        self.graph = graph
        self.colors = [0]
        self.coloring = np.zeros(self.graph.num_nodes)
        self.visited = []
        self.unvisited = list(self.graph.adjacency_list.keys())
        self.current_node = self.choose_start()

    def choose_start(self):
        start_node = random.choice(self.unvisited)
        self.visit(start_node)
        return start_node

    def walk_graph(self):
        while self.unvisited: 
            random.shuffle(self.unvisited) # randomize order of unvisited nodes
            max_pheromone = 0
            next_node = -1
            for node in self.unvisited: # choose highest-pheromone option
                pheromone = self.graph.pheromones[self.current_node][node] #
                if pheromone > max_pheromone: 
                    max_pheromone = pheromone
                    next_node = node 
            if next_node == -1:
                next_node = self.unvisited[0]
            self.visit(next_node)


    def visit(self, node):
        self.visited.append(node)
        self.unvisited.remove(node)

        available_colors = self.colors.copy()
        for neighbor in self.graph.adjacency_list[node]:
            if neighbor in self.visited and self.coloring[neighbor] in available_colors:
                available_colors.remove(self.coloring[neighbor])
        if available_colors:
            self.coloring[node] = random.choice(available_colors) 
        else:
            new_color = max(self.colors) + 1
            self.coloring[node] = new_color
            self.colors.append(new_color)
        
        self.current_node = node





def run(input_path, no_ants=10, pheromone_decay=0.2, update_amt=1):
    graph = Graph(input_path, pheromone_decay)
    while True:
        ants = []
        for _ in range(no_ants):
            ant = Ant(graph)
            ant.walk_graph()
            ants.append(ant)
        
        ants.sort(key=lambda x: len(x.colors))
        min_colors = len(ants[0].colors)
        if min_colors <= 4:
            break # Found a 4-coloring (or better), stop search

        # pheromone update
        graph.decay()
        for ant in ants:
            coloring = ant.coloring
            if len(ant.colors) > min_colors: break  # only update pheromones for ants that found the best coloring so far
            for i in range(len(coloring)):
                for j in range(i+1, len(coloring)): # ignore self-loops
                    if coloring[i] == coloring[j]:
                        graph.pheromones[i][j] += update_amt  
                        graph.pheromones[j][i] += update_amt
    return ants[0].coloring

