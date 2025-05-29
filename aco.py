import argparse
from pathlib import Path
import random
import numpy as np
import time


class Graph(object):
    def __init__(self, input_path, decay):
        self.decay_parameter = decay

        self.adjacency_list = {}
        adjacency_matrix = np.loadtxt(input_path, dtype=int)
        for u in range(adjacency_matrix.shape[0]):
            self.adjacency_list[u] = set()
            for v in range(adjacency_matrix.shape[1]):
                if adjacency_matrix[u][v] == 1 and u != v: # ignore self-loops
                    self.adjacency_list[u].add(v)
                    if v not in self.adjacency_list:
                        self.adjacency_list[v] = set()
                    self.adjacency_list[v].add(u)
        
        self.num_nodes = len(self.adjacency_list)
        
        self.pheromones = 1 - adjacency_matrix # pheromone[i][j] := desirability for nodes i, j to have same col.
        np.fill_diagonal(self.pheromones, 0) # no self-loops
                    
    
    def decay(self):
        self.pheromones *= self.decay_parameter


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
            self.coloring[node] = random.choice(available_colors) # better to choose randomly or favor first one?
        else:
            new_color = max(self.colors) + 1
            self.coloring[node] = new_color
            self.colors.append(new_color)
        
        self.current_node = node





def run(input_path, no_ants=10, ratio_elites=0.2, pheromone_decay=0.2, update_amt=1):
    graph = Graph(input_path, pheromone_decay)
    no_elites = max(int(no_ants * ratio_elites), 1)
    while True:
        ants = []
        for _ in range(no_ants):
            ant = Ant(graph)
            ant.walk_graph()
            # add pheromones only of best ant or weighted of best k ants?
            ants.append(ant)
        
        ants.sort(key=lambda x: len(x.colors))
        if len(ants[0].colors) <= 4:
            break # Found a 4-coloring (or better), stop search

        # pheromone update
        graph.decay()
        for ant in ants[:no_elites]:
            coloring = ant.coloring
            for i in range(len(coloring)):
                for j in range(i+1, len(coloring)): # ignore self-loops
                    if coloring[i] == coloring[j]:
                        graph.pheromones[i][j] += update_amt  # how much, addivite or multiplicative?
                        graph.pheromones[j][i] += update_amt
        
    return ants[0].coloring
