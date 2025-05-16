import argparse
import random
import numpy as np

class Graph(object):
    def __init__(self, input_path):
        self.adjacency_list = self.create_adjacency_list(input_path)
        self.pheromones = self.initialize_pheromones()

    def create_adjacency_list(self, input_path):
        pass


class Ant(object):
    def __init__(self, graph):
        self.graph = graph
        self.colors = [1]
        self.coloring = np.zeros(len(self.graph.nodes))
        self.visited = []
        self.unvisited = graph.nodes
        self.current_node = self.choose_start()

    def choose_start(self):
        start_node = random.choice(list(self.graph.nodes))
        self.visit(start_node)
        return start_node

    def walk_graph(self):
        while self.unvisited: # keep some randomization or always choose highest pheromone node?
            next_candidates = []
            pheromones = []
            for node in self.unvisited:
                pheromone = self.graph.pheromones[self.current_node][node] # pheromone kinda needs to be reflexive...
                if pheromone > 0: # not neighbors
                    next_candidates.append(node)
                    pheromones.append(pheromone)
            next = random.choice(
                                next_candidates,
                                weights=pheromones,
                                k=len(next_candidates))
            self.visit(next)


    def visit(self, node):
        self.visited.append(node)
        self.unvisited.remove(node)

        available_colors = self.colors.copy()
        for vertex in self.graph.adjacency_list[node]:
            if vertex in self.visited and self.coloring[vertex] in available_colors:
                available_colors.remove(self.coloring[vertex])
        
        if available_colors:
            self.coloring[node] = available_colors[np.random.randint(0, len(available_colors))] # better to choose randomly or favor first one?
        else:
            new_color = self.colors[-1] + 1
            self.coloring[node] = new_color
            self.colors.append(new_color)
        
        self.current_node = node





def main(args):
    # Hyperparameters
    input_path = args.input_graph
    no_ants = len(graph.nodes)
    pheromone_decay = 0.2
    no_elites = 1
    graph = Graph(input_path, pheromone_decay)

    while True:
        ants = []
        for _ in range(no_ants):
            ant = Ant(graph)
            ant.move()
            # add pheromones only of best ant or weighted of best k ants?
            ants.append(ant)
        
        ants.sort(key=lambda x: len(x.colors))
        if ants[0].colors == 4:
            print("Found a 4-coloring")
            break
        # pheromone update
        graph.decay()
        for ant in ants[:no_elites]:
            coloring = ant.coloring
            for i in range(len(coloring)):
                for j in range(i+1, len(coloring)): # ignore self-loops
                    if coloring[i] == coloring[j]:
                        graph.pheromones[i][j] += 1
                        graph.pheromones[j][i] += 1
        
    # TODO: visualization for small graphs
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a planar graph from a set of points.")
    parser.add_argument(
        "--input_graph",
        type=str,
        required=True,
        help="Path to the file containing the input graph.",
    )
    args = parser.parse_args()

    random.seed(42)
    main(args)

     