from math import sqrt
from typing import List
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import random
r = random.Random()
r.seed("AI")
import math


# region SearchAlgorithms
class Stack:

    def __init__(self):
        self.stack = []

    def push(self, value):
        if value not in self.stack:
            self.stack.append(value)
            return True
        else:
            return False

    def exists(self, value):
        if value not in self.stack:
            return True
        else:
            return False

    def pop(self):
        if len(self.stack) <= 0:
            return ("The Stack == empty")
        else:
            return self.stack.pop()

    def top(self):
        return self.stack[0]


class Node:
    id = None
    up: 'Node' = None
    down: 'Node' = None
    left: 'Node' = None
    right: 'Node' = None
    previousNode: 'Node' = None
    edgeCost = None
    gOfN = None  # total edge cost
    hOfN = None  # heuristic value
    heuristicFn = None
    pos_i: None
    pos_j: None

    def __init__(self, value):
        self.value = value

    @property
    def neighbors(self):
        return [self.up, self.down, self.left, self.right]

    @property
    def f_of_n(self):
        return self.gOfN + self.hOfN

    def set_g_of_n(self, distance_so_far):
        self.gOfN = self.edgeCost + distance_so_far

    def calculate_heuristic(self, si, sj):
        # Manhattan Distance
        self.hOfN = abs(si - self.pos_i) + abs(sj - self.pos_j)

class Maze:
    goal_symbol = "E"
    start_symbol = "S"
    obstacle_symbol = "#"

    matrix = []
    start: 'Node' = None
    goal: 'Node' = None
    n_rows = 0
    n_cols = 0

    def __init__(self, maze_str, edge_cost):
        self.parse_maze(maze_str, edge_cost)
        self.connect_neighbors()

    def parse_maze(self, maze_str, edge_cost):
        rows = maze_str.split()
        self.n_rows = len(rows)
        self.n_cols = rows[0].count(",") + 1
        id_latest = 0
        for i, row in enumerate(rows):
            cols = row.split(",")
            new_row = []
            for j, col in enumerate(cols):
                node = Node(col)
                node.id = id_latest
                node.pos_i = i
                node.pos_j = j
                node.edgeCost = edge_cost[i * self.n_cols + j]
                id_latest += 1
                if col == self.start_symbol:
                    self.start = node
                elif col == self.goal_symbol:
                    self.goal = node
                new_row.append(node)
            self.matrix.append(new_row)

    def connect_neighbors(self):
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if i - 1 >= 0:
                    (self.matrix[i][j]).up = self.matrix[i - 1][j]
                if i + 1 < self.n_rows:
                    (self.matrix[i][j]).down = self.matrix[i + 1][j]
                if j - 1 >= 0:
                    (self.matrix[i][j]).left = self.matrix[i][j - 1]
                if j + 1 < self.n_cols:
                    (self.matrix[i][j]).right = self.matrix[i][j + 1]


class SearchAlgorithms:
    ''' * DON'T change Class, Function or Parameters Names and Order
        * You can add ANY extra functions,
          classes you need as long as the main
          structure is left as is '''
    path = []  # Represents the correct path from start node to the goal node.
    fullPath = []  # Represents all visited nodes from the start node to the goal node.
    totalCost = None
    maze: Maze = None

    def __init__(self, mazeStr, edgeCost=None):
        ''' mazeStr contains the full board
         The board is read row wise,
        the nodes are numbered 0-based starting
        the leftmost node'''
        self.maze = Maze(mazeStr, edgeCost)

    def AstarManhattanHeuristic(self):
        extended = [self.maze.start]
        closed = []
        self.maze.start.gOfN = 0
        self.maze.start.hOfN = 0
        distance_so_far = self.maze.start.edgeCost
        self.totalCost = 0

        while len(extended) > 0:
            extended.sort(key=lambda x: x.f_of_n)
            n = extended.pop(0)
            closed.append(n)
            distance_so_far += n.edgeCost
            # n is Goal Node
            if n == self.maze.goal:
                while n != self.maze.start:
                    # push to the list top to avoid reversing
                    self.path.insert(0, n.id)
                    self.totalCost += n.edgeCost
                    n = n.previousNode
                self.path.insert(0, self.maze.start.id)

                for cn in closed:
                    self.fullPath.append(cn.id)

            # Check Neighbors
            for neighbor in n.neighbors:
                if neighbor is None:
                    continue
                neighbor.calculate_heuristic(n.pos_i, n.pos_j)
                neighbor.set_g_of_n(n.gOfN)
                # if neighbor.value == "#":
                #     continue
                if neighbor in closed:
                    continue
                if self.add_to_extended(extended, neighbor):
                    neighbor.previousNode = n
                    extended.append(neighbor)

        return self.fullPath, self.path, self.totalCost

    def add_to_extended(self, extended: List[Node], neighbor: Node):
        for n in extended:
            if n == neighbor and neighbor.f_of_n >= n.f_of_n:
                return False
        return True

# endregion

# region KNN
class KNN_Algorithm:
    def __init__(self, K):
        self.K = K

    def euclidean_distance(self, p1, p2):
        sigma = 0
        for i in range(len(p1)):
            sigma += (p1[i] - p2[i]) ** 2
        return sqrt(sigma)

    def KNN(self, X_train, X_test, Y_train, Y_test):
        c = 0
        for i in range(len(X_test)):
            distances = []
            for j in range(len(X_train)):
                euclidean_distance = self.euclidean_distance(X_test[i], X_train[j])
                distances.append((Y_train[j], euclidean_distance))
            
            distances.sort(key=lambda x: x[1])
            k_neighbors = []
            for k in range(self.K):
                k_neighbors.append(distances[k][0])
            freq_0 = k_neighbors.count(0)
            freq_1 = k_neighbors.count(1)
            response = 1
            if freq_0 > freq_1:
                response = 0
            if response == Y_test[i]:
                c += 1
        return c * 100 / len(Y_test)

# endregion


# region GeneticAlgorithm
class GeneticAlgorithm:
    Cities = [1, 2, 3, 4, 5, 6]
    DNA_SIZE = len(Cities)
    POP_SIZE = 20
    GENERATIONS = 5000
    mutation_chance = 0.01

    """
    - Chooses a random element from items, where items is a list of tuples in
       the form (item, weight).
    - weight determines the probability of choosing its respective item. 
     """
    def weighted_choice(self, items):
        weight_total = sum((item[1] for item in items))
        n = r.uniform(0, weight_total)
        for item, weight in items:
            if n < weight:
                return item
            n = n - weight
        return item

    """ 
      Return a random character between ASCII 32 and 126 (i.e. spaces, symbols, 
       letters, and digits). All characters returned will be nicely printable. 
    """
    def random_char(self):
        return chr(int(r.randrange(32, 126, 1)))

    """ 
       Return a list of POP_SIZE individuals, each randomly generated via iterating 
       DNA_SIZE times to generate a string of random characters with random_char(). 
    """
    def random_population(self):
        pop = []
        for i in range(1, 21):
            x = r.sample(self.Cities, len(self.Cities))
            if x not in pop:
                pop.append(x)
        return pop

    """ 
      For each gene in the DNA, this function calculates the difference between 
      it and the character in the same position in the OPTIMAL string. These values 
      are summed and then returned. 
    """

    def cost(self, city1, city2):
        if (city1 == 1 and city2 == 2) or (city1 == 2 and city2 == 1):
            return 10
        elif (city1 == 1 and city2 == 3) or (city1 == 3 and city2 == 1):
            return 20
        elif (city1 == 1 and city2 == 4) or (city1 == 4 and city2 == 1):
            return 23
        elif (city1 == 1 and city2 == 5) or (city1 == 5 and city2 == 1):
            return 53
        elif (city1 == 1 and city2 == 6) or (city1 == 6 and city2 == 1):
            return 12
        elif (city1 == 2 and city2 == 3) or (city1 == 3 and city2 == 2):
            return 4
        elif (city1 == 2 and city2 == 4) or (city1 == 4 and city2 == 2):
            return 15
        elif (city1 == 2 and city2 == 5) or (city1 == 5 and city2 == 2):
            return 32
        elif (city1 == 2 and city2 == 6) or (city1 == 6 and city2 == 2):
            return 17
        elif (city1 == 3 and city2 == 4) or (city1 == 4 and city2 == 3):
            return 11
        elif (city1 == 3 and city2 == 5) or (city1 == 5 and city2 == 3):
            return 18
        elif (city1 == 3 and city2 == 6) or (city1 == 6 and city2 == 3):
            return 21
        elif (city1 == 4 and city2 == 5) or (city1 == 5 and city2 == 4):
            return 9
        elif (city1 == 4 and city2 == 6) or (city1 == 6 and city2 == 4):
            return 5
        else:
            return 15

    '''
        * The fitness function is sum of distance between every two cities in the list.
    '''
    # complete fitness function
    def fitness(self, dna):
        fit = 0
        for i in range(len(dna) - 1):
            fit += self.cost(dna[i], dna[i + 1])
        if fit == 0:
            print(dna)
        return fit

    """ 
       For each gene in the DNA, there is a 1/mutation_chance chance that it will be 
       switched out with a random character. This ensures diversity in the 
       population, and ensures that is difficult to get stuck in local minima. 
       """

    def mutate(self, dna, random1, random2):
        mutation = dna[:]
        for i in range(self.DNA_SIZE):
            if random1 <= self.mutation_chance:
                swap_with = int(random2 * self.DNA_SIZE)
                if swap_with < self.DNA_SIZE:
                    mutation[i], mutation[swap_with] = mutation[swap_with], mutation[i]
        return mutation

        '''
       Slices both dna1 and dna2 into two parts at a random index within their 
       length and merges them. Both keep their initial sublist up to the crossover 
       index, but their ends are swapped. 
       '''

    def crossover(self, dna1, dna2, random1, random2):
        offspring1 = []
        offspring2 = []
        index_a = int(random1 * self.DNA_SIZE)
        index_b = int(random2 * self.DNA_SIZE)
        start = min(index_a, index_b)
        end = max(index_a, index_b)

        for i in range(start, end):
            offspring1.append(dna1[i])
            offspring2.append(dna2[i])

        offspring1 = offspring1 + [item for item in dna2 if item not in offspring1]
        offspring2 = offspring2 + [item for item in dna1 if item not in offspring2]

        return offspring1, offspring2


# endregion
#################################### Algorithms Main Functions #####################################
# region Search_Algorithms_Main_Fn
def SearchAlgorithm_Main():
    searchAlgo = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.',
                                  [0, 15, 2, 100, 60, 35, 30,
                                   3, 100, 2, 15, 60, 100, 30,
                                   2, 100, 2, 2, 2, 40, 30, 2,
                                   2, 100, 100, 3, 15, 30, 100,
                                   2, 100, 0, 2, 100, 30]
                                  )

    fullPath, path, TotalCost = searchAlgo.AstarManhattanHeuristic()
    print('**ASTAR with Manhattan Heuristic ** Full Path:' + str(fullPath) + '\nPath is: ' + str(path)
          + '\nTotal Cost: ' + str(TotalCost) + '\n\n')


# endregion

# region KNN_MAIN_FN
'''The dataset classifies tumors into two categories (malignant and benign) (i.e. malignant = 0 and benign = 1)
    contains something like 30 features.
'''


def KNN_Main():
    BC = load_breast_cancer()
    X = []

    for index, row in pd.DataFrame(BC.data, columns=BC.feature_names).iterrows():
        temp = []
        temp.append(row['mean area'])
        temp.append(row['mean compactness'])
        X.append(temp)
    y = pd.Categorical.from_codes(BC.target, BC.target_names)
    y = pd.get_dummies(y, drop_first=True)
    YTemp = []
    for index, row in y.iterrows():
        YTemp.append(row[1])
    y = YTemp;
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1024)
    KNN = KNN_Algorithm(7);
    accuracy = KNN.KNN(X_train, X_test, y_train, y_test)
    print("KNN Accuracy: " + str(accuracy))


# endregion

# region Genetic_Algorithm_Main_Fn
def GeneticAlgorithm_Main():
    genetic = GeneticAlgorithm();
    population = genetic.random_population()
    for generation in range(genetic.GENERATIONS):
        # print("Generation %s... Random sample: '%s'" % (generation, population[0]))
        weighted_population = []

        for individual in population:
            fitness_val = genetic.fitness(individual)

            pair = (individual, 1.0 / fitness_val)
            weighted_population.append(pair)
        population = []

        for _ in range(int(genetic.POP_SIZE / 2)):
            ind1 = genetic.weighted_choice(weighted_population)
            ind2 = genetic.weighted_choice(weighted_population)
            ind1, ind2 = genetic.crossover(ind1, ind2, r.random(), r.random())
            population.append(genetic.mutate(ind1, r.random(), r.random()))
            population.append(genetic.mutate(ind2, r.random(), r.random()))

    fittest_string = population[0]
    minimum_fitness = genetic.fitness(population[0])
    for individual in population:
        ind_fitness = genetic.fitness(individual)
    if ind_fitness <= minimum_fitness:
        fittest_string = individual
        minimum_fitness = ind_fitness

    print(fittest_string)
    print(genetic.fitness(fittest_string))


# endregion
######################## MAIN ###########################33
if __name__ == '__main__':
    # SearchAlgorithm_Main()
    KNN_Main()
    # GeneticAlgorithm_Main()
