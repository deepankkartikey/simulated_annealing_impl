
import csv
import numpy as np
import math
import copy
import sys
import random
from timeit import default_timer as timer


class State:
    # State constructor
    def __init__(self, route: [], distance: int = 0):
        self.route = route
        self.distance = distance

    # Compare 2 states
    def __eq__(self, other):
        for i in range(len(self.route)):
            if (self.route[i] != other.route[i]):
                return False
        return True

    # Is current state's distance shorter than other state's distance
    def __lt__(self, other):
        return self.distance < other.distance

    # Printout a state
    def __repr__(self):
        return ('({0},{1})\n'.format(self.route, self.distance))

    # Copy state
    def copy(self):
        return State(self.route, self.distance)

    # Deeply Copy state
    def deepcopy(self):
        return State(copy.deepcopy(self.route), copy.deepcopy(self.distance))

    # Update distance
    def update_distance(self, matrix, home):
        # Reset distance
        self.distance = 0
        # City of departure
        from_index = home
        # Walk through each city in the current route
        for i in range(len(self.route)):
            self.distance += matrix[from_index][self.route[i]]
            from_index = self.route[i]
        # Add the distance back to home
        self.distance += matrix[from_index][home]

    # Random routes shuffling
    def shuffle_route(self, matrix, home):
        random.shuffle(self.route)
        self.update_distance(matrix, home)

# This class represent a city (used when we need to delete cities)


class City:
    # City constructor
    def __init__(self, index: int, distance: int):
        self.index = index
        self.distance = distance

    # Is current city's distance shorter than other city's distance
    def __lt__(self, other):
        return self.distance < other.distance

# Get the best random solution from a population
def get_best_solution_by_population(matrix: [], home: int, city_indexes: [], size: int):
    # Create a list with city indexes
    cities = city_indexes.copy()
    # Remove the home city
    cities.pop(home)
    # Create a population
    population = []
    for i in range(size):
        # Shuffle cities at random
        random.shuffle(cities)
        # Create a state
        state = State(cities[:])
        state.update_distance(matrix, home)
        # Add an individual to the population
        population.append(state)
    # Sort population
    population.sort()
    # Return the best solution
    return population[0]

# Get the best random solution by distance
def get_best_solution_by_distance(matrix: [], home: int):
    route = []
    from_index = home
    length = len(matrix) - 1

    while len(route) < length:
        # Matrix row
        row = matrix[from_index]
        # Create a dictionary of cities
        cities = {}
        for i in range(len(row)):
            cities[i] = City(i, row[i])
        # Remove cities that already is assigned to the route
        del cities[home]
        for i in route:
            del cities[i]
        # Create list of cities
        sorted = list(cities.values())
        # Sort cities
        sorted.sort()
        # Add the city with the shortest distance
        from_index = sorted[0].index
        route.append(from_index)
    # Create a new state and update the distance
    state = State(route)
    state.update_distance(matrix, home)
    # Return a state
    return state

# Mutate a solution
def mutate(matrix: [], home: int, state: State, mutation_rate: float = 0.01):
    # Create a copy of the state
    mutated_state = state.deepcopy()
    # Loop all the states in a route
    for i in range(len(mutated_state.route)):
        # Check if we should do a mutation
        if (random.random() < mutation_rate):
            # Swap two cities
            j = int(random.random() * len(state.route))
            city_1 = mutated_state.route[i]
            city_2 = mutated_state.route[j]
            mutated_state.route[i] = city_2
            mutated_state.route[j] = city_1
    # Update the distance
    mutated_state.update_distance(matrix, home)
    # Return a mutated state
    return mutated_state

# Hill climbing algorithm


def hill_climbing(matrix: [], home: int, initial_state: State, max_iterations: int, mutation_rate: float = 0.01):
    # Keep track of the best state
    best_state = initial_state
    # An iterator can be used to give the algorithm more time to find a solution
    iterator = 0
    # Create an infinite loop
    while True:
        # Mutate the best state
        neighbor = mutate(matrix, home, best_state, mutation_rate)
        # Check if the distance is less than in the best state
        if (neighbor.distance >= best_state.distance):
            iterator += 1
            if (iterator > max_iterations):
                break
        if (neighbor.distance < best_state.distance):
            best_state = neighbor
    # Return the best state
    return best_state

# Reduce possibility of local optimum


def restart_random(matrixofcities, home, initial_state, max_iterations, mutation_rate):
    state = initial_state.copy()
    Fitness = [0] * max_iterations
    count = 0
    while count < max_iterations:
        best_state = hill_climbing(
            matrixofcities, home, initial_state, max_iterations, mutation_rate)
        Fitness[count] = best_state.distance
        initial_state = best_state
        count += 1
    return [best_state, Fitness]

# Get the TSP file as input


def get_file_input(fn):

    # open file in read mode
    my_file = open(fn, 'r')

    NAME = my_file.readline().strip().split()[1]  # NAME
    TYPE = my_file.readline().strip().split()[1]  # TYPE
    COMMENT = my_file.readline().strip()  # COMMENT
    DIMENSION = my_file.readline().strip().split()[-1]  # DIMENSION
    EDGE_WEIGHT_TYPE = my_file.readline().strip().split()[1]  # EDGE_WEIGHT_TYPE
    my_file.readline()

    # Read node list
    nodelist = []
    N = int(DIMENSION)
    for i in range(0, N):
        x, y = my_file.readline().strip().split()[1:]
        nodelist.append([float(x), float(y)])
    # Close stream
    my_file.close()

    # Calculation of Matrix of cities' distances
    output_matrix = [[0 for x in range(N)] for y in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if i > j:
                output_matrix[i][j] = output_matrix[j][i]
            else:
                output_matrix[i][j] = math.sqrt(
                    ((nodelist[j][0] - nodelist[i][0])**2) + ((nodelist[j][1] - nodelist[i][1])**2))

    return output_matrix

# The main entry point for this module


def main():
    # Start location
    home = 2
    # Max iterations
    max_iterations = 1000
    # Distances in miles between cities, same indexes (i, j) as in the cities array
    file_name = sys.argv[1]
    matrix = get_file_input(file_name)
    N = np.shape(matrix)[0]
    city_indexes = list(range(N))
    cities_int = range(N)
    cities = [str(x) for x in cities_int]

    # state_population as the first approach to Random Restart
    state_population = get_best_solution_by_population(matrix, home, city_indexes, 200)
    # state_distance as the second approach to Random Restart
    state_distance = get_best_solution_by_distance(matrix, home)

    print('-- Start Iterations ... --')
    print("Running...")

    start = timer()
    [state, fitness] = restart_random(matrix, home, state_distance, max_iterations, 0.01)

    # open the file in the write mode
    f = open('solution.csv', 'w', newline='')

    # create the csv writer
    writer = csv.writer(f)

    print('-- HC solution --')
    print(cities[home], end='')
    writer.writerow(cities[home])
    for i in range(0, len(state.route)):
        print(' -> ' + cities[state.route[i]], end='')

    print(' -> ' + cities[home], end='')
    arr = np.array(state.route)
    newarr = arr.reshape(len(state.route), 1)
    writer.writerows(newarr)
    # writer.writerow(cities[home])
    f.close()
    end = timer()
    print('\n\nDistance: {0} miles'.format(state.distance))
    print()
    print('Time: ', round(end-start, 5), ' sec(s)')



if __name__ == "__main__":
    main()
