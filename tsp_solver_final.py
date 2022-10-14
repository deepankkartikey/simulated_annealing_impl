import pprint
import sys
import math
import copy
import random
import csv
import numpy as np

from timeit import default_timer as timer

class State:
    """ A state is a representation of route from one city to other """
    def __init__(self, route: [], distance: int = 0):
        self.route = route
        self.distance = distance

    def __eq__(self, other):
        for i in range(len(self.route)):
            if (self.route[i] != other.route[i]):
                return False
        return True

    def __lt__(self, other):
        return self.distance < other.distance

    def __repr__(self):
        return ('({0},{1})\n'.format(self.route, self.distance))

    def copy(self):
        return State(self.route, self.distance)

    def deepcopy(self):
        return State(copy.deepcopy(self.route), copy.deepcopy(self.distance))

    def update_distance(self, matrix, home):
        """ calculates distance of current state """
        self.distance = 0
        from_index = home
        for i in range(len(self.route)):
            self.distance += matrix[from_index][self.route[i]]
            from_index = self.route[i]
        self.distance += matrix[from_index][home]

    def shuffle_route(self, matrix, home):
        """ brings randomness in the state """
        random.shuffle(self.route)
        self.update_distance(matrix, home)


class City:
    def __init__(self, index: int, distance: int):
        self.index = index
        self.distance = distance

    def __lt__(self, other):
        return self.distance < other.distance


def get_data_from_tsp_file(fn):
    """ Reads co_ordinates from the tsp file. \
        Calculates ditance matrix between cities in the dataset. \
        Returns distance matrix and co_ordinates dictionary """
    my_file = open(fn, 'r')

    NAME = my_file.readline().strip().split()[1]  # NAME
    TYPE = my_file.readline().strip().split()[1]  # TYPE
    COMMENT = my_file.readline().strip()  # COMMENT
    DIMENSION = my_file.readline().strip().split()[-1]  # DIMENSION
    # print("DIMENSION: ", DIMENSION)
    EDGE_WEIGHT_TYPE = my_file.readline().strip().split()[
        1]  # EDGE_WEIGHT_TYPE
    my_file.readline()

    distancesMatrix = []
    distancesList = []
    co_ordinates = {}
    N = int(DIMENSION)
    for i in range(0, N):
        x, y = my_file.readline().strip().split()[1:]
        co_ordinates[i+1] = (float(x), float(y))
        distancesList.append([float(x), float(y)])
    # pprint.pprint(co_ordinates)
    my_file.close()

    # Calculation of Matrix of cities' distances
    distancesMatrix = [[0 for x in range(N)] for y in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if i > j:
                distancesMatrix[i][j] = distancesMatrix[j][i]
            else:
                distancesMatrix[i][j] = math.sqrt(
                    ((distancesList[j][0] - distancesList[i][0])**2)
                    + ((distancesList[j][1] - distancesList[i][1])**2))

    # pprint.pprint(distancesMatrix)
    return co_ordinates, distancesMatrix


def get_random_solution_from_population(matrix: [], home: int, city_indexes: [], size: int):
    """ Returns best solution State from a randomly generated population \
        created using all cities indexes except the home city. """
    cities = city_indexes.copy()
    cities.pop(home)
    population = []
    for i in range(size):
        random.shuffle(cities)
        state = State(cities[:])
        state.update_distance(matrix, home)
        population.append(state)
    population.sort()
    return population[0]


def probability(p):
    """ Returns random probability to be used for Simulated Annealing """
    return p > random.uniform(0.0, 1.0)


def exp_schedule(k=20, lam=0.005, limit=1000):
    """ Calculates Cooling Schedule in Simulated Annealing """
    return lambda t: (k * np.exp(-lam * t) if t < limit else 0)


def mutate(matrix: [], home: int, state: State, mutation_rate: float = 0.01):
    """ Returns a mutated state based on mutation rate to further optimize solution """
    mutated_state = state.deepcopy()
    for i in range(len(mutated_state.route)):
        if random.random() < mutation_rate:
            j = int(random.random() * len(state.route))
            city_1 = mutated_state.route[i]
            city_2 = mutated_state.route[j]
            mutated_state.route[i] = city_2
            mutated_state.route[j] = city_1
    mutated_state.update_distance(matrix, home)
    return mutated_state


def simulated_annealing(matrix: [], home: int, initial_state: State, mutation_rate: float = 0.01, schedule=exp_schedule()):
    """ Core logic for Simulated Annealing along with cooling schedule and mutation to get a neighbor """
    best_state = initial_state
    for t in range(sys.maxsize):
        T = schedule(t)
        if T == 0:
            return best_state
        neighbor = mutate(matrix, home, best_state, mutation_rate)
        delta_e = best_state.distance - neighbor.distance
        if delta_e > 0 or probability(np.exp(delta_e / T)):
            best_state = neighbor


def random_restart(algorithm, matrixofcities, home, initial_state, max_iterations, mutation_rate):
    """ Returns best_state based on fitness to reduce possibility of local maxima/optima """
    state = initial_state.copy()
    Fitness = [0] * max_iterations
    count = 0
    algorithm = algorithm.lower()
    while count < max_iterations:
        best_state = simulated_annealing(
                matrixofcities, home, initial_state, mutation_rate)
        Fitness[count] = best_state.distance
        initial_state = best_state
        count += 1
        # print("Iteration: {0}, Current Fitness: {1}, Best Fitness: {2}"\
        #     .format(count, initial_state.distance, best_state.distance))
    return [best_state, Fitness]


def writeToFile(cities, home, state):
    """ Write Solution to CSV file"""
    f = open('solution.csv', 'w', newline='')
    writer = csv.writer(f)

    print('-- Simulated Annealing solution --')
    print(cities[home], end='')
    writer.writerow(cities[home])
    for i in range(0, len(state.route)):
        print(' -> ' + cities[state.route[i]], end='')

    print(' -> ' + cities[home], end='')
    arr = np.array(state.route)
    newarr = arr.reshape(len(state.route), 1)
    writer.writerows(newarr)
    writer.writerow(cities[home])
    print()
    f.close()


def main():
    """Driver function for tsp_solver. \
        Gets dataset file_name from command line and calls other utility functions"""
    home = 2
    max_iterations = 1000
    file_name = sys.argv[1]
    co_ordinates, distancesMatrix = get_data_from_tsp_file(file_name)

    # pprint.pprint(distancesMatrix)
    # print("distancesMatrix: ",len(distancesMatrix[0]))

    N = np.shape(distancesMatrix)[0]
    city_indexes = list(range(N))
    cities_int = range(N)
    cities = [str(x) for x in cities_int]

    state1 = get_random_solution_from_population(distancesMatrix, home, city_indexes, 200)
    # print("state1: ", state1)

    # state2 = get_best_solution_by_distance(distancesMatrix, home)
    # print("state2: ", state2)

    print('\n\n-- Applying Random Restart on Simulated Annealing ... --')
    start = timer()
    algorithm = "Simulated_Annealing"
    [state, fitness] = random_restart(algorithm, distancesMatrix, home, state1, max_iterations, 0.01)
    print('\n\n-- Simulated Annealing with Random Restart completed !!! --')
    end = timer()
    print('\n\nTotal distance: {0} miles'.format(state.distance))
    print()
    print('Time Taken: ', round(end-start, 5), ' seconds\n\n')

    writeToFile(cities, home, state)


if __name__ == "__main__":
    main()