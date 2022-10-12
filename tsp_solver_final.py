import pprint
import sys
import math
import copy
import random
import csv
import numpy as np

from timeit import default_timer as timer

class State:
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

    # Update distance
    def update_distance(self, matrix, home):
        # Reset distance
        self.distance = 0
        # Keep track of departing city
        from_index = home
        # Loop all cities in the current route
        for i in range(len(self.route)):
            self.distance += matrix[from_index][self.route[i]]
            from_index = self.route[i]
        # Add the distance back to home
        self.distance += matrix[from_index][home]

    # Shuffle routes
    def shuffle_route(self, matrix, home):
        random.shuffle(self.route)
        self.update_distance(matrix, home)


class City:
    # Create a new city
    def __init__(self, index: int, distance: int):
        self.index = index
        self.distance = distance

    # Sort cities
    def __lt__(self, other):
        return self.distance < other.distance


def get_data_from_tsp_file(fn):
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
    # print("N: ", N)
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


def get_best_solution_by_distance(matrix: [], home: int):
    # Variables
    route = []
    from_index = home
    length = len(matrix) - 1
    # Loop until route is complete
    while len(route) < length:
        # Get a matrix row
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


def probability(p):
    return p > random.uniform(0.0, 1.0)


def exp_schedule(k=20, lam=0.005, limit=1000):
    return lambda t: (k * np.exp(-lam * t) if t < limit else 0)


def mutate(matrix: [], home: int, state: State, mutation_rate: float = 0.01):
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
    best_state = initial_state
    for t in range(sys.maxsize):
        T = schedule(t)
        if T == 0:
            return best_state
        neighbor = mutate(matrix, home, best_state, mutation_rate)
        delta_e = best_state.distance - neighbor.distance
        if delta_e > 0 or probability(np.exp(delta_e / T)):
            best_state = neighbor


def hill_climbing(matrix: [], home: int, initial_state: State, max_iterations: int, mutation_rate: float = 0.01):
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


def random_restart(algorithm, matrixofcities, home, initial_state, max_iterations, mutation_rate):
    state = initial_state.copy()
    Fitness = [0] * max_iterations
    count = 0
    algorithm = algorithm.lower()
    while count < max_iterations:
        if algorithm == 'simulated_annealing':
            best_state = simulated_annealing(
                matrixofcities, home, initial_state, mutation_rate)
        elif algorithm == 'hill_climbing':
            best_state = hill_climbing(
                matrixofcities, home, initial_state, max_iterations, mutation_rate)
        Fitness[count] = best_state.distance
        initial_state = best_state
        count += 1
        # print("Iteration: {0}, Current Fitness: {1}, Best Fitness: {2}"\
        #     .format(count, initial_state.distance, best_state.distance))

    return [best_state, Fitness]


def writeToFile(cities, home, state):
    f = open('solution.csv', 'w', newline='')

    # create the csv writer
    writer = csv.writer(f)

    print('\n\n-- Simulated Annealing solution --')
    print(cities[home], end='')
    writer.writerow(cities[home])
    for i in range(0, len(state.route)):
        print(' -> ' + cities[state.route[i]], end='')

    print(' -> ' + cities[home], end='')
    arr = np.array(state.route)
    newarr = arr.reshape(len(state.route), 1)
    writer.writerows(newarr)
    writer.writerow(cities[home])
    f.close()


def main():
    # Index of start location
    home = 2
    # Max iterations
    max_iterations = 1000
    # Distances in miles between cities, same indexes (i, j) as in the cities array
    file_name = sys.argv[1]
    co_ordinates, distancesMatrix = get_data_from_tsp_file(file_name)
    # pprint.pprint(distancesMatrix)

    # print("distancesMatrix: ",len(distancesMatrix[0]))

    N = np.shape(distancesMatrix)[0]
    # print("N: ", N)
    # return None
    city_indexes = list(range(N))
    cities_int = range(N)
    cities = [str(x) for x in cities_int]

    state1 = get_random_solution_from_population(distancesMatrix, home, city_indexes, 200)
    # print("state1: ", state1)

    state2 = get_best_solution_by_distance(distancesMatrix, home)
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


    print('\n\n-- Applying Random Restart on Hill Climbing ... --')
    start = timer()
    algorithm = "Hill_Climbing"
    [state, fitness] = random_restart(algorithm, distancesMatrix, home, state1, max_iterations, 0.01)
    print('\n\n-- Hill Climbing with Random Restart completed !!! --')
    end = timer()

    # write solution to csv file
    # writeToFile(cities, home, state)

    print('\n\nTotal distance: {0} miles'.format(state.distance))
    print()
    print('Time Taken: ', round(end-start, 5), ' seconds\n\n')



if __name__ == "__main__":
    main()