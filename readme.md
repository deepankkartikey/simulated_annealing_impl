## SIMULATED ANNEALING Implementation

#### Simulated Annealing - a way to select worse neighbor based on some probability metric so that eventually global maxima can be reached.

#### How to execute ?
`python3 tsp_solver.py "dataset_name"`
- tsp_solver.py - contains the logic for solving a travelling salesman problem
- dataset_name - file name of dataset like a280.tsp , pr76.tsp

- once the execution is complete a solution.csv file is generated containing travelled path of the best solution

- Probability is based on loss and temperature. e^ (-loss/temperature)
  - Loss is how much worse a neighbor is as compared to current state.
  - Temperature is value representing current temperature
  - Higher the loss, lower is the probability of moving to that state.
  - Higher the temperature, greater the probability of taking that state.
- Repeated iterations to be called on check with slightly lower temperature ('cooling_rate')

- With very high temperature (1000) and infinitely slow cooling rate (0.99), we will always find global maxima.
