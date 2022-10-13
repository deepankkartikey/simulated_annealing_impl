### Assignment-1

### CSI-5137 - Software Verification and Testing

- Hill climbing algorithm is a local search algorithm which continuously moves in the direction of increasing elevation/value to find the peak of the mountain or best solution to the problem. It terminates when it reaches a peak value where no neighbor has a higher value.

- An optimization problem is one that has many solutions and each solution has a different score. The goal is to find the best score.

- In a `State`,
- A 'route' is the order of cities, the salesman tries to visit.
- 'distance' in a state is length of the path taken by the salesman.

- Each State has neighbor state which can be better or worse than the current state.

- In Hill Climbing, we always try to move to a better state by checking left and right neighbor of current state. Check current with left and right and move to whichever is better.

- Problem of local maxima/optima arises in Hill Climbing search. When current state has no better neighbor (left or right) then it gets stuck at local maxima. Hence, most optimal solution is not found.

- Simulated Annealing - a way to select worse neighbor based on some probability metric so that eventually global maxima can be reached.

- Probability is based on loss and temperature. e^ (-loss/temperature)
  - Loss is how much worse a neighbor is as compared to current state.
  - Temperature is value representing current temperature
  - Higher the loss, lower is the probability of moving to that state.
  - Higher the temperature, greater the probability of taking that state.
- Repeated iterations to be called on check with slightly lower temperature ('cooling_rate')

- With very high temperature (1000) and infinitely slow cooling rate (0.99), we will always find global maxima.
