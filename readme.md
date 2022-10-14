### Assignment-1

### CSI-5137 - Software Verification and Testing

- Simulated Annealing - a way to select worse neighbor based on some probability metric so that eventually global maxima can be reached.

- Probability is based on loss and temperature. e^ (-loss/temperature)
  - Loss is how much worse a neighbor is as compared to current state.
  - Temperature is value representing current temperature
  - Higher the loss, lower is the probability of moving to that state.
  - Higher the temperature, greater the probability of taking that state.
- Repeated iterations to be called on check with slightly lower temperature ('cooling_rate')

- With very high temperature (1000) and infinitely slow cooling rate (0.99), we will always find global maxima.
