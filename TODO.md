# ACO Edge Detection - Remove Unnecessary Ants

## Plan

- [x] Analyze aco_edge.cpp to understand ant behavior
- [x] Add heuristic-biased spawning (ants start proportional to gradient)
- [x] Add early termination for ants in flat regions (MIN_ETA threshold)
- [x] Conditional pheromone deposit (only on edges)
- [x] Build and test

## Changes Summary

1. **Heuristic-biased spawning**: Build cumulative distribution from `eta` map so ants spawn on/near edges
2. **Early termination**: Ants die when `eta < MIN_ETA` instead of taking all `ANT_STEPS`
3. **Conditional deposit**: Only deposit pheromone when `eta[nidx] > MIN_ETA`
