#### Setup
The command `source setup_env.sh` will create a python virtual environment, install prerequisites, and then activate the virtual environment. 
The created environment is located at `../air_berlin_group1_env/`

#### Usage
Use `source activate_env.sh` to activate the existing environment if it is not already active.
```
./run_air_berlin.py < AB10.sample > AB10.sol
```
In this case, `AB10.sample` is the input file, and`AB10.sol` is the output file.
Use `deactivate` to exit the active virtual environment if needed.

#### Directories and files
1. air_berlin
    + `expanded.py`: `Expanded` class
    + `fringe.py`: `Fringe`, `BeamFringe`, and `MinMaxFringe` classes.
    + `game.py`: contains Air Berlin game functions that start, evaluate and control the game.
    + `heuristics.py`: contains the different attempted heuristics.
    + `min_max_heap.py`: `MinMaxHeap` class, used for `MinMaxFringe`.
    + `search.py`: contains variations of search functions.
    + `tests.py`:  tests if a hueristic is not admissable/consistent.
    + `utils.py`: different functions used to help with debugging/visualization.
    + `visualizations.py`: `RingPlot` class. 
2. solutions: contains saved solutions to random examples. 
3. `run_air_berlin.py`: takes in the input for the small and large discs and prints the solution found.