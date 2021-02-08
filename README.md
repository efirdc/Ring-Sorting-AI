### Results

Video results are viewable at the following links:  
n=5 https://www.youtube.com/watch?v=6XSouVbOamo&feature=youtu.be
n=9 https://www.youtube.com/watch?v=w8gSbvMyN5Q&feature=youtu.be

#### Setup
The command `source setup_env.sh` will create a python virtual environment, install prerequisites, 
and then activate the virtual environment.   
The created environment is located at `../air_berlin_group1_env/`

#### Usage
Use `source activate_env.sh` to activate the existing environment if it is not already active.   
Then use `python run_air_berlin.py <num_disks>` to run the program.  
Use `deactivate` to exit the active virtual environment if needed.  

Example
```
(air_berlin_group1_env) $ python run_air_berlin.py 10 < AB10.sample > AB10.sol
(air_berlin_group1_env) $ ./ABC 10 AB10.check < AB10.sol
Move 1 OK
Move 2 OK
Move 3 OK
SOLUTION OK
```

#### Directories and files
- `./air_berlin/`
    + `expanded.py`: Data structure for closed nodes.
    + `fringe.py`: Data structures for the fringe.
    + `game.py`: Functions for game logic and game data structures.
    + `heuristics.py`: Different heuristics we tried.
    + `min_max_heap.py`: `MinMaxHeap` class, used for `MinMaxFringe`.
    + `search.py`: Search procedures.
    + `tests.py`:  Some tests for heuristics and `Expanded`.
    + `utils.py`: Misc. functions used for debugging/visualization.
    + `visualizations.py`: A class for plotting game states and generating video frames.
- `run_air_berlin.py`: Main program.
- `cory_notebook.ipynb`: Messy notebook for experiments.
- `setup_env.sh`: Sets up the virtual environment.
- `activate_env.sh`: Activate the virtual environment.
- `README.md`: You are here.
