This search-based AI solver was developed over a 2-3 week period for our first group project in the AI course at MacEwan university.

#### Results

<a href="https://www.youtube.com/watch?v=jye2id7gr3c">
    <img alt="asdfasdf" src="https://i.imgur.com/3H9Cj41.png" width="75%">
</a>

#### Game Description 
The object of this game is to sort the dots into a rainbow colour pattern. The potential moves at each iteration are shown with thick white lines, while the chosen move is a red line. The white dot is always allowed to swap with the directly adjacent dots, but each location in the ring is also randomly assigned a jump size J of 1, 2, 3, or 4 at the start of the game. The white dot is also allowed to swap with colored dots that are J spots away from it. So if J=1 then there are no new moves. However, if J is 2, 3, or 4 then there are two extra possible moves at that location.

A value n is chosen to determine the size of the game. For example in the first game in the video n=5. This means there will be 5 different colors, each with 5 dots. When you include the white dot then there are 5\*5 + 1 = 26 dots total in the ring. In the second game of the video n=9 so there are 9\*9 + 1 = 82 dots in the ring.

#### Method

We implemented a number of heuristic based search algorithms including A\*, weighted A\*, beam search, SMA\*, and IDA\*.

More details can be found in our report:  
https://drive.google.com/file/d/1C5Dvy-799mfjAU4lcljcCjWvq8kI5DWi/view?usp=sharing

The visualization was made with matplotlib. In short it's a scatter plot with a rainbow colormap, black background, and bezier curves to show moves. The frames were saved to the disk as .png files and then combined with ffmpeg.

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
