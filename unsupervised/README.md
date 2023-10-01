# Project structure
1. requirements.txt has all the required packages to run the project.
2. run ```pip install .``` on this folder to create a new environment with all requirements installed.
    - currently, the automatic environment creation only works on Windows.

# Recreating report files
1. run main.py from the src folder.
    - main module receives a data file from the data folder as input.
    - place the data file in the data folder and change the input file name in main.py.

2. output: main returns several classification matrix in a .txt format.

# Additional features and interactive figures.
1. figures of 3D datapoints colored by class (displaying first 3 dimensions) [.html]
2. distance matrix figures created with each norm 
3. run notebook on Google Colab for interactive in-notebook visualizations
