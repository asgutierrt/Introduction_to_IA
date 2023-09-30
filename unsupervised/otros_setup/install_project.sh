#!/bin/bash

# Create and activate the virtual environment
python -m venv my-venv  # Replace with your desired venv directory name
source my-venv/bin/activate

# Install project dependencies
pip install -r requirements.txt

# Install python 3.7
pip install python==3.7

# Start a new shell session with the activated environment
bash

# The project itself is already installed as part of the setup process
