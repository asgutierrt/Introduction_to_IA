# Basic imports for creating the virtual environment
from venv import create
from os.path import join, expanduser, abspath
from os import getcwd
from subprocess import run
import platform

# virtual environment directory
dir = join(getcwd(), "my-env")

activate_script = ["source", join(dir, "bin", "activate")]
if platform.system()=="Windows": 
   activate_script = [join(dir, "Scripts", "activate.bat")]

# Creation of the virtual environment
create(dir, with_pip=True)
# Activation of the virtual environment
run(activate_script)
# Updating of pip
run([join(dir,'Scripts','python.exe'), "-m", "pip", "install", "--upgrade", "pip"])
# Installation of the necessary packages according to 'requirements.txt'
run([join(dir,'Scripts','pip.exe'), "install", "-r", abspath("src/requirements.txt")], cwd=dir)