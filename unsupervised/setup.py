import os
import subprocess
import platform
from setuptools import setup, find_packages
from setuptools.command.install import install

# Specify project requirements
with open('requirements.txt', encoding='utf-16') as f:
    requirements = f.read().splitlines()

# Define a custom command for setup.py
class CustomInstallCommand(install):
    def run(self):
        # Define the virtual environment directory
        venv_dir = 'ana_venv'  # Replace with your desired venv directory name
        venv_dir = os.path.join(os.getcwd(), venv_dir)
        activate_script = ["source", os.path.join(venv_dir, "bin", "activate")]
        pip_exe = os.path.join(venv_dir, "Scripts", "pip3.exe")

        if platform.system()=="Windows": 
            venv_dir = venv_dir.replace("/", "\\")
            activate_script = [os.path.join(venv_dir, "Scripts", "activate.bat").replace("/", "\\")]
            pip_exe = pip_exe.replace("/", "\\")
        
        # Create a virtual environment
        subprocess.run(["python", "-m", "venv", venv_dir], check=True)
        # Activate the virtual environment
        subprocess.run(activate_script, shell=True, check=True)
        # Install project requirements
        subprocess.run([pip_exe, "install", "-r", "requirements.txt"], check=True)

        # Continue with the standard installation
        super().run()

setup(
    name='unsupervised',
    version='0.1.0',
    description='A short description of your project',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.7',
    cmdclass={'install': CustomInstallCommand},
)