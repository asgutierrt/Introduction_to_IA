from os import path, getcwd
import subprocess
import platform
from setuptools import setup, find_packages
from setuptools.command.install import install

# Specify project requirements
with open('requirements.txt', encoding='utf-16') as f:
    requirements = f.read().splitlines()

# Define a custom command for setup.py
class CustomInstallCommand(install):
    def run(self, venv_name = 'ana_venv'):
        # Define the virtual environment directory
        venv_dir = path.join(getcwd(), venv_name)
        activate_script = ["source", path.join(venv_dir, "bin", "activate")]
        pip_exe = path.join(venv_dir, "Scripts", "pip3.exe")
        python_exe=path.join(venv_dir, "Scripts", "python.exe")

        if platform.system()=="Windows": 
            #venv_dir = venv_dir.replace("/", "\\")
            activate_script = [path.join(venv_dir, "Scripts", "activate.bat")]
            #pip_exe = pip_exe.replace("/", "\\")
        
        # Create a virtual environment
        subprocess.run(["python", "-m", "venv", venv_dir], check=True)
        # Activate the virtual environment
        subprocess.run(activate_script, check=True)
        # Install project requirements
        subprocess.run([pip_exe, "install", "-r", "requirements.txt"], check=True)
        # Continue with the standard installation
        super().run()

setup(
    name='unsupervised',
    version='0.1.0',
    author='anasofia_gt',
    description='A short description of your project',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.7',
    cmdclass={'install': CustomInstallCommand},
)