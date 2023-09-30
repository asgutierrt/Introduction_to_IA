@echo off
setlocal

rem Create and activate the virtual environment. Replace with desired venv name
python -m venv my-venv

rem Activate the virtual environment
call my-venv\Scripts\activate

rem Install project dependencies
pip install -r requirements.txt

rem Install python 3.7
pip install python==3.7

rem Start a new shell session with the activated environment
cmd.exe /K

endlocal
