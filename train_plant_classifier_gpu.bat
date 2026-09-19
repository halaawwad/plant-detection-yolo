@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" train_plant_classifier.py %*
