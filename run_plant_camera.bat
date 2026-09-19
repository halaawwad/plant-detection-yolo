@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" live_plant_camera.py %*
