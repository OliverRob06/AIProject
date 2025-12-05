@echo off
title Dependency Installer
echo =================================================
echo Found requirements.txt. Installing libraries now...
echo This may take a while depending on internet speed.
echo =================================================
echo.

:: This command runs the installation
pip install -r requirements.txt

echo.
echo =================================================
echo Installation Complete! 
echo You can now run the Python script.
echo =================================================
pause