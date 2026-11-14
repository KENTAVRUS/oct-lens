@echo off
setlocal
cd /d "%~dp0"
py -3.13 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Setup complete. Open run.bat to start OCT Lens.
pause
exit /b 0
:failed
echo Setup failed. Install Python 3.13 and check your internet connection.
pause
exit /b 1
