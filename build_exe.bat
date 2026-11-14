@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto failed
".venv\Scripts\python.exe" -m pip install pyinstaller
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --onedir --windowed --name OCTLens --collect-data customtkinter app.py
if errorlevel 1 goto failed
copy /y best_model.pth "dist\OCTLens\best_model.pth" >nul
if errorlevel 1 goto failed
echo Ready: dist\OCTLens\OCTLens.exe
echo Keep the whole OCTLens folder when moving the application.
pause
exit /b 0
:failed
echo Build failed. Run setup_env.bat and check the error above.
pause
exit /b 1
