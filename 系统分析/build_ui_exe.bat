@echo off
setlocal
cd /d "%~dp0"
echo [1/4] Installing base dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo [2/4] Installing PySide6 UI dependencies...
python -m pip install -r requirements-ui.txt
if errorlevel 1 goto :failed

echo [3/4] Installing PyInstaller...
python -m pip install pyinstaller
if errorlevel 1 goto :failed

echo [4/4] Building Windows executable...
pyinstaller --noconfirm --clean --onefile --windowed ^
  --name "STM32G431_AI_Frequency_Response" ^
  --add-data "diagnosis_knowledge_base.json;." ^
  main_window.py
if errorlevel 1 goto :failed

echo.
echo Build completed:
echo "%~dp0dist\STM32G431_AI_Frequency_Response.exe"
echo.
pause
exit /b 0

:failed
echo.
echo Build failed. Please check the messages above.
pause
exit /b 1
