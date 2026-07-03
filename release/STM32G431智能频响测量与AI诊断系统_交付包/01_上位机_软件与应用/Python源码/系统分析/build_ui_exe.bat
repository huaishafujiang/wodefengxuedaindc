@echo off
setlocal
cd /d "%~dp0"
echo [1/3] Installing base dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo [2/3] Installing PyInstaller...
python -m pip install pyinstaller
if errorlevel 1 goto :failed

echo [3/3] Building Windows executable from main.py...
pyinstaller --noconfirm --clean --onefile --windowed ^
  --name "STM32G431_AI_Frequency_Response" ^
  --add-data "system_analysis\data\diagnosis_knowledge_base.json;system_analysis\data" ^
  --icon "assets\icons\frequency_response_icon.ico" ^
  main.py
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
