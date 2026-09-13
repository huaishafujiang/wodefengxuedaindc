@echo off
setlocal
cd /d "%~dp0"
echo Installing PySide6 UI runtime dependencies...
python -m pip install -r requirements-ui.txt
if errorlevel 1 goto :failed
echo.
echo UI runtime is ready. You can now run start_ui.bat.
pause
exit /b 0

:failed
echo.
echo UI dependency installation failed. Please check the messages above.
pause
exit /b 1
