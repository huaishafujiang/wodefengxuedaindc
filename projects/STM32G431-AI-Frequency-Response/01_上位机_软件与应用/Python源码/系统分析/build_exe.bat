pip install pyinstaller
pyinstaller --onefile --windowed --add-data "system_analysis\data\diagnosis_knowledge_base.json;system_analysis\data" main.py
pause
