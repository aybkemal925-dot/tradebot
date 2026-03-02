@echo off
setlocal
cd /d "%~dp0"
if not exist "config.json" (
  copy /Y "config.example.json" "config.json" >nul
)
where pythonw >nul 2>nul
if %ERRORLEVEL%==0 (
  start "" pythonw app.py
) else (
  start "" /min python app.py
)
endlocal
