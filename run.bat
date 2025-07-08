
@echo off
echo Starting FedEx Backlog Recovery System...
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run the Streamlit app
echo Launching Streamlit application...
streamlit run app_with_ml.py

pause
