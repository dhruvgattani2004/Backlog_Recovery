
@echo off
echo Setting up FedEx Backlog Recovery System...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

echo.
echo ✅ Setup completed successfully!
echo.
echo To run the system:
echo 1. Activate the virtual environment: .venv\Scripts\activate.bat
echo 2. Run the Streamlit app: streamlit run app_with_ml.py
echo.
echo For ML prediction setup:
echo 1. Place your Excel files in a folder
echo 2. Run: python calculate_rollover.py
echo 3. Run: python train_rollover_model.py
echo.
pause
