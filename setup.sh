#!/bin/bash
echo "Setting up FedEx Backlog Recovery System..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

echo "✅ Python found"
echo

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo
echo "✅ Setup completed successfully!"
echo
echo "To run the system:"
echo "1. Activate the virtual environment: source .venv/bin/activate"
echo "2. Run the Streamlit app: streamlit run app_with_ml.py"
echo
echo "For ML prediction setup:"
echo "1. Place your Excel files in a folder"
echo "2. Run: python calculate_rollover.py"
echo "3. Run: python train_rollover_model.py"
echo
