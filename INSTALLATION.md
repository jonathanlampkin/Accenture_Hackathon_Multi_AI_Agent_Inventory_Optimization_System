# Installation Instructions

This document provides detailed instructions for setting up the Multi-Agent Inventory Optimization System on different environments.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Setting Up the Environment

### 1. Install Python Dependencies

First, you need to install pip if it's not already installed:

```bash
# On Ubuntu/Debian
sudo apt update
sudo apt install python3-pip

# On CentOS/RHEL
sudo yum install python3-pip

# On macOS (with Homebrew)
brew install python3  # Includes pip3
```

Then, install the required packages:

```bash
pip3 install pandas==2.0.0 numpy==1.24.3 scikit-learn==1.3.0 matplotlib==3.7.1 seaborn==0.12.2 prophet==1.1.4 statsmodels==0.14.0 tensorflow==2.13.0 langchain==0.0.267 openai==0.28.0 tiktoken==0.4.0 pymilvus==2.3.0 faiss-cpu==1.7.4
```

### 2. Create Required Directories

The system expects certain directories to exist:

```bash
mkdir -p output
mkdir -p models
```

### 3. Running with Limited Dependencies

If you are unable to install all the required packages, you can still run the system with minimal functionality by modifying `src/config.py` to disable certain features.

Edit `src/config.py` and set:

```python
USE_ADVANCED_MODELS = False
```

This will disable features that require TensorFlow, Prophet, and other advanced packages.

## Running in a Virtual Environment

For a more isolated environment, you can use Python's virtual environment feature:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the system
python src/main.py
```

## Docker Installation (Alternative)

If you prefer using Docker:

```bash
# Build the Docker image
docker build -t inventory-optimization .

# Run the container
docker run -it --rm -v $(pwd):/app inventory-optimization
```

## CrewAI with Ollama Integration

This project now supports using CrewAI with Ollama models for local LLM execution. Follow these steps to set it up:

1. Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Ollama setup script:
   ```bash
   ./setup_ollama.sh
   ```
   This script will:
   - Install Ollama if not already installed
   - Start the Ollama service
   - Pull the specified model from the .env file (default: llama2)

3. Test the CrewAI with Ollama integration:
   ```bash
   python test_crewai_ollama.py
   ```

4. If you want to use a different Ollama model, edit the `.env` file and change the `OLLAMA_MODEL` value.

## Troubleshooting

If you encounter issues with package installation:

1. Try installing packages one by one to identify problematic dependencies
2. Check for OS-specific requirements (e.g., some packages may need additional system libraries)
3. Consult the error messages for specific guidance

For specific package errors:

- **Prophet**: May require additional dependencies. Try `pip install prophet --no-binary prophet`
- **TensorFlow**: If installation fails, try the CPU-only version: `pip install tensorflow-cpu`
- **Matplotlib**: On Linux, you might need: `sudo apt-get install python3-matplotlib`

If you encounter issues with the CrewAI and Ollama integration:

1. Make sure Ollama is running: `ollama serve`
2. Check available models: `ollama list`
3. If a model is missing, pull it manually: `ollama pull llama2` (or your model of choice)
4. Verify your environment variables in the `.env` file
5. Check that the virtual environment is activated: `source venv/bin/activate` 