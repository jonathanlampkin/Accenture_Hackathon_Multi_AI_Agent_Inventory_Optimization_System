#!/bin/bash
# Run script for the Multi-Agent Inventory Optimization System

# Set the script to exit on first error
set -e

# Define colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Display header
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Multi-Agent Inventory Optimization System Runner Script   ${NC}"
echo -e "${BLUE}============================================================${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

echo -e "${GREEN}Setting up environment...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}No virtual environment found. Creating one...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created.${NC}"
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Check if requirements are installed
if [ ! -f "venv/.requirements_installed" ]; then
    echo -e "${YELLOW}Installing requirements...${NC}"
    pip install -r requirements.txt
    touch venv/.requirements_installed
    echo -e "${GREEN}Requirements installed.${NC}"
else
    echo -e "${GREEN}Requirements already installed.${NC}"
fi

# Check for data files and set them up
echo -e "${GREEN}Setting up data files...${NC}"
python3 setup_data.py

# Check if setup was successful
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: Some data files may be missing. The system may not work correctly.${NC}"
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Exiting as requested.${NC}"
        exit 1
    fi
fi

# Parse command line arguments
OPTIMIZE_FOR="balanced"
SIMPLE_MODE=""
FOCUS_PRODUCT=""
FOCUS_STORE=""
ITERATIONS="5"
OUTPUT_DIR=""
SUMMARY_ONLY=""

# Function to display help
function show_help {
    echo -e "\n${BLUE}Usage:${NC}"
    echo -e "  ./run_system.sh [options]"
    echo -e "\n${BLUE}Options:${NC}"
    echo -e "  -h, --help              Show this help message and exit"
    echo -e "  -s, --simple            Run in simple mode without advanced ML models"
    echo -e "  -o, --optimize VALUE    Optimization target: cost, availability, balanced (default: balanced)"
    echo -e "  -p, --product ID        Focus on specific product ID"
    echo -e "  -t, --store ID          Focus on specific store ID"
    echo -e "  -i, --iterations NUM    Number of optimization iterations (default: 5)"
    echo -e "  -d, --output-dir DIR    Custom output directory"
    echo -e "  --summary               Only display data summary without running optimization"
    echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_help
      exit 0
      ;;
    -s|--simple)
      SIMPLE_MODE="--simple-mode"
      shift
      ;;
    -o|--optimize)
      OPTIMIZE_FOR="$2"
      shift 2
      ;;
    -p|--product)
      FOCUS_PRODUCT="--product-id $2"
      shift 2
      ;;
    -t|--store)
      FOCUS_STORE="--store-id $2"
      shift 2
      ;;
    -i|--iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    -d|--output-dir)
      OUTPUT_DIR="--output-dir $2"
      shift 2
      ;;
    --summary)
      SUMMARY_ONLY="--summary-only"
      shift
      ;;
    *)
      echo -e "${RED}Error: Unknown option $1${NC}"
      show_help
      exit 1
      ;;
  esac
done

# Build the command
CMD="python3 main.py $SIMPLE_MODE --optimize-for $OPTIMIZE_FOR --iterations $ITERATIONS $FOCUS_PRODUCT $FOCUS_STORE $OUTPUT_DIR $SUMMARY_ONLY"

# Run the system
echo -e "${GREEN}Running Multi-Agent Inventory Optimization System...${NC}"
echo -e "${YELLOW}Command: $CMD${NC}\n"

eval $CMD

# Deactivate virtual environment
deactivate

echo -e "\n${GREEN}Run completed. Exiting...${NC}"
exit 0 