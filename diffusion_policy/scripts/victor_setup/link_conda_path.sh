#!/bin/bash

# Script to setup PYTHONPATH for robotool conda environment
# This script creates both activation and deactivation scripts

set -e

# Get the current project directory (where this script is run from)
PROJECT_DIR=${DP_PROJECT_ROOT:-$(pwd)}

# Get conda environment path
CONDA_PREFIX_PATH=$(conda info --base)/envs/robodiff_vic

# Check if conda environment exists
if [ ! -d "$CONDA_PREFIX_PATH" ]; then
    echo "Error: Conda environment 'robodiff_vic' not found at $CONDA_PREFIX_PATH"
    echo "Please create the environment first with: mamba env create -f env.yaml"
    exit 1
fi

# Create conda activation/deactivation script directories
mkdir -p "$CONDA_PREFIX_PATH/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX_PATH/etc/conda/deactivate.d"

echo "Setting up PYTHONPATH for project directory: $PROJECT_DIR"

# Create PYTHONPATH activation script
cat > "$CONDA_PREFIX_PATH/etc/conda/activate.d/pythonpath.sh" << 'ACTIVATION_EOF'
#!/bin/bash

# Add project directory to PYTHONPATH
PROJECT_DIR="PROJECT_DIR_PLACEHOLDER"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Source .env file if it exists in the project directory
if [ -f "$PROJECT_DIR/.env" ]; then
    # Check if we've already sourced this .env file to avoid duplicates
    if [[ ! "$SOURCED_ENV_FILES" == *"$PROJECT_DIR/.env"* ]]; then
        set -a  # automatically export all variables
        source "$PROJECT_DIR/.env"
        set +a  # turn off automatic export
        export SOURCED_ENV_FILES="$SOURCED_ENV_FILES:$PROJECT_DIR/.env"
        echo "Sourced environment variables from $PROJECT_DIR/.env"
    fi
fi
ACTIVATION_EOF

# Replace the placeholder with actual project directory
sed -i "s|PROJECT_DIR_PLACEHOLDER|$PROJECT_DIR|g" "$CONDA_PREFIX_PATH/etc/conda/activate.d/pythonpath.sh"

# Create PYTHONPATH deactivation script to clean up
cat > "$CONDA_PREFIX_PATH/etc/conda/deactivate.d/pythonpath.sh" << 'DEACTIVATION_EOF'
#!/bin/bash

# Remove project directory from PYTHONPATH
PROJECT_DIR="PROJECT_DIR_PLACEHOLDER"
export PYTHONPATH=$(echo "$PYTHONPATH" | sed "s|$PROJECT_DIR:||g" | sed "s|:$PROJECT_DIR||g" | sed "s|^$PROJECT_DIR$||g")

# Remove this .env file from the sourced files list
if [[ "$SOURCED_ENV_FILES" == *"$PROJECT_DIR/.env"* ]]; then
    export SOURCED_ENV_FILES=$(echo "$SOURCED_ENV_FILES" | sed "s|:$PROJECT_DIR/.env||g" | sed "s|$PROJECT_DIR/.env:||g" | sed "s|^$PROJECT_DIR/.env$||g")
fi
DEACTIVATION_EOF

# Replace the placeholder with actual project directory
sed -i "s|PROJECT_DIR_PLACEHOLDER|$PROJECT_DIR|g" "$CONDA_PREFIX_PATH/etc/conda/deactivate.d/pythonpath.sh"

# Make scripts executable
chmod +x "$CONDA_PREFIX_PATH/etc/conda/activate.d/pythonpath.sh"
chmod +x "$CONDA_PREFIX_PATH/etc/conda/deactivate.d/pythonpath.sh"

echo "✓ PYTHONPATH activation script created"
echo "✓ PYTHONPATH deactivation script created"
echo ""
echo "The project directory will be automatically added to PYTHONPATH when you activate the 'robotool' environment"
echo "and removed when you deactivate it."
echo ""
if [ -f "$PROJECT_DIR/.env" ]; then
    echo "✓ Found .env file - environment variables will be sourced automatically"
else
    echo "ℹ No .env file found. You can create one at $PROJECT_DIR/.env to set environment variables like:"
    echo "  ROS_IP=10.10.10.255"
    echo "  ROS_DOMAIN_ID=24"
fi
echo ""
echo "You can also manually edit the activation script for additional customization:"
echo "  $CONDA_PREFIX_PATH/etc/conda/activate.d/pythonpath.sh"
