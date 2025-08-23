#!/bin/bash

# Script to setup Isaac Sim extension symlinks
# This script creates symlinks from the project's Isaac Sim extensions to Isaac Sim's extension directory

set -e

# Get the current project directory
PROJECT_DIR=${DP_PROJECT_ROOT:-$(pwd)}

# Source .env file if it exists to get ISAAC_SIM_PATH
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a  # automatically export all variables
    source "$PROJECT_DIR/.env"
    set +a  # turn off automatic export
    echo "Sourced environment variables from .env"
fi

# Check if ISAAC_SIM_PATH is set - error out immediately if not
if [ -z "$ISAAC_SIM_PATH" ]; then
    echo "Error: ISAAC_SIM_PATH is not set"
    echo "Please set ISAAC_SIM_PATH in your .env file, for example:"
    echo "  ISAAC_SIM_PATH=\$HOME/Applications/isaac-sim-4.5.0"
    exit 1
fi

# Expand environment variables in the path
ISAAC_SIM_PATH=$(eval echo "$ISAAC_SIM_PATH")

# Check if Isaac Sim installation exists
if [ ! -d "$ISAAC_SIM_PATH" ]; then
    echo "Error: Isaac Sim installation not found at $ISAAC_SIM_PATH"
    echo "Please check your ISAAC_SIM_PATH setting in .env file"
    exit 1
fi

# Create extsUser directory if it doesn't exist
mkdir -p "$ISAAC_SIM_PATH/extsUser"

# Function to create symlink for an extension
create_extension_symlink() {
    local ext_name=$1
    local source_dir=$2
    
    local symlink_path="$ISAAC_SIM_PATH/extsUser/$ext_name"
    
    # Check if symlink already exists
    if [ -L "$symlink_path" ]; then
        echo "Symlink already exists at $symlink_path"
        echo "Removing existing symlink..."
        rm "$symlink_path"
    fi
    
    # Create symlink from Isaac Sim to your extension
    ln -s "$PROJECT_DIR/$source_dir" "$symlink_path"
    echo "✓ Created symlink for $ext_name"
    echo "  Symlink: $symlink_path"
    echo "  Points to: $PROJECT_DIR/$source_dir"
}

# Create symlinks for extension
create_extension_symlink "isaacsim.armsdg" "diffusion_policy/env/isaacsim.armsdg"

echo ""
echo "✓ All Isaac Sim extension symlinks created successfully"
echo "You can now enable the 'isaacsim.armsdg' extension in Isaac Sim"