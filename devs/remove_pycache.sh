#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [-y] <path>"
    echo "  -y    Remove __pycache__ directories instead of just listing them"
    exit 1
}

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    usage
fi

# Initialize variables
REMOVE=false

# Parse options
while getopts ":y" opt; do
    case ${opt} in
        y )
            REMOVE=true
            ;;
        \? )
            usage
            ;;
    esac
done
shift $((OPTIND -1))

# Get the path argument
TARGET_PATH=$1

# Find and process __pycache__ directories
if $REMOVE; then
    find "$TARGET_PATH" -type d -name "__pycache__" -exec rm -rf {} +
    echo "All __pycache__ directories have been removed."
else
    find "$TARGET_PATH" -type d -name "__pycache__"
fi

