#!/bin/bash

# Check CPU information
echo "CPU Information:"
lscpu

echo "--------------------------------------"

# Check RAM information
echo "RAM Information:"
free -h

echo "--------------------------------------"

# Check GPU Information
echo "GPU Information:"
# For NVIDIA GPUs (optional, uncomment if needed)
# nvidia-smi

# General method for GPU information
lspci | grep -E "VGA|3D|Display"

# Additional GPU info using glxinfo (optional, uncomment if needed)
# glxinfo | grep "Device:"