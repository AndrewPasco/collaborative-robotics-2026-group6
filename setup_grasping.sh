#!/bin/bash
# =============================================================================
# TidyBot2 Grasping Setup Script
# Installs dependencies for PointNetGPD (CPU-based grasp detection)
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_status() { echo -e "${GREEN}[*]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' not found. Please run ./setup.sh first to install base tools."
    exit 1
fi

print_status "Setting up Grasping Environment..."

# 1. Install System Dependencies
print_status "Installing system dependencies (pcl-tools)..."
sudo apt update && sudo apt install -y pcl-tools

# 2. Install Python Dependencies into existing .venv
# We assume .venv exists from main setup
if [ ! -d ".venv" ]; then
    print_warning ".venv not found! Running uv sync..."
    uv sync
fi

POINTNET_ROOT="$SCRIPT_DIR/ros2_ws/src/third_party/pointnet_gpd"

if [ -d "$POINTNET_ROOT" ]; then
    print_status "Installing PointNetGPD python requirements..."
    
    # Base requirements
    uv pip install -r "$POINTNET_ROOT/requirements.txt"
    
    # PyTorch (CPU) and Open3D
    print_status "Installing PyTorch (CPU) and Open3D..."
    uv pip install open3d torch torchvision --index-url https://download.pytorch.org/whl/cpu
    
    # Compatibility fixes for legacy dex-net code
    print_status "Applying version compatibility fixes..."
    uv pip install "matplotlib<3.6" "ruamel.yaml<0.18"
    
    # Vendored sub-packages (Editable install)
    print_status "Installing meshpy and dex-net..."
    uv pip install -e "$POINTNET_ROOT/meshpy"
    uv pip install -e "$POINTNET_ROOT/dex-net"
    
    print_status "Grasping environment setup complete!"
else
    echo -e "${RED}[ERROR]${NC} PointNetGPD source not found at $POINTNET_ROOT"
    exit 1
fi
