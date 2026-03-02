# Contact-GraspNet Integration (CPU-Only / Pure TF)

This directory contains the integration of NVIDIA's **Contact-GraspNet** into the TidyBot2 ROS 2 pipeline. 

## 🚀 Overview of Changes

To support CPU-only inference and modern Python environments (Python 3.10+), the following modifications were made:

1.  **Pure TensorFlow Implementation**: The original PointNet++ "TF Ops" (custom C++ kernels) were GPU-only and required CUDA 10.1. We replaced `tf_sampling.py`, `tf_grouping.py`, and `tf_interpolate.py` with pure TensorFlow implementations.
    *   **Benefit**: No compilation required. Works on any CPU. No CUDA/GPU driver dependencies.
    *   **Trade-off**: Slightly slower inference compared to the optimized CUDA kernels.
2.  **ROS 2 Wrapper**: Created `contact_graspnet_node.py` which interfaces with the `GraspEstimator` and handles ROS 2 `PointCloud2` and `PoseStamped` messages.
3.  **Launch Integration**: Created `contact_graspnet.launch.py` to automate point cloud generation and model startup.

---

## 🛠 Reproduction Guide

To set this up on a new installation or another device, follow these steps:

### 1. Clone the Repository & Submodules
Ensure you clone with submodules to get the `contact_graspnet` code:
```bash
git clone --recursive <your-repo-url>
# OR if already cloned:
git submodule update --init --recursive
```

### 2. Environment Setup (using `uv`)
Install the core deep learning dependencies:
```bash
uv pip install tensorflow scipy trimesh pyyaml
```
*Note: TensorFlow 2.18+ is recommended for Python 3.10+ compatibility.*

### 3. Download Model Weights
1.  Download the pretrained models from [this Drive folder](https://drive.google.com/drive/folders/1tBHKf60K8DLM5arm-Chyf7jxkzOr5zGl).
2.  Extract the `scene_test_2048_bs3_hor_sigma_001` folder into:
    `ros2_ws/src/tidybot_bringup/scripts/vision-manipulation/contact_graspnet/checkpoints/`

### 4. Verification
Run the simulation launch file to ensure everything loads:
```bash
# In terminal 1 (Simulation)
ros2 launch tidybot_bringup sim.launch.py

# In terminal 2 (Contact-GraspNet)
ros2 launch tidybot_bringup contact_graspnet.launch.py use_sim:=true
```

---

## 📝 Important Notes for Developers

### How to Reproduce the "Pure TF" Patch
If you ever need to re-apply the patches to a fresh clone of the original NVIDIA repository, the files modified are:
*   `pointnet2/tf_ops/sampling/tf_sampling.py`
*   `pointnet2/tf_ops/grouping/tf_grouping.py`
*   `pointnet2/tf_ops/3d_interpolation/tf_interpolate.py`

These files now use `tf.while_loop`, `tf.gather_nd`, and `tf.nn.top_k` instead of trying to load `tf_sampling_so.so`.

### Troubleshooting
*   **Memory Usage**: Pure TF implementations of Farthest Point Sampling (FPS) can be memory-intensive. If the node crashes with an OOM (Out of Memory) error, reduce the `num_input_points` in the `config.yaml`.
*   **Performance**: If inference is too slow, ensure `oneDNN` is enabled (TensorFlow usually does this by default on modern CPUs).
