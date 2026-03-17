# PPRL-Mask

Official implementation of **PPRL-Mask: Reinforcement Learning for Adaptive Mask Selection in Semi-Supervised Segmentation**.

PPRL-Mask is a lightweight and plug-and-play reinforcement learning module for mask selection in teacher-student semi-supervised segmentation. It constructs policy states from semantic features, prediction uncertainty, and teacher-student disagreement, and optimizes the policy with a boundary-aware reward. The module is used only during training and introduces no inference-time overhead.

## Highlights
- Adaptive mask selection via reinforcement learning
- Boundary-aware reward for improved contour refinement
- Plug-and-play integration into teacher-student semi-supervised segmentation frameworks
- Validated on **ACDC (2D)** and **LA (3D)** datasets

## Recommended environment

The repository does not currently include a locked environment file.  
The following is a **recommended reproduction environment** consistent with the released codebase:

- Python 3.10
- PyTorch 2.1.0
- torchvision 0.16.0
- CUDA 11.8
- numpy 1.24.4
- scipy 1.10.1
- scikit-image 0.21.0
- opencv-python 4.8.1.78
- matplotlib 3.7.2
- imageio 2.31.6
- tensorboardX 2.6.2
- h5py 3.9.0
- nibabel 5.1.0
- SimpleITK 2.3.1
- medpy 0.4.0
- tqdm 4.66.1
