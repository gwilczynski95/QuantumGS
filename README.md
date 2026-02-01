# QuantumGS

This repository contains the implementation for the paper **QuantumGS** [Paper Link Placeholder].

### Abstract

Recent advances in neural rendering, particularly 3D Gaussian Splatting (3DGS), have enabled real-time rendering of complex scenes. However, standard 3DGS relies on spherical harmonics, which often struggle to accurately capture high-frequency view-dependent effects such as sharp reflections and transparency. While hybrid approaches like Viewing Direction Gaussian Splatting (VDGS) mitigate this limitation using classical Multi-Layer Perceptrons (MLPs), they remain limited by the expressivity of classical networks in low-parameter regimes. In this paper, we introduce QuantumGS, a novel hybrid framework that integrates Variational Quantum Circuits (VQC) into the Gaussian Splatting pipeline. We propose a unique encoding strategy that maps the viewing direction directly onto the Bloch sphere, leveraging the natural geometry of qubits to represent 3D directional data. By replacing classical color-modulating networks with quantum circuits generated via a hypernetwork or conditioning mechanism, we achieve higher expressivity and better generalization. Source code is available in the supplementary material.

## Installation

To install the necessary environment, please follow these steps:

1.  **Create a Conda environment:**
    Ensure you have conda installed.

    ```bash
    conda env create -f environment.yml
    conda activate quantum_gs
    ```

2.  **Install specific 3DGS and Quantum modules:**
    
    *   **TorchQuantum (0.1.8):** Follow instructions from the official repository or install via pip if available.
    *   **Diff Gaussian Rasterization & Simple KNN:** It is recommended to install these from the mother project for compatibility:
        [ViewingDirectionGaussianSplatting](https://github.com/gmum/ViewingDirectionGaussianSplatting)

    ```bash
    # Example for installing submodules if they are present in the repo or cloned separately
    # pip install ./submodules/diff-gaussian-rasterization
    # pip install ./submodules/simple-knn
    ```

## To run experiments

Just use sh scripts main directory. Make sure to change paths to your own. 

- `run_db.sh` - Use it to run experiments on Deep Blending dataset
- `run_tandt.sh` - Use it to run experiments on Tanks and Temples dataset
- `run_mipnerf.sh` - Use it to run experiments on Mip-NeRF360 dataset
- `run_nerfsynth.sh` - Use it to run experiments on NeRF Synthetic dataset
