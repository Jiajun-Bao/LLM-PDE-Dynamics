# Text-Trained LLMs Can Zero-Shot Extrapolate PDE Dynamics, Revealing a Three-Stage In-Context Learning Mechanism
This repository includes Jupyter notebooks, utility functions, and generated datasets used to reproduce all numerical experiments in the paper “Text-Trained LLMs Can Zero-Shot Extrapolate PDE Dynamics, Revealing a Three-Stage In-Context Learning Mechanism” by Jiajun Bao, Nicolas Boullé, Toni J.B. Liu, Raphaël Sarfati, and Christopher J. Earls.


https://github.com/user-attachments/assets/ae7e51e2-76b1-477e-9f49-b4da0a0546c9


## Paper Abstract
Large language models (LLMs) have demonstrated emergent in-context learning (ICL) capabilities across a range of tasks, including zero-shot time-series forecasting. We show that text-trained foundation models can accurately extrapolate spatiotemporal dynamics from discretized partial differential equation (PDE) solutions without fine-tuning or natural language prompting. Predictive accuracy improves with longer temporal contexts but degrades at finer spatial discretizations. In multi-step rollouts, where the model recursively predicts future spatial states over multiple time steps, errors grow algebraically with the time horizon, reminiscent of global error accumulation in classical finite-difference solvers. We interpret these trends as in-context neural scaling laws, where prediction quality varies predictably with both context length and output length. To better understand how LLMs are able to internally process PDE solutions so as to accurately roll them out, we analyze token-level output distributions and uncover a consistent three-stage ICL progression: beginning with syntactic pattern imitation, transitioning through an exploratory high-entropy phase, and culminating in confident, numerically grounded predictions.

## Repository Contents
The materials are organized into three folders corresponding to the main text and Supplementary Material sections. Each notebook contains an annotation at the beginning specifying the corresponding study.

## Environment Details
For the computations, the following environment was used:
- Python version: 3.12.11
- PyTorch version: 2.7.1
- Transformers version: 4.55.0
- NumPy version: 2.2.5
- SciPy version: 1.16.1
- Matplotlib version: 3.10.5
- OS info: Linux-6.8.0-59-generic-x86_64-with-glibc2.39
- CPU cores: 64
- GPUs: 3 × NVIDIA RTX A4000
