# SDR-Net: Deep Polar Modeling for Robust Oriented Ship Detection in Dense SAR Scenarios

This repository contains the core implementation of **SDR-Net**, a polar coordinate-based detector for oriented ship detection in SAR images.

## Files

| File | Description |
|------|-------------|
| `SDR-head.py` | Detection head with three parallel branches: heatmap prediction, center offset regression, and polar parameter regression (Mr, α1, α2, θm). |
| `SDR-loss.py` | Loss functions: dense regression heatmap loss (smooth L1), center offset loss, and polar parameter loss (cosine-based for angles). |
| `SDR-tal.py` | Target assignment utilities including covariance-adaptive rotated Gaussian heatmap generation and scale-based dense regression (SDR) assigner. |
| `SDR-Net.pdf` | Full paper with methodology, experiments, and ablation studies. |
| `data.yaml` | Example dataset configuration template. |

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- torchvision
- numpy
- math (standard library)

## Note

Our pre-trained weights will be uploaded after the paper is accepted.This code is directly related to the manuscript submitted to The Visual Computer.
