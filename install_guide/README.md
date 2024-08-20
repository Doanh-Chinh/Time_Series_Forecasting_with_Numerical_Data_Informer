# Experiment Guide

This guideline provides a step-by-step approach to running the experiment, including installation, data preparation, model training, and parameter information.

## Requirements

- Python >= 3.7
- PaddlePaddle
- PaddleTS

## Installation

Follow the steps below to install the necessary packages for the experiment.

### Table. Installation Script for PaddleTS Informer

| NO. | Step                                 | Script                                                                                       |
|-----|--------------------------------------|----------------------------------------------------------------------------------------------|
|  1  | Create a virtual environment using Conda | `conda create -n paddlets python=3.9`                                                         |
|  2  | Activate the new environment           | `conda activate paddlets`                                                                     |
|  3  | Install PaddlePaddle                   | `python -m pip install paddlepaddle==2.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple`       |
|  4  | Install PaddleTS                       | `python -m pip install paddlets`                                                              |

