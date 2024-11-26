# Towards Robust Federated Learning: Investigating Poisoning Attacks Under Clients Data Heterogeneity

This study investigates the effects of poisoning attacks under data heterogeneity for federated learning. We conducted experiments to evaluate the impact of varying malicious client fractions and poison concentration levels on the model's accuracy. In addition, we explore the effects of poisoning attacks on **FedAvg** and **FedNova** models using medical imaging tasks. Our findings reveal that increasing data heterogeneity aggravates the effects of poisoning, with FedNova demonstrating greater resilience than FedAvg. We found that the number of malicious clients plays a more significant role in degrading performance than the ratio of poisoning samples shared by each attacker, suggesting that even modest levels of poisoning can be tolerated by most algorithms.

# Federated Learning with PyTorch and Flower

This project demonstrates federated learning (FL) setup using PyTorch for training a MobileNetV2 model, with the Flower framework to manage the federated learning process. The project also incorporates differential privacy for enhanced security, using local differential privacy (DP) mechanisms.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
  - [Setting Up the Environment](#setting-up-the-environment)
  - [Environment Configuration](#environment-configuration)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Running the Federated Learning Simulation](#running-the-federated-learning-simulation)
- [Differential Privacy](#differential-privacy)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In this project, we leverage federated learning to train a neural network model in a distributed manner across multiple clients. Each client trains the model locally using its own data and shares only the updated model parameters with the central server. No raw data is shared, ensuring privacy. Additionally, the project employs **differential privacy** mechanisms to further safeguard the model training process.

### Key Components:
- **Flower**: A federated learning framework used to simulate the FL process.
- **PyTorch**: For building and training the MobileNetV2 model.
- **Differential Privacy**: Ensuring local privacy of client updates using `LocalDpMod`.

## Features

- **Federated Learning Simulation**: Manage multiple clients and a central server to orchestrate model training.
- **Differential Privacy**: Protect data during training with configurable DP mechanisms.
- **MobileNetV2**: Utilize a pre-trained MobileNetV2 model, which is fine-tuned on custom datasets.
- **Data Preprocessing**: Includes image transformations such as resizing, normalization, and grayscale conversion.
  
## Installation

### Prerequisites
- Conda (Anaconda or Miniconda)
- Python 3.8+

### Setting Up the Environment

To set up the environment using Conda, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
    ```

2. **Create the Conda environment**
    ```bash
   conda env create -f environment.yml
    ```
   
4. **Activate the environment**:

   ```bash
   conda activate federated-learning
   ```

4. **Verify installation**
    ```bash
   conda list
    ```
