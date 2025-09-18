# ModularMLP

**ModularMLP** is a Java library for building and training **Multi-Layer Perceptrons (MLPs) from scratch** with a fully **modular design**. It uses a *builder pattern* for easy configuration of networks and trainers.  

## Features

- Fully configurable MLPs: number of layers, neurons per layer, activation functions.  
- Trainers with support for different optimizers (Adam, SGD, etc.) and loss functions (Cross-Entropy, MSE…).  
- Batch support and training on custom datasets.  
- Optional regularization: L1, L2, ElasticNet.  
- Implemented **from scratch**, no heavy external dependencies.  
- **Modular design**: every component (layer, optimizer, dataset, trainer) can be swapped or extended easily.  

## Installation

Clone the repository:  
```bash
git clone https://github.com/your-username/ModularMLP.git
cd ModularMLP
