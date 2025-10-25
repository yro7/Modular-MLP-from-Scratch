# ModularMLP

**ModularMLP** is a Java library for building and training **Multi-Layer Perceptrons (MLPs) from scratch** with a fully **modular design**. 
The primary goal of this implementation is to **prioritize simplicity** and ease of use for the end user, allowing the construction and training of complex MLPs with minimal code.

This library is for **educational purposes only** and shouldn't be considered for big networks. More specially, the matrix implementation is particularly inefficient and doesn't uses standard java librairies such as EJML or ND4J.

## Features

- Fully configurable MLPs: number of layers, neurons per layer, activation functions.  
- Trainers with support for different optimizers (Adam, SGD, etc.) and loss functions (Cross-Entropy, MSE…).  
- Batch support and training on custom datasets.  
- Optional regularization: L1, L2, ElasticNet.  
- Implemented **from scratch**, no external dependencies.  
- **Modular design**: every component (layer, optimizer, dataset, trainer) can be swapped or extended easily.  

## Usage example

```java
// Create the trainer
Trainer mnistTrainer = Trainer.builder() 
        .setLossFunction(CE) 
        .setOptimizer(new Adam(0.001, 0.99, 0.999))
        .setDataset(mnistDataset)
        .setEpoch(30)
        .setBatchSize(7_000)
        .build();

// Build and train the MLP
MLP mnistMLP = MLP.builder(784)
        .setRandomSeed(420)
        .addLayer(256, ReLU)
        .addLayer(128, ReLU)
        .addLayer(10, SoftMax)
        .build()
        .train(mnistTrainer);

```

## Installation

Clone the repository:  
```bash
git clone https://github.com/yro7/ModularMLP.git
cd ModularMLP
```

## Library overhead

Here's the call tree of the project, when training a basic MNIST mlp resolver.
As expected, the matrix multiplication takes ~95% of CPU time. 
![image](https://github.com/user-attachments/assets/f9b8339b-b829-4b84-aec5-349e66da833f)


Note that the application only ran for a few minutes, JIT by the JVM might change results while training larger models.
