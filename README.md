# GPU Specialization Capstone Project

## cuDNN-Accelerated Neural Network Inference Pipeline

A GPU-accelerated deep learning inference pipeline built using NVIDIA cuDNN and CUDA libraries. This project demonstrates forward-pass inference of a Convolutional Neural Network (CNN) on image datasets entirely on the GPU, applying concepts learned throughout the CUDA at Scale for the Enterprise specialization.

---

## Table of Contents

* Overview
* Features
* Requirements
* Project Structure
* Installation
* Building the Project
* Running the Project
* Command Line Arguments
* Dataset
* Results
* References

---

## Overview

This project implements a GPU-accelerated CNN inference pipeline using NVIDIA cuDNN. The application performs a multi-layer forward pass consisting of:

* Convolution Layer
* ReLU Activation Layer
* Max Pooling Layer

The pipeline processes batches of images entirely on the GPU and demonstrates how cuDNN primitives can be used to accelerate deep learning inference workloads.

### Workflow

1. Load batches of input images into GPU memory.
2. Apply convolution operations using cuDNN.
3. Apply ReLU activation functions.
4. Perform max pooling operations.
5. Generate output feature maps.
6. Record execution statistics and timing information.

---

## Features

* GPU-accelerated CNN inference using cuDNN
* Convolution operations via `cudnnConvolutionForward`
* ReLU activation via `cudnnActivationForward`
* Max pooling via `cudnnPoolingForward`
* Batch image processing support
* Command-line configuration options
* GPU execution logging and performance metrics
* Modular project structure

---

## Requirements

### Hardware

* NVIDIA GPU
* Compute Capability 3.5 or higher

### Software

* CUDA Toolkit 11.0+
* cuDNN 8.0+
* CMake 3.18+
* GCC 7+ or MSVC 2019+

---

## Installing cuDNN

Download cuDNN from NVIDIA Developer and follow the official installation instructions.

Verify installation:

```bash
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

---

## Project Structure

```text
GPU-Specialization-Capstone/
├── src/
│   ├── main.cpp
│   ├── inference.cu
│   └── inference.h
├── data/
│   ├── input/
│   └── output/
├── logs/
├── CMakeLists.txt
├── Makefile
└── README.md
```

### File Description

| File         | Description                                 |
| ------------ | ------------------------------------------- |
| main.cpp     | Program entry point and argument parsing    |
| inference.cu | CNN forward-pass implementation using cuDNN |
| inference.h  | Function and class declarations             |
| data/input   | Input image dataset                         |
| data/output  | Generated feature maps and outputs          |
| logs         | Execution logs and timing information       |

---

## Installation

```bash
git clone https://github.com/shanmugavasanth/GPU-Specialization-Capstone.git
cd GPU-Specialization-Capstone
```

---

## Building the Project

### Using CMake

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Using Makefile

```bash
make
```

---

## Running the Project

### Basic Usage

```bash
./cudnn_inference --input data/input/ --output data/output/ --batch 32
```

### Full Example

```bash
./cudnn_inference \
  --input data/input/ \
  --output data/output/ \
  --batch 32 \
  --layers 3 \
  --log logs/run.log
```

---

## Command Line Arguments

| Argument | Description           | Default      |
| -------- | --------------------- | ------------ |
| --input  | Input image directory | data/input/  |
| --output | Output directory      | data/output/ |
| --batch  | Batch size            | 32           |
| --layers | Number of CNN layers  | 3            |
| --log    | Output log file       | logs/run.log |

---

## Dataset

This project uses image datasets from:

* USC SIPI Image Database
* MNIST Handwritten Digits Dataset

The implementation is designed to process hundreds of images in batches using GPU acceleration.

Place all images inside:

```text
data/input/
```

before executing the program.

---

## Results

The generated outputs are stored in:

```text
data/output/
```

Execution logs are stored in:

```text
logs/run.log
```

The log file contains:

* GPU device information
* Batch processing statistics
* Layer execution times
* cuDNN API usage details
* Overall inference performance metrics

### Example Output

```text
GPU: NVIDIA Tesla T4

Processing Batch Size: 32

Layer 1 (Convolution): 1.24 ms
Layer 2 (ReLU): 0.38 ms
Layer 3 (Max Pooling): 0.29 ms

Total Inference Time: 1.91 ms

Batch Processing Completed Successfully
```

---

## References

* NVIDIA cuDNN Developer Guide
* CUDA C++ Programming Guide
* USC SIPI Image Database
* MNIST Dataset
* Google C++ Style Guide

---

## Author

*Pradeep E*

Johns Hopkins University
CUDA at Scale for the Enterprise Specialization
GPU Specialization Capstone Project
