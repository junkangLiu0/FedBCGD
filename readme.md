



# FedBCGD: Communication-Efficient Accelerated Block Coordinate Gradient Descent for Federated Learning

<p align="center">
  <b>Block-wise communication for scalable, fast, and bandwidth-efficient federated optimization.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-Federated%20Learning-EE4C2C?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Ray-Parallel%20Simulation-028CF0">
  <img src="https://img.shields.io/badge/Communication-d%2FN-blue">
  <img src="https://img.shields.io/badge/Backbone-LeNet%20%7C%20ResNet-green">
  <img src="https://img.shields.io/badge/Datasets-CIFAR10%20%7C%20CIFAR100%20%7C%20MNIST%20%7C%20EMNIST-orange">
</p>

---
本仓库提供论文 **“FedBCGD: Communication-Efficient Accelerated Block Coordinate Gradient Descent for Federated Learning”** 的官方实现。
FedBCGD 是一种**通信高效的加速块坐标梯度下降算法**，通过在联邦学习中对模型参数进行分块更新和动量式聚合，加快收敛速度并降低通信代价。

* 一张4090或者两张2080ti即可训练！！发顶会！！代码问题或者讨论+vx 15653218567

* 我的其他论文也都是这一套代码配置，均可复现！

* 个人主页：https://junkangliu0.github.io/
*  Email: junkangliukk@gmail.com
  
## News

This repository provides the implementation of:

> **FedBCGD: Communication-Efficient Accelerated Block Coordinate Gradient Descent for Federated Learning**

FedBCGD is a communication-efficient federated learning framework that reduces client-to-server upload cost by transmitting only a selected parameter block instead of the entire model in each communication round.

The implementation is built with **PyTorch** and **Ray**, supporting scalable federated simulation with multiple clients, partial client participation, non-IID Dirichlet partitioning, checkpointing, logging, TensorBoard visualization, and communication-efficient block-wise aggregation.

---

## Highlights

### Block-wise Communication

FedBCGD partitions model parameters into multiple blocks.  
Each selected client trains the full local model but uploads only its assigned parameter block and a lightweight shared block.

This reduces the uploaded parameters from:

```text
FedAvg:   d
FedBCGD:  d / N
````

where `d` is the number of model parameters and `N` is the number of blocks.

### Full Local Training, Partial Upload

A naive block-coordinate method may freeze non-selected blocks during local training, which can cause severe block drift.
FedBCGD avoids this by allowing clients to update all parameters locally, while restricting only the upload stage.

### Shared Block Strategy

The implementation uses a shared block for important lightweight layers:

```text
LeNet-5:   fc2 + fc3
ResNet-18: linear classifier
```

This improves stability and helps the server maintain a consistent global classifier.

### Server-side Momentum

FedBCGD+ uses momentum-based server aggregation to compensate for missing block updates and accelerate convergence.

### Drift Control for FedBCGD+

FedBCGD+ further integrates a SCAFFOLD-style control variate mechanism to mitigate client drift under non-IID data.

### Ray-based Parallel Federated Simulation

The code uses Ray actors to simulate:

```text
ParameterServer
DataWorker
```

This enables parallel local training and flexible GPU allocation through `--num_gpus_per`.

---

## Repository Structure

```text
.
├── FedBCGD.py                  # Main training entry
├── dirichlet_data.py           # Dirichlet non-IID data partition
├── models/
│   ├── resnet.py               # ResNet with GroupNorm-style variants
│   └── resnet_bn.py            # ResNet with BatchNorm variants
├── data/                       # Dataset directory
├── log/                        # Training logs
├── checkpoint/                 # Saved checkpoints
├── plot/                       # Saved numpy curves
└── runs/                       # TensorBoard logs
```

---

## Installation

### 1. Create Environment

```bash
conda create -n fedbcgd python=3.8 -y
conda activate fedbcgd
```

### 2. Install Dependencies

```bash
pip install torch torchvision
pip install ray==1.0.0 tensorboardX filelock numpy matplotlib torchsummary
```

A typical environment:

```text
Python >= 3.8
PyTorch >= 1.10
torchvision
ray == 1.0.0
tensorboardX
numpy
matplotlib
filelock
torchsummary
```

---

## Supported Algorithms

The current implementation supports the following algorithms through `--alg`:

| Algorithm   | Status      | Description                                                        |
| ----------- | ----------- | ------------------------------------------------------------------ |
| `FedBCGD`   | Recommended | Proposed block-wise federated optimizer                            |
| `FedBCGD+`  | Recommended | Accelerated FedBCGD with control variates and momentum aggregation |
| `FedAvg`    | Supported   | Standard federated averaging baseline                              |
| `FedMoment` | Supported   | FedAvg with server-side momentum                                   |
| `SCAFFOLDM` | Supported   | SCAFFOLD-style control variate with momentum aggregation           |

The code also contains placeholders or partial entries for additional algorithms such as `FedAdam`, `FedDC`, `FedDyn`, and `FedCM`.
For clean reproducibility, we recommend using the five algorithms listed above unless you complete the corresponding server-side aggregation logic.

---

## Supported Datasets

| Dataset            | `--data_name` | Notes                                     |
| ------------------ | ------------- | ----------------------------------------- |
| CIFAR-10           | `CIFAR10`     | 10-class image classification             |
| CIFAR-100          | `CIFAR100`    | 100-class image classification            |
| EMNIST             | `EMNIST`      | ByClass split                             |
| MNIST-style EMNIST | `MNIST`       | Implemented through EMNIST balanced split |

The dataset will be loaded from:

```text
./data
```

For CIFAR-100 and EMNIST/MNIST-style experiments, the code enables automatic download.
For CIFAR-10, make sure the dataset exists locally or set `download=True` manually in the dataset loader.

---

## Supported Models

The current codebase provides stable model selection through `--CNN`:

| Model               | `--CNN`            | Supported Dataset    |
| ------------------- | ------------------ | -------------------- |
| LeNet-5             | `lenet5`           | CIFAR-10 / CIFAR-100 |
| ResNet-10           | `resnet10`         | CIFAR-10 / CIFAR-100 |
| ResNet-18           | `resnet18`         | CIFAR-10 / CIFAR-100 |
| Simple Linear Model | automatically used | MNIST-style EMNIST   |
| EMNIST CNN          | automatically used | EMNIST               |

Normalization can be selected by:

```text
--normalization BN
--normalization GN
```

For ResNet experiments:

```text
BN: ResNet10BN / ResNet18BN
GN: ResNet10 / ResNet18
```

---

## Quick Start

### FedBCGD on CIFAR-100 with LeNet-5

```bash
python FedBCGD.py --alg FedBCGD --data_name CIFAR100 --CNN lenet5 --normalization BN --num_workers 100 --selection 0.1 --epoch 1000 --E 5 --batch_size 50 --lr 0.1 --lr_decay 0.998 --alpha_value 0.6 --block 5 --gpu 0 --num_gpus_per 0.1 --extname CIFAR100_lenet5_FedBCGD --print 1
```

### FedBCGD+ on CIFAR-100 with LeNet-5

```bash
python FedBCGD.py --alg FedBCGD+ --data_name CIFAR100 --CNN lenet5 --normalization BN --num_workers 100 --selection 0.1 --epoch 1000 --E 5 --batch_size 50 --lr 0.1 --lr_decay 0.998 --alpha_value 0.6 --block 5 --gamma 0.45 --gpu 0 --num_gpus_per 0.1 --extname CIFAR100_lenet5_FedBCGDPlus --print 1
```

### FedBCGD on CIFAR-10 with ResNet-18

```bash
python FedBCGD.py --alg FedBCGD --data_name CIFAR10 --CNN resnet18 --normalization BN --num_workers 100 --selection 0.1 --epoch 1000 --E 5 --batch_size 50 --lr 0.05 --lr_decay 0.998 --alpha_value 0.6 --block 5 --gpu 0 --num_gpus_per 0.1 --extname CIFAR10_resnet18_FedBCGD --print 1
```

### FedBCGD+ on CIFAR-100 with ResNet-18

```bash
python FedBCGD.py --alg FedBCGD+ --data_name CIFAR100 --CNN resnet18 --normalization BN --num_workers 100 --selection 0.1 --epoch 1000 --E 5 --batch_size 50 --lr 0.05 --lr_decay 0.998 --alpha_value 0.6 --block 5 --gamma 0.45 --gpu 0 --num_gpus_per 0.1 --extname CIFAR100_resnet18_FedBCGDPlus --print 1
```

---

## Baseline Commands

### FedAvg

```bash
python FedBCGD.py --alg FedAvg --data_name CIFAR100 --CNN lenet5 --normalization BN --num_workers 100 --selection 0.1 --epoch 1000 --E 5 --batch_size 50 --lr 0.1 --lr_decay 0.998 --alpha_value 0.6 --block 5 --gpu 0 --num_gpus_per 0.1 --extname CIFAR100_lenet5_FedAvg --print 1
```

### FedMoment

```bash
python FedBCGD.py --alg FedMoment --data_name CIFAR100 --CNN lenet5 --normalization BN --num_workers 100 --selection 0.1 --epoch 1000 --E 5 --batch_size 50 --lr 0.1 --lr_decay 0.998 --alpha_value 0.6 --block 5 --gamma 0.45 --gpu 0 --num_gpus_per 0.1 --extname CIFAR100_lenet5_FedMoment --print 1
```

### SCAFFOLDM

```bash
python FedBCGD.py --alg SCAFFOLDM --data_name CIFAR100 --CNN lenet5 --normalization BN --num_workers 100 --selection 0.1 --epoch 1000 --E 5 --batch_size 50 --lr 0.1 --lr_decay 0.998 --alpha_value 0.6 --block 5 --gamma 0.45 --gpu 0 --num_gpus_per 0.1 --extname CIFAR100_lenet5_SCAFFOLDM --print 1
```

---

## Reproducing Main Experiments

### CIFAR-100, LeNet-5, Moderate Non-IID

```bash
python FedBCGD.py --alg FedBCGD --data_name CIFAR100 --CNN lenet5 --normalization BN --num_workers 100 --selection 0.1 --epoch 1000 --E 5 --batch_size 50 --lr 0.1 --lr_decay 0.998 --alpha_value 0.6 --block 5 --gpu 0 --num_gpus_per 0.1 --extname main_cifar100_lenet5_alpha06 --print 1
```

### CIFAR-100, ResNet-18, Moderate Non-IID

```bash
python FedBCGD.py --alg FedBCGD --data_name CIFAR100 --CNN resnet18 --normalization BN --num_workers 100 --selection 0.1 --epoch 1000 --E 5 --batch_size 50 --lr 0.05 --lr_decay 0.998 --alpha_value 0.6 --block 5 --gpu 0 --num_gpus_per 0.1 --extname main_cifar100_resnet18_alpha06 --print 1
```

### CIFAR-100, ResNet-18, Strong Non-IID

```bash
python FedBCGD.py --alg FedBCGD+ --data_name CIFAR100 --CNN resnet18 --normalization BN --num_workers 100 --selection 0.1 --epoch 1000 --E 5 --batch_size 50 --lr 0.05 --lr_decay 0.998 --alpha_value 0.1 --block 5 --gamma 0.45 --gpu 0 --num_gpus_per 0.1 --extname main_cifar100_resnet18_alpha01 --print 1
```

---

## Important Arguments

| Argument          | Description                   | Recommended Value                        |
| ----------------- | ----------------------------- | ---------------------------------------- |
| `--alg`           | Federated algorithm           | `FedBCGD`, `FedBCGD+`, `FedAvg`          |
| `--data_name`     | Dataset name                  | `CIFAR10`, `CIFAR100`, `MNIST`, `EMNIST` |
| `--CNN`           | Backbone model                | `lenet5`, `resnet10`, `resnet18`         |
| `--normalization` | Normalization type for ResNet | `BN`, `GN`                               |
| `--num_workers`   | Total number of clients       | `100`                                    |
| `--selection`     | Client participation ratio    | `0.1`                                    |
| `--epoch`         | Communication rounds          | `1000`                                   |
| `--E`             | Local training epochs         | `5`                                      |
| `--batch_size`    | Local batch size              | `50`                                     |
| `--lr`            | Client learning rate          | `0.05` or `0.1`                          |
| `--lr_decay`      | Learning-rate decay per round | `0.998`                                  |
| `--alpha_value`   | Dirichlet non-IID parameter   | `0.6` or `0.1`                           |
| `--block`         | Number of parameter blocks    | `5`                                      |
| `--gamma`         | Server momentum coefficient   | `0.45`                                   |
| `--gpu`           | Visible GPU ids               | `0` or `0,1`                             |
| `--num_gpus_per`  | GPU fraction per Ray worker   | `0.1`, `0.2`, `1`                        |
| `--extname`       | Experiment name suffix        | custom                                   |
| `--check`         | Resume from checkpoint        | `0` or `1`                               |
| `--print`         | Print training logs           | `1`                                      |

---

## Non-IID Data Partition

The code uses Dirichlet distribution to simulate label distribution skew across clients:

```bash
--alpha_value 0.6
```

Typical settings:

```text
alpha_value = 0.6   moderate heterogeneity
alpha_value = 0.1   strong heterogeneity
```

Smaller `alpha_value` indicates stronger non-IID data heterogeneity.

---

## Checkpointing

The best checkpoint is automatically saved when test accuracy improves.

Checkpoint path:

```text
./checkpoint/ckpt-{alg}-{lr}-{extname}-{alpha_value}
```

To resume training:

```bash
python FedBCGD.py --check 1 --alg FedBCGD --data_name CIFAR100 --CNN lenet5 --lr 0.1 --alpha_value 0.6 --extname CIFAR100_lenet5_FedBCGD
```

The checkpoint stores:

```text
model state_dict
client control variates
parameter server state
Dirichlet data partition
current epoch
```

---

## Logging and Visualization

Training logs are saved to:

```text
./log/{alg}-{data_name}-{lr}-{num_workers}-{batch_size}-{E}-{lr_decay}.txt
```

TensorBoard logs are written to:

```text
./runs/
```

Launch TensorBoard:

```bash
tensorboard --logdir runs
```

Training curves are saved as NumPy files:

```text
./plot/alg_{alg}-E_{E}-#wk_{num_workers}-ep_{epoch}-lr_{lr}-alpha_value_{alpha_value}-selec_{selection}-alpha{alpha}-{extname}-gamma{gamma}.npy
```

The saved file contains:

```text
x: evaluation rounds
result: test accuracy curve
result_loss: train loss curve and final runtime
test_list_loss: test loss curve
```

---

## Recommended Hardware

For the default CIFAR experiments:

```text
1 × RTX 4090
or
2 × RTX 2080 Ti
```

For memory-friendly Ray simulation, use fractional GPU allocation:

```bash
--num_gpus_per 0.1
```

Example:

```bash
python FedBCGD.py --alg FedBCGD --data_name CIFAR100 --CNN lenet5 --num_workers 100 --selection 0.1 --num_gpus_per 0.1 --gpu 0
```

---

## Practical Recommendations

### Learning Rate

```text
LeNet-5 on CIFAR:     lr = 0.1
ResNet-18 on CIFAR:   lr = 0.05
```

### Local Epochs

```text
E = 5
```

This is a good default for CIFAR experiments.

### Block Number

```text
block = 5
```

The current code manually defines five blocks for LeNet-5 and ResNet-18.

### Server Momentum

```text
gamma = 0.45
```

The implementation uses `gamma` for server-side momentum aggregation in FedBCGD+ and FedMoment-style algorithms.

---

## Implementation Notes

The current code follows a synchronous parameter-server training pipeline:

```text
1. Initialize global model on ParameterServer.
2. Sample participating clients.
3. Assign each client to a parameter block.
4. Each DataWorker performs local training.
5. Each client uploads only selected block updates.
6. ParameterServer aggregates block-wise updates.
7. Evaluate global model every 10 rounds.
8. Save checkpoint when accuracy improves.
```

For LeNet-5, the block assignment is manually defined as:

```text
block1: conv1 + shared classifier layers
block2: conv2 + shared classifier layers
block3: fc1   + shared classifier layers
block4: fc2   + fc3
block5: fc3   + fc2
```

For ResNet-18, the block assignment follows network stages:

```text
block1: stem + layer1 + classifier
block2: layer2 + classifier
block3: layer3 + classifier
block4: layer4.0 + classifier
block5: layer4.1 + classifier
```

---

## Known Notes

The codebase contains some legacy entries for algorithms and models that are not fully activated in the current main script. For strict reproducibility, we recommend using:

```text
Algorithms: FedBCGD, FedBCGD+, FedAvg, FedMoment, SCAFFOLDM
Models:     lenet5, resnet10, resnet18
Datasets:   CIFAR10, CIFAR100, MNIST, EMNIST
```

If you want to reproduce paper-level VGG or ViT experiments, please ensure the corresponding model definitions and block partitions are added to `FedBCGD.py`.

---

## Citation

If this repository is useful for your research, please consider citing:

```bibtex
@inproceedings{fedbcgd,
  title     = {FedBCGD: Communication-Efficient Accelerated Block Coordinate Gradient Descent for Federated Learning},
  author    = {Anonymous Authors},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year      = {2025}
}
```

Please update the author information and venue after the official camera-ready version is released.

---

## Contact

For questions, discussion, or collaboration, please contact the authors or open an issue in this repository.

---

## Acknowledgements

This repository builds upon PyTorch, torchvision, Ray, and the federated learning literature including FedAvg, SCAFFOLD, FedAvgM, FedAdam, FedDC, and block coordinate optimization.

---

## License

This project is released for academic research purposes. Please check the final license file before commercial use.

```
```




