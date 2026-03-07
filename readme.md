
# FedBCGD: Communication-Efficient Accelerated Block Coordinate Gradient Descent for Federated Learning

本仓库提供论文 **“FedBCGD: Communication-Efficient Accelerated Block Coordinate Gradient Descent for Federated Learning”** 的官方实现。
FedBCGD 是一种**通信高效的加速块坐标梯度下降算法**，通过在联邦学习中对模型参数进行分块更新和动量式聚合，加快收敛速度并降低通信代价。

* 一张4090或者两张2080ti即可训练！！发顶会！！代码问题或者讨论+vx 15653218567

* 我的其他论文也都是这一套代码配置，均可复现！

* 个人主页：https://junkangliu0.github.io/
* 
---

## 🧩 主要特性

* **分块坐标更新（Block Coordinate Update）**：每次通信仅更新部分参数，减少带宽占用。
* **动量式聚合（Accelerated Aggregation）**：`FedBCGD+` 版本在服务器端引入动量机制。
* **多算法框架**：兼容多种联邦优化算法，包括 `FedAvg`, `FedMoment`, `SCAFFOLDM`, `FedAdam` 等。
* **Ray 并行模拟**：基于 Ray 框架实现参数服务器与客户端的分布式并行。
* **多数据集支持**：`MNIST`, `EMNIST`, `CIFAR10`, `CIFAR100`。
* **多网络结构支持**：`LeNet-5`, `ResNet10`, `ResNet18`, `ResNet50`（含 BN / GN）。

---

## 🛠 环境依赖

```bash
python >= 3.8
torch >= 1.10
torchvision
ray==1.0.0
numpy
matplotlib
tensorboardX
filelock
```

安装依赖：

```bash
pip install torch torchvision ray tensorboardX filelock numpy matplotlib
```

---

## 📂 数据集准备

首次运行会自动下载数据集并放置到 `./data` 目录下。
支持的数据集包括：

* **CIFAR10 / CIFAR100**
* **EMNIST**（byclass / balanced）
* **MNIST**（通过 EMNIST balanced 模式实现）

---

## 🚀 快速开始

### 运行示例

#### 基础实验：FedBCGD on CIFAR100

```bash
python FedBCGD.py \
  --alg FedBCGD \
  --data_name CIFAR100 \
  --CNN lenet5 \
  --normalization BN \
  --num_workers 100 \
  --selection 0.1 \
  --epoch 1000 \
  --E 5 \
  --batch_size 50 \
  --lr 0.1 \
  --lr_decay 0.998 \
  --alpha_value 0.6 \
  --gpu 0 \
  --extname CIFAR100 \
  --print 1
```

#### 动量增强版本：FedBCGD+

```bash
python FedBCGD.py \
  --alg FedBCGD+ \
  --data_name CIFAR10 \
  --CNN resnet18 \
  --normalization BN \
  --epoch 1000 \
  --E 5 \
  --lr 0.05 \
  --gamma 0.45 \
  --selection 0.1
```

---

## ⚙️ 参数说明

| 参数名                | 说明                                                                 | 默认值      |
| ------------------ | ------------------------------------------------------------------ | -------- |
| `--alg`            | 联邦算法 (`FedBCGD`, `FedBCGD+`, `FedAvg`, `FedMoment`, `SCAFFOLDM` 等) | `FedAvg` |
| `--data_name`      | 数据集 (`MNIST`, `EMNIST`, `CIFAR10`, `CIFAR100`)                     | `MNIST`  |
| `--CNN`            | 模型结构 (`lenet5`, `resnet10`, `resnet18`)                            | `lenet5` |
| `--normalization`  | 归一化类型 (`BN` 或 `GN`)                                                | `BN`     |
| `--num_workers`    | 客户端数量                                                              | `100`    |
| `--selection`      | 每轮选取的客户端比例                                                         | `0.1`    |
| `--E`              | 每个客户端本地训练轮数                                                        | `1`      |
| `--batch_size`     | 本地批大小                                                              | `50`     |
| `--lr`             | 学习率                                                                | `0.1`    |
| `--lr_decay`       | 学习率衰减                                                              | `1`      |
| `--alpha_value`    | Dirichlet 非IID 参数                                                  | `0.6`    |
| `--gamma`          | 动量系数（FedBCGD+）                                                     | `0.45`   |
| `--tau`, `--lr_ps` | FedAdam/FedDyn相关参数                                                 | 见默认值     |
| `--extname`        | 日志/模型文件附加标识                                                        | `EM`     |
| `--check`          | 是否加载 checkpoint                                                    | `0`      |
| `--num_gpus_per`   | 每个 Ray worker 占用 GPU 比例                                            | `1`      |

---

## 📊 输出与日志

* **日志文件**：存于 `./log/{alg}-{data}-{lr}-{num_workers}-{batch_size}-{E}-{lr_decay}.txt`
* **TensorBoard 日志**：默认写入 `runs/` 文件夹
* **模型 checkpoint**：保存至 `./checkpoint/ckpt-{alg}-{lr}-{extname}-{alpha_value}`
* **绘图数据**：训练结束后存入 `./plot/*.npy`，包括 accuracy、loss、epoch 曲线

运行 TensorBoard：

```bash
tensorboard --logdir runs
```

---

## 🧠 可复现实验建议

1. 固定随机种子（脚本已内置 `set_random_seed(42)`）。
2. 使用相同的 `--alpha_value`、`--selection` 参数以确保数据划分一致性。
3. 多次运行（建议 3 次以上）并计算平均与方差结果。
4. 若使用 GPU 多卡，可通过 `--num_gpus_per` 控制每个 worker 占用比例。

---

## ❓ 常见问题

**Q1：如何减少显存占用？**
降低 `--batch_size` 或减少 `--num_workers`；
若使用 ResNet18，可切换为 `resnet10` 或 `lenet5`。

**Q2：模型不收敛或震荡？**
降低 `--lr`，或增大 `--lr_decay`。
`FedBCGD+` 模式下可调整 `--gamma`。

**Q3：非IID 设置？**
通过 `--alpha_value` 控制数据异质性，值越小非IID程度越高。

---

## 🧾 引用

如果本代码对您的研究有帮助，请引用原论文：

```bibtex
@inproceedings{fedbcgd2025,
  title={FedBCGD: Communication-Efficient Accelerated Block Coordinate Gradient Descent for Federated Learning},
  author={Author Name and Coauthors},
  booktitle={Proceedings of ...},
  year={2025}
}
```

---

是否希望我为你生成一个可直接保存的 `README.md` 文件（UTF-8 编码）？我可以帮你导出。


