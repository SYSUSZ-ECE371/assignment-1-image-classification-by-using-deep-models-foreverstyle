<!--
 * @Author: foreverstyle
 * @Date: 2025-05-13 22:54:43
 * @LastEditTime: 2025-05-14 01:02:01
 * @Description:
 * @FilePath: \ECE371\Assignment1\README.md
-->

# ECE371 神经网络与深度学习 - 作业 1

这个仓库包含 ECE371 第一次作业的相关材料，主要关注使用深度学习模型进行图像分类。本次作业涉及使用两种不同的方法对花卉图像进行分类：使用 MMClassification 框架微调预训练模型，以及使用 PyTorch 实现训练脚本。

仓库结构组织如下，包含完成两个练习和最终报告所需的必要文件和结果。

## 项目结构

- `Ex1/`: 与练习 1 相关的文件（使用 MMClassification 进行微调）

  - `work_dirs/`: 包含 MMClassification 训练过程中保存的模型检查点（`.pth` 文件）和日志的目录，其中`work_dirs/20250508_183257/20250508_183257.log` 为 MMClassification 训练生成的日志文件
  - `my_resnet50_8xb32_in1k.py`: 用于微调的 MMClassification 原始的配置文件

- `Ex2/`: 与练习 2 相关的文件（PyTorch 脚本实现）

  - `work_dir/`: 包含保存的模型`best_model.pth`以及运行 `main.py` 脚本时的控制台输出保存下的日志`EX2.result.txt`
  - `main.py`: 完成的 PyTorch 脚本，用于训练和验证分类模型。
  - `report.pdf`: 使用 LaTeX 编写编译生成的实验报告最终 PDF 文件

- `other/`: 用于处理的数据和对结果进行验证的相关文件

  - `pre_tained/`: 文件夹用于存放 EX1 训练时候需要的预训练模型`resnet50_8xb32_in1k_20210831-ea4938fc.pth`
  - `ECE371_Assignment1.pdf`: 原始作业文档，详细说明了要求
  - `process_dataset`用于将压缩后的花卉数据集组织为 ImageNet 格式，用于 EX1 的微调训练
  - `verify_model.py`: 一个独立的脚本，用于验证 EX2 中保存的 `.pth` 模型文件

## 开始步骤

1.  确保已经设置好了必要的深度学习环境（Python, PyTorch, torchvision, MMClassification, MMEngine 等）。
2.  对于 EX1，将处理后的花卉数据集，组织为 ImageNet 格式，包含训练集和验证集划分、标注文件（`train.txt`, `val.txt`, `classes.txt`）以及类别子文件夹（`train/`, `val/`）的文件夹`flower_dataset/`放在 `mmpretrain` 的`/data`下, 将`other/`中的`pre_tained/`文件夹放在 `mmpretrain`下，将`Ex1/`中的文件放在 `mmpretrain/configs/` 下，便可按照要求运行 Exercise 1 的训练脚本
3.  对于 EX2 ，将直接解压的花卉数据集放在`EX2/flower_dataset/`可以直接按照要求运行 Exercise 1 和 Exercise 2 的训练脚本。

_注意：请务必根据你的项目实际结构，在 Exercise 1 的配置文件和 Exercise 2 的脚本文件`main.py`中，正确配置 `flower_dataset` 的路径。_

**关于大型 `.pth` 文件上传的注意事项：**

由于模型检查点文件（`.pth`）可能比较大，而 GitHub 对单个文件有 100MB 的硬限制。标准 Git 不适合直接处理大型二进制文件。对于大于 100MB 的文件，使用 **Git Large File Storage (Git LFS)** 来进行管理和上传。Git LFS 会在仓库中存储文件的指针，而实际内容存储在 LFS 服务器上。

**Git LFS** 的免费额度为 1G，所以对于 EX1，我选择了最后的一个 epoch.pth 进行上传，其他的不进行上传

---
