# FCIL：联邦类别增量学习（Federated Class-Incremental Learning, FCIL）框架

欢迎来到**联邦类别增量学习（FCIL）** 代码仓库！本项目旨在提供一个全面的框架，用于解决联邦学习和持续学习场景中的挑战，重点应用于医学影像和预测性维护领域。以下是对该仓库的结构、组件、数据集和使用指南的详细概述。


## 仓库概述
FCIL仓库旨在整合联邦学习、持续学习及相关领域（如生成建模、联邦学习和图神经网络）的最先进技术。它完全基于Python实现，确保跨平台兼容性，并便于与现有机器学习管道集成。


## 目录结构与核心组件
仓库通过文件夹和文件进行组织，分别处理从模型实现到数据处理和管道执行的特定功能。

### 文件夹
- **DDPM**：包含与**去噪扩散概率模型（Denoising Diffusion Probabilistic Models, DDPM）** 相关的代码，这是一类用于数据合成的生成模型。在标签数据稀缺的联邦场景中，合成数据可用于扩充训练样本，因此该模型尤为实用。*状态：开发中*。

- **MAML**：实现**模型无关元学习（Model-Agnostic Meta-Learning, MAML）**，这是一种流行的联邦学习算法。MAML通过优化模型以快速适应具有少量数据的新任务，训练模型“学会学习”。该文件夹可能包含元训练循环、任务采样和适应机制。*状态：开发中*。

- **SMPC**：专注于**安全多方计算（Secure Multi-Party Computation, SMPC）**，这是一种协作式机器学习技术，允许多方在不共享原始数据的情况下联合训练模型。这对于隐私保护应用至关重要，例如医学影像（如CheXpert、MIMIC-CXR-JPG），其中数据保密性至关重要。*状态：开发中*。

- **capgnn**：将**图神经网络（Graph Neural Networks, GNNs）** 与**胶囊网络（Capsule Networks）** 相结合，用于处理结构化数据（如具有空间依赖性或层次特征的医学影像模态）。胶囊网络通过保留空间关系和部分-整体层次结构，改进了传统卷积神经网络（CNNs），使其在BraTS2021中的肿瘤分割等任务中更为有效。*描述：“带胶囊的GNN”*。


### 核心文件
- **contrastive_learning.py**：实现对比学习技术，这是一种自监督学习方法，通过对比相似和不相似的数据点来学习有意义的表示。在联邦场景中，这一技术很有价值，因为它可以在有限标签样本上进行微调之前，在无标签数据上预训练模型以捕捉鲁棒特征。*提交记录：初始测试代码*。

- **data_loader.py**：管理项目数据集（CheXpert、MIMIC-CXR-JPG、BraTS2021、PHM2012）的数据加载和预处理。它可能包含以下工具：
  - 加载医学图像（如MIMIC-CXR-JPG的DICOM格式、BraTS2021的NIfTI格式）和表格数据（如PHM2021）。
  - 应用数据增强（如旋转、归一化）以提高模型泛化能力。
  - 将数据拆分为训练集、验证集和测试集，并支持联邦任务配置。
  *提交记录：初始测试代码*。

- **dpmm_model.py**：实现**狄利克雷过程混合模型（Dirichlet Process Mixture Models, DPMM）**，这是一种用于聚类的贝叶斯非参数方法。在类别/任务数量未预先定义的持续学习场景中，DPMM非常实用，因为它们可以动态适应新的聚类（如医学影像中的新疾病类别）。*提交记录：初始测试代码*。

- **drift_detector.py**：检测流数据中的**概念漂移**，这是持续学习中的常见挑战，即输入数据的分布或任务目标随时间变化（如医学数据中不断演变的疾病模式）。当检测到漂移时，该组件通过触发重新训练或适应，确保模型保持稳健性。*提交记录：初始测试代码*。

- **run_pipeline.py**：作为执行端到端FCIL管道的主要入口点。它可能协调以下操作：
  - 通过`data_loader.py`加载数据。
  - 模型初始化（如MAML、capGNN）。
  - 带有对比学习或元学习的训练循环。
  - 通过`drift_detector.py`进行漂移检测和适应。
  - 在联邦/持续学习任务上的评估。
  *提交记录：初始测试代码*。

- **utils.py**：提供整个仓库中使用的实用函数，例如：
  - 指标计算（如分类任务的准确率、F1分数；分割任务的Dice分数）。
  - 日志和可视化工具。
  - 模型检查点、超参数解析或数据转换的辅助函数。
  *最后更新：2025年6月9日*（截至文档编写时的最新提交）。


## 使用的数据集
该项目利用四个关键数据集（涵盖医学影像和预测性维护领域）来验证联邦学习和持续学习能力：

1. **CheXpert**：一个大型胸部X光数据集，包含带标签的检查结果（如肺炎、心脏肥大）。它广泛用于联邦医学影像分类，因为它包含具有不同样本量的多种病理。

2. **MIMIC-CXR-JPG**：一个胸部X光数据集，配有来自MIMIC-III重症监护数据库的临床笔记。它支持疾病分类和报告生成等任务，由于罕见疾病类别的样本不平衡，因此重点关注联邦学习。

3. **BraTS2021**：一个脑肿瘤分割基准数据集，包括MRI扫描（T1、T2、FLAIR模态）以及胶质瘤亚区域的标注。此处的联邦分割涉及在有限标签扫描的情况下适应新的肿瘤类型或患者群体。

4. **PHM2012**：一个用于预测性维护的数据集，包含旋转机械（如发动机、轴承）的传感器数据。它用于测试联邦/持续学习在异常检测或剩余使用寿命（RUL）预测中的应用，其中新的机器状态（任务）会随时间出现。


## 快速开始

### 前置条件
- Python 3.8及以上版本
- 所需库（通过`requirements.txt`安装；待添加，典型依赖包括`torch`、`torchvision`、`numpy`、`pandas`、`scikit-learn`、`monai`（用于医学影像）、`dicom2jpg`（用于MIMIC-CXR处理））。

### 安装步骤
1. 克隆仓库：
   ```bash
   git clone https://github.com/SaeedIqbal/FCIL.git
   cd FCIL
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt  # 若存在requirements.txt；否则，手动安装所列库。
   ```

### 运行管道
使用`run_pipeline.py`执行端到端FCIL管道，指定数据集和任务参数（示例）：
```bash
python run_pipeline.py --dataset CheXpert --task few_shot_classification --epochs 50
```
*注意：确切参数可能有所不同；请参考`run_pipeline.py`查看支持的参数。*


## 项目状态与未来工作
- **当前进展**：核心组件（数据加载、对比学习、漂移检测）处于初始测试阶段。关键文件夹（DDPM、MAML、SMPC、capgnn）正在积极开发中。


## 贡献者
- **Saeed Iqbal**（仓库所有者）。欢迎通过GitHub issues或pull requests提供贡献和反馈。


## 许可证
本仓库为公开仓库，但请参考原始数据集的许可证（例如，CheXpert、MIMIC-CXR-JPG需要机构批准）以了解使用限制。代码可能以开源许可证（如MIT）发布，但具体细节待定。


如需咨询或合作，请联系（https://github.com/SaeedIqbal）。

# 说明文档：联邦类别增量学习（Federated Class-Incremental Learning, FCIL）

## 概述
本仓库专注于联邦类别增量学习（FCIL），这是机器学习中的一个挑战性场景，其中模型必须随时间适应新类别，且每个新类别仅有有限的示例。该方法整合了持续学习和联邦学习的思想，以解决灾难性遗忘问题，并实现对新概念的高效学习。代码库包含在多种数据集（包括医学影像和工业数据集）上的实现和实验。

## 核心特性
- **联邦类别增量学习（FCIL）**：使模型能够从少量示例中学习新类别，同时保留对先前学习类别的知识。
- **多数据集支持**：包括在CheXpert、MIMIC-CXR-JPG、BraTS2021和PHM2012数据集上的实验。
- **最先进的基线**：实现并对比领先的FCIL方法。
- **全面评估**：提供详细的模型性能指标和可视化。

## 安装步骤
要搭建项目环境，请遵循以下步骤：

1. 克隆仓库：
   ```bash
   git clone https://github.com/SaeedIqbal/FCIL.git
   cd FCIL
   ```

2. 安装所需依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. （可选）为数据集路径设置环境变量：
   ```bash
   export CHEXPERT_PATH="/path/to/chexpert"
   export MIMIC_PATH="/path/to/mimic-cxr-jpg"
   export BRATS_PATH="/path/to/brats2021"
   export PHM_PATH="/path/to/phm2012"
   ```

## 数据集
### 1. CheXpert
- **描述**：包含14种疾病类别的大型胸部X光数据集。
- **来源**：[CheXpert数据集](https://stanfordmlgroup.github.io/competitions/chexpert/)
- **准备**：下载数据集，并使用`data/chexpert/`中提供的脚本进行预处理。

### 2. MIMIC-CXR-JPG
- **描述**：带有结构化标签的胸部X光片数据集。
- **来源**：[MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
- **准备**：需要PhysioNet访问权限。请遵循`data/mimic-cxr/`中的说明进行预处理。

### 3. BraTS2021
- **描述**：脑肿瘤分割挑战数据集，包含多模态MRI扫描。
- **来源**：[BraTS2021](https://www.med.upenn.edu/sbia/brats2021.html)
- **准备**：下载数据集，并使用`data/brats2021/`中的脚本进行预处理。

### 4. PHM2012
- **描述**：用于旋转机械的预测与健康管理数据集。
- **来源**：[PHM2012挑战赛](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- **准备**：数据准备说明位于`data/phm2012/`中。

## 使用方法
### 训练
要在特定数据集上训练模型，请使用以下命令结构：

```bash
python train.py --dataset [dataset_name] --method [method_name] --config [config_file]
```

在CheXpert上训练的示例：
```bash
python train.py --dataset chexpert --method our_method --config configs/chexpert.yaml
```

### 评估
要评估训练好的模型：
```bash
python evaluate.py --checkpoint [checkpoint_path] --dataset [dataset_name]
```

### 超参数调优
使用`tune.py`脚本进行超参数搜索：
```bash
python tune.py --dataset [dataset_name] --method [method_name]
```

## 联系方式
如需咨询或反馈，请联系saeediqbalkhattak@gmail.com。
