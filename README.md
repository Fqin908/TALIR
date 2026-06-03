# TALIR-CIOD 复现

> 论文：*Learning Task-Aware Language-Image Representation for Class-Incremental Object Detection* (AAAI-24)
>
> 本指南面向**组员/测试组**，从零开始搭建环境并复现实验。
---

## 一、交付文件清单

应包含以下内容：

```
├──TALIR-main/
│   ├── README.md                      # 文档
│   ├── configs                        # mmdetection_configs
│   ├── mmdet                          # mmdetection
│   ├── data/
│   │   ├──  voc2007                   # 数据集
│   │   ├──  voc_glip_dataset.py       # 数据集加载
│   ├── data_preprocess                # 数据预处理
│   ├── pretrained                     # 预训练权重
│   ├── work_dirs                      # 日志输出、权重保存
│   ├── engine/
│   │   ├── config.json					   # 日志输出、权重保存
│   │   ├── talir_head.py
│   │   ├── train_glip_voc.py
│   │   ├── train_task1.py
│   │   ├── eval_sis.py
│   │   ├── eval_der.py
│   │   ├── train_single.py
│   │   ├── eval_task1_fixed.py
```

**不打包的内容（需要自行下载）：**

- **VOC2007 数据集** → 从官网下载 [open-mmlab/mmdetection: OpenMMLab Detection Toolbox and Benchmark](https://github.com/open-mmlab/mmdetection)

- **GLIP、BERT 预训练模型** → 

  BERT pretrained中脚本直接运行自动下载（[google-bert/bert-base-uncased at main](https://huggingface.co/google-bert/bert-base-uncased/tree/main)），包括config.json, pytorch_model.bin, special_tokens_map.json, tokenizer_config.json, vocab.txt

  GLIP:https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth

---

## 二、环境搭建（从零开始）

### 2.1 创建 conda 环境

```bash
conda create -n mmdet python=3.10 -y
conda activate mmdet
```

### 2.2 安装 PyTorch

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

### 2.3 安装 mmdetection 3.3.0

```bash
pip install -U openmim
mim install mmengine==0.10.4
mim install mmcv==2.1.0
mim install mmdet==3.3.0
```

**验证版本：**
```bash
python -c "import torch; print(torch.__version__)"    # 期望: 2.1.2+cu121
python -c "import mmcv; print(mmcv.__version__)"      # 期望: 2.1.0
python -c "import mmdet; print(mmdet.__version__)"    # 期望: 3.3.0
```

### 2.4 安装其他依赖

```bash
pip install transformers==4.36.0 pillow numpy==1.24.3
```

---

## 三、代码部署

### 3.1 放入预训练权重（如交付包中包含）

```bash
# 预训练权重放在  TALIR-main/pretrained
cp ~/Downloads/glip_tiny_a_mmdet-b3654169.pth ./pretrained

#Bert下载
python ./pretrained/download_Bert.py
```

> 如未获得权重文件，可从 [OpenMMLab 模型库](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/glip/README.md) 下载 `glip_tiny_mmdet.pth`，或直接点击下面的链接下载权重。
>
> https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth

### 3.2 放入训练好的权重

```
通过百度网盘分享的文件：mmdetection-TALIR
链接: https://pan.baidu.com/s/1WPK0o30rKm3XMU1Q9_POVw 提取码: r8xa
提供了voc2007在SEED=42下训练好的权重，包括10+10，15+5，19+1三种增量策略，以及在10+10增量策略下训练的baseline，和用于消融实验的_notam
链接中的work_dirs可直接复制到该项目的根目录下，pretrained中包含了下载的Bert和GLIP权重，可根据需求进行下载。
```

---

## 四、数据准备

### 4.1 下载 VOC2007（可在官网自行下载）

```bash
# 在 TALIR-main 同级目录创建 data 文件夹(最后不包含VOCdevkit这一层文件夹)
# 下载 voc2007
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

# 解压
tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
tar -xf VOCdevkit_08-Jun-2007.tar
```

最终目录结构应为：
```
TALIR-main/data/voc2007/
├── Annotations/
├── ImageSets/
│   └── Main/
│       ├── trainval.txt
│       └── test.txt
└── JPEGImages/
```

### 4.2 增量分割文件

运行preprocess_voc.py，默认当前的位置是 ~/mmdetection/

```
cd /your/path/to/TALIR-main
python data_preprocess/preprocess_voc.py --protocol 10_10 --seed 42
python data_preprocess/preprocess_voc.py --protocol 19_1 --seed 42
python data_preprocess/preprocess_voc.py --protocol 15_5 --seed 42
```

运行generate_test_split.py

```
cd /your/path/to/TALIR-main
python data_preprocess/generate_test_split.py --protocol 10_10 --seed 42
python data_preprocess/generate_test_split.py --protocol 19_1 --seed 42
python data_preprocess/generate_test_split.py --protocol 15_5 --seed 42
```

该目录应包含：

```
cd /your/path/to/TALIR-main
incremental_10_10/seed42/
├── category_split.json
├── task0_trainval.txt
├── task1_trainval.txt
├── task0_test.txt
└── task1_test.txt
```

> 如缺少 test split，运行 `python projects/TALIR/generate_test_split.py` 生成。

---

## 五、路径修改

我们的代码中有几处**硬编码的相对路径**，可根据自己的需求修改,在配置文件./engine/config.json和具体脚本中修改。

### 5.1 数据、模型权重路径

```
...
"paths": {
        "data_root": "./data/voc2007",
        "pretrained_dir": "./pretrained",
        "bert_model_name": "./pretrained/bert-base-uncased",
        "pretrained_checkpoint": ".//pretrained/glip_tiny_a_mmdet-b3654169.pth",
        "glip_config": "./configs/glip/glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py",
        "work_dir": "./work_dirs/talir_voc"
    },
 ...
```

### 5.2 GPU 设备

所有脚本头部硬编码了：
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

如需使用其他 GPU，将 `0` 改为对应的设备号（如 `1`）。

## 六、训练全流程

### Step 1: Task 0 基类训练

```bash
cd /your/path/to/TALIR-main
python engine/train_glip_voc.py  --protocol 10_10 --seed 42 --epochs 2 --batch-size 4 --lr 5e-5  --eval-interval 1
```

```
cd /your/path/to/TALIR-main
python engine/train_glip_voc.py  --protocol 15_5 --seed 42 --epochs 2 --batch-size 4 --lr 5e-5 --eval-interval 1
```

```
cd /your/path/to/TALIR-main
python engine/train_glip_voc.py --protocol 19_1 --seed 42 --epochs 2 --batch-size 4 --lr 5e-5 --eval-interval 1
```

**输出位置：**

```
work_dirs/talir_voc/10_10/seed42/
├── task0_best.pth          # test mAP50 最高的 checkpoint
├── task0_epoch1.pth
└── task0_epoch2.pth
```

---

### Step 2: Task 1 增量训练（TAM 版本）

```bash
python engine/train_task1.py --protocol 10_10 --seed 42 --task0-ckpt work_dirs/talir_voc/10_10/seed42/task0_best.pth --epochs 2 --batch-size 4 --lr 5e-5 --eval-interval 1
```

```
python engine/train_task1.py --protocol 15_5 --seed 42 --task0-ckpt work_dirs/talir_voc/15_5/seed42/task0_best.pth --epochs 2 --batch-size 4 --lr 5e-5 --eval-interval 1
```

```
python engine/train_task1.py --protocol 19_1 --seed 42 --task0-ckpt work_dirs/talir_voc/19_1/seed42/task0_best.pth --epochs 2 --batch-size 4 --lr 5e-5 --eval-interval 1
```

**输出位置：**

```
work_dirs/talir_voc/10_10/seed42/
├── task1_best.pth          # test All mAP50 最高的 checkpoint
├── task1_epoch1.pth
└── task1_epoch2.pth
```

---

### Step 4: 多 Seed 实验

如需补跑 seed 0 和 seed 1，取 3-seed 平均：

```bash
for SEED in 0 1; do
  # Task 0
  python engine/train_glip_voc.py --protocol 10_10 --seed $SEED --epochs 2 --batch-size 4 --lr 5e-5
  
  # Task 1 TAM
  python engine/train_task1.py --protocol 10_10 --seed $SEED --task0-ckpt work_dirs/talir_voc/10_10/seed${SEED}/task0_best.pth --epochs 2 --batch-size 4 --lr 5e-5 --eval-interval 1
  
  # Task 1 Baseline
  python engine/train_task1.py --protocol 10_10 --seed $SEED --task0-ckpt work_dirs/talir_voc/10_10/seed${SEED}/task0_best.pth --epochs 2 --batch-size 4 --lr 5e-5 --eval-interval 1 --no-tam
done
```

---

## 七、独立评估（备用）

训练脚本内部已包含正确的评估逻辑（test split, filter_task_classes=False）。如需单独评估某个 checkpoint：

```bash
# 评估 TAM 版本
python engine/eval_sis.py test

# 评估 Baseline 版本
python engine/eval_sis.py test baseline

# 评估解冻 backbone 版本
python engine/eval_sis.py test unfreeze
```



# **提供给测试小组的执行命令：**

### VOC2007（主表Table 2：10+10、15+5、19+1）

step1+step2，分别按照三种增量策略执行命令

### Step 1: Task 0 基类训练 + Step 2: Task 1 增量训练

```
#10+10
cd /your/path/to/TALIR-main
python engine/train_glip_voc.py --protocol 10_10 --seed 42 --epochs 2 --batch-size 4 --lr 5e-5 --eval-interval 1

python engine/train_task1.py --protocol 10_10 --seed 42 --task0-ckpt work_dirs/talir_voc/10_10/seed42/task0_best.pth --epochs 2 --batch-size 4 --lr 5e-5 --eval-interval 1

#如已下载训练好的对应增量策略的权重，可直接跳过前两个命令，直接执行下面这条命令
python engine/eval_sis.py test --protocol 10_10  --inference-strategy rowmax
```

```
#15+5
cd /your/path/to/TALIR-main
python engine/train_glip_voc.py --protocol 15_5 --seed 42 --epochs 2 --batch-size 4 --lr 5e-5 --eval-interval 1

python engine/train_task1.py --protocol 15_5 --seed 42 --task0-ckpt work_dirs/talir_voc/15_5/seed42/task0_best.pth --epochs 2 --batch-size 4 --lr 5e-5 --eval-interval 1

#如已下载训练好的对应增量策略的权重，可直接跳过前两个命令，直接执行下面这条命令
python engine/eval_sis.py test --protocol 15_5  --inference-strategy rowmax
```

```
#19+1
cd /your/path/to/TALIR-main
python engine/train_glip_voc.py --protocol 19_1 --seed 42 --epochs 2 --batch-size 4 --lr 5e-5 --eval-interval 1

python engine/train_task1.py --protocol 19_1 --seed 42 --task0-ckpt work_dirs/talir_voc/19_1/seed42/task0_best.pth --epochs 2 --batch-size 4 --lr 5e-5 --eval-interval 1

#如已下载训练好的对应增量策略的权重，可直接跳过前两个命令，直接执行下面这条命令
python engine/eval_sis.py test --protocol 19_1  --inference-strategy rowmax
```

### 消融实验（Table 4）

#### 论文中只在VOC2007数据集上和10+10策略上进行了消融实验，总共九个实验

### Baseline 

```
# Task 0 (无 TAM baseline)
python engine/train_glip_voc.py --protocol 10_10 --seed 42 --task-id 0 --epochs 2 --batch-size 4 --lr 5e-5 --no-tam

# Task 1 (无 TAM baseline)
python engine/train_task1.py --protocol 10_10 --task0-ckpt work_dirs\talir_voc\10_10\seed42_baseline\task0_best.pth --seed 42 --epochs 2 --batch-size 4 --lr 5e-5 --eval-interval 1 --no-tam

#如已下载训练好的10+10的baseline权重，可直接跳过前两个命令，直接执行下面这条命令
python engine/eval_sis.py test baseline --protocol 10_10  --inference-strategy none
```

### ROWMAX/elemax/elemean

```
# Step 1: 训练 Task 0 模型（旧类）
python engine/train_single.py --task-id 0 --protocol 10_10 --seed 42

# Step 2: 训练 Task 1 模型（新类）
python engine/train_single.py --task-id 1 --protocol 10_10 --seed 42

#如已下载训练好的seed_der里的no_tam权重，可直接跳过前两个命令，直接执行下面三条命令
# Step 3: Der-like SIS 评估
python engine/eval_der.py --protocol 10_10 --seed 42 --sis rowmax
python engine/eval_der.py --protocol 10_10 --seed 42 --sis elemax
python engine/eval_der.py --protocol 10_10 --seed 42 --sis elemean
```

### Vision Only + ROWMAX
```
#默认已有训练好的10+10权重
python engine/eval_sis.py test --tam-mode vision_only --protocol 10_10 --inference-strategy rowmax
```

### Language Only + ROWMAX
```
#默认已有训练好的10+10权重
python engine/eval_sis.py test --tam-mode language_only --protocol 10_10 --inference-strategy rowmax
```
### Vision+Language + ROWMAX/elemax/elemean

```
#默认已有训练好的10+10权重
python engine/eval_sis.py test --protocol 10_10  --inference-strategy elemax
python engine/eval_sis.py test --protocol 10_10  --inference-strategy elemean
python engine/eval_sis.py test --protocol 10_10  --inference-strategy rowmax
```

---

## 八、核心代码说明

| 文件 | 作用 |
|------|------|
| `talir_head.py` | TALIRHead：Visual+Language TAM 通道掩码 + SIS ROWMX/ELEMAX/ELEMEAN 融合 |
| `voc_glip_dataset.py` | VOC 数据集适配器；训练时 `filter_task_classes=True`，评估时 `False` |
| `train_glip_voc.py` | Task 0 基类训练脚本 |
| `train_task1.py` | Task 1 增量训练；支持 `--no-tam`（消融）、`--unfreeze-backbone` |
| `eval_sis.py` | SIS ROWMX/ELEMAX/ELEMEAN 融合评估脚本 |
| `eval_task1_fixed.py` | 分别评估 Old/New 的独立脚本 |
| `preprocess_voc.py` | VOC2007数据预处理 |
| `generate_test_split.py` | 从 VOC test set 生成 task0_test.txt / task1_test.txt |
| `train_single.py` | 消融实验中，训练单独任务的模型 |
| `eval_der.py` | 加载两个单独任务的模型进行SIS ROWMX/ELEMAX/ELEMEAN 融合评估 |

---

## 九、关键注意事项

1. **GPU 设置**：脚本默认 `CUDA_VISIBLE_DEVICES='0'`，请根据实际情况修改。
2. **batch_size**：默认 4（单卡 A6000 48GB）。论文用 16，如有多卡或更大显存可提升。
3. **冻结策略**：默认冻结 backbone + language model，只训练 neck/head。实验表明冻结 backbone 效果优于解冻。
4. **评估坐标**：必须使用 `rescale=False`，否则预测框和 GT 坐标空间不匹配。
6. **Deep Fusion**：当前未启用（`early_fuse=False`），预训练权重缺少 VLFuse 参数。这是和论文报告值差距的主要可能来源。

---

## 十、实验结果（seed 42, test split）

| 方法 | Old mAP50 | New mAP50 | All mAP50 | Avg |
|:-----|:---------:|:---------:|:---------:|:---------:|
| 15+5：TAM (V+L+ROWMAX)  主表 Table 2 | 58.24% | 77.65% | 63.35% | 66.41% |
| 19+1：TAM (V+L+ROWMAX)  主表 Table 2 | 68.68% | 64.66% | 68.32% | 67.22% |
| 10+10：TAM (V+L+ROWMAX) 主表 Table 2 | 55.28%                                    | 75.74%                    | 65.51%                    | 65.51%              |
| 10+10：TAM (V+L+ELEMAX) 消融Table 4 | 55.49%                                    | 75.93%                    | 65.71%                   | -                  |
| 10+10：TAM (V+L+ELEMEAN) 消融Table 4 | 56.09% | 72.53% | 64.31% | - |
| 10+10：Baseline 消融Table 4 | 52.79% | 76.88% | 64.84% | - |
| 10+10：ROWMAX 消融Table 4 | 63.49%    | 79.32%    | 71.36%     | -    |
| 10+10：ELEMAX 消融Table 4 | 62.80%    | 79.38%    | 71.09%     | -    |
| 10+10：ELEMEAN 消融Table 4 | 61.13%    | 75.67%    | 68.40%     | -    |
| 10+10：TAM (V+ROWMAX) 消融Table 4 | 55.32%                                    | 75.15%                    | 65.24%                | -               |
| 10+10：TAM (L+ROWMAX) 消融Table 4 | 58.57% | 76.85% | 67.71% | - |

---

