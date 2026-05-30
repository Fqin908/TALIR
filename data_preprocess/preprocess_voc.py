#!/usr/bin/env python3
"""Pascal VOC 2007 增量数据集预处理脚本。

论文依据 (Datasets and Evaluation Metrics):
    - 数据集: Pascal VOC 2007, 20个目标类别
    - 训练集: trainval子集; 测试集: test子集
    - 评估指标: mAP@0.5 IoU阈值
    - 划分方式: 与CIOD (Dong et al. 2023)保持一致
    - 增量协议: 19+1, 15+5, 10+10
    - 论文运行3次不同随机顺序取平均

使用方法:
    python tools/preprocess_voc.py \
        --data-root ./data/voc2007 \
        --protocol 10_10 \
        --seed 42

输出:
    - data/voc2007/incremental_{protocol}/
        - seed42/
            - task0_trainval.txt
            - task1_trainval.txt
            - category_split.json
        - seed123/
        - seed2024/
"""
import os
import json
import argparse
import random
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Set
from collections import defaultdict

# 加载配置文件
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_SCRIPT_DIR, 'config.json'), 'r', encoding='utf-8') as f:
    _cfg_data = json.load(f)
_PATHS = _cfg_data['paths']


class VOCIncrementalSplitter:
    """Pascal VOC 2007增量数据集划分器。

    按CIOD的划分方式生成任务特定的图像列表和标注。
    论文运行3次不同随机顺序，报告平均mAP。

    Attributes:
        data_root (str): VOC 2007数据集根目录
        protocol (str): 增量协议，如 "10_10", "15_5", "19_1"
        seed (int): 随机种子
        num_classes (int): VOC总类别数，20
    """

    # Pascal VOC 2007 20个类别名称（标准顺序）
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    def __init__(self, data_root: str, protocol: str, seed: int = 42):
        """初始化VOC增量划分器。

        Args:
            data_root: VOC 2007数据集根目录
            protocol: 增量协议字符串，如 "10_10", "15_5", "19_1"
            seed: 随机种子
        """
        self.data_root = data_root
        self.protocol = protocol
        self.seed = seed
        self.num_classes = 20

        # 解析协议
        self.task_splits = self._parse_protocol(protocol)
        self.num_tasks = len(self.task_splits)

        # 设置随机种子并打乱类别顺序
        random.seed(seed)
        self.shuffled_classes = self.VOC_CLASSES.copy()
        random.shuffle(self.shuffled_classes)

        # 构建类别到任务的映射
        self.task_classes = []
        idx = 0
        for task_id, num_cls in enumerate(self.task_splits):
            task_cls = self.shuffled_classes[idx:idx + num_cls]
            self.task_classes.append(task_cls)
            idx += num_cls

    def _parse_protocol(self, protocol: str) -> List[int]:
        """解析协议字符串。

        Args:
            protocol: 如 "10_10", "15_5", "19_1"

        Returns:
            List[int]: 每个任务的类别数
        """
        parts = protocol.split('_')
        return [int(parts[0]), int(parts[1])]

    def _parse_voc_xml(self, xml_path: str) -> Tuple[List[str], List[List[float]]]:
        """解析VOC XML标注文件。

        Args:
            xml_path: XML文件路径

        Returns:
            Tuple[List[str], List[List[float]]]: 类别列表和bbox列表
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        classes = []
        boxes = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
            if difficult:
                continue

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            classes.append(name)
            boxes.append([xmin, ymin, xmax, ymax])

        return classes, boxes

    def _get_image_list(self, split: str) -> List[str]:
        """获取指定split的图像ID列表。

        Args:
            split: 'trainval' 或 'test'

        Returns:
            List[str]: 图像ID列表
        """
        list_file = os.path.join(
            self.data_root, 'ImageSets', 'Main', f'{split}.txt'
        )
        with open(list_file, 'r') as f:
            lines = f.readlines()
        return [line.strip().split()[0] for line in lines]

    def generate_splits(self):
        """生成增量划分。

        为每个task生成图像列表文件（.txt），同时保存类别划分JSON。
        每个图像可能包含多个类别的实例，但只保留当前task类别的标注。
        """
        output_dir = os.path.join(
            self.data_root, f'incremental_{self.protocol}', f'seed{self.seed}'
        )
        os.makedirs(output_dir, exist_ok=True)

        # 获取trainval图像列表
        image_ids = self._get_image_list('trainval')

        # 解析所有图像的标注，建立类别到图像的映射
        class_to_images = defaultdict(set)
        image_to_annotations = {}

        for img_id in image_ids:
            xml_path = os.path.join(self.data_root, 'Annotations', f'{img_id}.xml')
            if not os.path.exists(xml_path):
                continue

            classes, boxes = self._parse_voc_xml(xml_path)
            image_to_annotations[img_id] = {'classes': classes, 'boxes': boxes}

            for cls in set(classes):
                class_to_images[cls].add(img_id)

        # 保存类别划分信息
        splits_info = {
            'protocol': self.protocol,
            'seed': self.seed,
            'num_tasks': self.num_tasks,
            'task_splits': self.task_splits,
            'task_classes': self.task_classes,
            'shuffled_order': self.shuffled_classes,
        }
        with open(os.path.join(output_dir, 'category_split.json'), 'w') as f:
            json.dump(splits_info, f, indent=2)

        # 为每个task生成训练图像列表
        for task_id in range(self.num_tasks):
            target_classes = self.task_classes[task_id]
            target_classes_set = set(target_classes)

            # 收集包含当前task类别的图像
            task_images = set()
            for cls in target_classes:
                task_images.update(class_to_images.get(cls, set()))

            # 写入图像列表文件
            out_file = os.path.join(
                output_dir, f'task{task_id}_trainval.txt'
            )
            with open(out_file, 'w') as f:
                for img_id in sorted(task_images):
                    f.write(f"{img_id}\n")

            print(f"[INFO] Task {task_id}: {len(task_images)} images, {len(target_classes)} classes")
            print(f"       Classes: {target_classes}")
            print(f"       Saved to {out_file}")

        print(f"[SUCCESS] VOC splits generated in {output_dir}")


def main():
    """主入口函数。"""
    parser = argparse.ArgumentParser(
        description='Pascal VOC 2007 Incremental Dataset Preprocessing'
    )
    parser.add_argument(
        '--data-root', type=str, default=_PATHS['data_root'],
        help='VOC 2007 dataset root directory'
    )
    parser.add_argument(
        '--protocol', type=str, default='10_10',
        choices=['19_1', '15_5', '10_10'],
        help='Incremental protocol: B+I'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for category shuffling'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Pascal VOC 2007 Incremental Splitter")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Protocol: {args.protocol}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    splitter = VOCIncrementalSplitter(
        data_root=args.data_root,
        protocol=args.protocol,
        seed=args.seed,
    )
    splitter.generate_splits()

    print("[DONE] VOC preprocessing completed.")


if __name__ == '__main__':
    main()
