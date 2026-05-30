"""VOC 2007 GLIP 格式数据集.

预处理好的增量分割，输出 mmdetection GLIP 所需的格式.
"""
import os
import json
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def parse_voc_xml(xml_path: str) -> Tuple[np.ndarray, List[str]]:
    """解析 VOC XML，返回 boxes 和 class names."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    classes = []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip().lower()
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        classes.append(name)
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), []
    return np.array(boxes, dtype=np.float32), classes


def resize_image_and_boxes(img: Image.Image, boxes: np.ndarray,
                           max_long: int = 1333, max_short: int = 800):
    """保持长宽比 resize，短边不超过 max_short，长边不超过 max_long."""
    w, h = img.size
    ratio = max_short / min(h, w)
    new_h = int(round(h * ratio))
    new_w = int(round(w * ratio))
    if max(new_h, new_w) > max_long:
        ratio = max_long / max(new_h, new_w)
        new_h = int(round(new_h * ratio))
        new_w = int(round(new_w * ratio))
    
    img = img.resize((new_w, new_h), Image.BILINEAR)
    if len(boxes) > 0:
        scale_x = new_w / w
        scale_y = new_h / h
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
    return img, boxes, (new_w, new_h)


class VOCGLIPDataset(Dataset):
    """VOC 2007 增量数据集，适配 GLIP 训练.
    
    Args:
        data_root: VOC 2007 根目录（如 ./data/voc2007）
        task_id: 当前任务 ID（0=基类，1=增量类）
        protocol: 增量协议，如 "10_10"
        seed: 随机种子
        split: trainval 或 test
        filter_task_classes: 是否只保留当前 task 的类别（训练时 True，评估时 False）
    """
    
    def __init__(self, data_root: str, task_id: int, protocol: str,
                 seed: int = 42, split: str = 'trainval',
                 filter_task_classes: bool = True):
        self.data_root = data_root
        self.task_id = task_id
        self.split = split
        self.filter_task_classes = filter_task_classes
        
        # 读取类别分割
        split_dir = os.path.join(data_root, f'incremental_{protocol}', f'seed{seed}')
        with open(os.path.join(split_dir, 'category_split.json'), 'r') as f:
            split_info = json.load(f)
        
        self.task_classes = split_info['task_classes']
        self.current_classes = self.task_classes[task_id]
        
        # GLIP prompt: 类别用 ". " 连接，尾部加 "."
        self.prompt = '. '.join(self.current_classes) + '.'
        self.class_to_idx = {name: i for i, name in enumerate(self.current_classes)}
        
        # 读取图像列表
        img_list_file = os.path.join(split_dir, f'task{task_id}_{split}.txt')
        with open(img_list_file, 'r') as f:
            all_img_ids = [line.strip() for line in f.readlines()]
        
        # 过滤：只保留包含当前 task 类别的图像（训练时）
        if split == 'trainval':
            self.img_ids = []
            for img_id in all_img_ids:
                ann_path = os.path.join(data_root, 'Annotations', f'{img_id}.xml')
                _, classes = parse_voc_xml(ann_path)
                if any(c in self.class_to_idx for c in classes):
                    self.img_ids.append(img_id)
        else:
            self.img_ids = all_img_ids
        
        print(f"[VOCGLIP] Task {task_id} {split}: {len(self.img_ids)} images, "
              f"{len(self.current_classes)} classes")
        print(f"[VOCGLIP] Prompt: {self.prompt}")
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        # 读取图像
        img_path = os.path.join(self.data_root, 'JPEGImages', f'{img_id}.jpg')
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        
        # 读取标注
        ann_path = os.path.join(self.data_root, 'Annotations', f'{img_id}.xml')
        boxes, classes = parse_voc_xml(ann_path)
        
        # 过滤当前 task 类别（训练时过滤，评估时保留全部用于 mAP 计算）
        if self.filter_task_classes:
            filtered_boxes = []
            filtered_labels = []
            for box, cls in zip(boxes, classes):
                if cls in self.class_to_idx:
                    filtered_boxes.append(box)
                    filtered_labels.append(self.class_to_idx[cls])
            if len(filtered_boxes) == 0:
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)
            else:
                boxes = np.array(filtered_boxes, dtype=np.float32)
                labels = np.array(filtered_labels, dtype=np.int64)
        else:
            # 评估模式：保留所有 20 类的 GT
            all_class_to_idx = {name: i for i, name in enumerate(VOC_CLASSES)}
            filtered_boxes = []
            filtered_labels = []
            for box, cls in zip(boxes, classes):
                if cls in all_class_to_idx:
                    filtered_boxes.append(box)
                    filtered_labels.append(all_class_to_idx[cls])
            if len(filtered_boxes) == 0:
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)
            else:
                boxes = np.array(filtered_boxes, dtype=np.float32)
                labels = np.array(filtered_labels, dtype=np.int64)
        
        # Resize
        img, boxes, (new_w, new_h) = resize_image_and_boxes(img, boxes)
        
        # 转为 tensor: [3, H, W], uint8, RGB 顺序（DetDataPreprocessor 会做 BGR 转换）
        img_np = np.array(img)  # [H, W, 3]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [3, H, W]
        
        # 构建 DetDataSample
        data_sample = DetDataSample()
        data_sample.text = self.prompt
        data_sample.custom_entities = True  # 关键：跳过 NLTK NER
        data_sample.set_metainfo({
            'img_id': img_id,
            'img_path': img_path,
            'ori_shape': (orig_h, orig_w),
            'img_shape': (new_h, new_w),
            'scale_factor': (new_w / orig_w, new_h / orig_h),
        })
        
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.from_numpy(boxes)
        gt_instances.labels = torch.from_numpy(labels)
        data_sample.gt_instances = gt_instances
        
        return {'inputs': img_tensor, 'data_samples': data_sample}
