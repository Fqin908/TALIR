"""SIS 评估：自动识别协议/seed，加载对应权重和类别.

Usage:
    python projects/TALIR/eval_sis.py test --protocol 10_10 --seed 42
    python projects/TALIR/eval_sis.py test --protocol 15_5 --seed 42 --inference-strategy elemax --tam-mode vision_only
    python projects/TALIR/eval_sis.py test baseline --protocol 19_1 --seed 42
"""
import os
import json
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np
from mmengine.config import Config
from mmdet.apis import init_detector
from mmdet.models.data_preprocessors import DetDataPreprocessor
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.voc_glip_dataset import VOCGLIPDataset, VOC_CLASSES
from talir_head import TALIRHead
from mmdet.registry import MODELS

MODELS.register_module(module=TALIRHead, force=True)

# 加载配置文件

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_SCRIPT_DIR, 'config.json'), 'r', encoding='utf-8') as f:
    _cfg_data = json.load(f)
_PATHS = _cfg_data['paths']

device = torch.device('cuda:0')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='SIS Evaluation')
    parser.add_argument('split', nargs='?', default='test', choices=['test', 'trainval'],
                        help='Dataset split to evaluate on')
    parser.add_argument('variant', nargs='?', default='tam', choices=['tam', 'baseline', 'unfreeze'],
                        help='Model variant: tam / baseline / unfreeze')
    parser.add_argument('epoch', nargs='?', type=int, default=1,
                        help='Checkpoint epoch number')
    parser.add_argument('--protocol', default='10_10',
                        help='Incremental protocol (10_10, 15_5, 19_1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--inference-strategy', default='rowmax',
                        choices=['rowmax', 'elemax', 'elemean', 'none'],
                        help='SIS fusion strategy (none = standard per-class, no SIS)')
    parser.add_argument('--tam-mode', default='full',
                        choices=['full', 'vision_only', 'language_only'],
                        help='TAM branch ablation mode')
    return parser.parse_args()


args = parse_args()
protocol = args.protocol
seed = args.seed

# 加载类别划分
split_file = os.path.join(_PATHS['data_root'], f'incremental_{protocol}', f'seed{seed}', 'category_split.json')
with open(split_file) as f:
    split_info = json.load(f)
task_classes = split_info['task_classes']
old_classes = task_classes[0]
new_classes = task_classes[1]
all_classes = old_classes + new_classes
print(f"[INFO] Protocol={protocol}, Seed={seed}")
print(f"[INFO] Old ({len(old_classes)}): {old_classes}")
print(f"[INFO] New ({len(new_classes)}): {new_classes}")

# 构建模型
cfg = Config.fromfile(_PATHS['glip_config'])
cfg.model.bbox_head.type = 'TALIRHead'
cfg.model.bbox_head.num_classes = 20
cfg.model.bbox_head.lang_model_name = _PATHS['bert_model_name']
cfg.model.language_model.name = _PATHS['bert_model_name']
model = init_detector(cfg, _PATHS['pretrained_checkpoint'], device=device)

# 加载 checkpoint
suffix = '' if args.variant == 'tam' else ('_baseline' if args.variant == 'baseline' else '_unfreeze')
work_subdir = os.path.join(_PATHS['work_dir'], protocol, f'seed{seed}')
ckpt_path = os.path.join(work_subdir, f'task1_best.pth')
print(f"[INFO] Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

# TAM 配置
total_classes = sum(len(t) for t in task_classes)
if args.variant == 'baseline':
    model.bbox_head.tam_cfg = None
    print("[BASELINE] TAM disabled for evaluation")
else:
    total_ch = 256
    cumsum = 0
    visual_splits = {}
    language_splits = {}
    for tid, tcls in enumerate(task_classes):
        n_cls = len(tcls)
        n_ch = int(total_ch * n_cls / total_classes)
        visual_splits[tid] = (cumsum, cumsum + n_ch)
        language_splits[tid] = (cumsum, cumsum + n_ch)
        cumsum += n_ch
    model.bbox_head.tam_cfg = {
        'visual_splits': visual_splits,
        'language_splits': language_splits,
    }
    print(f"[TAM] Visual splits: {visual_splits}")
    print(f"[TAM] Language splits: {language_splits}")

model.bbox_head.inference_strategy = args.inference_strategy
model.bbox_head.tam_mode = args.tam_mode
print(f"[INFO] Inference strategy: {args.inference_strategy}, TAM mode: {args.tam_mode}")

preprocessor = DetDataPreprocessor(
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    bgr_to_rgb=False,
    pad_size_divisor=32,
).to(device)

# 评估数据集
print(f"[INFO] Evaluating on {args.split} split")
dataset = VOCGLIPDataset(
    data_root=_PATHS['data_root'],
    task_id=1, protocol=protocol, seed=seed, split=args.split,
    filter_task_classes=False,
)


def collate_fn(batch):
    return {'inputs': [item['inputs'] for item in batch],
            'data_samples': [item['data_samples'] for item in batch]}


loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

all_prompt = '. '.join(all_classes) + '.'


@torch.no_grad()
def evaluate_all(model, prompt, loader, device):
    model.eval()
    print(f"\n[EVAL] All prompt: {prompt[:50]}...")

    all_predictions = []
    all_gts = {}

    for i, data in enumerate(loader):
        for ds in data['data_samples']:
            img_id = ds.metainfo['img_id']
            all_gts[img_id] = {
                'boxes': ds.gt_instances.bboxes.cpu().numpy(),
                'labels': ds.gt_instances.labels.cpu().numpy(),
            }

        data = preprocessor(data, training=False)
        inputs = data['inputs'].to(device)
        data_samples = data['data_samples']
        for ds in data_samples:
            ds.text = prompt
            ds.custom_entities = True

        results = model.predict(inputs, data_samples, rescale=False)
        for ds in results:
            img_id = ds.metainfo['img_id']
            pred = ds.pred_instances
            if len(pred) > 0:
                for box, score, label in zip(pred.bboxes.cpu().numpy(), pred.scores.cpu().numpy(), pred.labels.cpu().numpy()):
                    if 0 <= label < len(all_classes):
                        all_predictions.append({
                            'image_id': img_id,
                            'class_name': all_classes[label],
                            'bbox': box.tolist(),
                            'score': float(score),
                        })
        if i % 20 == 0 or i == len(loader) - 1:
            print(f"  [{i}/{len(loader)}] preds={len(all_predictions)}")

    print(f"[EVAL] Total predictions: {len(all_predictions)}")

    from collections import defaultdict
    pred_by_class = defaultdict(list)
    for p in all_predictions:
        pred_by_class[p['class_name']].append(p)

    gt_by_class = defaultdict(lambda: defaultdict(list))
    for img_id, gt in all_gts.items():
        for box, label in zip(gt['boxes'], gt['labels']):
            class_name = VOC_CLASSES[label]
            gt_by_class[class_name][img_id].append(box)

    def calc_ap(preds, gt_dict):
        if len(preds) == 0:
            return 0.0, 0
        npos = sum(len(v) for v in gt_dict.values())
        if npos == 0:
            return None, 0
        preds = sorted(preds, key=lambda x: x['score'], reverse=True)
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        matched = {img_id: [False] * len(boxes) for img_id, boxes in gt_dict.items()}

        for j, p in enumerate(preds):
            img_id = p['image_id']
            if img_id not in gt_dict or len(gt_dict[img_id]) == 0:
                fp[j] = 1
                continue
            ious = [compute_iou(np.array(p['bbox']), gt_box) for gt_box in gt_dict[img_id]]
            max_iou = max(ious)
            max_idx = ious.index(max_iou)
            if max_iou >= 0.5 and not matched[img_id][max_idx]:
                tp[j] = 1
                matched[img_id][max_idx] = True
            else:
                fp[j] = 1

        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        rec, prec = tp_cum / npos, tp_cum / (tp_cum + fp_cum + 1e-6)
        ap = sum(np.max(prec[rec >= t]) if np.any(rec >= t) else 0 for t in np.linspace(0, 1, 11)) / 11
        return ap, npos

    old_aps = []
    new_aps = []
    all_aps = []

    for cls in all_classes:
        ap, npos = calc_ap(pred_by_class[cls], gt_by_class[cls])
        if ap is not None:
            tag = "Old" if cls in old_classes else "New"
            print(f"  [{tag}] {cls}: AP={ap:.4f}, preds={len(pred_by_class[cls])}, GT={npos}")
            all_aps.append(ap)
            if cls in old_classes:
                old_aps.append(ap)
            else:
                new_aps.append(ap)

    print(f"\n{'=' * 60}")
    print(f"SIS Old mAP50 = {np.mean(old_aps):.4f}")
    print(f"SIS New mAP50 = {np.mean(new_aps):.4f}")
    print(f"SIS All mAP50 = {np.mean(all_aps):.4f}")
    print(f"{'=' * 60}")


def compute_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1, a2 = (box1[2] - box1[0]) * (box1[3] - box1[1]), (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-6)


evaluate_all(model, all_prompt, loader, device)