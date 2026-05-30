"""Der-like 单任务独立训练脚本（无 TAM，无增量机制）.

用于论文 Table 4 Lines 2-4：两个独立模型各自在自己的类别子集上微调，
推理时通过 SIS 策略融合结果，不共享参数，不使用 TAM。

用法:
    python projects/TALIR/train_single.py --task-id 0 --protocol 10_10 --seed 42
    python projects/TALIR/train_single.py --task-id 1 --protocol 10_10 --seed 42
"""
import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import argparse
import time
from typing import Dict, List

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmengine.config import Config
from mmdet.apis import init_detector
from mmdet.models.data_preprocessors import DetDataPreprocessor

from data.voc_glip_dataset import VOCGLIPDataset, VOC_CLASSES
from talir_head import TALIRHead
from mmdet.registry import MODELS

MODELS.register_module(module=TALIRHead, force=True)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_SCRIPT_DIR, 'config.json'), 'r', encoding='utf-8') as f:
    _cfg_data = json.load(f)
_PATHS = _cfg_data['paths']


def parse_args():
    parser = argparse.ArgumentParser(description='Der-like single-task training (no TAM)')
    parser.add_argument('--task-id', type=int, required=True,
                        help='Task ID: 0 for old/base classes, 1 for new/incremental classes')
    parser.add_argument('--data-root', default=_PATHS['data_root'])
    parser.add_argument('--protocol', default='10_10')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr-language', type=float, default=0.0)
    parser.add_argument('--lr-backbone', type=float, default=0.0)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--freeze-backbone', action='store_true', default=True)
    parser.add_argument('--freeze-language', action='store_true', default=True)
    parser.add_argument('--work-dir', default=_PATHS['work_dir'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--eval-interval', type=int, default=1)
    return parser.parse_args()


def build_model(device: str):
    cfg = Config.fromfile(_PATHS['glip_config'])
    cfg.model.bbox_head.type = 'TALIRHead'
    cfg.model.bbox_head.num_classes = 20
    lang_model_name = _PATHS['bert_model_name']
    cfg.model.bbox_head.lang_model_name = lang_model_name
    cfg.model.language_model.name = lang_model_name
    model = init_detector(cfg, _PATHS['pretrained_checkpoint'], device=device)
    return model


def collate_fn(batch: List[Dict]) -> Dict:
    inputs = [item['inputs'] for item in batch]
    data_samples = [item['data_samples'] for item in batch]
    return {'inputs': inputs, 'data_samples': data_samples}


def train_one_epoch(model, preprocessor, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_loss_cls = 0.0
    total_loss_bbox = 0.0
    total_loss_centerness = 0.0
    num_batches = 0

    start_time = time.time()
    for i, data in enumerate(dataloader):
        data = preprocessor(data, training=True)
        inputs = data['inputs'].to(device)
        data_samples = data['data_samples']

        losses = model.loss(inputs, data_samples)
        loss = losses['loss_cls'] + losses['loss_bbox'] + losses['loss_centerness']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_loss_cls += losses['loss_cls'].item()
        total_loss_bbox += losses['loss_bbox'].item()
        total_loss_centerness += losses['loss_centerness'].item()
        num_batches += 1

        if i % 50 == 0 or i == len(dataloader) - 1:
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            print(f"[Epoch {epoch}] [{i}/{len(dataloader)}] LR: {lr:.2e} | "
                  f"Loss: {loss.item():.4f} (Cls:{losses['loss_cls'].item():.4f} "
                  f"Bbox:{losses['loss_bbox'].item():.4f} "
                  f"Ctr:{losses['loss_centerness'].item():.4f}) | "
                  f"Time: {elapsed:.1f}s")

    return {
        'loss': total_loss / num_batches,
        'loss_cls': total_loss_cls / num_batches,
        'loss_bbox': total_loss_bbox / num_batches,
        'loss_centerness': total_loss_centerness / num_batches,
    }


@torch.no_grad()
def evaluate(model, preprocessor, dataloader, device, target_classes):
    """评估模型在目标类别上的 mAP@0.5."""
    model.eval()

    prompt = '. '.join(target_classes) + '.'
    predictions = []
    ground_truths = {}

    for data in dataloader:
        for ds in data['data_samples']:
            img_id = ds.metainfo['img_id']
            gt_boxes = ds.gt_instances.bboxes.cpu().numpy()
            gt_labels = ds.gt_instances.labels.cpu().numpy()
            ground_truths[img_id] = {'boxes': gt_boxes, 'labels': gt_labels}

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
                for box, score, label in zip(pred.bboxes.cpu().numpy(),
                                             pred.scores.cpu().numpy(),
                                             pred.labels.cpu().numpy()):
                    # predict 返回的 label 是 prompt 内局部索引，映射到真实类别名
                    if 0 <= label < len(target_classes):
                        predictions.append({
                            'image_id': img_id,
                            'class_name': target_classes[label],
                            'bbox': box.tolist(),
                            'score': float(score),
                        })
    return compute_ap(predictions, ground_truths, target_classes)


def compute_ap(predictions, ground_truths, target_classes):
    from collections import defaultdict
    pred_by_class = defaultdict(list)
    for pred in predictions:
        if pred['class_name'] in target_classes:
            pred_by_class[pred['class_name']].append(pred)

    gt_by_class = defaultdict(lambda: defaultdict(list))
    for img_id, gt in ground_truths.items():
        for box, label in zip(gt['boxes'], gt['labels']):
            class_name = VOC_CLASSES[label]
            if class_name in target_classes:
                gt_by_class[class_name][img_id].append(box)

    aps = []
    for class_name in target_classes:
        preds = sorted(pred_by_class[class_name], key=lambda x: x['score'], reverse=True)
        gt_dict = gt_by_class[class_name]
        npos = sum(len(v) for v in gt_dict.values())
        if npos == 0:
            continue

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        matched = {img_id: [False] * len(boxes) for img_id, boxes in gt_dict.items()}

        for j, pred in enumerate(preds):
            img_id = pred['image_id']
            if img_id not in gt_dict or len(gt_dict[img_id]) == 0:
                fp[j] = 1
                continue
            pred_box = np.array(pred['bbox'])
            ious = [compute_iou(pred_box, gt_box) for gt_box in gt_dict[img_id]]
            max_iou, max_idx = max(ious), ious.index(max(ious))
            if max_iou >= 0.5 and not matched[img_id][max_idx]:
                tp[j] = 1
                matched[img_id][max_idx] = True
            else:
                fp[j] = 1

        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        rec, prec = tp_cum / npos, tp_cum / (tp_cum + fp_cum + 1e-6)
        ap = sum(np.max(prec[rec >= t]) if np.any(rec >= t) else 0
                 for t in np.linspace(0, 1, 11)) / 11
        aps.append(ap)

    return np.mean(aps) if len(aps) > 0 else 0.0


def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Training Task {args.task_id} (single, no TAM)")

    # ── 加载类别信息 ──
    split_file = os.path.join(args.data_root, f'incremental_{args.protocol}',
                              f'seed{args.seed}', 'category_split.json')
    with open(split_file) as f:
        split_info = json.load(f)
    task_classes = split_info['task_classes'][args.task_id]
    print(f"[INFO] Task classes ({len(task_classes)}): {task_classes}")

    # ── 构建模型（无 TAM）──
    model = build_model(args.device)
    model.bbox_head.tam_cfg = None
    model.bbox_head.tam_mode = 'vision_only'
    model.bbox_head.inference_strategy = 'none'
    model.to(device)

    # ── 冻结策略 ──
    if args.freeze_backbone:
        print("[FREEZE] Backbone frozen")
        for param in model.backbone.parameters():
            param.requires_grad = False
    if args.freeze_language:
        print("[FREEZE] Language model frozen")
        for param in model.language_model.parameters():
            param.requires_grad = False

    # ── 优化器 ──
    backbone_params, language_params, neck_head_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        elif 'language_model' in name or 'bert' in name:
            language_params.append(param)
        else:
            neck_head_params.append(param)

    param_groups = []
    if neck_head_params and args.lr > 0:
        param_groups.append({'params': neck_head_params, 'lr': args.lr})
    if backbone_params and args.lr_backbone > 0:
        param_groups.append({'params': backbone_params, 'lr': args.lr_backbone})
    if language_params and args.lr_language > 0:
        param_groups.append({'params': language_params, 'lr': args.lr_language})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # ── Data Preprocessor ──
    preprocessor = DetDataPreprocessor(
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32,
    ).to(device)

    # ── 数据集 ──
    train_dataset = VOCGLIPDataset(
        data_root=args.data_root, task_id=args.task_id,
        protocol=args.protocol, seed=args.seed,
        split='trainval', filter_task_classes=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    # 测试集：不按 task 过滤，评估对应类别的 mAP
    test_dataset = VOCGLIPDataset(
        data_root=args.data_root, task_id=args.task_id,
        protocol=args.protocol, seed=args.seed,
        split='test', filter_task_classes=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    # ── 输出目录 ──
    output_dir = os.path.join(args.work_dir, args.protocol, f'seed{args.seed}_der')
    os.makedirs(output_dir, exist_ok=True)

    # ── 训练循环 ──
    best_map = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, preprocessor, train_loader, optimizer, device, epoch)
        print(f"[Epoch {epoch}] Train: loss={train_metrics['loss']:.4f} "
              f"(cls={train_metrics['loss_cls']:.4f} bbox={train_metrics['loss_bbox']:.4f} "
              f"ctr={train_metrics['loss_centerness']:.4f})")

        # 每个 epoch 评估并显示 mAP
        cur_map = evaluate(model, preprocessor, test_loader, device, task_classes)
        print(f"[Epoch {epoch}] Task{args.task_id} mAP@0.5 = {cur_map:.4f} (best={best_map:.4f})")

        if cur_map > best_map:
            best_map = cur_map
            best_epoch = epoch
            save_path = os.path.join(output_dir, f'task{args.task_id}_notam.pth')
            torch.save({'model_state_dict': model.state_dict()}, save_path)
            print(f"[Model Saved] {save_path} (epoch={epoch}, mAP={best_map:.4f})")

        # 每个 epoch 都保存 checkpoint（用于 resume）
        ckpt_path = os.path.join(output_dir, f'task{args.task_id}_epoch{epoch}.pth')
        torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, ckpt_path)

    print(f"\n{'='*60}")
    print(f"[DONE] Task {args.task_id} single training completed.")
    print(f"       Best epoch: {best_epoch}, mAP@0.5 = {best_map:.4f}")
    print(f"       Output: {output_dir}/task{args.task_id}_notam.pth")


if __name__ == '__main__':
    main()