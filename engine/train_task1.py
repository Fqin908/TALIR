"""TALIR Task 1 增量训练脚本.

加载 Task 0 最优 checkpoint，切换 TAM 到 Task 1，冻结旧参数，只训练新类.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import argparse
import json
import time
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmengine.config import Config
from mmdet.apis import init_detector
from mmdet.models.data_preprocessors import DetDataPreprocessor
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


def parse_args():
    parser = argparse.ArgumentParser(description='TALIR Task 1 Incremental Training')
    parser.add_argument('--data-root', default=_PATHS['data_root'], help='VOC 2007 root')
    parser.add_argument('--protocol', default='10_10', help='Incremental protocol')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--task0-ckpt', required=True, help='Task 0 best checkpoint path')
    parser.add_argument('--epochs', type=int, default=2, help='Max epochs for Task 1')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for head/neck')
    parser.add_argument('--lr-language', type=float, default=0.0, help='Language model LR (0=frozen)')
    parser.add_argument('--lr-backbone', type=float, default=0.0, help='Backbone LR (0=frozen)')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--freeze-backbone', action='store_true', default=True, help='Freeze backbone')
    parser.add_argument('--freeze-language', action='store_true', default=True, help='Freeze language model')
    parser.add_argument('--unfreeze-backbone', action='store_true', default=False, help='Unfreeze backbone (paper setting)')
    parser.add_argument('--no-tam', action='store_true', default=False, help='Disable TAM (baseline without masking)')
    parser.add_argument('--tam-mode', default='full', choices=['full', 'vision_only', 'language_only'],
                        help='TAM branch ablation mode (default: full)')
    parser.add_argument('--work-dir-suffix', default='', help='Suffix for work dir to avoid overwriting')
    parser.add_argument('--work-dir', default=_PATHS['work_dir'], help='Output directory')
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--eval-interval', type=int, default=1, help='Evaluate every N epochs')
    return parser.parse_args()


def build_model(checkpoint_path: str, device: str):
    config_file = _PATHS['glip_config']
    cfg = Config.fromfile(config_file)
    cfg.model.bbox_head.type = 'TALIRHead'
    cfg.model.bbox_head.num_classes = 20
    lang_model_name = _PATHS['bert_model_name']
    cfg.model.bbox_head.lang_model_name = lang_model_name
    cfg.model.language_model.name = lang_model_name
    model = init_detector(cfg, checkpoint_path, device=device)
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
def evaluate(model, preprocessor, dataloader, device, all_classes, old_classes, new_classes):
    """评估 old / new / all 的 mAP@0.5."""
    model.eval()
    
    old_prompt = '. '.join(old_classes) + '.'
    new_prompt = '. '.join(new_classes) + '.'
    all_prompt = '. '.join(all_classes) + '.'
    
    # 分别用三个 prompt 评估
    def _eval_with_prompt(prompt):
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
                    boxes = pred.bboxes.cpu().numpy()
                    scores = pred.scores.cpu().numpy()
                    labels = pred.labels.cpu().numpy()
                    for box, score, label in zip(boxes, scores, labels):
                        if 0 <= label < len(all_classes):
                            predictions.append({
                                'image_id': img_id,
                                'class_name': all_classes[label],
                                'bbox': box.tolist(),
                                'score': float(score),
                            })
        return predictions, ground_truths
    
    # 用 all_prompt 评估（对应论文 Overall）
    all_predictions, all_ground_truths = _eval_with_prompt(all_prompt)
    
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
        
        import numpy as np
        aps = []
        for class_name in target_classes:
            preds = sorted(pred_by_class[class_name], key=lambda x: x['score'], reverse=True)
            gt_dict = gt_by_class[class_name]
            npos = sum(len(v) for v in gt_dict.values())
            if npos == 0:
                continue
            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))
            matched = {img_id: [False]*len(boxes) for img_id, boxes in gt_dict.items()}
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
            ap = sum(np.max(prec[rec >= t]) if np.any(rec >= t) else 0 for t in np.linspace(0, 1, 11)) / 11
            aps.append(ap)
        return np.mean(aps) if len(aps) > 0 else 0.0
    
    old_map = compute_ap(all_predictions, all_ground_truths, old_classes)
    new_map = compute_ap(all_predictions, all_ground_truths, new_classes)
    all_map = compute_ap(all_predictions, all_ground_truths, all_classes)
    
    return {'old_mAP50': old_map, 'new_mAP50': new_map, 'all_mAP50': all_map}


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # 加载 Task 0 checkpoint
    print(f"[INFO] Loading Task 0 checkpoint: {args.task0_ckpt}")
    model = build_model(_PATHS['pretrained_checkpoint'], args.device)
    
    # 加载 Task 0 训练权重
    ckpt = torch.load(args.task0_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print("[INFO] Loaded Task 0 weights")
    
    model.to(device)
    
    # 配置 TAM
    total_ch = 256
    split_file = os.path.join(args.data_root, f'incremental_{args.protocol}', f'seed{args.seed}', 'category_split.json')
    with open(split_file) as f:
        split_info = json.load(f)
    all_task_classes = split_info['task_classes']
    total_classes = sum(len(t) for t in all_task_classes)
    
    cumsum = 0
    visual_splits = {}
    language_splits = {}
    for tid, tcls in enumerate(all_task_classes):
        n_cls = len(tcls)
        n_ch = int(total_ch * n_cls / total_classes)
        visual_splits[tid] = (cumsum, cumsum + n_ch)
        language_splits[tid] = (cumsum, cumsum + n_ch)
        cumsum += n_ch
    
    if args.no_tam:
        model.bbox_head.tam_cfg = None
        model.bbox_head.tam_mode = 'vision_only'  # tam_cfg=None 时 tam_mode 无影响，此处显式设值
        print("[BASELINE] TAM disabled (mask all 1s)")
    else:
        model.bbox_head.tam_cfg = {
            'visual_splits': visual_splits,
            'language_splits': language_splits,
        }
        model.bbox_head.tam_mode = args.tam_mode
        print(f"[TAM] Visual splits: {visual_splits}")
        print(f"[TAM] Language splits: {language_splits}")
        print(f"[TAM] tam_mode: {args.tam_mode}")
    model.bbox_head.current_task_id = 1  # 切换到 Task 1
    print(f"[TAM] Current task_id: 1 (incremental)")
    
    # 冻结策略（论文默认：整个模型都更新，不冻结）
    if args.unfreeze_backbone:
        args.freeze_backbone = False
    if args.freeze_backbone:
        print("[FREEZE] Backbone frozen")
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        print("[UNFREEZE] Backbone trainable (paper setting)")
    if args.freeze_language:
        print("[FREEZE] Language model frozen")
        for param in model.language_model.parameters():
            param.requires_grad = False
    
    # 构建优化器（只包含 requires_grad=True 的参数）
    backbone_params = []
    language_params = []
    neck_head_params = []
    
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
    if len(neck_head_params) > 0 and args.lr > 0:
        param_groups.append({'params': neck_head_params, 'lr': args.lr})
    if len(backbone_params) > 0 and args.lr_backbone > 0:
        param_groups.append({'params': backbone_params, 'lr': args.lr_backbone})
    if len(language_params) > 0 and args.lr_language > 0:
        param_groups.append({'params': language_params, 'lr': args.lr_language})
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    print(f"[INFO] Optimizer groups: neck/head={len(neck_head_params)}, "
          f"backbone={len(backbone_params)}, language={len(language_params)}")
    
    preprocessor = DetDataPreprocessor(
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32,
    ).to(device)
    
    # Task 1 数据集
    train_dataset = VOCGLIPDataset(
        data_root=args.data_root,
        task_id=1,
        protocol=args.protocol,
        seed=args.seed,
        split='trainval',
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # 评估用 Task 0 + Task 1 全类别
    old_classes = all_task_classes[0]
    new_classes = all_task_classes[1]
    all_classes = old_classes + new_classes
    
    val_dataset = VOCGLIPDataset(
        data_root=args.data_root,
        task_id=1,
        protocol=args.protocol,
        seed=args.seed,
        split='test',
        filter_task_classes=False,  # 评估时保留所有类别 GT，避免 old mAP=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    if args.work_dir_suffix:
        suffix = args.work_dir_suffix
    elif args.no_tam:
        suffix = '_baseline'
    elif args.unfreeze_backbone:
        suffix = '_unfreeze'
    else:
        suffix = ''
    work_dir = os.path.join(args.work_dir, args.protocol, f'seed{args.seed}{suffix}')
    os.makedirs(work_dir, exist_ok=True)
    
    print(f"\n[INFO] Starting Task 1 incremental training...")
    print(f"[INFO] Old classes ({len(old_classes)}): {old_classes}")
    print(f"[INFO] New classes ({len(new_classes)}): {new_classes}")
    best_metric = -1.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        train_metrics = train_one_epoch(
            model, preprocessor, train_loader, optimizer, device, epoch)
        
        print(f"\n[Summary Epoch {epoch}] "
              f"Loss: {train_metrics['loss']:.4f} "
              f"(Cls:{train_metrics['loss_cls']:.4f} "
              f"Bbox:{train_metrics['loss_bbox']:.4f} "
              f"Ctr:{train_metrics['loss_centerness']:.4f})")
        
        ckpt_path = os.path.join(work_dir, f'task1_epoch{epoch}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'args': vars(args),
        }, ckpt_path)
        print(f"[CHECKPOINT] Saved to {ckpt_path}")
        
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            print(f"\n[INFO] Evaluating (old / new / all)...")
            val_metrics = evaluate(
                model, preprocessor, val_loader, device,
                all_classes=all_classes,
                old_classes=old_classes,
                new_classes=new_classes,
            )
            print(f"[EVAL] Old mAP50: {val_metrics['old_mAP50']:.4f}")
            print(f"[EVAL] New mAP50: {val_metrics['new_mAP50']:.4f}")
            print(f"[EVAL] All mAP50: {val_metrics['all_mAP50']:.4f}")
            
            if val_metrics['all_mAP50'] > best_metric:
                best_metric = val_metrics['all_mAP50']
                best_path = os.path.join(work_dir, 'task1_best.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_metric': best_metric,
                    'args': vars(args),
                }, best_path)
                print(f"[BEST] New best model saved: all_mAP50={best_metric:.4f}")
    
    print(f"\n[INFO] Task 1 training completed!")
    print(f"[INFO] Best all_mAP50: {best_metric:.4f}")


if __name__ == '__main__':
    main()
