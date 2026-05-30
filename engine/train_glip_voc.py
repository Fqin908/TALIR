"""基于 mmdetection GLIP 的 VOC 2007 增量训练脚本 (TALIRHead 修复版).

Usage:
    cd mmdetection
    conda activate mmdet
    python projects/TALIR/train_glip_voc.py \
        --data-root ../TALIR-CIOD-v2/data/voc2007 \
        --protocol 10_10 --seed 42 --task-id 0 \
        --epochs 12 --batch-size 4 --lr 5e-5
"""
import os

# 强制使用物理 GPU 1（避开 GPU 0 上的 ollama）
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

# 注册 TALIRHead
MODELS.register_module(module=TALIRHead, force=True)

# 加载配置文件
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_SCRIPT_DIR, 'config.json'), 'r', encoding='utf-8') as f:
    _cfg_data = json.load(f)
_PATHS = _cfg_data['paths']


def parse_args():
    parser = argparse.ArgumentParser(description='Train GLIP on VOC 2007 Incremental')
    parser.add_argument('--data-root', default=_PATHS['data_root'], help='VOC 2007 root')
    parser.add_argument('--protocol', default='10_10', help='Incremental protocol')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--task-id', type=int, default=0, help='Task ID')
    parser.add_argument('--epochs', type=int, default=12, help='Max epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--lr-language', type=float, default=5e-6, help='Language model LR')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--resume', default=None, help='Resume checkpoint path')
    parser.add_argument('--work-dir', default=_PATHS['work_dir'], help='Output directory')
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--eval-interval', type=int, default=2, help='Evaluate every N epochs')
    parser.add_argument('--tam-mode', default='full', choices=['full', 'vision_only', 'language_only'],
                        help='TAM branch ablation mode (default: full)')
    parser.add_argument('--no-tam', action='store_true', default=False,
                        help='Disable TAM (baseline without masking)')
    return parser.parse_args()


def build_model(checkpoint_path: str, device: str):
    config_file = _PATHS['glip_config']
    cfg = Config.fromfile(config_file)
    
    # 关键：使用 TALIRHead（修复了 centerness weight 维度 bug + TAM）
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
def evaluate(model, preprocessor, dataloader, device, all_classes):
    """评估 mAP@0.5（与 train_task1.py 一致的 compute_ap 实现）."""
    model.eval()
    
    all_prompt = '. '.join(all_classes) + '.'
    
    # 收集预测结果
    all_predictions = []
    all_ground_truths = {}
    
    for data in dataloader:
        for ds in data['data_samples']:
            img_id = ds.metainfo['img_id']
            gt_boxes = ds.gt_instances.bboxes.cpu().numpy()
            gt_labels = ds.gt_instances.labels.cpu().numpy()
            all_ground_truths[img_id] = {'boxes': gt_boxes, 'labels': gt_labels}
        
        data = preprocessor(data, training=False)
        inputs = data['inputs'].to(device)
        data_samples = data['data_samples']
        for ds in data_samples:
            ds.text = all_prompt
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
                        all_predictions.append({
                            'image_id': img_id,
                            'class_name': all_classes[label],
                            'bbox': box.tolist(),
                            'score': float(score),
                        })
    
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
    
    mAP = compute_ap(all_predictions, all_ground_truths, all_classes)
    return {'mAP50': mAP}


def compute_iou(box1, box2):
    """计算两个框的 IoU. box: [x1, y1, x2, y2]"""
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
    
    print("[INFO] Loading GLIP model with TALIRHead...")
    # model = build_model('pretrained/glip_tiny_mmdet.pth', args.device)
    model = build_model(_PATHS['pretrained_checkpoint'], args.device)
    model.to(device)
    
    # 构建优化器
    language_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'language_model' in name or 'bert' in name:
            language_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': args.lr},
        {'params': language_params, 'lr': args.lr_language},
    ], weight_decay=args.weight_decay)
    
    print(f"[INFO] Optimizer: {len(other_params)} other groups, {len(language_params)} language groups")
    
    preprocessor = DetDataPreprocessor(
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32,
    ).to(device)
    
    # 数据集
    train_dataset = VOCGLIPDataset(
        data_root=args.data_root,
        task_id=args.task_id,
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
    
    # 验证集（用 test 集合评估）
    # 注意：VOC 2007 没有单独的 val，论文用 test 评估
    val_dataset = VOCGLIPDataset(
        data_root=args.data_root,
        task_id=args.task_id,
        protocol=args.protocol,
        seed=args.seed,
        split='test',
        filter_task_classes=False,  # 评估时保留所有类别 GT
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    work_dir = os.path.join(args.work_dir, args.protocol,
                           f'seed{args.seed}{"_baseline" if args.no_tam else ""}')
    os.makedirs(work_dir, exist_ok=True)
    
    # 读取类别划分（TAM 和评估都需要）
    split_file = os.path.join(args.data_root, f'incremental_{args.protocol}', f'seed{args.seed}', 'category_split.json')
    with open(split_file) as f:
        split_info = json.load(f)
    all_task_classes = split_info['task_classes']
    
    # 设置 TAM 配置（VOC 10+10: 总20类，Task0=10类=50% channel）
    if args.no_tam:
        model.bbox_head.tam_cfg = None
        model.bbox_head.tam_mode = 'vision_only'
        print("[BASELINE] TAM disabled (mask all 1s)")
    elif hasattr(model.bbox_head, 'tam_cfg') and model.bbox_head.tam_cfg is None:
        # FPN 输出 channel = 256
        total_ch = 256
        total_classes = sum(len(t) for t in all_task_classes)
        
        # 计算每个 task 的 channel 分割点
        cumsum = 0
        visual_splits = {}
        language_splits = {}
        for tid, tcls in enumerate(all_task_classes):
            n_cls = len(tcls)
            n_ch = int(total_ch * n_cls / total_classes)
            visual_splits[tid] = (cumsum, cumsum + n_ch)
            language_splits[tid] = (cumsum, cumsum + n_ch)
            cumsum += n_ch
        
        model.bbox_head.tam_cfg = {
            'visual_splits': visual_splits,
            'language_splits': language_splits,
        }
        print(f"[TAM] Visual channel splits: {visual_splits}")
        print(f"[TAM] Language channel splits: {language_splits}")
    
    model.bbox_head.tam_mode = args.tam_mode
    print(f"[TAM] tam_mode: {args.tam_mode}")
    
    # 设置当前 task_id
    model.bbox_head.current_task_id = args.task_id
    print(f"[TAM] Current task_id: {args.task_id}")
    
    print(f"[INFO] Starting training Task {args.task_id}...")
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
        
        ckpt_path = os.path.join(work_dir, f'task{args.task_id}_epoch{epoch}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'args': vars(args),
        }, ckpt_path)
        print(f"[CHECKPOINT] Saved to {ckpt_path}")
        
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            print(f"\n[INFO] Evaluating...")
            val_metrics = evaluate(
                model, preprocessor, val_loader, device,
                all_classes=train_dataset.current_classes)
            print(f"[EVAL] mAP50: {val_metrics['mAP50']:.4f}")
            
            if val_metrics['mAP50'] > best_metric:
                best_metric = val_metrics['mAP50']
                best_path = os.path.join(work_dir, f'task{args.task_id}_best.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_metric': best_metric,
                    'args': vars(args),
                }, best_path)
                print(f"[BEST] New best model saved: mAP50={best_metric:.4f}")
    
    print(f"\n[INFO] Task {args.task_id} training completed!")
    print(f"[INFO] Best mAP50: {best_metric:.4f}")


if __name__ == '__main__':
    main()
