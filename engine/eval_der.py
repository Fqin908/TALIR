"""Der-like SIS 评估：两个独立模型（Task 0 + Task 1）+ SIS 融合.

论文 Table 4 Lines 2-4:
    Line 2: 双模型 + ELEmean (Eq.6)
    Line 3: 双模型 + ELEmax  (Eq.5)
    Line 4: 双模型 + ROWmax  (Eq.7)

Usage:
    python projects/TALIR/eval_der.py --protocol 10_10 --seed 42 --sis rowmax
    python projects/TALIR/eval_der.py --protocol 10_10 --seed 42 --sis elemax
    python projects/TALIR/eval_der.py --protocol 10_10 --seed 42 --sis elemean
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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_SCRIPT_DIR, 'config.json'), 'r', encoding='utf-8') as f:
    _cfg_data = json.load(f)
_PATHS = _cfg_data['paths']

device = torch.device('cuda:0')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Der-like SIS Evaluation')
    parser.add_argument('--protocol', default='10_10')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sis', default='rowmax',
                        choices=['rowmax', 'elemax', 'elemean'],
                        help='SIS fusion strategy')
    parser.add_argument('--split', default='test', choices=['test', 'trainval'])
    return parser.parse_args()


args = parse_args()
protocol = args.protocol
seed = args.seed

# ── 加载类别划分 ──
split_file = os.path.join(_PATHS['data_root'], f'incremental_{protocol}', f'seed{seed}', 'category_split.json')
with open(split_file) as f:
    split_info = json.load(f)
old_classes = split_info['task_classes'][0]
new_classes = split_info['task_classes'][1]
all_classes = old_classes + new_classes
print(f"[INFO] Protocol={protocol}, Seed={seed}")
print(f"[INFO] Old ({len(old_classes)}): {old_classes}")
print(f"[INFO] New ({len(new_classes)}): {new_classes}")
print(f"[INFO] SIS strategy: {args.sis}")

# ── 构建两个独立模型（无 TAM）──
cfg = Config.fromfile(_PATHS['glip_config'])
cfg.model.bbox_head.type = 'TALIRHead'
cfg.model.bbox_head.num_classes = 20
cfg.model.bbox_head.lang_model_name = _PATHS['bert_model_name']
cfg.model.language_model.name = _PATHS['bert_model_name']

# Model 0: Task 0 (old classes), 无 TAM
model_0 = init_detector(cfg, _PATHS['pretrained_checkpoint'], device=device)
model_0.bbox_head.tam_cfg = None
model_0.bbox_head.inference_strategy = 'none'

# Model 1: Task 1 (new classes), 无 TAM
model_1 = init_detector(cfg, _PATHS['pretrained_checkpoint'], device=device)
model_1.bbox_head.tam_cfg = None
model_1.bbox_head.inference_strategy = 'none'

# ── 加载权重（der 子目录下的独立训练结果）──
der_dir = os.path.join(_PATHS['work_dir'], protocol, f'seed{seed}_der')
ckpt_0 = os.path.join(der_dir, 'task0_notam.pth')
ckpt_1 = os.path.join(der_dir, 'task1_notam.pth')

print(f"[INFO] Model 0 (old classes): {ckpt_0}")
ckpt0 = torch.load(ckpt_0, map_location=device)
model_0.load_state_dict(ckpt0['model_state_dict'], strict=False)
model_0.eval()

print(f"[INFO] Model 1 (new classes): {ckpt_1}")
ckpt1 = torch.load(ckpt_1, map_location=device)
model_1.load_state_dict(ckpt1['model_state_dict'], strict=False)
model_1.eval()

# ── Preprocessor ──
preprocessor = DetDataPreprocessor(
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    bgr_to_rgb=False,
    pad_size_divisor=32,
).to(device)

# ── 数据集 ──
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

prompt_0 = '. '.join(old_classes) + '.'
prompt_1 = '. '.join(new_classes) + '.'
all_prompt = '. '.join(all_classes) + '.'


@torch.no_grad()
def der_sis_eval(model_0, model_1, loader, sis_strategy):
    model_0.eval()
    model_1.eval()

    # 两个模型都用全类别 prompt，各自输出 20 类 logit，方便 element-wise SIS
    all_prompt = '. '.join(all_classes) + '.'
    lang_all = model_0.language_model([all_prompt] * 1)
    token_map_all = model_0.get_tokens_positive_and_prompts(all_prompt, True, None, None)[0]

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
        B = len(data_samples)
        batch_img_metas = [ds.metainfo for ds in data_samples]

        # 共享视觉特征（backbone 冻结，完全相同）
        visual_feats = model_0.extract_feat(inputs)

        # 两个模型各自用全类别 prompt 推理 → logit 形状相同
        lang_batch = model_0.language_model([all_prompt] * B)
        cls_0, bbox_0, cent_0 = model_0.bbox_head.forward(visual_feats, lang_batch)
        cls_1, bbox_1, cent_1 = model_1.bbox_head.forward(visual_feats, lang_batch)

        # ── SIS 融合（logit → logit，形状和 token 严格对应）──
        num_levels = len(cls_0)
        combined_logits = []
        combined_token_maps = []

        for lvl in range(num_levels):
            c0 = cls_0[lvl]  # [B, Q, T]
            c1 = cls_1[lvl]  # [B, Q, T]

            if sis_strategy == 'rowmax':
                max_0 = c0.sigmoid().max(dim=-1, keepdim=True)[0]  # [B, Q, 1]
                max_1 = c1.sigmoid().max(dim=-1, keepdim=True)[0]
                mask = (max_0 > max_1).float()
                combined = c0 * mask + c1 * (1 - mask)
            elif sis_strategy == 'elemax':
                combined = torch.max(c0, c1)
            elif sis_strategy == 'elemean':
                combined = (c0 + c1) / 2.0

            combined_logits.append(combined)
            # token map 相同（同一个 prompt），直接复用
            combined_token_maps.append([token_map_all] * B)

        batch_token_maps = combined_token_maps[-1]

        # bbox / centerness 平均
        bbox_preds = [(b0 + b1) / 2 for b0, b1 in zip(bbox_0, bbox_1)]
        centerness = [(c0 + c1) / 2 for c0, c1 in zip(cent_0, cent_1)]

        pred_instances_list = model_1.bbox_head.predict_by_feat(
            tuple(combined_logits), tuple(bbox_preds), tuple(centerness),
            batch_img_metas=batch_img_metas,
            batch_token_positive_maps=batch_token_maps,
            rescale=False)

        # predict_by_feat 返回 InstanceData，挂回 DataSample 保留 metainfo
        for ds, pred in zip(data_samples, pred_instances_list):
            ds.pred_instances = pred

        for ds in data_samples:
            img_id = ds.metainfo['img_id']
            pred = ds.pred_instances
            if len(pred) > 0:
                for box, score, label in zip(pred.bboxes.cpu().numpy(),
                                             pred.scores.cpu().numpy(),
                                             pred.labels.cpu().numpy()):
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

    # ── mAP 计算 ──
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
        ap = sum(np.max(prec[rec >= t]) if np.any(rec >= t) else 0
                 for t in np.linspace(0, 1, 11)) / 11
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
    print(f"Der-like SIS ({sis_strategy}) — Old mAP50 = {np.mean(old_aps):.4f}")
    print(f"Der-like SIS ({sis_strategy}) — New mAP50 = {np.mean(new_aps):.4f}")
    print(f"Der-like SIS ({sis_strategy}) — All mAP50 = {np.mean(all_aps):.4f}")
    print(f"{'=' * 60}")


def compute_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1, a2 = (box1[2] - box1[0]) * (box1[3] - box1[1]), (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-6)


der_sis_eval(model_0, model_1, loader, args.sis)