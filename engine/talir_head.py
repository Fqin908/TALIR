"""TALIR Head: 继承 ATSSVLFusionHead，修复 centerness weight 维度问题 + TAM + SIS."""
import torch
from torch import Tensor
from typing import Optional, Tuple, Dict

from mmdet.registry import MODELS
from mmdet.models.dense_heads.atss_vlfusion_head import ATSSVLFusionHead


@MODELS.register_module()
class TALIRHead(ATSSVLFusionHead):
    """TALIR Detection Head.

    基于 ATSSVLFusionHead：
    1. 修复 _loss_by_feat 中 centerness_targets 与 bbox loss 的广播维度不匹配
    2. TAM（Task-Aware Masking）— 双分支消融：
       - Visual TAM:  在 visual feature channel 层面 mask
       - Language TAM: 在分类 logit 层面做 task-aware class scaling
       两者正交互补，可独立消融
    3. 实现 SIS（Selective Inference Strategy）多策略融合
    """

    def __init__(self, *args, tam_cfg: Optional[Dict] = None,
                 inference_strategy: str = 'rowmax',
                 tam_mode: str = 'full', **kwargs):
        """Args:
            tam_cfg: TAM 配置，示例：
                {
                    'visual_splits': {0: (0, 128), 1: (128, 256)},
                    'language_splits': {0: (0, 128), 1: (128, 256)},
                }
            inference_strategy: SIS 融合策略，可选：
                - 'rowmax' (默认): ROWmax，按行取 max 后选择任务
                - 'elemax': ELEMax，逐元素取 max
                - 'elemean': ELEMean，逐元素取均值
                - 'none': 标准单次推理，跳过 SIS
            tam_mode: TAM 分支消融，可选：
                - 'full' (默认):      Visual TAM + Language TAM
                - 'vision_only':      仅 Visual TAM
                - 'language_only':    仅 Language TAM"""
        super().__init__(*args, **kwargs)
        self.tam_cfg = tam_cfg
        self.current_task_id = 0  # 训练前由外部设置
        self.inference_strategy = inference_strategy
        self.tam_mode = tam_mode
        self.lang_tam_strength = 0.15  # 语言 TAM 缩放强度
        assert inference_strategy in ('rowmax', 'elemax', 'elemean', 'none'), \
            f"Unknown inference_strategy: {inference_strategy}"
        assert tam_mode in ('full', 'vision_only', 'language_only'), \
            f"Unknown tam_mode: {tam_mode}"

    def apply_tam_visual(self, visual_feats: Tuple[Tensor]) -> Tuple[Tensor]:
        """对多尺度视觉特征做 channel masking.

        Args:
            visual_feats: tuple of [B, C, H, W]
        Returns:
            masked visual_feats
        """
        if self.tam_cfg is None:
            return visual_feats

        task_id = self.current_task_id
        start, end = self.tam_cfg['visual_splits'][task_id]

        masked = []
        for feat in visual_feats:
            # feat: [B, C, H, W]
            C = feat.shape[1]
            mask = torch.zeros(C, device=feat.device, dtype=feat.dtype)
            mask[start:end] = 1.0
            masked.append(feat * mask.view(1, -1, 1, 1))
        return tuple(masked)

    def apply_tam_language(self, language_feats: dict) -> dict:
        """语言 TAM：不对 BERT embedding 做 channel mask.

        原因：embedded 是 BERT 输出的类别语义向量，直接 mask channel 会破坏所有类别的语义完整性，
        导致分类投影层（dot_product_projection_text）接收到被截断的输入，性能下降。

        视觉 TAM 已经通过 DyHead 交叉注意力使语言特征变得 task-aware 
        （masked visual features → cross-attention → task-aware language hidden states），
        语言侧无需再做 channel masking。

        该函数保留为 pass-through，供未来基于 language projection 或 cross-attention
        内部实现语言 TAM 时使用。
        """
        return language_feats

    # ======================== SIS 融合策略（消融实验） ========================

    def _fuse_rowmax(self, cls_logits_0: Tuple[Tensor],
                     cls_logits_1: Tuple[Tensor]) -> Tuple[Tensor]:
        """ROWmax 融合：对每个 query（行），比较两个 task 的 max score，
        选择 max 更大的 task 的整行分类 logits.

        Args:
            cls_logits_0: Task 0 分类 logits，每层 shape [B, num_queries, num_tokens]
            cls_logits_1: Task 1 分类 logits

        Returns:
            融合后的 cls_logits
        """
        fused = []
        for c0, c1 in zip(cls_logits_0, cls_logits_1):
            max_0 = c0.max(dim=-1, keepdim=True)[0]
            max_1 = c1.max(dim=-1, keepdim=True)[0]
            mask = max_0 > max_1
            fused.append(torch.where(mask, c0, c1))
        return tuple(fused)

    def _fuse_elemax(self, cls_logits_0: Tuple[Tensor],
                     cls_logits_1: Tuple[Tensor]) -> Tuple[Tensor]:
        """ELEMax 融合：逐元素取 max，两个 task 的分类 logits 在同一位置取较大值.

        Args:
            cls_logits_0: Task 0 分类 logits
            cls_logits_1: Task 1 分类 logits

        Returns:
            融合后的 cls_logits
        """
        return tuple(
            torch.max(c0, c1) for c0, c1 in zip(cls_logits_0, cls_logits_1)
        )

    def _fuse_elemean(self, cls_logits_0: Tuple[Tensor],
                      cls_logits_1: Tuple[Tensor]) -> Tuple[Tensor]:
        """ELEMean 融合：逐元素取均值，两个 task 的分类 logits 在同一位置取平均.

        Args:
            cls_logits_0: Task 0 分类 logits
            cls_logits_1: Task 1 分类 logits

        Returns:
            融合后的 cls_logits
        """
        return tuple(
            (c0 + c1) / 2.0 for c0, c1 in zip(cls_logits_0, cls_logits_1)
        )

    def _fuse_cls_logits(self, cls_logits_0: Tuple[Tensor],
                         cls_logits_1: Tuple[Tensor]) -> Tuple[Tensor]:
        """根据 self.inference_strategy 派发融合策略.

        支持三种消融策略：
            - 'rowmax': ROWmax（论文默认）
            - 'elemax': ELEMax — 逐元素 max，导致跨任务混淆 → false positive
            - 'elemean': ELEMean — 逐元素均值，稀释置信度
        """
        if self.inference_strategy == 'rowmax':
            return self._fuse_rowmax(cls_logits_0, cls_logits_1)
        elif self.inference_strategy == 'elemax':
            return self._fuse_elemax(cls_logits_0, cls_logits_1)
        elif self.inference_strategy == 'elemean':
            return self._fuse_elemean(cls_logits_0, cls_logits_1)
        else:
            raise ValueError(f"Unknown inference_strategy: {self.inference_strategy}")

    # =====================================================================

    def _lang_tam_class_scale(self, num_classes: int) -> Tensor:
        """计算语言 TAM 的逐类缩放因子.

        根据 tam_cfg 中的 channel 比例反推 old/new 类别数，生成缩放向量：
        - Task 0: old 类 × (1+strength), new 类 × (1-strength)
        - Task 1: old 类 × (1-strength), new 类 × (1+strength)
        """
        if self.tam_cfg is None or 'language_splits' not in self.tam_cfg:
            return torch.ones(num_classes)

        task_id = self.current_task_id
        start0, end0 = self.tam_cfg['language_splits'][0]
        start1, end1 = self.tam_cfg['language_splits'][1]
        total_ch = float(end1)  # 256
        num_old = int(num_classes * (end0 - start0) / total_ch)
        num_new = num_classes - num_old

        strength = getattr(self, 'lang_tam_strength', 0.15)

        scale = torch.ones(num_classes)
        if task_id == 0:
            scale[:num_old] = 1.0 + strength
            scale[num_old:] = 1.0 - strength
        else:
            scale[:num_old] = 1.0 - strength
            scale[num_old:] = 1.0 + strength
        return scale

    def _apply_language_tam_to_logits(self, cls_logits: Tuple[Tensor]) -> Tuple[Tensor]:
        """语言 TAM：对每层分类 logits 做 task-aware class-level scaling.

        Visual TAM 在 feature channel 层面 mask → 控制哪些 visual 信息参与分类。
        Language TAM 在 logit 层面 scale  → 控制各类别在分类空间中的置信度偏置。
        两者正交且互补，可独立消融。
        """
        num_classes = cls_logits[0].shape[-1]
        scale = self._lang_tam_class_scale(num_classes).to(cls_logits[0].device)
        return tuple(logit * scale for logit in cls_logits)

    def forward(self, visual_feats: Tuple[Tensor],
                language_feats: dict) -> Tuple:
        """Forward（训练/推理共用）— 仅 visual TAM.

        Language TAM (logit scale) 只在 predict() 推理阶段施加，
        不参与训练，避免模型学习补偿 logit 偏置。
        """
        if self.tam_mode in ('full', 'vision_only'):
            visual_feats = self.apply_tam_visual(visual_feats)
        return super().forward(visual_feats, language_feats)

    def loss(self, visual_feats: Tuple[Tensor], language_feats: dict,
             batch_data_samples, **kwargs) -> dict:
        """Loss with visual TAM (TAM 在 forward 中已应用)."""
        # 父类 loss 会调用 self.forward，forward 中已经做了 TAM
        return super().loss(visual_feats, language_feats,
                            batch_data_samples, **kwargs)

    def predict(self, visual_feats: Tuple[Tensor],
                language_feats: dict,
                batch_data_samples,
                rescale: bool = True):
        """Predict with SIS + TAM（语言 TAM 仅在推理时施加，不参与训练）.
        
        当 inference_strategy='none' 时，跳过 SIS，做标准单次推理。
        """
        batch_img_metas = [ds.metainfo for ds in batch_data_samples]
        batch_token_positive_maps = [ds.token_positive_map for ds in batch_data_samples]
        
        apply_lang_tam = (self.tam_mode in ('full', 'language_only')
                          and self.tam_cfg is not None)

        if self.inference_strategy == 'none':
            # 标准单次推理，不跑 SIS 双路融合
            cls_logits, bbox_preds, centerness = self.forward(visual_feats, language_feats)
            if apply_lang_tam:
                cls_logits = self._apply_language_tam_to_logits(cls_logits)
            return self.predict_by_feat(
                cls_logits, bbox_preds, centerness,
                batch_img_metas=batch_img_metas,
                batch_token_positive_maps=batch_token_positive_maps,
                rescale=rescale)
        
        # SIS: Run twice with different task masks
        self.current_task_id = 0
        cls_logits_0, bbox_preds_0, centerness_0 = self.forward(visual_feats, language_feats)
        if apply_lang_tam:
            cls_logits_0 = self._apply_language_tam_to_logits(cls_logits_0)
        
        self.current_task_id = 1
        cls_logits_1, bbox_preds_1, centerness_1 = self.forward(visual_feats, language_feats)
        if apply_lang_tam:
            cls_logits_1 = self._apply_language_tam_to_logits(cls_logits_1)
        
        # Check if shapes match (all prompt should have same num_tokens)
        shapes_match = all(
            c0.shape == c1.shape for c0, c1 in zip(cls_logits_0, cls_logits_1)
        )
        
        if shapes_match:
            # SIS 融合策略派发（消融实验：ROWmax / ELEMax / ELEMean）
            cls_logits = self._fuse_cls_logits(cls_logits_0, cls_logits_1)
            
            # Average bbox and centerness
            bbox_preds = [(b0 + b1) / 2 for b0, b1 in zip(bbox_preds_0, bbox_preds_1)]
            centerness = [(c0 + c1) / 2 for c0, c1 in zip(centerness_0, centerness_1)]
        else:
            # Fall back to single task (different prompts have different num_tokens)
            cls_logits = cls_logits_0
            bbox_preds = bbox_preds_0
            centerness = centerness_0
        
        return self.predict_by_feat(
            cls_logits, bbox_preds, centerness,
            batch_img_metas=batch_img_metas,
            batch_token_positive_maps=batch_token_positive_maps,
            rescale=rescale)

    def predict_sis_task_specific(
        self,
        visual_feats: Tuple[Tensor],
        lang_feats_task0: dict,
        token_map_task0: list,
        lang_feats_task1: dict,
        token_map_task1: list,
        batch_data_samples,
        rescale: bool = True,
    ):
        """SIS with task-specific prompts — 论文标准方式.

        每个 task 只分类自己的类别（不同的 prompt），然后 SIS 融合。
        这样即使 baseline (无 TAM) 也能看出策略差异：

        - ELEmax/ELEmean: 每 query 同时得到 old + new 类分数 → false positive
        - ROWmax: 每 query 只选出获胜 task 的类别 → 避免跨任务混淆

        Args:
            visual_feats: 视觉特征（两个 task 共享）
            lang_feats_task0: Task 0 (old-class) 语言特征
            token_map_task0: Task 0 token_positive_map (每个 batch element 一个 dict)
            lang_feats_task1: Task 1 (new-class) 语言特征
            token_map_task1: Task 1 token_positive_map
            batch_data_samples: 数据样本（需要 img_metas）
            rescale: 是否 rescale bboxes

        Returns:
            list[:obj:`InstanceData`]: 检测结果
        """
        batch_img_metas = [ds.metainfo for ds in batch_data_samples]

        apply_lang_tam = (self.tam_mode in ('full', 'language_only')
                          and self.tam_cfg is not None)

        # Task 0 forward: old classes only
        self.current_task_id = 0
        cls_0, bbox_0, cent_0 = self.forward(visual_feats, lang_feats_task0)
        if apply_lang_tam:
            cls_0 = self._apply_language_tam_to_logits(cls_0)

        # Task 1 forward: new classes only
        self.current_task_id = 1
        cls_1, bbox_1, cent_1 = self.forward(visual_feats, lang_feats_task1)
        if apply_lang_tam:
            cls_1 = self._apply_language_tam_to_logits(cls_1)

        # 构建融合后的 cls_logits [B, Q, T0+T1] 和 combined token_positive_map
        B = cls_0[0].shape[0]
        num_levels = len(cls_0)
        combined_logits = []
        combined_token_maps = []

        for level_idx in range(num_levels):
            c0_lvl = cls_0[level_idx]  # [B, Q, T0]
            c1_lvl = cls_1[level_idx]  # [B, Q, T1]
            T0 = c0_lvl.shape[-1]
            T1 = c1_lvl.shape[-1]

            combined_lvl_list = []
            for b in range(B):
                c0_b = c0_lvl[b:b + 1]  # [1, Q, T0]
                c1_b = c1_lvl[b:b + 1]  # [1, Q, T1]

                if self.inference_strategy == 'rowmax':
                    # ROWmax: 每 query 比较 max，选获胜 task 的整行
                    max_0 = c0_b.max(dim=-1, keepdim=True)[0]  # [1, Q, 1]
                    max_1 = c1_b.max(dim=-1, keepdim=True)[0]
                    mask = (max_0 > max_1).float()              # [1, Q, 1]
                    combined = torch.zeros(1, c0_b.shape[1], T0 + T1,
                                           device=c0_b.device, dtype=c0_b.dtype)
                    combined[:, :, :T0] = c0_b * mask
                    combined[:, :, T0:] = c1_b * (1 - mask)
                else:
                    # ELEmax/ELEmean: 柱不重叠 → 本质 = concatenation
                    # 每个 query 同时获得 old + new 类分数 → false positive
                    combined = torch.cat([c0_b, c1_b], dim=-1)

                combined_lvl_list.append(combined)

            combined_logits.append(torch.cat(combined_lvl_list, dim=0))

        # 构建 combined token_positive_map（offset task1 的 token 索引和 label key）
        # Task 0 的 token 占 [0, T0)，label key 保持 0..n_old-1
        # Task 1 的 token 占 [T0, T0+T1)，label key offset 为 n_old..n_old+n_new-1
        n_old = len(token_map_task0[0])
        for b in range(B):
            tm0 = token_map_task0[min(b, len(token_map_task0) - 1)]
            tm1 = token_map_task1[min(b, len(token_map_task1) - 1)]
            combined_tm = dict(tm0)  # shallow copy, task 0 token indices unchanged
            for label_j, token_indices in tm1.items():
                combined_tm[label_j + n_old] = [t + T0 for t in token_indices]
            combined_token_maps.append(combined_tm)

        # bbox 和 centerness 取平均（class-agnostic）
        bbox_preds = [(b0 + b1) / 2 for b0, b1 in zip(bbox_0, bbox_1)]
        centerness = [(c0 + c1) / 2 for c0, c1 in zip(cent_0, cent_1)]

        results_list = self.predict_by_feat(
            tuple(combined_logits), tuple(bbox_preds), tuple(centerness),
            batch_img_metas=batch_img_metas,
            batch_token_positive_maps=combined_token_maps,
            rescale=rescale)

        # 与 GLIP.predict() 一致：把 pred_instances 挂回 DataSample
        for ds, pred in zip(batch_data_samples, results_list):
            ds.pred_instances = pred
        return batch_data_samples

    def _loss_by_feat(self, anchors: Tensor, cls_score: Tensor,
                      bbox_pred: Tensor, centerness: Tensor, labels: Tensor,
                      label_weights: Tensor, bbox_targets: Tensor,
                      avg_factor: float) -> dict:
        """Calculate the loss of all scale level based on the features.

        与父类唯一区别：centerness_targets 先 unsqueeze(1)，
        使其从 [N_pos] 变为 [N_pos, 1]，从而与 [N_pos, 4] 的 bbox loss
        正确广播。
        """
        anchors = anchors.reshape(-1, 4)

        pos_inds = (labels.sum(-1) > 0).reshape(-1)

        assert (self.text_masks.dim() == 2)
        text_mask = (self.text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, cls_score.size(1), 1)
        cls_score = torch.masked_select(cls_score, text_mask).contiguous()
        labels = torch.masked_select(labels, text_mask)
        label_weights = label_weights[...,
                                      None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)

        bbox_pred = bbox_pred.reshape(-1, 4)
        centerness = centerness.reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # classification loss
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor)

        if pos_inds.sum() > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            centerness_targets = self.centerness_target(
                pos_anchors, pos_bbox_targets)

            if torch.isnan(centerness_targets).any():
                print('=====Centerness includes NaN=====')
                mask = ~torch.isnan(centerness_targets)
                centerness_targets = centerness_targets[mask]
                pos_centerness = pos_centerness[mask]
                pos_anchors = pos_anchors[mask]
                pos_bbox_targets = pos_bbox_targets[mask]
                pos_bbox_pred = pos_bbox_pred[mask]

                if pos_bbox_targets.shape[0] == 0:
                    loss_bbox = bbox_pred.sum() * 0
                    loss_centerness = centerness.sum() * 0
                    centerness_targets = bbox_targets.new_tensor(0.)
                    return loss_cls, loss_bbox, loss_centerness, \
                        centerness_targets.sum()

            # The decoding process takes the offset into consideration.
            pos_anchors[:, 2:] += 1
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)

            # ===== 关键修复 =====
            # centerness_targets: [N_pos] -> [N_pos, 1]
            # 使其能与 [N_pos, 4] 的 bbox loss 正确广播
            centerness_targets = centerness_targets.unsqueeze(1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_bbox_targets,
                weight=centerness_targets,
                avg_factor=1.0)

            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness, centerness_targets.squeeze(1),
                avg_factor=avg_factor)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()
