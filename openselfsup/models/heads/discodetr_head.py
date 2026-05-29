
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple

import openselfsup.utils.box_ops as box_ops
from openselfsup.utils.misc import accuracy
from .. import builder
from ..necks.helper import MLP
from ..necks.conditional_tr import pred_bbox_with_reference
from ..registry import HEADS
from ..utils.hungarian_matcher import build_matcher


def loss_contra(pred_v1: Tensor, proj_v2: Tensor, size_average: bool = True) -> Tensor:
    pred_v1_norm = F.normalize(pred_v1, dim=1)
    proj_v2_norm = F.normalize(proj_v2, dim=1)

    loss = 2 * (pred_v1_norm * proj_v2_norm).sum()
    if size_average:
        loss /= pred_v1_norm.size(0)
    return 2 - loss


def resort_matched_indices(indices_v1, indices_v2):
    new_indices = []
    for indice_v1, indice_v2 in zip(indices_v1, indices_v2):
        _, tgt_v1 = indice_v1
        src_v2, tgt_v2 = indice_v2

        sort_idx = torch.argsort(tgt_v2)
        new_indices.append((src_v2[sort_idx][tgt_v1], tgt_v1))
    return new_indices


@HEADS.register_module
class DisCoDETRPredictHead(nn.Module):

    def __init__(self,
                 aux_loss=False,
                 hidden_dim=None,
                 size_average=True,
                 feature_recon: bool = False,
                 backbone_channels: int = 2048,
                 contrast_use: bool = False,
                 all_layers: bool = False,
                 part_contrast: bool = True,
                 batch_contrast: bool = False,
                 positive_from_cosine: bool =False,
                 contrast_transform: bool = False,
                 part_contrast_transform: bool = False,
                 only_positive: bool = False,
                 part_contrast_transform_proj: bool = False,
                 ):
                 #matcher_cfg: Dict = None):
        super().__init__()
        self.all_layers = all_layers
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.feature_recon = feature_recon
        self.aux_loss = aux_loss
        self.size_average = size_average
        self.part_contrast = part_contrast
        self.batch_contrast = batch_contrast
        self.positive_from_cosine = positive_from_cosine
        self.contrast_transform =contrast_transform
        self.part_contrast_transform = part_contrast_transform
        self.part_contrast_transform_proj = part_contrast_transform_proj
        self.only_positive = only_positive
        # matcher = build_matcher(matcher_cfg)
        self.hidden_dim=hidden_dim
        self.class_embed = nn.Linear(hidden_dim, 2 + 1)  # 0 or 1
        if self.feature_recon:
            # self.feature_align = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
            self.feature_align = MLP(hidden_dim, hidden_dim, backbone_channels, 2)
        self.contrast_use=contrast_use
        losses = ['labels', 'cardinality', 'boxes']
        if self.feature_recon:
            losses.append('feature')
        if self.contrast_use:
            losses.append('contrast')
        self.criterion = DisCoDETRCriterion(aux_loss, losses=losses, part_contrast=self.part_contrast, batch_contrast=self.batch_contrast, positive_from_cosine=self.positive_from_cosine, contrast_transform=self.contrast_transform, part_contrast_transform=self.part_contrast_transform, only_positive=self.only_positive, hidden_dim=self.hidden_dim)
        self.criterion.bbox_embed = self.bbox_embed
        self.criterion.contrast_use = contrast_use
        # self.criterion = SiamDETRCriterion(matcher, aux_loss, losses=losses)

    def _loss_box(self, coord, target):
        '''
        coord: cxcyhw
        target: xyxy
        '''
        bs, num_queries = coord.shape[:2]
        num_boxes = bs * num_queries

        loss_bbox = F.l1_loss(coord, box_ops.box_xyxy_to_cxcywh(target), reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(coord.view(-1, 4)),
            target.view(-1, 4)))
        loss_giou = loss_giou.sum() / num_boxes
        return dict(loss_bbox=loss_bbox, loss_giou=loss_giou)

    def forward(self, hs, ref=None, target_box=None, img_size=None, recon_gt=None, ori_idx=None, ori_idx_v2=None, hs_v2=None):
        """Head for SiameseDETR
        Args:
            hs (Tensor): [num_dec, N, num_queries, C]
            target_box (Tensor): [N, num_patches, 4], unnormalized
        """
        num_dec, bs, num_queries, c = hs.shape
        device = hs.device
        out = {}
        out['ori_idx'] = ori_idx[-1]
        contrast_really_use = False
        if self.contrast_use and hs_v2 is not None:
            out['ori_idx2'] = ori_idx_v2[-1]
            out['ori_outputsv1'] = hs[-1]
            out['ori_outputsv2'] = hs_v2[-1]
            contrast_really_use = True
        # out['ori_idx'] = ori_idx
        target = [{} for _ in range(bs)]

        outputs_class = self.class_embed(hs)
        out['pred_logits'] = outputs_class[-1]
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss('pred_logits', outputs_class)
        for t in target:
            t['labels'] = torch.ones((target_box.shape[1], ), dtype=torch.long, device=device)

        h, w = img_size[:, 0].view(-1, 1), img_size[:, 1].view(-1, 1)
        scale_fct = torch.cat([w, h, w, h], dim=1).float().to(target_box.device)
        target_box = target_box / scale_fct[:, None, :]  # norm
        outputs_coord = pred_bbox_with_reference(lambda x: self.bbox_embed(x), hs, ref)
        out['pred_boxes'] = outputs_coord[-1]
        if self.aux_loss:
            aux_boxes_outputs = self._set_aux_loss('pred_boxes', outputs_coord)
            for i, aux_out in enumerate(aux_boxes_outputs):
                out['aux_outputs'][i].update(aux_out)
            for i in range(len(ori_idx) - 1):   # aux loss
                out['aux_outputs'][i].update({'ori_idx': ori_idx[i]})
            if ori_idx_v2 is not None:
                for i in range(len(ori_idx_v2) - 1):   # aux loss
                    out['aux_outputs'][i].update({'ori_idx2': ori_idx_v2[i]})
        for i, t in enumerate(target):
            t['boxes'] = box_ops.box_xyxy_to_cxcywh(target_box[i])

        if self.feature_recon:
            outputs_feature = self.feature_align(hs)
            out.update({
                'gt_feature': recon_gt,
                'pred_feature': outputs_feature[-1]})

            if self.aux_loss:
                for i in range(len(outputs_class) - 1):
                    out['aux_outputs'][i].update({'pred_feature': outputs_feature[i]})
                    out['aux_outputs'][i].update({'gt_feature': recon_gt})
        if contrast_really_use and self.all_layers:
            for i in range(len(outputs_class) - 1):
                out['aux_outputs'][i].update({'ori_outputsv1': hs[i]})
                out['aux_outputs'][i].update({'ori_outputsv2': hs_v2[i]})
        loss_dict = self.criterion(out, target, contrast_really_use, all_layers=self.all_layers)
        return loss_dict

    @torch.jit.unused
    def _set_aux_loss(self, key, value):
        aux_out = []
        for i in range(len(value) - 1):
            aux_dict = {key: value[i]}
            aux_out.append(aux_dict)
        return aux_out


@HEADS.register_module
class DisCoDETREncoderGlobalLatentHead(nn.Module):
    '''
    Global contrastive loss on encoder output
    '''

    def __init__(self, projector: Dict, predictor: Dict, size_average: bool = True):
        super().__init__()

        self.projector = builder.build_neck(projector)
        self.predictor = builder.build_neck(predictor)
        self.size_average = size_average

    def proj(self, feat: Tensor) -> Tuple[Tensor, Tensor]:
        proj_feat = self.projector([feat])[0]
        pred_feat = self.predictor([proj_feat])[0]
        return proj_feat, pred_feat

    def forward(self, feat_v1: Tensor, feat_v2: Tensor, box_v1=None, box_v2=None, spatial_shapes=None) -> Dict:
        proj_v1, pred_v1 = self.proj(feat_v1)
        proj_v2, pred_v2 = self.proj(feat_v2)
        loss = 0.5 * loss_contra(pred_v1, proj_v2.clone().detach()) + \
            0.5 * loss_contra(pred_v2, proj_v1.clone().detach())
        return {'loss_enc_global_contra': loss}


@HEADS.register_module
class DisCoDETRDecoderLocalLatentHead(nn.Module):

    def __init__(self, projector: Dict, predictor: Dict, size_average: bool = True, aux_loss: bool = False):
        super().__init__()

        self.projector = builder.build_neck(projector)
        self.predictor = builder.build_neck(predictor)
        self.size_average = size_average
        self.aux_loss = aux_loss

    def proj(self, hs: Tensor) -> Tuple[Tensor]:
        proj_feat = self.projector([hs])[0]
        pred_feat = self.predictor([proj_feat])[0]
        return proj_feat.reshape(*hs.shape), pred_feat.reshape(*hs.shape)

    def forward(self, hs_v1: Tensor, hs_v2: Tensor, indices_v1: List[Tuple[Tensor]], indices_v2: List[Tuple[Tensor]]):
        proj_v1, pred_v1 = self.proj(hs_v1)
        proj_v2, pred_v2 = self.proj(hs_v2)

        new_indices_v2 = resort_matched_indices(indices_v1, indices_v2)
        indices = [(v1[0], v2[0]) for v1, v2 in zip(indices_v1, new_indices_v2)]

        proj_v1 = torch.stack([proj_v1[:, i, v1] for i, (v1, _) in enumerate(indices)]).transpose(0, 1)
        pred_v1 = torch.stack([pred_v1[:, i, v1] for i, (v1, _) in enumerate(indices)]).transpose(0, 1)
        proj_v2 = torch.stack([proj_v2[:, i, v2] for i, (_, v2) in enumerate(indices)]).transpose(0, 1)
        pred_v2 = torch.stack([pred_v2[:, i, v2] for i, (_, v2) in enumerate(indices)]).transpose(0, 1)

        proj_v1_detach = proj_v1.clone().detach()
        proj_v2_detach = proj_v2.clone().detach()

        c = proj_v1.shape[-1]
        loss = 0.5 * loss_contra(pred_v1[-1].reshape(-1, c), proj_v2_detach[-1].reshape(-1, c)) + \
            0.5 * loss_contra(pred_v2[-1].reshape(-1, c), proj_v1_detach[-1].reshape(-1, c))
        losses = {'loss_dec_local_contra': loss}
        if self.aux_loss:
            for i in range(len(proj_v1) - 1):
                loss = 0.5 * loss_contra(pred_v1[i].reshape(-1, c), proj_v2_detach[i].reshape(-1, c)) + \
                    0.5 * loss_contra(pred_v2[i].reshape(-1, c), proj_v1_detach[i].reshape(-1, c))
                losses.update({f'loss_dec_local_contra_{i}': loss})
        return losses


class DisCoDETRCriterion(nn.Module):

    def __init__(self, aux_loss, losses, part_contrast=True, batch_contrast=False, positive_from_cosine=False, contrast_transform=False, only_positive=False, part_contrast_transform=False, part_contrast_transform_proj=False, hidden_dim=256, eos_coef=0.1):
        super().__init__()
        # self.matcher = matcher
        self.only_positive = only_positive
        self.eos_coef = eos_coef
        self.losses = losses
        self.aux_loss = aux_loss
        self.num_classes = 2
        self.part_contrast = part_contrast
        self.batch_contrast = batch_contrast
        self.contrast_projector = None
        self.contrast_predictor = None
        self.contrast_use = None
        if contrast_transform and self.contrast_use:
            self.contrast_projector = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=False),
                                            nn.BatchNorm1d(hidden_dim),
                                            nn.ReLU(inplace=True), # first layer
                                            nn.Linear(hidden_dim, hidden_dim, bias=False),
                                            nn.BatchNorm1d(hidden_dim),
                                            nn.ReLU(inplace=True), # second layer
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.BatchNorm1d(hidden_dim, affine=False)) # output layer
            self.contrast_projector[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

            # build a 2-layer predictor
            self.contrast_predictor = nn.Sequential(nn.Linear(hidden_dim, 1024, bias=False),
                                            nn.BatchNorm1d(1024),
                                            nn.ReLU(inplace=True), # hidden layer
                                            nn.Linear(1024, hidden_dim)) # output layer
        if part_contrast_transform and self.contrast_use:
            self.contrast_predictor = nn.Sequential(nn.Linear(hidden_dim, 1024, bias=False),
                                            nn.BatchNorm1d(1024),
                                            nn.ReLU(inplace=True), # hidden layer
                                            nn.Linear(1024, hidden_dim)) # output layer
        if part_contrast_transform_proj and self.contrast_use:
            self.contrast_projector = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=False),
                                            nn.BatchNorm1d(hidden_dim),
                                            nn.ReLU(inplace=True), # first layer
                                            nn.Linear(hidden_dim, hidden_dim, bias=False),
                                            nn.BatchNorm1d(hidden_dim),
                                            nn.ReLU(inplace=True), # second layer
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.BatchNorm1d(hidden_dim, affine=False)) # output layer
            self.contrast_projector[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.positive_from_cosine = positive_from_cosine
        self.register_buffer('empty_weight', empty_weight)
        self.bbox_embed=None

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_feature(self, outputs, targets, indices):
        """Compute the mse loss between normalized features.
        """
        bs = len(targets)
        num_patches = len(targets[0]['boxes'])
        num_boxes = bs * num_patches

        """
        recon_gt torch.Size([80, 2048])
        outputs_feature torch.Size([6, 8, 100, 2048])
        outputs_class torch.Size([6, 8, 100, 3])
        """
        target_feature = outputs['gt_feature']
        idx = self._get_src_permutation_idx(indices)
        batch_size = len(indices)
        target_feature = target_feature.view(batch_size, target_feature.shape[0] // batch_size, -1)

        src_feature = outputs['pred_feature'][idx]
        target_feature = torch.cat([t[i] for t, (_, i) in zip(target_feature, indices)], dim=0)

        # l2 normalize the feature
        src_feature = F.normalize(src_feature, dim=1)
        target_feature = F.normalize(target_feature, dim=1)

        loss_feature = F.mse_loss(src_feature, target_feature, reduction='none')
        losses = {'loss_feature': loss_feature.sum() / num_boxes}

        return losses

    def loss_boxes(self, outputs, targets, indices):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        bs = len(targets)
        num_patches = len(targets[0]['boxes'])
        num_boxes = bs * num_patches

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    def compute_similarities(self, features1, features2, temperature):
        return F.cosine_similarity(features1, features2, dim=-1) / temperature
    
    def compute_loss(self, sim_positive, sim_negative):
        exp_sim_positive = torch.exp(sim_positive)
        
        exp_sim_negative = torch.exp(sim_negative)

        numerator = exp_sim_positive
        denominator = exp_sim_negative.sum(dim=-1, keepdim=True)

        loss = -torch.log(numerator / denominator)
        return loss.mean()

    def loss_contrast(self, outputs, targets, indices, ori_idx1=None, ori_idx2=None):
        num_patches=int(targets[0]['boxes'].shape[0])

        pred_features_v1 = outputs['ori_outputsv1']
        pred_features_v2 = outputs['ori_outputsv2']
        batch_size, num_queries, feature_dim = pred_features_v1.shape
        # num_matches= int(num_queries/num_patches)
        device = pred_features_v1.device

        def process_contrast_idx(contrast_idx):
            query_indices = torch.stack([idxs[0] for idxs in contrast_idx])
            patch_indices = torch.stack([idxs[1] for idxs in contrast_idx])

            return query_indices.view(batch_size, num_patches), patch_indices.view(batch_size, num_patches)
        # 这里修改匹配方法

        query_indices1, patch_indices1 = process_contrast_idx(ori_idx1)
        query_indices2, patch_indices2 = process_contrast_idx(ori_idx2)

        def get_patch_features(query_indices, patch_indices, version='v1'):
            batch_indices = torch.arange(batch_size, device=device)[:, None]
            if version == 'v1':
                patch_features = pred_features_v1[batch_indices, query_indices]
            else:
                patch_features = pred_features_v2[batch_indices, query_indices]
            return patch_features  # Shape: [batch_size, num_patches, num_matches, feature_dim]

        patch_features1 = get_patch_features(query_indices1, patch_indices1, version='v1')
        patch_features2 = get_patch_features(query_indices2, patch_indices2, version='v2')

        if self.contrast_predictor is not None and self.contrast_projector is not None and self.only_positive is True: 

            patch_features1_z1 = self.contrast_projector(patch_features1.flatten(0, 1))
            patch_features2_z2 = self.contrast_projector(patch_features2.flatten(0, 1))

            positive_p1 = self.contrast_predictor(patch_features1_z1).view(batch_size, num_patches, -1)
            positive_p2 = self.contrast_predictor(patch_features2_z2).view(batch_size, num_patches, -1)

            positive_z1 = patch_features1_z1.detach().view(batch_size, num_patches, -1)
            positive_z2 = patch_features2_z2.detach().view(batch_size, num_patches, -1)

            loss1 = 1.0 - self.compute_similarities(positive_p1.unsqueeze(2), positive_z2.unsqueeze(2), 1)
            loss2 = 1.0 - self.compute_similarities(positive_p2.unsqueeze(2), positive_z1.unsqueeze(2), 1)
        elif self.contrast_predictor is not None and self.contrast_projector is None and self.only_positive is True:

            patch_features1_z1 = patch_features1.flatten(0, 1)
            patch_features2_z2 = patch_features2.flatten(0, 1)

            positive_p1 = self.contrast_predictor(patch_features1_z1).view(batch_size, num_patches, -1)
            positive_p2 = self.contrast_predictor(patch_features2_z2).view(batch_size, num_patches, -1)

            positive_z1 = patch_features1_z1.detach().view(batch_size, num_patches, -1)
            positive_z2 = patch_features2_z2.detach().view(batch_size, num_patches, -1)

            loss1 = 1.0 - self.compute_similarities(positive_p1.unsqueeze(2), positive_z2.unsqueeze(2), 1)
            loss2 = 1.0 - self.compute_similarities(positive_p2.unsqueeze(2), positive_z1.unsqueeze(2), 1)
        elif self.contrast_predictor is None and self.contrast_projector is not None and self.only_positive is True:
            # print('only positive')

            patch_features1_z1 = self.contrast_projector(patch_features1.flatten(0, 1))
            patch_features2_z2 = self.contrast_projector(patch_features2.flatten(0, 1))

            positive_p1 = patch_features1_z1.view(batch_size, num_patches, -1)
            positive_p2 = patch_features2_z2.view(batch_size, num_patches, -1)

            positive_z1 = positive_p1.detach().view(batch_size, num_patches, -1)
            positive_z2 = positive_p2.detach().view(batch_size, num_patches, -1)

            loss1 = 1.0 - self.compute_similarities(positive_p1.unsqueeze(2), positive_z2.unsqueeze(2), 1)
            loss2 = 1.0 - self.compute_similarities(positive_p2.unsqueeze(2), positive_z1.unsqueeze(2), 1)
        elif self.only_positive is True:
            patch_features1 = patch_features1[:, :, :]
            patch_features2 = patch_features2[:, :, :]

            positive_p1 = patch_features1
            positive_p2 = patch_features2

            positive_z1 = patch_features1.detach()
            positive_z2 = patch_features2.detach()

            loss1 = 1.0 - self.compute_similarities(positive_p1.unsqueeze(2), positive_z2.unsqueeze(2), 1)
            loss2 = 1.0 - self.compute_similarities(positive_p2.unsqueeze(2), positive_z1.unsqueeze(2), 1)
        # Average the losses
        total_loss = (loss1 + loss2) / 2
        losses = {'loss_contrast': total_loss}
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'feature': self.loss_feature,
            'contrast': self.loss_contrast,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        if loss == 'contrast':
            return loss_map[loss](outputs, targets, indices, **kwargs)
        return loss_map[loss](outputs, targets, indices)

    def forward(self, outputs, targets, contrast_use, all_layers=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs_without_aux, targets)
        indices = outputs['ori_idx']
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if contrast_use and loss == 'contrast':
                losses.update(self.get_loss(loss, outputs, targets, indices, ori_idx1=outputs['ori_idx'], ori_idx2=outputs['ori_idx2']))
            elif loss != 'contrast':
                losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = aux_outputs['ori_idx']
                for loss in self.losses:
                    if 'contrast' in loss and contrast_use and all_layers:
                            kwargs = {'ori_idx1':aux_outputs['ori_idx'], 'ori_idx2':aux_outputs['ori_idx2']}
                    elif 'contrast' in loss and (contrast_use is False or all_layers != True):
                        continue
                    else:
                        kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
