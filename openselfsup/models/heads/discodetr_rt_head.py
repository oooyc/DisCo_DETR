
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
from torch import Tensor
from typing import Dict, List, Tuple
from ..necks.helper import MLP
from .. import builder
# from torchvision.ops import box_convert, generalized_box_iou
import openselfsup.utils.box_ops as box_ops
from ..registry import HEADS
from utils.dist import get_world_size, is_dist_available_and_initialized

@HEADS.register_module
class DisCoRTDETRDecoderLocalLatentHead(nn.Module):
    def __init__(self, hidden_dim = 256):
        super().__init__()
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
            
    def compute_similarities(self, features1, features2, temperature):
        return F.cosine_similarity(features1, features2, dim=-1) / temperature
    
    
    def forward(self, num_patches, hs_v1, hs_v2, idx1, idx2):
        # num_patches=int(targets[0]['boxes'].shape[0])
        batch_size, num_queries, feature_dim = hs_v1.shape
        # num_matches= int(num_queries/num_patches)
        device = hs_v1.device

        def process_idx(idx):
            query_indices = torch.stack([idxs[0] for idxs in idx])
            # patch_indices = torch.stack([idxs[1] for idxs in idx])

            return query_indices.view(batch_size, num_patches)#, patch_indices.view(batch_size, num_patches)
        # 这里修改匹配方法

        query_indices1 = process_idx(idx1)
        query_indices2 = process_idx(idx2)

        def get_patch_features(query_indices, version='v1'):
            batch_indices = torch.arange(batch_size, device=device)[:, None]
            if version == 'v1':
                patch_features = hs_v1[batch_indices, query_indices]
            else:
                patch_features = hs_v2[batch_indices, query_indices]
            return patch_features  # Shape: [batch_size, num_patches, num_matches, feature_dim]

        patch_features1 = get_patch_features(query_indices1, version='v1')
        patch_features2 = get_patch_features(query_indices2, version='v2')

        if self.contrast_predictor is not None and self.contrast_projector is not None and self.only_positive is True: 

            patch_features1_z1 = self.contrast_projector(patch_features1.flatten(0, 1))
            patch_features2_z2 = self.contrast_projector(patch_features2.flatten(0, 1))

            positive_p1 = self.contrast_predictor(patch_features1_z1).view(batch_size, num_patches, -1)
            positive_p2 = self.contrast_predictor(patch_features2_z2).view(batch_size, num_patches, -1)

            positive_z1 = patch_features1_z1.detach().view(batch_size, num_patches, -1)
            positive_z2 = patch_features2_z2.detach().view(batch_size, num_patches, -1)

            loss1 = 1.0 - self.compute_similarities(positive_p1.unsqueeze(2), positive_z2.unsqueeze(2), 1)
            loss2 = 1.0 - self.compute_similarities(positive_p2.unsqueeze(2), positive_z1.unsqueeze(2), 1)
        
        total_loss = (loss1 + loss2) / 2
        losses = {'loss_contrast': total_loss}
        return losses



@HEADS.register_module
class DisCoRTDETRPredictHead(nn.Module):

    def __init__(self,
                 aux_loss=False,
                 hidden_dim=None,
                 size_average=True,
                 feature_recon: bool = False,
                 backbone_channels: int = 2048,
                 num_classes=None,
                 ):
                 #matcher_cfg: Dict = None):
        super().__init__()
        # self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.feature_recon = feature_recon
        self.aux_loss = aux_loss
        self.size_average = size_average
        # matcher = build_matcher(matcher_cfg)
        self.hidden_dim=hidden_dim
        # self.class_embed = nn.Linear(hidden_dim, 2 + 1)  # 0 or 1
        if self.feature_recon:
            # self.feature_align = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
            self.feature_align = MLP(hidden_dim, hidden_dim, backbone_channels, 2)
        losses = ['vfl', 'cardinality', 'boxes']
        if self.feature_recon:
            losses.append('feature')
        self.num_classes = num_classes
        self.criterion = DisCoRTDETRSetCriterion(aux_loss, losses=losses, contrast_transform=self.contrast_transform, hidden_dim=self.hidden_dim, num_classes=self.num_classes)
        # self.criterion.bbox_embed = self.bbox_embed
        # self.criterion = SiamDETRCriterion(matcher, aux_loss, losses=losses)

    def forward(self, hs, out, recon_gt=None, target=None):
        """Head for SiameseDETR
        Args:
            hs (Tensor): [num_dec, N, num_queries, C]
            target_box (Tensor): [N, num_patches, 4], unnormalized
        """
        num_dec, bs, num_queries, c = hs.shape


        if self.feature_recon:
            outputs_feature = self.feature_align(hs)
            out.update({
                'gt_feature': recon_gt,
                'pred_feature': outputs_feature[-1]})
            if self.aux_loss:
                for i in range(len(num_dec) - 1):
                    out['aux_outputs'][i].update({'pred_feature': outputs_feature[i]})
                    out['aux_outputs'][i].update({'gt_feature': recon_gt})

        loss_dict, indices = self.criterion(out, target)
        return loss_dict, indices


class DisCoRTDETRSetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
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

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def compute_similarities(self, features1, features2, temperature):
        return F.cosine_similarity(features1, features2, dim=-1) / temperature

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'vfl': self.loss_labels_vfl,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        indices_arr = []
        indices_arr.append(indices)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of cdn auxiliary losses. For rtdetr
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if i == 6:
                    indices = aux_outputs['ori_idx']
                else:
                    indices = self.matcher(aux_outputs, targets)
                indices_arr.insert(0, indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, indices_arr

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        '''get_cdn_matched_indices
        '''
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices





@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




