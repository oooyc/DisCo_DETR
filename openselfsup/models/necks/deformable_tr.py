
import math
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_, normal_
from scipy.optimize import linear_sum_assignment
from openselfsup.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from openselfsup.models.utils.misc import inverse_sigmoid
from openselfsup.models.utils.ms_deform_op.modules import MSDeformAttn
from .helper import get_clones, get_activation_fn
from ..registry import NECKS
import numpy as np
import torch.nn.functional as F
import openselfsup.utils.box_ops as box_ops

@NECKS.register_module
class DeformableTR(nn.Module):
    def __init__(self, 
                 d_model=256, 
                 nhead=8,
                 num_encoder_layers=6, 
                 num_decoder_layers=6, 
                 dim_feedforward=1024, 
                 dropout=0.1,
                 activation="relu", 
                 return_intermediate_dec=False,
                 num_feature_levels=4, 
                 dec_n_points=4,  
                 enc_n_points=4,
                 two_stage=False, 
                 two_stage_num_proposals=300,
                 num_queries=300,
                 dist_use=False,
                 pattern=0,
                 match_layer1=False,
                 part_contrast=True,
                 dist_match=True):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        # self.match_dict = {}
        # for num in range(num_queries):
        #     self.match_dict[num] = 0
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward, dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        self.decoder.dist_use = dist_use
        self.decoder.dist_match = dist_match
        self.dist_match = dist_match
        self.decoder.part_contrast = part_contrast
        self.dist_use = dist_use
        self.decoder.pattern = pattern
        self.match_layer1 = match_layer1
        self.decoder.match_layer1 = match_layer1
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)
            self.tgt_embed = nn.Embedding(num_queries, d_model)

        self.decoder.num_queries = num_queries
    def init_weights(self):
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    # src_v2_proj, mask_v2, patch_v1, pos_v2
    # conditional detr src, mask, query_embed, pos_embed, decoder_mask = None
    def forward(self, srcs, masks, query_embed, pos_embeds, patch_feature=None, target_boxes=None, decoder_mask=None):
        assert self.two_stage or query_embed is not None
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            # query_embed = query_embed[:, 0]  
            # query_embed = query_embed[0,:,:]
            # query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)

            query_embed, tgt = query_embed, self.tgt_embed.weight
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            tgt1 = None
            if self.dist_match:
                with torch.no_grad():
                    reference_points = self.reference_points(query_embed).sigmoid()
                    batch_indices, query_indices, patch_indices = self.decoder.dist_based_ordered_patch_feature(patch_feature, reference_points, target_boxes, None)
                # if self.dist_use:
                tgt1 = torch.zeros(reference_points.shape[0], reference_points.shape[1], patch_feature.shape[-1], device=patch_feature.device)
                tgt1[batch_indices[:, None], query_indices, :] = patch_feature[batch_indices[:, None], patch_indices, :]
                reference_points = self.reference_points(query_embed + tgt1).sigmoid()
            elif self.dist_match is False:
                tgt1 = patch_feature
                reference_points = self.reference_points(query_embed + tgt1).sigmoid()
            # else:
            #     reference_points = self.reference_points(query_embed + patch_feature).sigmoid()

            init_reference_out = reference_points

        # decoder
        hs, inter_references, ori_idx = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten,  tgt_mask=decoder_mask, patch_feature=patch_feature, tgt_boxes=target_boxes, first_ordered_feature=tgt1)

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact

        start_ix = 0
        multi_scale_memorys = []
        memory = memory.permute(0, 2, 1)
        bs, hidden_dim, _len = memory.shape
        for shape in spatial_shapes:
            l = shape[0] * shape[1]
            multi_scale_memorys.append(memory[:, :, start_ix:  start_ix + l].view(bs, hidden_dim, shape[0], shape[1]).contiguous())
            start_ix += l
        # memory和hs需要

        assert start_ix == _len

        return hs, multi_scale_memorys, ori_idx, (init_reference_out, inter_references_out)

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None, tgt_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), attn_mask=tgt_mask,)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.dist_use = None
        self.match_layer1 = None
        self.part_contrast =None
        self.num_queries = None
        self.dist_match = None
    def expand_array_optimized(self, original_array):
        if len(original_array) != 10:
            raise ValueError("The original array must have exactly 10 elements")

        original_array = np.array(original_array)
        if not np.all((original_array >= 0) & (original_array <= 299)):
            raise ValueError("All elements in the original array must be between 0 and 299")

        # 创建一个 10x300 的网格，每行包含 0-299
        grid = np.tile(np.arange(self.num_queries), (10, 1))

        # 创建一个 10x300 的掩码，标记出每行中等于原始元素的位置
        mask = (grid == original_array[:, np.newaxis])

        # 将原始元素放在每个子数组的开头
        result = np.where(mask, 0, grid).flatten()
        result[::self.num_queries] = original_array

        return result

    def dist_based_ordered_patch_feature(self, patch_feature, reference_points, tgt_boxes, my_pred_out):

        # tgt_boxes = torch.stack(tgt_boxes, dim=0)
        batch_size, num_queries, _ = reference_points.shape
        num_queries_per_group = num_queries
        num_patches_per_group = tgt_boxes.shape[1]

        tgt_boxes = tgt_boxes.view(batch_size, num_patches_per_group, -1)
        centers = (tgt_boxes[:, :, :2]).contiguous()# * img_sizes.unsqueeze(1)).contiguous()
        sizes = (tgt_boxes[:, :, 2:]).contiguous()# * img_sizes.unsqueeze(1)).contiguous()
        
        ref_boxes = reference_points.view(batch_size, num_queries_per_group, -1)

        # Combine centers and sizes to form boxes
        if reference_points.shape[-1] == 2:
            patch_boxes = centers
        else:
            patch_boxes = torch.cat((centers, sizes), dim=-1)

        num_matches = num_queries_per_group / num_patches_per_group

        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        # start_event.record()
        if reference_points.shape[-1] == 2:
            ori_idx, contrast_idx = self.global_optimal_matching(patch_boxes, ref_boxes, batch_size, num_patches_per_group, num_queries_per_group, num_matches, my_pred_out)
        else:
            ori_idx = self.global_optimal_matching(patch_boxes, ref_boxes, batch_size, num_patches_per_group, num_queries_per_group, num_matches, my_pred_out)
        # end_event.record()
        # torch.cuda.synchronize()
        # print(f'optimal_matching_time: {start_event.elapsed_time(end_event)} ms')

        batch_indices = torch.arange(batch_size, device=patch_feature.device)

        if reference_points.shape[-1] == 2:
            query_indices = torch.stack([src for (src, _) in contrast_idx], dim=0)
            patch_indices = torch.stack([src for (_, src) in contrast_idx], dim=0)
            patch_indices = patch_indices.repeat_interleave(int(num_matches), dim=1)

        # elif self.pattern == 1:
        #     query_indices = torch.stack([src for (src, _) in ori_idx], dim=0)
        #     patch_indices = torch.stack([src for (_, src) in ori_idx], dim=0)

        # tgt1 = torch.zeros(batch_size, num_queries, patch_feature.shape[-1], device=patch_feature.device)

        # tgt1[batch_indices[:, None], query_indices, :] = patch_feature[batch_indices[:, None], patch_indices, :]
        
        # tgt1 = patch_feature.repeat_interleave(10, dim=1).repeat(1, 3, 1)
        if reference_points.shape[-1] == 2:
            return batch_indices, query_indices, patch_indices
        return batch_indices, ori_idx

    def insert_b_into_a(self, a, b, num_matches):
        result = []
        for i in range(0, len(a), num_matches-1):
            result.append(b[i//(num_matches-1)])
            result.extend(a[i:i+num_matches-1])
        return result

    def global_optimal_matching(self, patch_boxes, ref_boxes, batch_size, num_patches_per_group, num_queries_per_group, num_matches, my_pred_out):
        # count = 0
        if my_pred_out is not None:
            my_pred_out = my_pred_out.view(batch_size, num_queries_per_group, -1)

            # count += 1
            # patch_boxes 2,3,10,4
            # ref_boxes 2,3,300,4
        patch_boxes_group = patch_boxes
        ref_boxes_group = ref_boxes
        # Compute L1 loss
        if ref_boxes_group.shape[-1] == 4:
        # bs, num_queries = outputs["pred_logits"].shape[:2]

        # # We flatten to compute the cost matrices in a batch
        # out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        # out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
            my_pred_out = my_pred_out.flatten(0, 1).softmax(-1)
            ref_boxes = ref_boxes.flatten(0,1)
        # # Also concat the target labels and boxes
        # tgt_ids = torch.cat([v["labels"] for v in targets])
        # tgt_bbox = torch.cat([v["boxes"] for v in targets])
            tgt_ids = torch.ones(batch_size*num_patches_per_group, dtype=torch.long, device=patch_boxes.device)
        # # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # # but approximate it in 1 - proba[target class].
        # # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -my_pred_out[:, tgt_ids]

        # # Compute the L1 cost between boxes
            ref_boxes = ref_boxes.float()
            patch_boxes = patch_boxes.flatten(0, 1)
            ptach_boxes = patch_boxes.float()
            # out_bbox = out_bbox.float()
            # tgt_bbox = tgt_bbox.float()
            cost_bbox = torch.cdist(ref_boxes, ptach_boxes, p=1)

        # # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(ref_boxes), box_cxcywh_to_xyxy(ptach_boxes))

        # # Final cost matrix
            C = 5 * cost_bbox + 1 * cost_class + 2 * cost_giou
        elif ref_boxes_group.shape[-1] == 2:
            assert ref_boxes_group.shape[-1] == 2
            ref_boxes = ref_boxes.flatten(0,1)
            ref_boxes = ref_boxes.float()
            patch_boxes = patch_boxes.flatten(0, 1)
            ptach_boxes = patch_boxes.float()
            # out_bbox = out_bbox.float()
            # tgt_bbox = tgt_bbox.float()
            cost_bbox = torch.cdist(ref_boxes, ptach_boxes, p=1)
            C = cost_bbox

        C = C.view(batch_size, num_queries_per_group, -1).cpu()

        sizes = [int(num_patches_per_group) for i in range(batch_size)]
    # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
    # return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        C_split = [c[i].permute(1,0) for i,c in enumerate(C.split(sizes,-1))]
        arr_indices = [linear_sum_assignment(c) for c in C_split]
        indices = [(torch.as_tensor(j, dtype=torch.int64), torch.as_tensor(i, dtype=torch.int64)) for i, j in arr_indices]

        if ref_boxes_group.shape[-1] == 2:
            C_unmatched = [c.clone() for c in C_split]
            for i, (row, _) in enumerate(indices):
                C_unmatched[i][:, row] = float('inf')
            neg_indices = []
            for i, c in enumerate(C_unmatched):
                c_neg = c.repeat_interleave(int(num_matches)-1, dim=0)
                # c_neg = c_neg.permute(1,0)
                row_ind, col_ind = linear_sum_assignment(c_neg)
                row_ind = range(int(num_patches_per_group))
                col_ind = self.insert_b_into_a(col_ind, arr_indices[i][1], int(num_matches))
                neg_indices.append((torch.as_tensor(col_ind, dtype=torch.int64), torch.as_tensor(row_ind, dtype=torch.int64)))
            return indices, neg_indices
        return indices


    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None,  tgt_mask=None, patch_feature=None, tgt_boxes=None, first_ordered_feature=None):
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        ori_idx_list = []
        my_reference_points = None
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            if lid == 0:
                # with torch.no_grad():
                #     ordered_patch_feature, ori_idx, contrast_idx = self.dist_based_ordered_patch_feature(patch_feature, reference_points, tgt_boxes)
                # ordered_patch_feature.detach()
                # ori_idx_list.append(ori_idx)
                # contrast_idx_list.append(contrast_idx)
                if first_ordered_feature is not None:
                    input_query_pos = query_pos + first_ordered_feature# + ordered_patch_feature
                else:
                    input_query_pos = query_pos + patch_feature
            output = layer(output, input_query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask, tgt_mask=tgt_mask)

            # hack implementation for my distance based selfsupvised-learning #iterative bounding box refinement
            with torch.no_grad():
                if self.bbox_embed is not None and self.match_layer1 is False or (self.match_layer1 is True and self.bbox_embed is not None and lid == 0):# and my_reference_points is None:
                    tmp = self.bbox_embed(output)
                    if reference_points.shape[-1] == 4:
                        new_reference_points = tmp + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    else:
                        assert reference_points.shape[-1] == 2
                        new_reference_points = tmp
                        new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    my_reference_points = new_reference_points.detach()
                    my_pred_out = self.class_embed(output)

            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            # start_event.record()
            if self.match_layer1 is False or (self.match_layer1 is True and lid == 0):
                with torch.no_grad():
                    batch_indices, ori_idx = self.dist_based_ordered_patch_feature(patch_feature, my_reference_points, tgt_boxes, my_pred_out)
            # end_event.record()
            # torch.cuda.synchronize()
            # print(f'dist_based_ordered_patch_feature_time{lid}: {start_event.elapsed_time(end_event)} ms')
            ori_idx_list.append(ori_idx)
            # if (self.dist_use and self.match_layer1 is False) or (self.dist_use and self.match_layer1 is True and lid == 0):
            #     tgt1 = torch.zeros(reference_points.shape[0], reference_points.shape[1], patch_feature.shape[-1], device=patch_feature.device)
            #     tgt1[batch_indices[:, None], query_indices, :] = patch_feature[batch_indices[:, None], patch_indices, :]
            #     input_query_pos = query_pos + tgt1
                # 是否可以更改reference points input呢？
            
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:

            return torch.stack(intermediate), torch.stack(intermediate_reference_points), ori_idx_list

        return output, reference_points, ori_idx_list


