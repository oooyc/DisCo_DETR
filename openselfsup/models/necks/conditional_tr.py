
import math
import torch
from torch import nn, Tensor
from typing import Optional, List
from typing import Tuple
from .helper import MultiheadAttention, MLP, get_clones, get_activation_fn
from .vanilla_tr import VanillaEncoder, VanillaEncoderLayer
from ..registry import NECKS
import numpy as np
from openselfsup.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from scipy.optimize import linear_sum_assignment

def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def pred_bbox_with_reference(forward, hs, reference=None):
    if reference is None:
        outputs_coord = forward(hs).sigmoid()
    else:
        if type(reference) is tuple:
            init_reference, inter_references = reference
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                tmp = forward(hs[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
                outputs_coords.append(outputs_coord)
            outputs_coord = torch.stack(outputs_coords)

        else:
            reference = inverse_sigmoid(reference)
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                tmp = forward(hs[lvl])
                tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
                outputs_coords.append(outputs_coord)
            outputs_coord = torch.stack(outputs_coords)


    return outputs_coord


@NECKS.register_module
class ConditionalTR(nn.Module):

    def __init__(self,
                 hidden_dim=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 return_intermediate_dec=False,
                 dist_use=False,
                 pattern=0,
                 num_queries=300,
                 match_layer1=False,
                 part_contrast=True,
                 tgt_use=False,
                 with_patch=False,
                 dist_match=True):
        super().__init__()

        encoder_layer = VanillaEncoderLayer(hidden_dim, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.encoder = VanillaEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = ConditionalDecoderLayer(hidden_dim, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = ConditionalDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          hidden_dim=hidden_dim)
        self.decoder.dist_use = dist_use
        self.decoder.part_contrast = part_contrast
        self.dist_use = dist_use
        self.decoder.pattern = pattern
        self.match_layer1 = match_layer1
        self.decoder.match_layer1 = match_layer1

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.tgt_use = tgt_use
        if tgt_use:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.with_patch = with_patch
        self.decoder.with_patch = with_patch
        self.decoder.dist_match = dist_match
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, patch_feature=None, target_boxes=None, decoder_mask=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        # query_embed = query_embed.permute(1, 0, 2)
        if self.tgt_use:
            tgt = self.tgt_embed.weight
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.permute(1, 0, 2)
        else:
            tgt = torch.zeros((query_embed.shape[1], query_embed.shape[0], query_embed.shape[2]), device=src.device)
        
        # tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs, references, ori_idx_list = self.decoder(
            tgt, memory, memory_key_padding_mask=mask, pos=pos_embed,
            query_pos=query_embed, tgt_mask=decoder_mask, patch_feature=patch_feature, target_boxes=target_boxes)
        return hs, memory.permute(1, 2, 0).view(bs, c, h, w), ori_idx_list, references


class ConditionalDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
                 hidden_dim=256):
        super().__init__()
        self.num_queries = None
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.query_scale = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.ref_point_head = MLP(hidden_dim, hidden_dim, 2, 2)
        for layer_id in range(num_layers - 1):
            self.layers[layer_id + 1].ca_qpos_proj = None
        self.bbox_embed = None
        self.class_embed = None
        self.dist_use = None
        self.match_layer1 = None
        self.part_contrast =None
        self.with_patch = None
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

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                patch_feature=None,
                target_boxes=None):
        output = tgt

        intermediate = []
        ori_idx_list = []
        # reference_points_before_sigmoid = self.ref_point_head(query_pos)  # [num_queries, batch_size, 2]
        # reference_points = reference_points_before_sigmoid.sigmoid().transpose(0, 1)


        if self.dist_match:
        # print('im here')
            with torch.no_grad():
                reference_points_before_sigmoid = self.ref_point_head(query_pos)  # [num_queries, batch_size, 2]
                reference_points = reference_points_before_sigmoid.sigmoid()#.transpose(0, 1)
                batch_indices, query_indices, patch_indices = self.dist_based_ordered_patch_feature(patch_feature, reference_points, target_boxes, None)
            # if self.dist_use:
            tgt1 = torch.zeros(reference_points.shape[0], reference_points.shape[1], patch_feature.shape[-1], device=patch_feature.device)
            tgt1[batch_indices[:, None], query_indices, :] = patch_feature[batch_indices[:, None], patch_indices, :]
            reference_points = self.ref_point_head(query_pos + tgt1).sigmoid()
            input_query_pos = query_pos + tgt1
            input_query_pos = input_query_pos.transpose(0, 1)
        else:
            reference_points = self.ref_point_head(query_pos + patch_feature).sigmoid()
            input_query_pos = query_pos + patch_feature
            input_query_pos = input_query_pos.transpose(0, 1)

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :2].transpose(0, 1)  # [num_queries, batch_size, 2]

            # For the first decoder layer, we do not apply transformation over p_s
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(output)

            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)
            # apply transformation
            query_sine_embed = query_sine_embed * pos_transformation
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=input_query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0))
            

            # hack implementation for my distance based selfsupvised-learning #iterative bounding box refinement
            with torch.no_grad():
                if self.bbox_embed is not None and self.match_layer1 is False or (self.match_layer1 is True and self.bbox_embed is not None and layer_id == 0):# and my_reference_points is None:
                    tmp = self.bbox_embed(output.transpose(0, 1))
                    if reference_points.shape[-1] == 4:
                        new_reference_points = tmp + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    else:
                        assert reference_points.shape[-1] == 2
                        new_reference_points = tmp
                        new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    my_reference_points = new_reference_points.detach()
                    my_pred_out = self.class_embed(output.transpose(0, 1))

            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            # start_event.record()
            if self.match_layer1 is False or (self.match_layer1 is True and layer_id == 0):
                with torch.no_grad():
                    batch_indices, ori_idx = self.dist_based_ordered_patch_feature(patch_feature, my_reference_points, target_boxes, my_pred_out)
            # end_event.record()
            # torch.cuda.synchronize()
            # print(f'dist_based_ordered_patch_feature_time{lid}: {start_event.elapsed_time(end_event)} ms')
            ori_idx_list.append(ori_idx)
            # if (self.dist_use and self.match_layer1 is False) or (self.dist_use and self.match_layer1 is True and layer_id == 0):
            #     # tgt1 = torch.zeros(reference_points.shape[0], reference_points.shape[1], patch_feature.shape[-1], device=patch_feature.device)
            #     tgt1[batch_indices[:, None], query_indices, :] = patch_feature[batch_indices[:, None], patch_indices, :]
            #     input_query_pos = query_pos + tgt1
            #     input_query_pos = input_query_pos.transpose(0, 1)
                # 是否可以更改reference points input呢？


            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate).transpose(1, 2), reference_points, ori_idx_list

        return output.unsqueeze(0).transpose(1, 2), reference_points, ori_idx_list


class ConditionalDecoderLayer(nn.Module):

    def __init__(self, hidden_dim, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(hidden_dim, hidden_dim)
        self.sa_qpos_proj = nn.Linear(hidden_dim, hidden_dim)
        self.sa_kcontent_proj = nn.Linear(hidden_dim, hidden_dim)
        self.sa_kpos_proj = nn.Linear(hidden_dim, hidden_dim)
        self.sa_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.self_attn = MultiheadAttention(hidden_dim, nhead, dropout=dropout, vdim=hidden_dim)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ca_qpos_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ca_kcontent_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ca_kpos_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ca_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ca_qpos_sine_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = MultiheadAttention(hidden_dim * 2, nhead, dropout=dropout, vdim=hidden_dim)

        self.nhead = nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed=None,
                     is_first=False):

        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.sa_qcontent_proj(tgt)  # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        q = q_content + q_pos
        k = k_content + k_pos

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                               key=k,
                               value=v, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed=None,
                is_first=False):
        if self.normalize_before:
            raise NotImplementedError
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask,
                                 pos, query_pos, query_sine_embed, is_first)
