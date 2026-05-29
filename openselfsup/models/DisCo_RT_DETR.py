
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import RoIAlign
from typing import Dict, List, Tuple, Union
import openselfsup.utils.box_ops as box_ops
from openselfsup.utils import print_log, parse_losses
from openselfsup.utils.nested_tensor import NestedTensor, nested_tensor_from_tensor_list, \
    nested_multiview_tensor_from_tensor_list
from . import builder
from .necks.helper import MLP
from .registry import MODELS
from .utils.position_encoding import build_position_encoding
import numpy as np

def decompose_multiview_tensor(tensor: NestedTensor) -> Tuple[NestedTensor, NestedTensor]:
    tensors, _ = tensor.decompose()
    if tensors.ndim != 5:  # bs, n_view, c, h, w
        raise RuntimeError(f'invalid multiview NestedTensor shape {tensors.shape}')
    
    img_v1 = tensors[:, 0, ...].contiguous()
    # mask_v1 = mask[:, 0, ...].contiguous()
    img_v2 = tensors[:, 1, ...].contiguous()
    # mask_v2 = mask[:, 1, ...].contiguous()
    return img_v1, img_v2


def decompose_multiview_box(box: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
    box_v1 = [b[0, ...].contiguous() for b in box]
    box_v2 = [b[1, ...].contiguous() for b in box]
    return box_v1, box_v2


def add_randomness_on_multiview_box(box_v1, box_v2, img_size, ratio=0.05, same_randomness=False):
    def _add(box, size, ratio, version=None, last_addon=None):
        w = (box[:, 2] - box[:, 0]).view(-1, 1)
        h = (box[:, 3] - box[:, 1]).view(-1, 1)
        if last_addon is None:
            randomness = torch.randn(w.shape[0], 4).sigmoid()
            randomness = randomness * ratio * 2 - ratio
            addon = torch.cat([w, h, w, h], dim=-1) * randomness.to(box.device)
            new_box = box + addon
        elif version == 'v2' and last_addon is not None:
            new_box = box + last_addon
        
        max_h, max_w = size.unbind()
        new_box[:, ::2].clamp_(min=0, max=max_w)
        new_box[:, 1::2].clamp_(min=0, max=max_h)
        if version == 'v1':
            return new_box, addon
        return new_box
    
    new_box_v1 = []
    new_box_v2 = []
    tmp_box = None
    tmp_addon = None
    for b1, b2, sz in zip(box_v1, box_v2, img_size):
        if same_randomness is False:
            new_box_v1.append(_add(b1, sz, ratio))
            new_box_v2.append(_add(b2, sz, ratio))
        else:
            tmp_box, tmp_addon = _add(b1, sz, ratio, version='v1')
            new_box_v1.append(tmp_box)
            new_box_v2.append(_add(b2, sz, ratio, version='v2', last_addon=tmp_addon))
    return new_box_v1, new_box_v2


@MODELS.register_module
class disCo_rt_detr(nn.Module):

    def __init__(self,
                 backbone: Dict,
                 transformer: Dict,
                 pred_head: Dict,
                 backbone_channels: int,
                 hidden_dim: int,
                 encoder_head: Union[Dict, List[Dict]] = None,
                 decoder_head: Union[Dict, List[Dict]] = None,
                 pretrained: str = None,
                 freeze_backbone: bool = False,
                 num_queries: int = 100,
                 num_patches: int = 100,
                 box_disturbance: float = 0,
                 query_shuffle: bool = False,
                 feature_recon: bool = False,
                 weight_dict: Dict = None,
                 multi_scale_features: bool = False,
                 multi_scale_features_backbone_strides: tuple = None,
                 multi_scale_features_backbone_num_channels: tuple = None,
                 num_encoder_layers = 1,
                 use_encoder_idx = [2],
                 eval_spatial_size = [640, 640],
                 pe_temperature=10000,
                 multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800],):
        super().__init__()
        self.multi_scale = multi_scale
        self.pe_temperature = pe_temperature
        self.hidden_dim = hidden_dim
        self.backbone = builder.build_backbone(backbone)
        self.patch2query = nn.Linear(backbone_channels, hidden_dim)
        self.transformer = builder.build_neck(transformer)
        self.pred_head = builder.build_head(pred_head)  # box + cls + recon
        self.enc_head, self.dec_head = None, None
        self.multi_scale_features = multi_scale_features
        self.tr_class = transformer['type']
        # channel projection
        self.multi_scale_features_backbone_strides = multi_scale_features_backbone_strides
        self.multi_scale_features_backbone_num_channels = multi_scale_features_backbone_num_channels

        #-----------------RT-DETR-----------------
        self.input_proj = nn.ModuleList()
        for in_channel in self.multi_scale_features_backbone_num_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        if encoder_head is not None:
            if not isinstance(encoder_head, List):
                encoder_head = [encoder_head]
            self.enc_head = nn.Sequential(*[builder.build_head(head) for head in encoder_head])
        if decoder_head is not None:
            if not isinstance(decoder_head, List):
                decoder_head = [decoder_head]
            self.dec_head = nn.Sequential(*[builder.build_head(head) for head in decoder_head])

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.multi_scale_features:
            self.align = []
            for stride in self.multi_scale_features_backbone_strides:
                roi_align = RoIAlign(output_size=(7, 7), spatial_scale=(1. / stride), sampling_ratio=-1)  # wo/ DC5
                self.align.append(roi_align)
        else:
            self.align = RoIAlign(output_size=(7, 7), spatial_scale=(1. / 32.), sampling_ratio=-1)  # wo/ DC5

        self.num_queries = num_queries
        self.num_patches = num_patches
        self.query_shuffle = query_shuffle
        self.box_disturbance = box_disturbance
        self.weight_dict = self.generate_weight_dict(weight_dict)
        self.feature_recon = feature_recon

        self.init_weights(pretrained=pretrained)
        self.freeze_backbone = freeze_backbone
        # additional
        self.num_encoder_layers = num_encoder_layers
        self.use_encoder_idx = use_encoder_idx
        self.eval_spatial_size = eval_spatial_size

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load backbone from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained, strict=False)
        self.transformer.init_weights()

    def generate_weight_dict(self, weight_dict):
        num_repeat = weight_dict.pop('num_repeat', 1)
        blocklist = ['loss_enc_global_contra']

        if self.pred_head.aux_loss:
            aux_weight_dict = {}
            for i in range(num_repeat):
                aux_weight_dict.update({f'{k}_{i}': v for k, v in weight_dict.items()
                                        if k not in blocklist})
                aux_weight_dict.update({f'{k}_dn_{i}': v for k, v in weight_dict.items()
                                            if k not in blocklist})
            weight_dict.update(aux_weight_dict)
        print_log(f'weight_dict: {json.dumps(weight_dict, indent=4)}', 'root')
        return weight_dict        


    def extract_feature_maps(self, view: Tensor):
        if self.freeze_backbone:
            with torch.no_grad():
                if self.multi_scale and self.training:
                    sz = np.random.choice(self.multi_scale)
                    x = F.interpolate(x, size=[sz, sz])
                out = [o.detach() for o in self.backbone(view)]
                assert len(out) == len(self.multi_scale_features_backbone_num_channels)
            return out     
        else:
            raise NotImplementedError       

    def extract_patch_feature(self, crop: Tensor, query_embed: Tensor = None, to_query = True) -> Tensor:
        bs, num_patch = crop.shape[:2]
        crop = crop.flatten(0, 1)

        if self.freeze_backbone:
            with torch.no_grad():
                patches = [o.detach() for o in self.backbone(crop)]
        else:
            patches = self.backbone(crop)

        if self.multi_scale_features:
            patch = [self.avgpool(o).flatten(1) for o in patches]
            patch = torch.cat(patch, dim=1)  # TODO 暂时用concat的方式结合multi scale特征, 作为recon的目标, 而query也用相似的方法得到
        else:
            patch = patches[-1]
            patch = self.avgpool(patch).flatten(1)  # bs, num_patch x c -> num_patch, bs, c

        if to_query:
            patch = self.patch2query(patch) \
                .view(bs, num_patch, -1) \
                .repeat_interleave(self.num_queries // self.num_patches, dim=1) \
                .permute(1, 0, 2) \
                .contiguous()

            if query_embed is not None:
                patch = patch + query_embed

        return patch

    def align_and_proj(self, feature_map: NestedTensor, box: List[Tensor], query_embed: Tensor = None) -> Tensor:
        bs = len(box)
        if self.multi_scale_features:
            # for deformable, 输入到query的feature只取backbone部分的3个feature进行align和concat
            assert len(feature_map) ==  len(self.align) # and len(self.align) == self.num_feature_levels - 1  # 3
            multi_scale_patch_features = []
            for feat_, align_ in zip(feature_map, self.align):
                patch_feat_ = align_(feat_, box)
                multi_scale_patch_features.append(patch_feat_)
            patch_feature = torch.cat(multi_scale_patch_features, dim=1)  # TODO 暂时用concat的方式结合multi scale特征，作为query输入
        else:
            patch_feature = self.align(feature_map, box)

        patch_feature = self.avgpool(patch_feature).flatten(1).view(bs, self.num_patches, -1)  # bs, n_patch, c
        patch_feature = self.patch2query(patch_feature) \
            .repeat_interleave(self.num_queries // self.num_patches, dim=1) \
            .contiguous()  # n_queries, bs, c

        if query_embed is not None:
            patch_feature = patch_feature # + query_embed

        return patch_feature

    def extract_reference_points(self, ori_idx, ref, name=None):
        # last_layer_indices_v1 = ori_idx[-1]

        batch_results = []

        # match_dict = {}

        for  batch_idx, (query_indices, _) in enumerate(ori_idx):
            batch_ref = ref[batch_idx]

            selected_refs = batch_ref[query_indices]

            batch_results.append(selected_refs)
            # # nam
            # if name is not None:
            #     for i, query in enumerate(query_indices):
            #         match_dict[f'{name[batch_idx]}_patch{i}'] = int(query.item())

        result = torch.stack(batch_results)

        return result#, match_dict

    def forward_train(self, img: NestedTensor, box: List[Tensor], img_size: Tensor, crop: Tensor, name):
        # Data preprocess and box jitter.
        img_v1, img_v2 = decompose_multiview_tensor(img)
        box_v1, box_v2 = decompose_multiview_box(box)
        disturbed_box_v1, disturbed_box_v2 = None, None
        if self.box_disturbance > 0 and self.same_randomness is False:
            disturbed_box_v1, disturbed_box_v2 = add_randomness_on_multiview_box(box_v1, box_v2, img_size, ratio=self.box_disturbance)
        elif self.box_disturbance > 0 and self.same_randomness is True:
            disturbed_box_v1, disturbed_box_v2 = add_randomness_on_multiview_box(box_v1, box_v2, img_size, ratio=self.box_disturbance)
            # disturbed_box_v2 = disturbed_box_v1

        # Extract global views.
        feat_v1 = self.extract_feature_maps(img_v1)  # in deformable, feat and pos correspond to 3 features from layer 2 3 4
        feat_v2 = self.extract_feature_maps(img_v2)

        # Extract patch views through ROIAlign.
        # Following UP-DETR, we add each patch view on every ten query embeddings.
        idx = torch.randperm(self.num_queries) if self.query_shuffle \
                else torch.arange(self.num_queries)
        # query_embed = self.query_embed.weight[idx, :].unsqueeze(0).expand(len(box_v1), -1, -1)

        recon_v1_gt = recon_v2_gt = None
        if self.feature_recon:
            assert crop is not None
            patch_v1 = self.align_and_proj(feat_v1, box_v1 if disturbed_box_v1 is None else disturbed_box_v1)
            patch_v2 = self.align_and_proj(feat_v2, box_v2 if disturbed_box_v2 is None else disturbed_box_v2)

            crop_v1 = crop[:, 0, ...].contiguous()
            crop_v2 = crop[:, 1, ...].contiguous()
            with torch.no_grad():
                recon_v1_gt = self.extract_patch_feature(crop_v1, query_embed=None, to_query=False)
                recon_v2_gt = self.extract_patch_feature(crop_v2, query_embed=None, to_query=False)
            recon_v1_gt = recon_v1_gt.detach()
            recon_v2_gt = recon_v2_gt.detach()
        else:
            patch_v1 = self.align_and_proj(feat_v1, box_v1 if disturbed_box_v1 is None else disturbed_box_v1)
            patch_v2 = self.align_and_proj(feat_v2, box_v2 if disturbed_box_v2 is None else disturbed_box_v2)
        # 这里的 patch_v1是添加了query_embed的


        # Perform Cross View Cross Attention: patch_v1 -> feat_v2, patch_v2 -> feat_v1.
        # Note that for DeformableDETR-MS, we use a list of four features:
        # Three backbone features and an additional scale of feature from input proj.
        batch_size = len(box_v1)
        target_boxes_v1 = torch.stack(box_v1)
        target_boxes_v2 = torch.stack(box_v2)
        target_v1 = [{} for _ in range(batch_size)]
        target_v2 = [{} for _ in range(batch_size)]

        for idx in len(batch_size):
            target_v1[idx]['labels'] = torch.ones((target_boxes_v1.shape[1], ), dtype=torch.long, device=img_v1.device)
            target_v2[idx]['labels'] = torch.ones((target_boxes_v2.shape[1], ), dtype=torch.long, device=img_v1.device)

        h, w = img_size[:, 0].view(-1, 1), img_size[:, 1].view(-1, 1)
        scale_fct = torch.cat([w, h, w, h], dim=1).float().to(target_boxes_v1.device)
        target_boxes_v1 = target_boxes_v1 / scale_fct[:, None, :]  # norm
        target_boxes_v2 = target_boxes_v2 / scale_fct[:, None, :]  # norm

        for idx in len(batch_size):
            target_v1[idx]['boxes'] = box_ops.box_xyxy_to_cxcywh(target_boxes_v1[idx])
            target_v2[idx]['boxes'] = box_ops.box_xyxy_to_cxcywh(target_boxes_v2[idx])

        hs_v1, mem_v2, out_dict_v1, ini_ref_v1 = self.transformer(feat_v2, patch_v1, target_boxes_v2, target_v2)  # ref is (init, inter), used for DeformableDETR only
        hs_v2, mem_v1, out_dict_v2, ini_ref_v2 = self.transformer(feat_v1, patch_v2, target_boxes_v1, target_v1)
        #这里的返回可能需要修改
        # 修改transformer部分，还有criterion部分（head部分）
        # print(f'!!!!target_boxes_v2.requuires_grad:{target_boxes_v2.requires_grad}')

        # ref_v1（匹配上的）最后一层，到tgt_boxesv2的距离
        # Compute loss function: region detection + local/global discrimination.
        losses = {}

        # Region detection loss and local discrimination loss.
        # 这里的indices_v1可以不用了
        # 处理一下得到图片和patch的字典
        loss_v1, indices_v1 = self.pred_head(hs_v1, out_dict_v1, recon_v1_gt, target_v2)
        loss_v2, indices_v2 = self.pred_head(hs_v2, out_dict_v2, recon_v2_gt, target_v1)
        losses.update({k: (loss_v1[k] * 0.5 + loss_v2[k] * 0.5) for k in loss_v1.keys()})

        # Encoder loss, i.e., global discrimination loss. 
        if self.enc_head is not None:
            for head in self.enc_head:
                loss_enc = head(mem_v1, mem_v2, 
                                box_v1 if disturbed_box_v1 is None else disturbed_box_v1,
                                box_v2 if disturbed_box_v2 is None else disturbed_box_v2)
                losses.update(loss_enc)

        # Decoder loss, not used.
        if self.dec_head is not None:
            for head in self.dec_head:
                loss_dec = head(self.num_patches, hs_v1, hs_v2, indices_v1[-1], indices_v2[-1])
                losses.update(loss_dec)
            
        my_boxes_v2 = target_boxes_v2.clone().detach()
        # print(f'my_boxes_v2.requuires_grad:{my_boxes_v2.requires_grad}')
        # print(f'target_boxes_v2.requuires_grad:{target_boxes_v2.requires_grad}')
        my_boxes_v1 = target_boxes_v1.clone().detach()

        with torch.no_grad():
            if self.tr_class == 'RTDETR':
                result_v1, match_dict_v1 = self.extract_reference_points(indices_v1[0], ini_ref_v1, name)
                result_v2, match_dict_v2 = self.extract_reference_points(indices_v2[0], ini_ref_v2, name)


            final_v1 = {'ref':result_v1.cpu().numpy().tolist(), 'tgt':my_boxes_v2.cpu().numpy().tolist()}
            final_v2 = {'ref':result_v2.cpu().numpy().tolist(), 'tgt':my_boxes_v1.cpu().numpy().tolist()}



        losses.update({'weight_dict': self.weight_dict})
        return losses, None, None, final_v1, match_dict_v1, final_v2, match_dict_v2

    def forward(self, img: NestedTensor, box: List[Tensor], img_size: Tensor, crop: Tensor = None, name: List[str] = None, mode='train', **kwargs):
        img = img.cuda()
        box = [b.cuda() for b in box]
        crop = crop.cuda() if crop is not None else None

        if mode == 'train':
            return self.forward_train(img, box, img_size, crop, name)
        elif mode == 'test':
            raise NotImplementedError
        elif mode == 'extract':
            raise NotImplementedError
        else:
            raise Exception("No such mode: {}".format(mode))

    def train_step(self, data, *args, **kwargs):
        losses, contrast_idx_v1, contrast_idx_v2, final_v1, match_dict_v1, final_v2, match_dict_v2 = self.forward(**data, mode='train')
        loss, log_vars = parse_losses(losses)
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data['img'].data))
        return outputs, contrast_idx_v1, contrast_idx_v2, final_v1, match_dict_v1, final_v2, match_dict_v2
