
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from typing import List, Tuple
# from data_sources import ImageNetWithUnsupbox
from .registry import DATASETS, PIPELINES
from .builder import build_datasource
from openselfsup.utils import build_from_cfg
import numpy as np
import cv2



def get_random_patch_from_img(img, min_pixel=8):
    """
    :param img: original image
    :param min_pixel: min pixels of the query patch
    :return: query_patch,x,y,w,h
    """
    w, h = img.size
    min_w, max_w = min_pixel, w - min_pixel
    min_h, max_h = min_pixel, h - min_pixel
    sw, sh = np.random.randint(min_w, max_w + 1), np.random.randint(min_h, max_h + 1)
    x, y = np.random.randint(w - sw) if sw != w else 0, np.random.randint(h - sh) if sh != h else 0
    # patch = img.crop((x, y, x + sw, y + sh))
    return x, y, x + sw, y+ sh


@DATASETS.register_module
class distDETRDataset(Dataset):

    def __init__(self, 
                 data_source,
                 base_pipeline,
                 view_pipeline,
                 ori_pipeline,
                 anchor_num=10,
                 return_crop=False,
                 crop_pipeline=None,
                 visualization=False):
        """
        Args:
            data_source:
            base_pipeline:
            view_pipeline:
            anchor_num:
            return_crop: 是否需要返回box所要crop的图片
            crop_pipeline: crop的图片需要经过的PA
        """
        self.data_source = build_datasource(data_source)
        # 得到了所有图片的路径和label
        self.anchor_num = anchor_num
        self.return_crop = return_crop
        self.visualization = visualization
        ComposeWithTarget = PIPELINES.get('ComposeWithTarget')
        self.base_pipeline = ComposeWithTarget([build_from_cfg(p, PIPELINES) for p in base_pipeline])
        self.view_pipeline = Compose([build_from_cfg(p, PIPELINES) for p in view_pipeline])
        self.ori_pipeline = Compose([build_from_cfg(p, PIPELINES) for p in ori_pipeline])
        if self.return_crop:
            assert isinstance(crop_pipeline, (List, Tuple))
            self.crop_pipeline = Compose([build_from_cfg(p, PIPELINES) for p in crop_pipeline])

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, index):
        try:
            sample, target = self.data_source.get_sample(index)
            w, h = sample.size
            target['orig_size'] = torch.as_tensor([int(h), int(w)])
            target['size'] = torch.as_tensor([int(h), int(w)])
        except:
            import random 
            print('raise exception in dataset, idx:', index)
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        # crop_oris = []
        crop_v1s = []
        crop_v2s = []
        crop_v3s = []
        while target['unsup_boxes'].shape[0] < self.anchor_num:
            if w<=16 or h<=16:
                return self[(index+1)%len(self)]
            x0, y0, x1, y1 = get_random_patch_from_img(sample)
            box = torch.tensor([x0, y0, x1, y1])
            target['unsup_boxes'] = torch.cat((target['unsup_boxes'], box.unsqueeze(0)), dim=0)

        for box in target['unsup_boxes'][:self.anchor_num]:
            crop_ori = sample.crop(box.tolist())
            # crop_v1, crop_v2 = self.view_pipeline(crop_ori), self.view_pipeline(crop_ori)
            # crop_v1, crop_v2 = self.crop_pipeline(crop_v1), self.crop_pipeline(crop_v2)
            # crop_ori = self.ori_pipeline(crop_ori)
            # crop_oris.append(crop_ori)
            # crop_v1s.append(crop_v1)
            # crop_v2s.append(crop_v2)

            crop_v1, crop_v2, crop_v3 = self.view_pipeline(crop_ori), self.view_pipeline(crop_ori), self.view_pipeline(crop_ori)
            crop_v1, crop_v2, crop_v3  = self.crop_pipeline(crop_v1), self.crop_pipeline(crop_v2), self.crop_pipeline(crop_v3)
            crop_v1s.append(crop_v1)
            crop_v2s.append(crop_v2)
            crop_v3s.append(crop_v3)

        # crops_ori = torch.stack(crop_oris, dim=0) # M x 3 x 128 x 128
        # crops_v1 = torch.stack(crop_v1s, dim=0) # M x 3 x 128 x 128
        # crops_v2 = torch.stack(crop_v2s, dim=0)  # M x 3 x 128 x 128

        crops_v1 = torch.stack(crop_v1s, dim=0) # M x 3 x 128 x 128
        crops_v2 = torch.stack(crop_v2s, dim=0) # M x 3 x 128 x 128
        crops_v3 = torch.stack(crop_v3s, dim=0)  # M x 3 x 128 x 128
        if target['unsup_boxes'].shape[0] < self.anchor_num:
            # crops_zeros = torch.zeros(self.anchor_num-len(target['unsup_boxes']), 3, 128, 128)
            # crops_ori = torch.cat((crops_ori, crops_zeros), dim=0)
            # crops_v1 = torch.cat((crops_v1, crops_zeros), dim=0)
            # crops_v2 = torch.cat((crops_v2, crops_zeros), dim=0)
            crops_zeros = torch.zeros(self.anchor_num-len(target['unsup_boxes']), 3, 128, 128)
            crops_v1 = torch.cat((crops_v1, crops_zeros), dim=0)
            crops_v2 = torch.cat((crops_v2, crops_zeros), dim=0)
            crops_v3 = torch.cat((crops_v3, crops_zeros), dim=0)
        target['boxes'] = target['unsup_boxes'][:self.anchor_num].repeat(3,1)
        del target['unsup_boxes']
        # target['area'] =( target['boxes'][..., 2] - target['boxes'][..., 0] )*( target['boxes'][..., 3] - target['boxes'][..., 1])
        # target['labels'] = torch.ones(self.anchor_num * 3).long()
        patches = []
        if self.return_crop:
            # patches.append(crops_ori)  # M x 3 x 128 x 128
            # patches.append(crops_v1)  # M x 3 x 128 x 128
            # patches.append(crops_v2)  # M x 3 x 128 x 128
            patches.append(crops_v1)  # M x 3 x 128 x 128
            patches.append(crops_v2)  # M x 3 x 128 x 128
            patches.append(crops_v3)  # M x 3 x 128 x 128
            patches = torch.stack(patches, dim=0)  # 3 x M x 3 x 128 x 128
        sample, target = self.base_pipeline(sample, target)
        result = dict(img=sample, box=target['boxes'], img_size=target['size'])
        if self.return_crop:
            result['crop'] = patches
        if self.visualization:
            result['ori_img_size'] = target['ori_size']
        return result