
import torch
from torchvision.ops.boxes import box_area

def batch_box_area(boxes):
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])

def batch_box_iou(boxes1, boxes2):
    area1 = batch_box_area(boxes1)
    area2 = batch_box_area(boxes2)

    lt = torch.max(boxes1[:, :, None, :2], boxes2[:, None, :, :2])
    rb = torch.min(boxes1[:, :, None, 2:], boxes2[:, None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, :, None] + area2[:, None, :] - inter

    iou = inter / union
    return iou, union

def batch_generalized_box_iou(boxes1, boxes2):
    """
    Compute batched generalized IoU between two sets of boxes.
    The boxes are expected to be in [x0, y0, x1, y1] format.

    Args:
        boxes1 (torch.Tensor): shape (batch_size, N, 4)
        boxes2 (torch.Tensor): shape (batch_size, M, 4)

    Returns:
        giou (torch.Tensor): shape (batch_size, N, M)
    """
    # Ensure the boxes are valid
    assert (boxes1[..., 2:] >= boxes1[..., :2]).all(), f"boxes1 is not valid: {boxes1}"
    assert (boxes2[..., 2:] >= boxes2[..., :2]).all(), f"boxes2 is not valid: {boxes2}"

    iou, union = batch_box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, :, None, :2], boxes2[:, None, :, :2])
    rb = torch.max(boxes1[:, :, None, 2:], boxes2[:, None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]

    return iou - (area - union) / area

def box_cxcywh_to_xyxy(x):
    if isinstance(x, list):
        x = torch.cat(x)
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), f"boxes1: {boxes1}"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), f"boxes2: {boxes2}"
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area
