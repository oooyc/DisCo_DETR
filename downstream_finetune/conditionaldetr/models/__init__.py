
from .conditional_detr import build


def build_model(args):
    return build(args)
