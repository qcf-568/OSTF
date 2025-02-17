"""
Dilated Neighborhood Attention Transformer.
https://arxiv.org/abs/2209.15001

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from mmengine.runner import load_checkpoint
from mmdet.utils import get_root_logger
from .nat import NAT
from mmdet.registry import MODELS

@MODELS.register_module()
class DiNAT(NAT):
    """
    DiNAT is NAT with dilations.
    It's that simple!
    """

    pass
