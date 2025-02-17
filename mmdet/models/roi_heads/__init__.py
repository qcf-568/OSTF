# Copyright (c) OpenMMLab. All rights reserved.
from .base_roi_head import BaseRoIHead
from .bbox_heads import (BBoxHead, ConvFCBBoxHead, Shared2FCBBoxHead, Shared4Conv1FCBBoxHead)
from .cascade_roi_head import CascadeRoIHead
# from .double_roi_head import DoubleHeadRoIHead
# from .dynamic_roi_head import DynamicRoIHead
# from .grid_roi_head import GridRoIHead
# from .htc_roi_head import HybridTaskCascadeRoIHead
# from .mask_heads import (CoarseMaskHead, FCNMaskHead, FeatureRelayHead,
#                          FusedSemanticHead, GlobalContextHead, GridHead,
#                          HTCMaskHead, MaskIoUHead, MaskPointHead,
#                          SCNetMaskHead, SCNetSemanticHead)
# from .mask_scoring_roi_head import MaskScoringRoIHead
# from .multi_instance_roi_head import MultiInstanceRoIHead
# from .pisa_roi_head import PISARoIHead
# from .point_rend_roi_head import PointRendRoIHead
from .roi_extractors import (BaseRoIExtractor, GenericRoIExtractor, SingleRoIExtractor)
# from .scnet_roi_head import SCNetRoIHead
# from .shared_heads import ResLayer
# from .sparse_roi_head import SparseRoIHead
from .standard_roi_head import StandardRoIHead
from .custom_roi_heads import CustomRoIHead
# from .c3ustom_roi_heads import C3ustomRoIHead
# from .c2ustom_roi_heads import C2ustomRoIHead
# from .ctustom_roi_heads import CTustomRoIHead
# from .ccustom_roi_heads import CCustomRoIHead
# from .cc2ustom_roi_heads import CC2ustomRoIHead
# from .mustom_roi_heads import MustomRoIHead
# from .trident_roi_head import TridentRoIHead
# from .dfpn_c2ustom_roi_heads import DFPNC2ustomRoIHead
# from .cin_roi_heads import CTinRoIHead 
# from .ctms_roi_heads import CTMSRoIHead
# from .ccnew_roi_heads import CCNewustomRoIHead
# from .grl2cc_roi_heads import GRL2CCRoIHead
from .dfpn_cmap3 import DFPNCMap3
from .dfpn_ori import DFPNOri
from .dfpn_sgl import DFPNSGL
from .cascade_cmap3 import CascadeCMap3

__all__ = [
    'BaseRoIHead', 'CascadeRoIHead', 'BBoxHead', 'ConvFCBBoxHead', 'Shared2FCBBoxHead',
    'StandardRoIHead', 'Shared4Conv1FCBBoxHead', 'BaseRoIExtractor', 'GenericRoIExtractor',
    'SingleRoIExtractor', 'CustomRoIHead', 'DFPNCMap3', 'DFPNOri', 'DFPNSGL', 'CascadeCMap3'
]
