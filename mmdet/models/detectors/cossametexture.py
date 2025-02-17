# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import Dict, List, Optional, Tuple, Union
from torchvision.ops.poolers import _setup_scales, _multiscale_roi_align, LevelMapper
import torch
from suploss import SupConLoss
from mmengine.config import Config
from torch import nn, Tensor
from torch.nn import functional as F
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector
from mmdet.structures.mask import mask_target

class MultiScaleRoIAlign(nn.Module):

    __annotations__ = {"scales": Optional[List[float]], "map_levels": Optional[LevelMapper]}

    def __init__(
        self,
        output_size = (28, 28),
        sampling_ratio: int = 2,
        *,
        canonical_scale: int = 224,
        canonical_level: int = 4,
    ):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        self.scales = None
        self.map_levels = None
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level

    def forward(self, x: List[Tensor], boxes: List[Tensor], image_shapes: List[Tuple[int, int]],) -> Tensor:
        self.scales, self.map_levels = _setup_scales(x, image_shapes, self.canonical_scale, self.canonical_level)
        return _multiscale_roi_align(x, boxes, self.output_size, self.sampling_ratio, self.scales, self.map_levels,)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(featmap_names={self.featmap_names}, "
            f"output_size={self.output_size}, sampling_ratio={self.sampling_ratio})"
        )

class BasicConv2d(nn.Module):
    def __init__(self,in_c,out_c,ks,stride=1,dilation=1,norm=True):
        super(BasicConv2d, self).__init__()
        if norm:
            self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=ks, padding = ks//2, stride=stride, dilation=1, bias=False),
                                   nn.BatchNorm2d(out_c),
                                   nn.ReLU(True))
        else:
            self.conv = nn.Conv2d(in_c, out_c, kernel_size=ks, padding = ks//2, stride=stride,bias=False)
    def forward(self,x):
        return self.conv(x)

class Texture(nn.Module):
    def __init__(self, in_channel, mid_channel):
        super(Texture, self).__init__()
        out_channel = in_channel
        self.conv11 = BasicConv2d(in_channel, mid_channel, 1)
        self.conv12 = BasicConv2d((mid_channel - 1) * (mid_channel)//2, out_channel, 1)
        self.fc = nn.Conv2d(out_channel*2, out_channel, 1, 1, 0)

    def gram_matrix(self, features):
        N, C, H, W = features.size()
        feat_reshaped = features.permute(0,2,3,1).contiguous().reshape(N*H*W,C,1)
        gram = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))
        gram = gram[torch.triu(torch.ones_like(gram,device=gram.device), diagonal=1)!=0].reshape(N,H,W,-1).permute(0,3,1,2).contiguous()
        return gram

    def forward(self, x0, mask=None):
        B,C,H,W = x0.shape
        x = self.conv11(x0)
        x = self.gram_matrix(F.normalize(x, 1))
        # x0 = self.gram_matrix(torch.cat((x0-x0.mean(1,keepdim=True),x0.mean(1,keepdim=True),(x0 * self.attn(x)).sum(1,keepdim=True)),1))
        x = self.conv12(x)
        return self.fc(torch.cat((x, x0), 1))

@MODELS.register_module()
class CosSameTextureDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 pretrained = None) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.bce = torch.nn.BCELoss()
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.msroialign = MultiScaleRoIAlign()
        self.backbone = MODELS.build(backbone)
        self.same_cfg = Config({'mask_size': (32, 32)})
        # self.texture = nn.ModuleList([Texture(96, 16), Texture(192, 24), Texture(384, 32), Texture(768, 32)])

        if neck is not None:
            self.neck = MODELS.build(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        'The `num_classes` should be 1 in RPN, but get '
                        f'{rpn_head_num_classes}, please set '
                        'rpn_head.num_classes = 1 in your config file.')
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading single-stage
        weights into two-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) != 0 and len(rpn_head_keys) == 0:
            for bbox_head_key in bbox_head_keys:
                rpn_head_key = rpn_head_prefix + \
                               bbox_head_key[len(bbox_head_prefix):]
                state_dict[rpn_head_key] = state_dict.pop(bbox_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        # print(batch_inputs.shape)
        # exit(0)
        batch_inputs = torch.cat((batch_inputs[:,:3], batch_inputs[:,3:]),0)
        x = self.backbone(batch_inputs)
        # x = [self.texture[i](x[i]) for i in range(len(x))]
        if self.with_neck:
            x = self.neck(x)
        # x = [self.texture[i](x[i]) for i in range(len(x))]
        return x

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        x = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        roi_outs = self.roi_head.forward(x, rpn_results_list, batch_data_samples)
        results = results + (roi_outs, )
        return results

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        # for i in range(len(batch_data_samples)):
        #     batch_data_samples[i].gt_instances.labels = batch_data_samples[i].gt_instances.labels0.clone()
        batch_data_samples.append(copy.deepcopy(batch_data_samples[0]))
        # batch_data_samples[0].gt_instances.labels = batch_data_samples[0].gt_instances.labels0.clone().long()
        # batch_data_samples[1].gt_iFloatnces.labels = batch_data_samples[1].gt_instances.labels1.clone().long()
        batch_data_samples[0].gt_instances.ttypes = batch_data_samples[0].gt_instances.tamper_type0.clone().long()
        batch_data_samples[1].gt_instances.ttypes = batch_data_samples[1].gt_instances.tamper_type1.clone().long()
        batch_data_samples[0].gt_instances.tamper = batch_data_samples[0].gt_instances.ttypes.clamp(-1,1) # (batch_data_samples[0].gt_instances.ttypes>=0).long()
        batch_data_samples[1].gt_instances.tamper = batch_data_samples[1].gt_instances.ttypes.clamp(-1,1) # (batch_data_samples[1].gt_instances.ttypes>=0).long()
        labels_zero = torch.zeros_like(batch_data_samples[0].gt_instances.labels).to(batch_data_samples[0].gt_instances.labels).long()
        batch_data_samples[0].gt_instances.labels = labels_zero
        batch_data_samples[1].gt_instances.labels = labels_zero
        device = batch_inputs.device

        img_shape = [batch_data_samples[0].img_shape]
        img_boxes = batch_data_samples[0].gt_instances.bboxes
        img_sames = torch.where(batch_data_samples[0].gt_instances.tamper_same.long()<=1, batch_data_samples[0].gt_instances.tamper_same.long(), 0)
        same_boxes = [b for bi,b in enumerate(img_boxes) if img_sames[bi]!=-1]
        if len(same_boxes)!=0:
            same_label = torch.cuda.FloatTensor([b for b in img_sames if b!=-1], device=device)
            same_boxes = [torch.stack(same_boxes).to(device)]
            same_index = [torch.cuda.LongTensor([bi for bi in range(len(img_boxes)) if img_sames[bi]!=-1], device=device)] #, device=same_boxes.device)
            flag = True
        else:
            flag = False
        # print('texture', batch_data_samples[0].gt_instances.ttypes, batch_data_samples[1].gt_instances.ttypes)
        x = self.extract_feat(batch_inputs)

        if flag:
          try:
            mask_rois = mask_target(same_boxes, same_index, [batch_data_samples[0].gt_instances.masks], self.same_cfg).unsqueeze(1)
            feat_roi0 = self.msroialign([xx[0:1] for xx in x], same_boxes, img_shape)
            feat_roi1 = self.msroialign([xx[1:2] for xx in x], same_boxes, img_shape)
            auth_roi0 = feat_roi0[:, :128] # self.roi_head.convert1(feat_roi0[:, :128])
            auth_roi1 = feat_roi1[:, :128] # self.roi_head.convert1(feat_roi1[:, :128])
            tamp_roi0 = feat_roi0[:, 128:] # self.roi_head.convert2(feat_roi0[:, 128:])
            tamp_roi1 = feat_roi1[:, 128:] # self.roi_head.convert2(feat_roi1[:, 128:])
            mask_rois_flat = mask_rois.flatten(1)
            # print('shape0', mask_rois.shape, auth_roi0.shape, tamp_roi1.shape)
            mask_rois_max = mask_rois_flat.max(1).values
            mask_rois_sum = mask_rois_flat.sum(1, keepdim=True)
            auth_roi0_feat = F.normalize((auth_roi0 * mask_rois).sum((2,3))/mask_rois_sum,1)
            auth_roi1_feat = F.normalize((auth_roi1 * mask_rois).sum((2,3))/mask_rois_sum,1)
            tamp_roi0_feat = F.normalize((tamp_roi0 * mask_rois).sum((2,3))/mask_rois_sum,1)
            tamp_roi1_feat = F.normalize((tamp_roi1 * mask_rois).sum((2,3))/mask_rois_sum,1)
            same_label0 = torch.ones_like(same_label)
            auth_cos = torch.abs(self.cos(auth_roi0_feat, auth_roi1_feat))
            if (auth_cos<=1).all():
                auth_bce = self.bce(auth_cos, same_label0)
                tamp_cos = torch.abs(self.cos(tamp_roi0_feat, tamp_roi1_feat))
                if (tamp_cos<=1).all():
                    tamp_bce = self.bce(tamp_cos, 1-same_label)
                else:
                    flag = False
            else:
                flag = False
          except:
            flag = False
        # cos_similarity? constrative learning? GRL?
        # print(ms_roi0.shape, ms_roi1.shape)
        if flag:
            losses = dict(auth_bce=auth_bce, tamp_bce=tamp_bce)
        else:
            losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)
            # print('rpn', len(x[0]), x[0][0].shape)#[0].shape, len(x))
            # print('xs',[xx.shape for xx in x])
            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        # print(batch_data_samples)
        # exit(0)
        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        # print('texture.py roi_loss', roi_losses)
        losses.update(roi_losses)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
