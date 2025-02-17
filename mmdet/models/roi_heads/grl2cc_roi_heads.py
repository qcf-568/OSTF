# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple
import torch
import random
from torch import nn
from torch import Tensor
from mmcv.cnn import ConvModule
from torch.nn import functional as F
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances, unpack_gt_instances
from .base_roi_head import BaseRoIHead

class SA(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x1, x2):
        return (x1 * self.sSE(x2))

from torch.autograd  import  Function

class GradReverse(torch.autograd.Function):
    def __init__(self):
        super(GradReverse, self).__init__()
    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)
    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None

class GradReverseLayer(torch.nn.Module):
    def __init__(self, lambd=0.01):
        super(GradReverseLayer,self).__init__()
        self.lambd = lambd
    def forward(self, x):
        lam = torch.tensor(self.lambd)
        return GradReverse.apply(x,lam)

class Texture(nn.Module):
    def __init__(self, in_channel, mid_channel):
        super(Texture, self).__init__()
        out_channel = in_channel
        self.conv11 = ConvModule(in_channel, mid_channel, 1) # BasicConv2d(in_channel, mid_channel, 1)
        self.conv12 = ConvModule((mid_channel - 1) * (mid_channel)//2, out_channel, 1) # BasicConv2d((mid_channel - 1) * (mid_channel)//2, out_channel, 1)

    def gram_matrix(self, features):
        N, C, H, W = features.size()
        feat_reshaped = features.permute(0,2,3,1).contiguous().reshape(N*H*W,C,1)
        gram = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))
        gram = gram[torch.triu(torch.ones_like(gram,device=gram.device), diagonal=1)!=0].reshape(N,H,W,-1).permute(0,3,1,2).contiguous()
        return gram

    def forward(self, x0):
        B,C,H,W = x0.shape
        x = self.conv11(x0)
        x = self.gram_matrix(F.normalize(x, 1))
        # x0 = self.gram_matrix(torch.cat((x0-x0.mean(1,keepdim=True),x0.mean(1,keepdim=True),(x0 * self.attn(x)).sum(1,keepdim=True)),1))
        x = self.conv12(x)
        return torch.cat((x, x0), 1)

@MODELS.register_module()
class GRL2CCRoIHead(BaseRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""
    def __init__(self,
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head1: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if shared_head is not None:
            self.shared_head = MODELS.build(shared_head)

        self.init_bbox_head(bbox_roi_extractor, bbox_head1)

        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)
        self.celoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.init_assigner_sampler()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def init_assigner_sampler(self) -> None:
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
            self.bbox_sampler = TASK_UTILS.build(
                self.train_cfg.sampler, default_args=dict(context=self))

    def init_bbox_head(self, bbox_roi_extractor: ConfigType,
        bbox_head1: ConfigType) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.bbox_head1 = MODELS.build(bbox_head1)
        self.convert1 = ConvModule(128, 256, 3, padding=1, conv_cfg={'type':'Conv2d'}, norm_cfg={'type': 'SyncBN'})
        self.txt = Texture(128, 24)

    def init_mask_head(self, mask_roi_extractor: ConfigType,
                       mask_head: ConfigType) -> None:
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict or ConfigDict): Config of mask roi
                extractor.
            mask_head (dict or ConfigDict): Config of mask in mask head.
        """
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = MODELS.build(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.grl = GradReverseLayer()
        self.mask_head = MODELS.build(mask_head)
        self.mask_head2 = MODELS.build(mask_head)
        mask_head['num_classes']=3
        self.adv_head = MODELS.build(mask_head)

    # TODO: Need to refactor later
    def forward(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList = None) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            x (List[Tensor]): Multi-level features that may have different
                resolutions.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
            the meta information of each image and corresponding
            annotations.

        Returns
            tuple: A tuple of features from ``bbox_head`` and ``mask_head``
            forward.
        """
        results = ()
        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        rois = bbox2roi(proposals)
        # bbox head
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            results = results + (bbox_results['cls_score'],
                                 bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            results = results + (mask_results['mask_preds'], )
        return results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            # print('assign_result', assign_result)
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)
            # print('custom_roi_heads.py', 'sampling results', sampling_results)
        losses = dict()
        # bbox head loss
        bbox_results, tamp_feats, this_inds = self.bbox_loss(x, sampling_results)
        losses.update(dict(loss_cls=bbox_results['loss_cls'], loss_bbox=bbox_results['loss_bbox']))
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        gt_label = torch.cat([res.tamper[pos_assigned_gt_inds[ri]] for ri,res in enumerate(batch_gt_instances)])
        adv_label = torch.cat([res.ttypes[pos_assigned_gt_inds[ri]] for ri,res in enumerate(batch_gt_instances)])
        # mask head forward and loss
        if self.with_mask and (len(tamp_feats)>0):
            mask_results, m2, advs = self.mask_loss(x, sampling_results, tamp_feats, batch_gt_instances, gt_label, adv_label, this_inds)
            losses.update(dict(loss_tamper=mask_results, loss_tamper2=m2, loss_adv=advs))

        return losses

    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
        this_inds = rois[:, 0] 
        if self.with_shared_head:
            assert False
            bbox_feats = self.shared_head(bbox_feats)
        reg_feats = self.convert1(bbox_feats[:,:128])
        tamp_feats = self.txt(bbox_feats[:,128:])
        cls_score, bbox_pred = self.bbox_head1(reg_feats)
        # cls_score = self.bbox_head2(cls_feats)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred, reg_feats=reg_feats)
        return bbox_results, tamp_feats, this_inds

    def bbox_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult]) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results, tamp_feats, this_inds = self._bbox_forward(x, rois)
        bbox_loss_and_target = self.bbox_head1.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            # domain_pd=bbox_results['domain_pd'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)
        # print(bbox_loss_and_target.keys(), bbox_results['cls_score'])
        bbox_results.update(dict(loss_cls=bbox_loss_and_target['loss_cls'], loss_bbox=bbox_loss_and_target['loss_bbox']))
        return bbox_results, tamp_feats, this_inds

    def mask_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult], bbox_feats: Tensor,
                  batch_gt_instances: InstanceList, gt_label, adv_label, this_inds) -> dict:
        """Perform forward propagation and loss calculation of the mask head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.
            bbox_feats (Tensor): Extract bbox RoI features.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
                - `mask_targets` (Tensor): Mask target of each positive\
                    proposals in the image.
                - `loss_mask` (dict): A dictionary of mask loss components.
        """
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
            mask_results, m2, advs = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            mask_results, m2, advs = self._mask_forward(x, pos_inds=pos_inds, bbox_feats=bbox_feats, this_inds=this_inds)
        # gt_label = torch.cat([res.labels[pos_assigned_gt_inds[ri]] for ri,res in enumerate(batch_gt_instances)])
        # print('fustom', pred_tamp, gt_tamp, gt_label)
        # print(mask_results.shape, gt_label.shape)
        # exit(0)
        # print('adv', adv_label)
        loss_adv = self.celoss(advs.float(), adv_label.long())
        loss_tamp = self.celoss(mask_results.float(), gt_label.long())
        loss_tamp2 = self.celoss(m2.float(), gt_label.long())
        if ((mask_results.shape[0]!=0) and (random.uniform(0,1)>0.98)):
            print('loss_tamper', loss_tamp, mask_results.max(1).indices, gt_label, loss_adv, adv_label)
        # print('lossf',loss_tamp, mask_results.shape)
        # print('mask_results.shape',mask_results.shape)
        # exit(0)
        '''
        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg)
        '''
        # mask_results.update(loss_tamper=loss_tamp)
        return loss_tamp, loss_tamp2, loss_adv # mask_results

    def _mask_forward(self,
                      x: Tuple[Tensor],
                      rois: Tensor = None,
                      pos_inds: Optional[Tensor] = None,
                      bbox_feats: Optional[Tensor] = None,
                      this_inds = None) -> dict:
        """Mask head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            pos_inds (Tensor, optional): Indices of positive samples.
                Defaults to None.
            bbox_feats (Tensor): Extract bbox RoI features. Defaults to None.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
        """
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]
            use_inds = this_inds[pos_inds]
            if use_inds.shape[0]!=0:
                gloabl_feats = self.txt(self.avgpool(x[3])[use_inds.long()][:,128:])
                cls_score, _ = self.mask_head(mask_feats-gloabl_feats)
                cls_score2, _ = self.mask_head2(mask_feats)
                adv_score, _ = self.adv_head(self.grl(mask_feats))
            else:
                cls_score, _ = self.mask_head(mask_feats)
                cls_score2, _ = self.mask_head2(mask_feats)
                adv_score, _ = self.adv_head(self.grl(mask_feats))
        return cls_score, cls_score2, adv_score

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        bbox_rescale = rescale # if not self.with_mask else False
        results_list, tamp_feats, this_inds = self.predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=bbox_rescale)
        tamp_score = self.predict_mask(x, batch_img_metas, results_list, tamp_feats, rescale=rescale, this_inds=this_inds)
        return self.zl(results_list, tamp_score)

    def zl(self, results_list, tamp_score):
        rsts = []
        for ri,r in enumerate(results_list):
            bbox_score = r['scores']
            bbox_use = (bbox_score>0.5)
            bboxes =  r['bboxes'][bbox_use]
            tampsc = tamp_score[ri][bbox_use]
            rsts.append((bboxes, tampsc))
        return rsts

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head.predict_box_type,
                num_classes=self.bbox_head.num_classes,
                score_per_cls=rcnn_test_cfg is None)

        bbox_results, tamp_feats, this_inds = self._bbox_forward(x, rois)

        # split batch bbox prediction back to each image
        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_preds will be None
        if bbox_preds is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_preds, torch.Tensor):
                bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
            else:
                bbox_preds = self.bbox_head.bbox_pred_split(
                    bbox_preds, num_proposals_per_img)
        else:
            bbox_preds = (None, ) * len(proposals)

        result_list = self.bbox_head1.predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)
        return result_list, tamp_feats, this_inds

    def predict_mask(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     results_list: InstanceList,
                     tamp_feats,
                     rescale: bool = False,
                     this_inds = None) -> InstanceList:
        """Perform forward propagation of the mask head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        # don't need to consider aug_test.
        # bboxes = [res.bboxes for res in results_list]
        # mask_rois = bbox2roi(bboxes)
        inds = [res.inds for res in results_list]
        if len(inds)==0: # mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        mask_preds = self._mask_forward(x, pos_inds=inds, bbox_feats=tamp_feats, this_inds=this_inds)
        # mask_preds = mask_results['mask_preds']
        # split batch mask prediction back to each image
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)
        return [F.softmax(m,1) for m in mask_preds]
        # TODO: Handle the case where rescale is false
        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)
        return results_list
