# _base_ = ['/DeepLearning/guitao_xu/temp/mmdetection-main/work_configs/tta.py']
# model settings
model = dict(
    type='FTDFPNTextureMaskRCNN',
    # pretrained='coco_pths/mask_rcnn_swin_small_patch4_window7.pth',#'xsza_swin_54.pth',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        use_abs_pos_embed=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 0.25],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox = dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        ),
    roi_head=dict(
        type='DFPNCMap3',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head1=dict(
                type='Shared2FCBBoxHead',
                with_cls=True,
                with_reg=True,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox = dict(type='SmoothL1Loss', beta=1.0,loss_weight=1.0),
                ),
         mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=8, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='MapShared3Conv1FCBBoxHead',
            with_cls=True,
            with_reg=False,
            in_channels=256,
            fc_out_channels=256,
            roi_feat_size=1,
            num_classes=1,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
        ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.1,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))


# dataset settings
# dataset_type = 'PTDataset'
# data_root = ''
dataset_type = 'OurDataset'
dataset_type2 = 'PTDataset'
data_root = 'mmacc/'
data_root2 = './'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    # dict(type='RangeResize', scale=[(1280, 768), (1280, 1280)], keep_ratio=True),
    # dict(type='Texture', revjpegpath='/media/dplearning1/chenfan/FBCNN-main/'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
        [{
            'type':
            'RandomChoiceResize',
            'scales': [(768, 1536), (800, 1536), (832, 1536), (864, 1536), (896, 1536), (928, 1536), (960, 1536), (992, 1536), (1024, 1536)],
            'keep_ratio':
            True
        }],
        [{
                      'type': 'RandomChoiceResize',
                      'scales': [(512, 1536), (640, 1536), (768, 1536)],
                      'keep_ratio': True
                  },], 
        [{
                        'type': 'RandomCrop',
                        'crop_type': 'absolute_range',
                        'crop_size': (512, 768),
                        'allow_negative_crop': True
                    }, {
                        'type':
                        'RandomChoiceResize',
                        'scales': [(768, 1536), (800, 1536), (832, 1536), (864, 1536), (896, 1536), (928, 1536), (960, 1536), (992, 1536), (1024, 1536)],
                        'keep_ratio':
                        True
        }]]),
    dict(type='CS'),
    dict(type='PackDetInputs', mode='test')
]


txt_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='TextureSG', revjpegpath='./revjpegs/', mins=16, bathres=512),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
        [{
            'type':
            'RandomChoiceResize',
            'scales': [(768, 1536), (800, 1536), (832, 1536), (864, 1536), (896, 1536), (928, 1536), (960, 1536), (992, 1536), (1024, 1536)],
            'keep_ratio':
            True
        }],
        [{
                      'type': 'RandomChoiceResize',
                      'scales': [(512, 1536), (640, 1536), (768, 1536)],
                      'keep_ratio': True
                  },],
        [{
                        'type': 'RandomCrop',
                        'crop_type': 'absolute_range',
                        'crop_size': (512, 768),
                        'allow_negative_crop': True
                    }, {
                        'type':
                        'RandomChoiceResize',
                        'scales': [(768, 1536), (800, 1536), (832, 1536), (864, 1536), (896, 1536), (928, 1536), (960, 1536), (992, 1536), (1024, 1536)],
                        'keep_ratio':
                        True
        }]]),
    dict(type='PackDetInputs')
]



test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1024,1536), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
        mode='test')
]

tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))

img_scales = [(768, 1536), (1024, 1536)]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=s, keep_ratio=True)
                for s in img_scales
            ],
            [
                # ``RandomFlip`` must be placed before ``Pad``, otherwise
                # bounding box coordinates after flipping cannot be
                # recovered correctly.
                dict(type='RandomFlip', prob=1.),
                dict(type='RandomFlip', prob=0.)
            ],
            [dict(type='LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'flip', 'flip_direction'))
            ]
        ])
]



art = dict(
        type=dataset_type2,
        data_root=data_root2,
        ann_file='pretrain/ArT/train.pk',
        data_prefix=dict(img=''),
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=txt_pipeline,
        backend_args=backend_args)

ic13 = dict(
        type='RepeatDataset',
        times=100,
        dataset=dict(type=dataset_type2,
        data_root=data_root2,
        ann_file='pretrain/ICDAR2013/train.pk',
        data_prefix=dict(img=''),
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=txt_pipeline,
        backend_args=backend_args)
)

ic15 = dict(
        type=dataset_type2,
        data_root=data_root2,
        ann_file='pretrain/ICDAR2015/train.pk',
        data_prefix=dict(img=''),
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=txt_pipeline,
        backend_args=backend_args)

ic17 = dict(
        type=dataset_type2,
        data_root=data_root2,
        ann_file='pretrain/ICDAR2017-MLT/train.pk',
        data_prefix=dict(img=''),
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=txt_pipeline,
        backend_args=backend_args)

rects = dict(
        type=dataset_type2,
        data_root=data_root2,
        ann_file='pretrain/ReCTS/train.pk',
        data_prefix=dict(img=''),
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=txt_pipeline,
        backend_args=backend_args)

lsvt = dict(
        type=dataset_type2,
        data_root=data_root2,
        ann_file='pretrain/LSVT/train.pk',
        data_prefix=dict(img=''),
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=txt_pipeline,
        backend_args=backend_args)

textocrpt = dict(
        type=dataset_type2,
        data_root=data_root2,
        ann_file='pretrain/TextOCR/train.pk',
        data_prefix=dict(img=''),
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=txt_pipeline,
        backend_args=backend_args)

ptdatas = dict(
        type='ConcatDataset',
        datasets = [ic13,ic15,ic17],
)

ftdatas = dict(
type='RepeatDataset',
times=120,
dataset = dict(
type=dataset_type,
data_root=data_root,
ann_file='mostel_train.pk',
data_prefix=dict(img='mostel/'),
pipeline=train_pipeline,
backend_args=backend_args)
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='ConcatDataset',
        datasets = [ptdatas, ftdatas],
    )
)

test_srnet = dict(
type=dataset_type,
data_root=data_root,
ann_file='srnet_test.pk',
data_prefix=dict(img='srnet/'),
test_mode=True,
pipeline=test_pipeline,
backend_args=backend_args
)

test_stefann = dict(
type=dataset_type,
data_root=data_root,
ann_file='stefann_test.pk',
data_prefix=dict(img='stefann/'),
test_mode=True,
pipeline=test_pipeline,
backend_args=backend_args
)

test_mostel = dict(
type=dataset_type,
data_root=data_root,
ann_file='mostel_test.pk',
data_prefix=dict(img='mostel/'),
test_mode=True,
pipeline=test_pipeline,
backend_args=backend_args
)

test_diffste = dict(
type=dataset_type,
data_root=data_root,
ann_file='diffste_test.pk',
data_prefix=dict(img='diffste/'),
test_mode=True,
pipeline=test_pipeline,
backend_args=backend_args
)

test_anytext = dict(
type=dataset_type,
data_root=data_root,
ann_file='anytext_test.pk',
data_prefix=dict(img='anytext/'),
test_mode=True,
pipeline=test_pipeline,
backend_args=backend_args
)

test_udifftext = dict(
type=dataset_type,
data_root=data_root,
ann_file='udifftext_test.pk',
data_prefix=dict(img='udifftext/'),
test_mode=True,
pipeline=test_pipeline,
backend_args=backend_args
)

test_derend = dict(
type=dataset_type,
data_root=data_root,
ann_file='derend_test.pk',
data_prefix=dict(img='derend/'),
test_mode=True,
pipeline=test_pipeline,
backend_args=backend_args
)

test_textocr = dict(
type=dataset_type,
data_root=data_root,
ann_file='textocr_test.pk',
data_prefix=dict(img='textocr/'),
test_mode=True,
pipeline=test_pipeline,
backend_args=backend_args
)

datas = dict(
        type='ConcatDataset',
        datasets = [test_srnet, test_stefann, test_mostel, test_derend, test_diffste, test_anytext, test_udifftext],
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=datas
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='FTIC15',
    ann_file=data_root + 'textocr_test.pk',
    metric=['bbox'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# training schedule for 1x
train_cfg = dict(type='IterBasedTrainLoop', max_iters=15000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='PolyLR',
        by_epoch=False,
        power=0.9,
        eta_min=1e-6)
]

# optimizer


optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        type='AdamW',
        lr=2e-5,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    clip_grad=dict(max_norm=36, norm_type=2),
    )

auto_scale_lr = dict(enable=False, base_batch_size=16)


default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=500, by_epoch=False, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

log_level = 'INFO'
load_from = 'ftuse.pth'#'xsza_swin_54.pth'
resume = None # 'work_dirs/test_faster/epoch_3.pth'#None
find_unused_parameters=True
