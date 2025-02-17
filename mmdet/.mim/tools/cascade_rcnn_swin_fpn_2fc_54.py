model = dict(
    type='CascadeRCNN',
    pretrained='coco_pths/swin_small_cascade_2fc.pth',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[
            2,
            2,
            18,
            2,
        ],
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        use_abs_pos_embed=False,
        patch_norm=True,
        out_indices=(
            0,
            1,
            2,
            3,
        )),
    neck=dict(
        type='FPN',
        in_channels=[
            96,
            192,
            384,
            768,
        ],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[
                8,
            ],
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[
            1,
            0.5,
            0.25,
        ],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[
                4,
                8,
                16,
                32,
            ]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.1,
                        0.1,
                        0.2,
                        0.2,
                    ]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.05,
                        0.05,
                        0.1,
                        0.1,
                    ]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.033,
                        0.033,
                        0.067,
                        0.067,
                    ]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
        ]),
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
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='OHEMSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='OHEMSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='OHEMSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
        ]),
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
dataset_type = 'MyDataset'
data_root = 'datas/afac2023/'
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', scale=(
        1024,
        1024,
    ), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Albu',
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
            ],
            min_visibility=0.0,
            filter_lost_elements=True),
        transforms=[
            dict(type='RandomRotate90', p=0.5),
            dict(
                type='ImageCompression',
                quality_lower=90,
                quality_upper=100,
                p=0.5),
            dict(type='RGBShift', p=0.1),
            dict(type='HueSaturationValue', p=0.1),
            dict(type='RandomBrightnessContrast', p=0.1),
            dict(type='GaussianBlur', p=0.01),
            dict(type='RandomGamma', p=0.1),
            dict(type='GaussNoise', p=0.01),
        ]),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(
        1532,
        1024,
    ), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        )),
]
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))
img_scales = [
    (
        1280,
        1280,
    ),
    (
        2560,
        2560,
    ),
]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=(
                    1280,
                    1280,
                ), keep_ratio=True),
                dict(type='Resize', scale=(
                    2560,
                    2560,
                ), keep_ratio=True),
            ],
            [
                dict(type='RandomFlip', prob=1.0),
                dict(type='RandomFlip', prob=0.0),
            ],
            [
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'flip',
                        'flip_direction',
                    )),
            ],
        ]),
]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='MyDataset',
        data_root='datas/afac2023/',
        ann_file='train/train.pk',
        data_prefix=dict(img='train/'),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
            dict(type='Resize', scale=(
                1024,
                1024,
            ), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Albu',
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                    ],
                    min_visibility=0.0,
                    filter_lost_elements=True),
                transforms=[
                    dict(type='RandomRotate90', p=0.5),
                    dict(
                        type='ImageCompression',
                        quality_lower=90,
                        quality_upper=100,
                        p=0.5),
                    dict(type='RGBShift', p=0.1),
                    dict(type='HueSaturationValue', p=0.1),
                    dict(type='RandomBrightnessContrast', p=0.1),
                    dict(type='GaussianBlur', p=0.01),
                    dict(type='RandomGamma', p=0.1),
                    dict(type='GaussNoise', p=0.01),
                ]),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='PackDetInputs'),
        ],
        backend_args=None))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MyDataset',
        data_root='datas/afac2023/',
        ann_file='train/val.pk',
        data_prefix=dict(img='train/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(
                1532,
                1024,
            ), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
            dict(
                type='PackDetInputs',
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                )),
        ],
        backend_args=None))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MyDataset',
        data_root='datas/afac2023/',
        ann_file='train/val.pk',
        data_prefix=dict(img='train/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(
                1532,
                1024,
            ), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
            dict(
                type='PackDetInputs',
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                )),
        ],
        backend_args=None))
val_evaluator = dict(
    type='SimpleAccuracy',
    ann_file='datas/afac2023/train/val.pk',
    metric=[
        'bbox',
    ],
    format_only=False,
    backend_args=None)
test_evaluator = dict(
    type='SimpleAccuracy',
    ann_file='datas/afac2023/train/val.pk',
    metric=[
        'bbox',
    ],
    format_only=False,
    backend_args=None)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='PolyLR', by_epoch=False, power=0.9, eta_min=1e-06),
]
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))),
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(
            0.9,
            0.999,
        ), weight_decay=0.05))
auto_scale_lr = dict(enable=False, base_batch_size=16)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = 'coco_pths/swin_small_cascade_2fc.pth'
resume = False
launcher = 'pytorch'
work_dir = './work_dirs/cascade_rcnn_swin_fpn_2fc_54'
