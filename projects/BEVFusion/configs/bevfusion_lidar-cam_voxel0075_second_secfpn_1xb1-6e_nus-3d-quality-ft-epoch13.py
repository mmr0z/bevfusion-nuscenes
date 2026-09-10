import os.path as osp

_base_ = ['./bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

# Fine-tune the lidar-camera BEVFusion model from the best local lidar-only
# checkpoint. Matching lidar/head weights are loaded from epoch_13, while the
# new camera branch is initialized from a local Swin-T nuImages pretrain.

lidar_pretrained = (
    'work_dirs/bevfusion_lidar_voxel0075_second_secfpn_1xb5-16e_nus-3d-quality/'
    'epoch_13.pth')
image_pretrained_local = 'checkpoints/swint-nuimages-pretrained.pth'
image_pretrained_url = (
    'https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/'
    'swint-nuimages-pretrained.pth')
image_pretrained = (
    image_pretrained_local
    if osp.isfile(image_pretrained_local) else image_pretrained_url)

load_from = lidar_pretrained
work_dir = (
    './work_dirs/'
    'bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-6e_nus-3d-quality-ft-epoch13'
)

model = dict(
    bbox_head=dict(num_proposals=200),
    img_backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=image_pretrained)))

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    pin_memory=False,
    prefetch_factor=2)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    pin_memory=False,
    drop_last=False)

test_dataloader = val_dataloader

# Keep the official nuScenes detection protocol explicit for reproducible
# reporting.  In particular, record that this model uses both sensor streams
# in the saved nuScenes result metadata (the inherited LiDAR config otherwise
# marks the evaluator output as LiDAR-only).
val_evaluator = dict(
    metric='bbox',
    eval_version='detection_cvpr_2019',
    modality=dict(use_lidar=True, use_camera=True))
test_evaluator = dict(
    metric='bbox',
    eval_version='detection_cvpr_2019',
    modality=dict(use_lidar=True, use_camera=True))

train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=1e-4, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.3),
            'pts_voxel_encoder': dict(lr_mult=0.2),
            'pts_middle_encoder': dict(lr_mult=0.2),
            'pts_backbone': dict(lr_mult=0.2),
            'pts_neck': dict(lr_mult=0.2),
            'bbox_head': dict(lr_mult=0.5),
        }),
    clip_grad=dict(max_norm=35, norm_type=2))

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=5000,
        max_keep_ckpts=10,
        save_last=True))
