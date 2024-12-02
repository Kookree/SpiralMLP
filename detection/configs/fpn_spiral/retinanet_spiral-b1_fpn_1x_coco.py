_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
#    '../_base_/datasets/coco_detection.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='RetinaNet',
    backbone=dict(
        _delete_=True,
        type='SpiralNet',
        embed_dims=64,
        num_layers=[2, 2, 4, 2],
        init_cfg=dict(checkpoint='./mumu_pretrained/spiral_b1_300.pth')),
    neck=dict(in_channels=[64, 128, 320, 512]))
find_unused_parameters = True
# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001))
