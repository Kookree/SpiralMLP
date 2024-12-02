_base_ = 'retinanet_spiral-b1_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        embed_dims=64,
        num_layers=[2, 3, 10, 3],
        init_cfg=dict(checkpoint='./mumu_pretrained/spiral_b2_300.pth')),
    neck=dict(in_channels=[64, 128, 320, 512]))
find_unused_parameters = True

