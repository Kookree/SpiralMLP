_base_ = [
    './spiral-b1-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
checkpoint_file = './mumu_pretrained/spiral_b2_300.pth'
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=64, 
        depths=[2, 3, 10, 3]),
#    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=150),
#    auxiliary_head=dict(in_channels=384, num_classes=150)
            )
