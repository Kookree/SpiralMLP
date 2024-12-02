_base_ = 'retinanet_spiral-b1_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        embed_dims=64,
        num_layers=[3, 8, 27, 3],
        init_cfg=dict(checkpoint='./mumu_pretrained/spiral_b4_300.pth')),
    neck=dict(in_channels=[64, 128, 320, 512]))
find_unused_parameters = True
# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, type='AdamW', lr=0.0001 / 1.4, weight_decay=0.0001))

# dataset settings
train_dataloader = dict(batch_size=4, num_workers=1)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (1 samples per GPU)
auto_scale_lr = dict(base_batch_size=4)
