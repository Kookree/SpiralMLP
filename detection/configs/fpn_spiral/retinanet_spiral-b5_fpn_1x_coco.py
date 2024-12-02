_base_ = 'retinanet_spiral-b1_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        embed_dims=96,
        num_layers=[3, 4, 24, 3],
        mlp_ratios=(4, 4, 4, 4),
        init_cfg=dict(checkpoint='./mumu_pretrained/spiral_b5_300.pth')),
    neck=dict(in_channels=[96, 192, 384, 768]))

## resume
#resume = True
#resume_from =  'work_dirs/retinanet_spiral-b5_fpn_1x_coco_bc/epoch_12.pth'

find_unused_parameters = True
#detect_anomalous_params = True
# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, type='AdamW', lr=0.0001 / 1.4, weight_decay=0.0001))

# dataset settings, each gpu has how many batch
train_dataloader = dict(batch_size=4, num_workers=1)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (1 samples per GPU)
# base_batch_size = number of gpus * batch size of each
# in mumu trial, the first epoch mAP when base_batch_size=4 is lower than that when base_batch_size=16
# the latest training which reaches 41.2% is when base_batch_size=16
# now try when base_batch_size=32
auto_scale_lr = dict(base_batch_size=32)
