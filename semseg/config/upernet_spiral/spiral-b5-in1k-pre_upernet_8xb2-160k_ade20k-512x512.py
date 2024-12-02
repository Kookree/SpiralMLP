_base_ = [
    './spiral-b1-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
checkpoint_file = './mumu_pretrained/spiral_b5_300.pth'
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=96,
        depths=[3, 4, 24, 3],
#        num_heads=[4, 8, 16, 32]
                ),
#    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=150),
#    auxiliary_head=dict(in_channels=512, num_classes=150)
            )
find_unused_parameters = True


#data = dict(samples_per_gpu=8, # it means 4 batch sizes are used in one GPU
#            workers_per_gpu=1,)

train_dataloader = dict(batch_size=8, 
                        num_workers=1,)
val_dataloader = dict(  batch_size=1, 
                        num_workers=1,)
test_dataloader = val_dataloader

