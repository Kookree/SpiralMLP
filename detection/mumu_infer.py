from mmdet.apis import DetInferencer
import os

directory = 'mumu_infer'
img_paths = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    img_paths.append(f)

inferencer = DetInferencer(model='configs/spiral/retinanet_spiral-b5_fpn_1x_coco.py', weights='work_dirs/retinanet_spiral-b5_fpn_1x_coco_0/epoch_31.pth', device='cpu')

inferencer(img_paths, out_dir='mumu_infer/outputs')
