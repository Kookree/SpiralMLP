To run in one GPU:
`python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --batch-size=128 2>&1 > train.log`
the --model is SpiralMLP_B1 by default

To finetune on the workstation:
`python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --batch-size=64 --resume result/first_300_epoch/checkpoint.pth --no-model-ema --epochs 600 2>&1 > train.log`

To chech the `train.log`:
`while true; do cat train.log; sleep 1; done`

