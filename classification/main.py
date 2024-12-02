#import wandb
import datetime
import json
from engine import train_one_epoch, evaluate
import time
from pathlib import Path
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import NativeScaler, get_state_dict, ModelEma
from losses import DistillationLoss
from timm.scheduler import create_scheduler
import torch.nn as nn
import os
import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from timm.optim import create_optimizer
from collections import Counter
from typing import Any, List
from timm.data import Mixup
from datasets import imagenet_dataloader
from torch.utils.data import DataLoader
from model import MODEL_REGISTRY
from contextlib import suppress
from fvcore.nn import flop_count
from fvcore.nn.jit_handles import get_shape, conv_flop_count
import torch.backends.cudnn as cudnn
import utils
import argparse
from args_config import get_args_parser



## start a new wandb run to track this script
#wandb.init(
#    # set the wandb project where this run will be logged
#    project="spiralmlp-image-classification",
#    
#    # track hyperparameters and run metadata
#    config={
#    "learning_rate": 'dynamic',
#    "architecture": "spiralmlp",
#    "dataset": "imagenet",
#    "epochs": 300,
#    }
#)

def main(args):
    utils.init_distributed_mode(args)
    
    """not using distillation"""
#    if args.distillation_type != 'none' and args.finetune and not args.eval:
#        raise NotImplementedError("Finetuning with distillation not yet supported")
    device = torch.device(args.device)

    """not using AMP => automatic mixed precision"""    
    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
#    if not args.no_amp:  # args.amp: Default  use AMP
#        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
#        if has_native_amp:
#            args.native_amp = True
#            args.apex_amp = False
#        elif has_apex:
#            args.native_amp = False
#            args.apex_amp = True
#        else:
#            raise ValueError("Warning: Neither APEX or native Torch AMP is available, using float32."
#                             "Install NVIDA apex or upgrade to PyTorch 1.6")
#    else:
#        args.apex_amp = False
#        args.native_amp = False
#    if args.apex_amp and has_apex:
#        use_amp = 'apex'
#    elif args.native_amp and has_native_amp:
#        use_amp = 'native'
#    elif args.apex_amp or args.native_amp:
#        print ("Warning: Neither APEX or native Torch AMP is available, using float32. "
#                        "Install NVIDA apex or upgrade to PyTorch 1.6")


    """set up the seed"""
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    """
    causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
    better used when have fixde input during training and large memory
    """
    cudnn.benchmark = True


#    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
#    dataset_val, _ = build_dataset(is_train=False, args=args)
#
#
#    if args.distributed: 
#        num_tasks = utils.get_world_size()
#        global_rank = utils.get_rank()
#        if args.repeated_aug:
#            sampler_train = RASampler(
#                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
#            )
#        else:
#            sampler_train = torch.utils.data.DistributedSampler(
#                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
#            )
#        if args.dist_eval:  # default False
#            if len(dataset_val) % num_tasks != 0:
#                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
#                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
#                      'equal num of samples per-process.')
#            sampler_val = torch.utils.data.DistributedSampler(
#                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
#        else:
#            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
#    else:
#        sampler_train = torch.utils.data.RandomSampler(dataset_train)
#        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
#
#
#    data_loader_train = DataLoader(
#        dataset_train, sampler=sampler_train,
#        batch_size=args.batch_size,
#        num_workers=args.num_workers,
#        pin_memory=args.pin_mem,
#        drop_last=True,
#    )
#
#    data_loader_val = DataLoader(
#        dataset_val, sampler=sampler_val,
#        batch_size=int(1.5 * args.batch_size),
#        num_workers=args.num_workers,
#        pin_memory=args.pin_mem,
#        drop_last=False
#    )

#    args.data_path = 'datasets/imagenet'
    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    data_loader_train, data_loader_val, num_classes = imagenet_dataloader(
                                                                train_dir=train_dir, 
                                                                val_dir=val_dir, 
                                                                batch_size=args.batch_size, 
                                                                val_batch_size=int(1.5 * args.batch_size), 
                                                                num_workers=args.num_workers, 
                                                                pin_memory=args.pin_mem, 
                                                                distributed=args.distributed, 
                                                                dist_eval=args.dist_eval, 
                                                                repeated_aug=args.repeated_aug, 
                                                                num_tasks=utils.get_world_size(), 
                                                                global_rank=utils.get_rank()
                                                                )

    """mixing, dataset augmentation"""
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=num_classes)


    """
    creating model
    note: need to relate to the args
    """
    print(f"Creating model: {args.model}")
    model = MODEL_REGISTRY[args.model]()    
#    model = create_model(
#        args.model,
#        pretrained=False,
#        num_classes=args.nb_classes,
#        drop_rate=args.drop,
#        drop_path_rate=args.drop_path,
#        drop_block_rate=None,
#    )

    
    """count flops"""
    if args.flops:
        #model_mode = model.training
        model.eval()
        dummy_input = torch.rand(1, 3, 224, 224)
        flops_dict, *_ = flop_count(model, dummy_input,
                                    supported_ops={"torchvision::deform_conv2d": utils.sfc_flop_jit})
        count = sum(flops_dict.values())
        model.train()
        print('=' * 30)
        print(f"fvcore MAdds: {count:.3f} G")

    """count number of parameters"""
    if args.num_params:
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Number of parameters: {num_params} M")
        print('=' * 30)

    """args.finetune:str => None or ckpt_path"""
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
        """note: relat to the torch.save()"""
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)
    
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)


    """setup automatic mixed-precision (AMP) loss scaling and op casting"""
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        print('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        print('Using native Torch AMP. Training in mixed precision.')
    else:
        print('AMP not enabled. Training in float32.')

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')


    raw_model = model
    if args.distributed:
#        if has_apex and use_amp != 'native':
#            # Apex DDP preferred unless native amp is activated
#            model = ApexDDP(model, delay_allreduce=True)
#        else:
#            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model = DDP(model, device_ids=[args.gpu])
        raw_model = model.module

    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    """not using"""
    teacher_model = None
#    if args.distillation_type != 'none':
#        assert args.teacher_path, 'need to specify teacher-path when using distillation'
#        print(f"Creating teacher model: {args.teacher_model}")
#        teacher_model = create_model(
#            args.teacher_model,
#            pretrained=False,
#            num_classes=args.nb_classes,
#            global_pool='avg',
#        )
#        if args.teacher_path.startswith('https'):
#            checkpoint = torch.hub.load_state_dict_from_url(
#                args.teacher_path, map_location='cpu', check_hash=True)
#        else:
#            checkpoint = torch.load(args.teacher_path, map_location='cpu')
#        teacher_model.load_state_dict(checkpoint['model'])
#        teacher_model.to(device)
#        teacher_model.eval()

    """
    wrap the criterion in our custom DistillationLoss, which
    just dispatches to the original criterion if args.distillation_type is 'none'
    args.distillation_type default is None
    """
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)

    """resume from a ckpt"""
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        raw_model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
#            if args.model_ema:
#                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
# loss_scaler is saved as None, does not have load_state_dict
#            if 'scaler' in checkpoint:
#                loss_scaler.load_state_dict(checkpoint['scaler'])


    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, amp_autocast=amp_autocast)
#        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        print(f"Accuracy of the network on the test images: {test_stats['acc1']:.1f}%")
        return


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
            amp_autocast=amp_autocast
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
#                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device, amp_autocast=amp_autocast)
#        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        print(f"Accuracy of the network on the test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
#    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('SpiralMLP training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

