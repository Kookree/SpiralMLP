from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, RandomSampler
from samplers import RASampler

def imagenet_dataloader(train_dir:str=None, 
                        val_dir:str=None, 
                        batch_size:int=32, 
                        val_batch_size:int=int(1.5*32), 
                        num_workers:int=4, 
                        pin_memory:bool=True, 
                        distributed:bool=True, 
                        dist_eval:bool=False, 
                        repeated_aug:bool=True, 
                        num_tasks:int=None, 
                        global_rank:int=None
                        ):

    train_dataset, val_dataset = None, None
    train_loader, val_loader = None, None
    num_classes = 0

    """preprocess dataset"""
    if train_dir is not None:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)

    if val_dir is not None:
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
        num_classes = len(val_dataset.class_to_idx)

    """note: remember to remove"""
    if train_dataset is None:
        train_dataset = val_dataset

    """prepare sampler"""
    if distributed:
        if repeated_aug:
            train_sampler = RASampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            train_sampler = DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        if dist_eval:
            if len(val_dataset) % num_tasks != 0:
                print("Warning: Enabling distributed evaluation with an eval dataset not divisible by process number.This will slightly alter validation results as extra duplicate entries are added to achieve equal num of samples per-process.")
                
            val_sampler = DistributedSampler(val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else: 
            val_sampler = SequentialSampler(val_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

    """build dataloader"""
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
#        shuffle=True,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=val_batch_size, 
#        shuffle=True,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    return train_loader, val_loader, num_classes




if __name__ == "__main__":
    train_dir = None
    val_dir = "imagenet/val"

    train_loader, val_loader, num_classes = imagenet_dataloader(
                                                                train_dir, 
                                                                val_dir, 
                                                                batch_size=32, 
                                                                val_batch_size=48, 
                                                                num_workers=4, 
                                                                pin_memory=True, 
                                                                distributed=True, 
                                                                dist_eval=False, 
                                                                repeated_aug=True, 
                                                                num_tasks=4, 
                                                                global_rank=1
                                                                )

    print(f"number of classes: {num_classes}")  # 1000
    for batch in val_loader:
        print(batch[0].shape)   #torch.Size([32, 3, 224, 224])
        break


