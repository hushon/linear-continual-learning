from logging import disable
import torch
from torch import nn, utils, optim, cuda
from torch.cuda import amp
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as T
from PIL import Image
import os, glob, re
import tqdm
import argparse
import numpy as np
import random
from torchvision import models
import torchvision.transforms.functional as VF
from models import resnet_cifar10, resnet_cifar100
from torch.nn.parallel import DistributedDataParallel
import torch.utils.data
import torch.distributed
import torch.multiprocessing
import torchmetrics, metrics
import atexit

torch.backends.cudnn.benchmark = True
SET_DETERMINISTIC = False
DATA_ROOT = '/workspace/Data/CLS-LOC/'
LOG_DIR = '/workspace/runs'
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.485, 0.456, 0.406)
CIFAR100_STD = (0.229, 0.224, 0.225)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
N_GPUS = torch.cuda.device_count()

if SET_DETERMINISTIC:
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    np.random.seed(123)
    random.seed(123)

parser = argparse.ArgumentParser()
parser.add_argument('-b', dest='batch_size', type=int, default=128)
parser.add_argument('-lr', dest='lr', type=float, default=1e-2)
parser.add_argument('-wd', dest='weight_decay', type=float, default=1e-5)
parser.add_argument('-e', dest='max_epoch', type=int, default=60)
parser.add_argument('-amp', dest='use_amp', action='store_true')
parser.add_argument('-cl', dest='channels_last', action='store_true')
args = parser.parse_args()

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def init_process(rank, world_size, backend=torch.distributed.Backend.NCCL):
    torch.distributed.init_process_group(
                                        backend,
                                        init_method='tcp://localhost:12345',
                                        rank=rank,
                                        world_size=world_size
                                        )

def cleanup():
    torch.distributed.destroy_process_group()

def train(rank, world_size):
    init_process(rank, world_size)

    transform_train = T.Compose([
        T.RandomResizedCrop(256),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    transform_test = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, 'train'), transform_train)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, 'val') ,transform_test)
    train_sampler = utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    test_sampler = utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=False)

    train_loader = utils.data.DataLoader(train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True)
    test_loader = utils.data.DataLoader(test_dataset,
            batch_size=args.batch_size*2,
            sampler=test_sampler,
            num_workers=8,
            pin_memory=True)

    torch.cuda.set_device(rank)
    # model = resnet_cifar100.resnet34(num_classes=100).cuda(rank)
    model = models.resnet34().cuda(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DistributedDataParallel(model, device_ids=[rank])
    if args.channels_last: model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch)
    lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.max_epoch, steps_per_epoch=len(train_loader))
    grad_scaler = amp.GradScaler(enabled=args.use_amp)
    loss_meter = metrics.Scalar().cuda(rank)
    acc_meter = torchmetrics.Accuracy().cuda(rank)

    def train_epoch():
        model.train()
        for i, (input, target) in enumerate(train_loader):
            if args.channels_last: input = input.contiguous(memory_format=torch.channels_last)
            input = input.cuda(rank)
            target = target.cuda(rank)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=args.use_amp):
                output = model(input)
                loss = criterion(output, target).mean()
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            lr_scheduler.step()
            if rank == 0 and i%10 == 0:
                print(f'[TRAIN][{i}/{len(train_loader)}] LR {lr_scheduler.get_last_lr()[-1]:.2e} | acc {acc_meter.compute():.4f}')

    @torch.no_grad()
    def evaluate(dataloader):
        model.eval()
        for input, target in dataloader:
            if args.channels_last: input = input.contiguous(memory_format=torch.channels_last)
            input = input.cuda(rank)
            target = target.cuda(rank)
            with amp.autocast(enabled=args.use_amp):
                output = model(input)
            loss = criterion(output, target)
        #     loss_meter(loss.detach())
            acc_meter(output.softmax(-1).detach(), target)
            if rank == 0:
                print('hi')
        if rank == 0:
            print(f'[VAL] {acc_meter.compute():.4f}')
        #     loss_meter.reset()
            acc_meter.reset()

    def save_pickle():
        pickle = {
                'model': model,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer,
                'optim_state_dict': optimizer.state_dict(),
                'args': args,
                'epoch': epoch,
                }
        pickle_path = os.path.join(LOG_DIR, f'state_dict.pt')
        torch.save(pickle, pickle_path)

    pbar = tqdm.trange(args.max_epoch, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', smoothing=1., disable=False if rank == 0 else True)
    for epoch in pbar:
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        # train_epoch()
        evaluate(test_loader)

    torch.distributed.barrier()
    if rank == 0:
        save_pickle()

    cleanup()


if __name__ == "__main__":
    torch.multiprocessing.spawn(train,
            args=(N_GPUS,),
            nprocs=N_GPUS,
            join=True)
