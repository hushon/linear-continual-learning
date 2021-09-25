from typing import NamedTuple
import torch
from torch import nn, utils, optim, cuda
from torch.cuda import amp
import torchvision.transforms as T
import torch.nn.functional as F
import os
from tqdm import tqdm, trange
import numpy as np
import random
from models.resnet_cifar100_jvplrelu import resnet18, resnet50
# from models.resnet_cifar100_lrelu import resnet18, resnet50
from torch.nn.parallel import DataParallel
from torchvision import datasets
import atexit
from PIL import Image
from utils import MultiEpochsDataLoader

# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

torch.backends.cudnn.benchmark = True
SET_DETERMINISTIC = False

if SET_DETERMINISTIC:
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    np.random.seed(123)
    random.seed(123)


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.485, 0.456, 0.406)
CIFAR100_STD = (0.229, 0.224, 0.225)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FLAGS(NamedTuple):
    DATA_ROOT = '/ramdisk/'
    # CHECKPOINT_DIR = '/workspace/runs/temp111'
    CHECKPOINT_DIR = '/workspace/runs/torch_imagenet32_resnet50_new'
    LOG_DIR = '/workspace/runs/temp111'
    BATCH_SIZE = 128
    INIT_LR = 1E-4
    WEIGHT_DECAY = 1E-5
    MAX_EPOCH = 100
    N_WORKERS = 4
    BN_UPDATE_STEPS = 1000
    USE_AMP = False
    SAVE = False


if FLAGS.SAVE:
    os.makedirs(FLAGS.LOG_DIR, exist_ok=True)


def tprint(obj):
    tqdm.write(obj.__str__())


def linear_schedule(max_iter):
    def f(i):
        return (max_iter - i)/max_iter
    return f


def icycle(iterable):
    while True:
        for x in iterable:
            yield x


def correct(pred: torch.Tensor, target: torch.Tensor, k=5):
    pred = pred.topk(k, -1).indices
    target = target[:, None].broadcast_to(pred.shape)
    correct = pred.eq(target).any(-1)
    return correct


def weight_decay(named_parameters, lam):
    for name, param in named_parameters:
        if 'bn' not in name and param.grad is not None:
            param.grad.data.add_(param.data, alpha=lam)


def main():

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        ])

    train_dataset = datasets.CIFAR100(FLAGS.DATA_ROOT, True, transform_train, download=True)
    test_dataset = datasets.CIFAR100(FLAGS.DATA_ROOT, False, transform_test, download=True)

    train_loader = MultiEpochsDataLoader(train_dataset,
            batch_size=FLAGS.BATCH_SIZE,
            shuffle=True,
            num_workers=FLAGS.N_WORKERS,
            )
    test_loader = MultiEpochsDataLoader(test_dataset,
            batch_size=FLAGS.BATCH_SIZE*2,
            shuffle=False,
            num_workers=FLAGS.N_WORKERS
            )

    # model = resnet18(num_classes=100).cuda()
    model = resnet50(num_classes=100).cuda()
    state_dict = torch.load(os.path.join(FLAGS.CHECKPOINT_DIR, 'state_dict.pt'))
    model.load_state_dict(state_dict, strict=False)

    # freeze feature extractor
    # for name, param in model.named_parameters():
    #     if 'fc' not in name:
    #         param.requires_grad_(False)

    # criterion = nn.MSELoss(reduction='none')
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.INIT_LR, weight_decay=FLAGS.WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, linear_schedule(FLAGS.MAX_EPOCH*len(train_loader)))
    grad_scaler = amp.GradScaler(enabled=FLAGS.USE_AMP)


    def train_epoch():
        losses = []
        corrects = []
        corrects_t5 = []
        model.eval()
        for i, (input, target) in enumerate(train_loader):
            input = input.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            with amp.autocast(enabled=FLAGS.USE_AMP):
                output = model(input)
                # loss = criterion(output, 15.*F.one_hot(target, num_classes=100).float()).sum(-1).mean()
                loss = criterion(output, target).mean()
            grad_scaler.scale(loss).backward()
            weight_decay(model.named_parameters(), FLAGS.WEIGHT_DECAY)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            lr_scheduler.step()

            with torch.no_grad():
                losses.append(loss.view(-1))
                corrects.append((target == output.max(-1).indices).view(-1))
                corrects_t5.append(correct(output, target, 5))

            if i%100 == 0:
                avg_loss = torch.cat(losses).mean().item()
                avg_acc = torch.cat(corrects).float().mean().item()*100
                avg_acc_t5 = torch.cat(corrects_t5).float().mean().item()*100
                tprint(f'[TRAIN][{i}/{len(train_loader)}] LR {lr_scheduler.get_last_lr()[-1]:.2e} | loss {avg_loss:.3f} | T1acc {avg_acc:.2f} | T5acc {avg_acc_t5:.2f}')
                losses.clear()
                corrects.clear()
                corrects_t5.clear()

    @torch.no_grad()
    def evaluate():
        losses = []
        corrects = []
        corrects_t5 = []
        model.eval()
        for input, target in test_loader:
            input = input.cuda()
            target = target.cuda()
            with amp.autocast(enabled=FLAGS.USE_AMP):
                output = model(input)
                # loss = criterion(output, 15.*F.one_hot(target, num_classes=100).float()).sum(-1)
                loss = criterion(output, target).mean()
            losses.append(loss.view(-1))
            corrects.append((target == output.max(-1).indices).view(-1))
            corrects_t5.append(correct(output, target, 5))
        avg_loss = torch.cat(losses).mean().item()
        avg_acc = torch.cat(corrects).float().mean().item()*100
        avg_acc_t5 = torch.cat(corrects_t5).float().mean().item()*100
        tprint(f'[TEST] loss {avg_loss:.3f} | T1acc {avg_acc:.2f} | T5acc {avg_acc_t5:.2f}')

    def save_pickle():
        pickle = model.state_dict()
        pickle_path = os.path.join(FLAGS.LOG_DIR, f'state_dict.pt')
        torch.save(pickle, pickle_path)
        tprint(f'[SAVE] Saved to {pickle_path}')


    # update BN stats
    loader = icycle(train_loader)
    with torch.no_grad():
        model.train()
        for _ in trange(FLAGS.BN_UPDATE_STEPS):
            input, _ = next(loader)
            input = input.cuda()
            with amp.autocast(enabled=FLAGS.USE_AMP):
                model(input)


    pbar = trange(FLAGS.MAX_EPOCH, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for epoch in pbar:
        train_epoch()
        if (epoch+1)%10 == 0:
            evaluate()

    if FLAGS.SAVE:
        save_pickle()


if __name__ == '__main__':
    main()
