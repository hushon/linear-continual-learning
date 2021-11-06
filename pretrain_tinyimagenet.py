from typing import NamedTuple
import torch
from torch import nn, utils, optim, cuda
from torch.cuda import amp
import torchvision.transforms as T
import os
from tqdm import tqdm, trange
import argparse
import numpy as np
import random
from models.resnet_cifar100_lrelu import resnet18, resnet50
from torch.nn.parallel import DataParallel
from torchvision import datasets
import atexit
from PIL import Image
from dataset import TinyImageNet
from utils import MultiEpochsDataLoader


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
    LOG_DIR = '/workspace/runs/pretrain_tinyimagenet_resnet18'
    BATCH_SIZE = 256*2
    INIT_LR = 2E-1
    WEIGHT_DECAY = 1E-4
    MAX_EPOCH = 200
    N_WORKERS = 8
    SAVE = True


if FLAGS.SAVE:
    os.makedirs(FLAGS.LOG_DIR, exist_ok=True)


def tprint(obj):
    tqdm.write(obj.__str__())



def train():

    transform_train = T.Compose([
        T.Resize(32),
        T.RandomCrop(32, padding=4),
        T.ColorJitter(0.1, 0.1, 0.1),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(TinyImageNet.MEAN, TinyImageNet.STD)
    ])
    transform_test = T.Compose([
        T.Resize(32),
        T.ToTensor(),
        T.Normalize(TinyImageNet.MEAN, TinyImageNet.STD)
    ])

    train_dataset = TinyImageNet(FLAGS.DATA_ROOT, transform=transform_train, train=True)
    test_dataset = TinyImageNet(FLAGS.DATA_ROOT, transform=transform_test, train=False)

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

    model = resnet18(num_classes=200)
    model = DataParallel(model).cuda()


    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.INIT_LR, momentum=0.9, weight_decay=FLAGS.WEIGHT_DECAY)
    # lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=FLAGS.INIT_LR, epochs=FLAGS.MAX_EPOCH, steps_per_epoch=len(train_loader), pct_start=0.1)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, FLAGS.MAX_EPOCH*len(train_loader))

    def train_epoch():
        model.train()
        for i, (input, target) in enumerate(train_loader):
            target = target.cuda()
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target).mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if i%1000 == 0:
                tprint(f'[TRAIN][{i}/{len(train_loader)}] LR {lr_scheduler.get_last_lr()[-1]:.2e} | loss {loss.cpu().item():.3f}')

    @torch.no_grad()
    def evaluate():
        losses = []
        corrects = []
        model.eval()
        for input, target in test_loader:
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            losses.append(loss.view(-1))
            corrects.append((target == output.max(-1).indices).view(-1))
        avg_loss = torch.cat(losses).mean().item()
        avg_acc = torch.cat(corrects).float().mean().item()*100
        tprint(f'[TEST] loss {avg_loss:.3f} | T1acc {avg_acc:.2f}')

    def save_pickle():
        pickle = model.module.state_dict()
        pickle.pop('fc.weight')
        pickle.pop('fc.bias')
        pickle_path = os.path.join(FLAGS.LOG_DIR, f'state_dict.pt')
        torch.save(pickle, pickle_path)
        tprint(f'[SAVE] Saved to {pickle_path}')

    if FLAGS.SAVE: atexit.register(save_pickle)

    pbar = trange(FLAGS.MAX_EPOCH, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', smoothing=1.)

    for epoch in pbar:
        train_epoch()
        if epoch%10 == 0:
            evaluate()
            save_pickle()

    if FLAGS.SAVE:
        save_pickle()

if __name__ == "__main__":
    train()