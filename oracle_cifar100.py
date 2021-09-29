from typing import NamedTuple
import torch
from torch import nn, utils, optim, cuda
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
import copy
from torch.utils.tensorboard.writer import SummaryWriter
from utils import MultiEpochsDataLoader
from dataset import TaskIncrementalTenfoldCIFAR100

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
    CHECKPOINT_DIR = '/workspace/runs/temp111'
    LOG_DIR = '/workspace/runs/torch_rbu_cifar_21'
    BATCH_SIZE = 128
    INIT_LR = 1e-4
    WEIGHT_DECAY = 1e-4
    # WEIGHT_DECAY = 0
    MAX_EPOCH = 200
    N_WORKERS = 4
    BN_UPDATE_STEPS = 0
    SAVE = True


if FLAGS.SAVE:
    os.makedirs(FLAGS.LOG_DIR, exist_ok=True)


def tprint(obj):
    tqdm.write(obj.__str__())


def linear_schedule(max_iter):
    def f(i):
        return (max_iter - i)/max_iter
    return f


def weight_decay(named_parameters, lam):
    for name, param in named_parameters:
        if 'bn' not in name:
            param.grad.data.add_(param.data, alpha=lam)


def make_dataloader(
        dataset: torch.utils.data.Dataset,
        train: bool = True,
        single_batch = False,
    ) -> torch.utils.data.DataLoader:
    data_loader = MultiEpochsDataLoader(
        dataset,
        batch_size=1 if single_batch else FLAGS.BATCH_SIZE,
        shuffle=train,
        drop_last=train,
        num_workers=FLAGS.N_WORKERS,
    )
    return data_loader

def icycle(iterable):
    while True:
        for x in iterable:
            yield x


class MultiHeadWrapper(nn.Module):
    def __init__(self, module, n_heads, in_features, out_features):
        super().__init__()
        self.module = module
        self.heads = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(n_heads)])
        self.head = None

    def forward(self, x):
        x = self.module(x)
        return [head(x) for head in self.heads]


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
    # target_transform = lambda x: 15.*F.one_hot(x, num_classes=10).float()

    train_dataset_sequence = [TaskIncrementalTenfoldCIFAR100(FLAGS.DATA_ROOT, task_id=i, train=True, transform=transform_train) for i in range(10)]
    test_dataset_sequence = [TaskIncrementalTenfoldCIFAR100(FLAGS.DATA_ROOT, task_id=i, train=False, transform=transform_test) for i in range(10)]
    test_loader_sequence = [make_dataloader(dset, train=False) for dset in test_dataset_sequence]

    model = resnet18(num_classes=0)
    state_dict = torch.load(os.path.join(FLAGS.CHECKPOINT_DIR, 'state_dict.pt'))
    model.load_state_dict(state_dict, strict=False)
    model = MultiHeadWrapper(model, 10, 512, 10)
    model.cuda()

    criterion = nn.MSELoss(reduction='none')

    @torch.no_grad()
    def update_batchnorm():
        train_loader_cycle = icycle(make_dataloader(datasets.CIFAR100(FLAGS.DATA_ROOT, train=True, transform=transform_train), True))
        model.train()
        for _ in trange(FLAGS.BN_UPDATE_STEPS):
            input, _ = next(train_loader_cycle)
            input = input.cuda()
            model(input)

    @torch.no_grad()
    def evaluate(data_loader, t):
        losses = []
        corrects = []
        model.eval()
        for input, target in data_loader:
            input = input.cuda()
            target = target.cuda()
            output = model(input)[t]
            loss = 0.5*criterion(output, 15.*F.one_hot(target, num_classes=10).float()).sum(-1)
            losses.append(loss.view(-1))
            corrects.append((target == output.max(-1).indices).view(-1))
        avg_loss = torch.cat(losses).mean().item()
        avg_acc = torch.cat(corrects).float().mean().item()*100
        return {
            'loss': avg_loss,
            'acc': avg_acc,
        }

    def evaluate_sequence(current_t):
        losses = dict()
        accs = dict()
        for t in range(current_t+1):
            data_loader = test_loader_sequence[t]
            summary = evaluate(data_loader, t)
            losses[str(t)] = summary['loss']
            accs[str(t)] = summary['acc']
            tprint(f"[TEST] loss {summary['loss']:.3f} | T1acc {summary['acc']:.2f}")
        if FLAGS.SAVE:
            summary_writer.add_scalars('test_loss/per_task', losses, global_step=global_step)
            summary_writer.add_scalars('test_acc/per_task', accs, global_step=global_step)
            summary_writer.add_scalar('test_loss/avg', np.mean(list(losses.values())), global_step=global_step)
            summary_writer.add_scalar('test_acc/avg', np.mean(list(accs.values())), global_step=global_step)

    def save_pickle():
        pickle = model.state_dict()
        pickle_path = os.path.join(FLAGS.LOG_DIR, f'state_dict.pt')
        torch.save(pickle, pickle_path)
        tprint(f'[SAVE] Saved to {pickle_path}')


    summary_writer = SummaryWriter(log_dir=FLAGS.LOG_DIR, max_queue=1)
    global_step = 0

    update_batchnorm()

    train_loader_sequence = [make_dataloader(dset, True) for dset in train_dataset_sequence]
    train_loader_sequence_cycle = [icycle(loader) for loader in train_loader_sequence]
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.INIT_LR)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, linear_schedule(FLAGS.MAX_EPOCH*len(train_loader_sequence[0])))

    model.eval()

    pbar = trange(FLAGS.MAX_EPOCH*len(train_loader_sequence[0]), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i in pbar:
        optimizer.zero_grad()
        for t in range(len(train_dataset_sequence)):
            input, target = next(train_loader_sequence_cycle[t])
            input = input.cuda()
            target = target.cuda()
            output = model(input)[t]
            mse_loss = 0.5*criterion(output, 15.*F.one_hot(target, num_classes=10).float()).sum(-1).mean()
            loss = mse_loss/10
            loss.backward()
        weight_decay(model.named_parameters(), FLAGS.WEIGHT_DECAY)
        optimizer.step()
        lr_scheduler.step()

        if (global_step+1)%50 == 0:
            tprint(f'[TRAIN][{i}/{len(pbar)-1}] LR {lr_scheduler.get_last_lr()[-1]:.2e} | loss {loss.cpu().item():.3f}')
            # summary_writer.add_scalar('lr', lr_scheduler.get_last_lr()[-1], global_step=global_step)

        if (global_step+1)%150 == 0:
            evaluate_sequence(t)

        global_step += 1

    evaluate_sequence(t)


    if FLAGS.SAVE:
        save_pickle()

    summary_writer.close()


if __name__ == '__main__':
    main()
