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
from dataset import DataIncrementalTenfoldCIFAR100, DataIncrementalHundredfoldCIFAR100
import shutil
from kfac import KFACRegularizer, EWCRegularizer
from models.modules import CustomConv2d, CustomLinear, CustomBatchNorm2d
import torchvision.transforms.functional as VF


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

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
    LOG_DIR = '/workspace/runs/torch_rbu_cifar_87'
    BATCH_SIZE = 128
    # INIT_LR = 1e-4
    # WEIGHT_DECAY = 1e-5
    INIT_LR = 1e-3
    WEIGHT_DECAY = 1e-4
    # WEIGHT_DECAY = 0
    MAX_STEP = 8000
    N_WORKERS = 4
    BN_UPDATE_STEPS = 1000
    SAVE = True
    METHOD = None


if FLAGS.SAVE:
    shutil.copytree('./', FLAGS.LOG_DIR, dirs_exist_ok=False)


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


def weight_decay_origin(model: nn.Module, lam: float = 1e-4):
    for module in model.modules():
        if type(module) in (CustomLinear, CustomConv2d):
            module.weight_tangent.grad.data.add_(module.weight.data + module.weight_tangent.data, alpha=lam)
            if module.bias_tangent is not None:
                module.bias_tangent.grad.data.add_(module.bias.data + module.bias_tangent.data, alpha=lam)
        elif type(module) in (nn.Linear, nn.Conv2d):
            module.weight.grad.data.add_(module.weight.data, alpha=lam)
            if module.bias is not None:
                module.bias.grad.data.add_(module.bias.data, alpha=lam)


# maybe regularize weights only?


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


def freeze_parameters(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad_(False)


class MultiHeadWrapper(nn.Module):
    def __init__(self, module, n_heads, in_features, out_features):
        super().__init__()
        self.module = module
        self.heads = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(n_heads)])
        self.t = None

    def set_head(self, t: int):
        self.t = t

    def forward(self, x):
        x = self.module(x)
        if self.t is None:
            return [head(x) for head in self.heads]
        else:
            return self.heads[self.t](x)


class CustomMSELoss(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5*F.mse_loss(input, target, reduction=self.reduction)


class LWF:
    def __init__(self, teacher_model: MultiHeadWrapper, criterion: nn.Module):
        self.teacher_model = copy.deepcopy(teacher_model)
        self.criterion = criterion

    def compute_loss(self, input: torch.Tensor, pred: torch.Tensor):
        with torch.no_grad():
            target = self.teacher_model(input)
        loss = self.criterion(pred, target).sum(-1).mean()
        return loss


def initialize_model(model: MultiHeadWrapper, sample_input: torch.Tensor):
    output = model(sample_input)
    loss = sum(o.sum() for o in output)
    loss.backward()
    model.zero_grad()


def get_target_transform_fn(num_classes: int = 10, alpha: float = 15.0):
    def transform_fn(target: torch.Tensor) -> torch.Tensor:
        # assert target.dtype == torch.long and target.ndim == 1
        return F.one_hot(target, num_classes=num_classes).mul_(alpha).float()
    return transform_fn


def main():

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        ])
    transform_train_jittered = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.Grayscale(num_output_channels=3),
        T.Lambda(lambda x: VF.rotate(x, 90)),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        ])
    # num_classes = 10
    # target_transform = get_target_transform_fn(num_classes=num_classes, alpha=15.0)

    train_dataset_sequence = [DataIncrementalTenfoldCIFAR100(FLAGS.DATA_ROOT, task_id=i, train=True, transform=transform_train) for i in range(10)]
    # train_dataset_sequence = [DataIncrementalHundredfoldCIFAR100(FLAGS.DATA_ROOT, task_id=i, train=True, transform=transform_train) for i in range(100)]
    test_dataset = datasets.CIFAR100(FLAGS.DATA_ROOT, train=False, transform=transform_test)
    test_loader = make_dataloader(test_dataset, train=False)

    model = resnet18(num_classes=100)
    state_dict = torch.load(os.path.join(FLAGS.CHECKPOINT_DIR, 'state_dict.pt'))
    model.load_state_dict(state_dict, strict=False)
    model.cuda()

    # freeze_parameters(model.module)

    # initialize grad attributes to zeros
    initialize_model(model, torch.zeros((1, 3, 32, 32), device='cuda'))

    criterion = CustomMSELoss(reduction='none')

    @torch.no_grad()
    def update_batchnorm():
        data_loader = make_dataloader(datasets.CIFAR100(FLAGS.DATA_ROOT, train=True, transform=transform_train), True)
        data_loader_cycle = icycle(data_loader)
        model.train()
        for _ in trange(FLAGS.BN_UPDATE_STEPS):
            input, _ = next(data_loader_cycle)
            input = input.cuda()
            model(input)

    @torch.no_grad()
    def evaluate(data_loader):
        losses = []
        corrects = []
        model.eval()
        for input, target in data_loader:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, 15.*F.one_hot(target, num_classes=100).float()).sum(-1)
            losses.append(loss.view(-1))
            corrects.append((target == output.max(-1).indices).view(-1))
        avg_loss = torch.cat(losses).mean().item()
        avg_acc = torch.cat(corrects).float().mean().item()*100
        summary_writer.add_scalar('test_loss', avg_loss, global_step=global_step)
        summary_writer.add_scalar('test_acc', avg_acc, global_step=global_step)
        tprint(f"[TEST] loss {avg_loss:.3f} | T1acc {avg_acc:.2f}")

    def save_pickle():
        pickle = model.state_dict()
        pickle_path = os.path.join(FLAGS.LOG_DIR, f'state_dict.pt')
        torch.save(pickle, pickle_path)
        tprint(f'[SAVE] Saved to {pickle_path}')


    summary_writer = SummaryWriter(log_dir=FLAGS.LOG_DIR, max_queue=1)
    global_step = 0

    update_batchnorm()

    # regularizer = EWCRegularizer(model.module, criterion)
    regularizer_list = []

    if FLAGS.METHOD == 'ORACLE':
        train_loader_list = [icycle(make_dataloader(dset, True)) for dset in train_dataset_sequence]

    for t in trange(len(train_dataset_sequence)):
        train_loader = make_dataloader(train_dataset_sequence[t], True)
        train_loader_cycle = icycle(train_loader)
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.INIT_LR)
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, linear_schedule(FLAGS.MAX_STEP))

        if FLAGS.METHOD == 'LWF':
            regularizer = LWF(model, criterion)
        elif FLAGS.METHOD == 'LWF+KFAC':
            regularizer_lwf = LWF(model, criterion)
        elif FLAGS.METHOD == 'LWF+EWC':
            regularizer_lwf = LWF(model, criterion)

        model.eval()
        for i in range(max_step := FLAGS.MAX_STEP):
            optimizer.zero_grad()

            if FLAGS.METHOD == 'LWF':
                input, target = next(train_loader_cycle)
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                mse_loss = criterion(output, 15.*F.one_hot(target, num_classes=100).float()).sum(-1).mean()
                reg_loss = regularizer.compute_loss(input, output)
                loss = (mse_loss + reg_loss*t)/(t+1)
                loss.backward()
            elif FLAGS.METHOD == 'EWC':
                input, target = next(train_loader_cycle)
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                mse_loss = criterion(output, 15.*F.one_hot(target, num_classes=100).float()).sum(-1).mean()
                # reg_loss = regularizer.compute_loss()
                reg_loss = sum(r.compute_loss() for r in regularizer_list) * 1.0
                loss = (mse_loss + reg_loss)/(t+1)
                loss.backward()
            elif FLAGS.METHOD == 'KFAC':
                input, target = next(train_loader_cycle)
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                mse_loss = criterion(output, 15.*F.one_hot(target, num_classes=100).float()).sum(-1).mean()
                reg_loss = sum(r.compute_loss() for r in regularizer_list)
                loss = (mse_loss + reg_loss)/(t+1)
            elif FLAGS.METHOD == 'LWF+KFAC':
                input, target = next(train_loader_cycle)
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                mse_loss = criterion(output, 15.*F.one_hot(target, num_classes=100).float()).sum(-1).mean()
                reg_loss = regularizer_lwf.compute_loss(input, output, t) + sum(r.compute_loss() for r in regularizer_list)
                reg_loss /= 2
                loss = (mse_loss + reg_loss)/(t+1)
                loss.backward()
            elif FLAGS.METHOD == 'LWF+EWC':
                input, target = next(train_loader_cycle)
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                mse_loss = criterion(output, 15.*F.one_hot(target, num_classes=100).float()).sum(-1).mean()
                reg_loss = regularizer_lwf.compute_loss(input, output, t) + sum(r.compute_loss() for r in regularizer_list) * 1.0
                reg_loss /= 2
                loss = (mse_loss + reg_loss)/(t+1)
                loss.backward()
            elif FLAGS.METHOD is None:
                input, target = next(train_loader_cycle)
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                mse_loss = criterion(output, 15.*F.one_hot(target, num_classes=100).float()).sum(-1).mean()
                reg_loss = 0.
                loss = mse_loss
                loss.backward()
            elif FLAGS.METHOD == 'ORACLE':
                for j in range(t+1):
                    input, target = next(train_loader_list[j])
                    input = input.cuda()
                    target = target.cuda()
                    output = model(input)
                    mse_loss = criterion(output, 15.*F.one_hot(target, num_classes=100).float()).sum(-1).mean()
                    (mse_loss/(t+1)).backward()
            else:
                raise NotImplementedError(FLAGS.METHOD)
            # weight_decay(model.module.named_parameters(), FLAGS.WEIGHT_DECAY)
            weight_decay_origin(model, FLAGS.WEIGHT_DECAY)
            optimizer.step()
            lr_scheduler.step()

            if global_step%100 == 0:
                tprint(f'[TRAIN][{i}/{max_step}] LR {lr_scheduler.get_last_lr()[-1]:.2e} | {mse_loss.cpu().item():.3f} | {reg_loss:.3f}')
                summary_writer.add_scalar('lr', lr_scheduler.get_last_lr()[-1], global_step=global_step)

            if global_step%500 == 0:
                evaluate(test_loader)

            global_step += 1

        evaluate(test_loader)


        if FLAGS.METHOD == 'EWC':
            # compute ewc state
            # old_ewc_state = regularizer.state_dict()
            # regularizer.compute_curvature(train_dataset_sequence[t], t, n_steps=10000)
            # regularizer.merge_regularizer(old_ewc_state)

            # compute ewc state
            regularizer = EWCRegularizer(model, criterion, [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, CustomLinear, CustomConv2d, CustomBatchNorm2d))])
            regularizer.compute_curvature(train_dataset_sequence[t], n_steps=1000) # TODO: not fair becuase of batch sizes
            regularizer_list.append(regularizer)

        if FLAGS.METHOD == 'KFAC':
            # compute kfac state
            regularizer = KFACRegularizer(model, criterion, [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, CustomLinear, CustomConv2d, CustomBatchNorm2d))])
            regularizer.compute_curvature(train_dataset_sequence[t], n_steps=1000)
            regularizer_list.append(regularizer)

        elif FLAGS.METHOD == 'LWF+KFAC':
            regularizer = KFACRegularizer(model, criterion, [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, CustomLinear, CustomConv2d, CustomBatchNorm2d))])
            regularizer.compute_curvature(train_dataset_sequence[t], n_steps=1000)
            regularizer_list.append(regularizer)

        elif FLAGS.METHOD == 'LWF+EWC':
            regularizer = EWCRegularizer(model, criterion, [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, CustomLinear, CustomConv2d, CustomBatchNorm2d))])
            regularizer.compute_curvature(train_dataset_sequence[t], n_steps=1000)
            regularizer_list.append(regularizer)


    if FLAGS.SAVE:
        save_pickle()

    summary_writer.close()


if __name__ == '__main__':
    main()
