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
import shutil
from kfac import KFACRegularizer
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
    LOG_DIR = '/workspace/runs/torch_rbu_cifar_68'
    BATCH_SIZE = 128
    INIT_LR = 1e-4
    WEIGHT_DECAY = 1e-5
    # WEIGHT_DECAY = 0
    MAX_STEP = 8000
    N_WORKERS = 4
    BN_UPDATE_STEPS = 1000
    SAVE = True


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
        if isinstance(module, (CustomLinear, CustomConv2d)):
            module.weight_tangent.grad.data.add_(module.weight.data + module.weight_tangent.data, alpha=lam)
            if module.bias_tangent is not None:
                module.bias_tangent.grad.data.add_(module.bias.data + module.bias_tangent.data, alpha=lam)


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
        self.head = None

    def forward(self, x):
        x = self.module(x)
        return [head(x) for head in self.heads]


class CustomMSELoss(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5*F.mse_loss(input, target, reduction=self.reduction)


class EWC:
    def __init__(self, model: nn.Module, criterion: nn.Module):
        self.model = model
        self.criterion = criterion
        self._reset_state()

    def _reset_state(self):
        self.importance = [torch.zeros_like(param) for param in self.model.parameters()]
        self.center = [torch.zeros_like(param) for param in self.model.parameters()]

    def _accumulate_curvature_step(self):
        for m_param, ewc_param in zip(self.model.parameters(), self.importance):
            ewc_param.add_(m_param.grad.data.square())

    def _update_center(self):
        self.center = [param.data.detach().clone() for param in self.model.parameters()]

    def compute_curvature(self, dataset: torch.utils.data.Dataset, t, n_steps):
        self._reset_state()
        self.model.register_full_backward_hook(self._backward_hook)
        data_loader_cycle = icycle(make_dataloader(dataset, False, True))
        self.model.eval()
        for _ in trange(n_steps):
            input, _ = next(data_loader_cycle)
            input = input.cuda()
            self.model.zero_grad()
            output = self.model(input)[t]
            pseudo_target = torch.normal(output.detach()) #TODO
            loss = self.criterion(output, pseudo_target).sum(-1).squeeze() #TODO
            loss.backward()
            self._accumulate_curvature_step()
        for ewc_param in self.importance:
            ewc_param.div_(n_steps)
        self._update_center()

    def compute_loss(self):
        loss = 0.
        for i, p, c in zip(self.importance, self.model.parameters(), self.center):
            loss += (i * torch.square(p - c)).sum()
        return 0.5*loss

    def merge_regularizer(self, old_state_dict):
        old_importance = old_state_dict['importance']
        self.importance = [x + y for x, y in zip(self.importance, old_importance)]

    def state_dict(self):
        return {
            'importance': copy.deepcopy(self.importance),
            'center': copy.deepcopy(self.center)
        }

    def load_state_dict(self, d: dict):
        self.importance = d['importance']
        self.center = d['center']


class LWF:
    def __init__(self, teacher_model: MultiHeadWrapper, criterion: nn.Module):
        self.teacher_model = copy.deepcopy(teacher_model)
        self.criterion = criterion

    def compute_loss(self, input: list, pred: list, t: int):
        with torch.no_grad():
            target = self.teacher_model(input)
        loss = sum(self.criterion(pred[i], target[i]).sum(-1).mean() for i in range(t))
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

    train_dataset_sequence = [TaskIncrementalTenfoldCIFAR100(FLAGS.DATA_ROOT, task_id=i, train=True, transform=transform_train) for i in range(10)]
    # train_dataset_sequence = [TaskIncrementalTenfoldCIFAR100(FLAGS.DATA_ROOT, task_id=i, train=True, transform=transform_train if i%2==0 else transform_train_jittered) for i in range(10)]
    test_dataset_sequence = [TaskIncrementalTenfoldCIFAR100(FLAGS.DATA_ROOT, task_id=i, train=False, transform=transform_test) for i in range(10)]
    test_loader_sequence = [make_dataloader(dset, train=False) for dset in test_dataset_sequence]

    model = resnet18(num_classes=0)
    state_dict = torch.load(os.path.join(FLAGS.CHECKPOINT_DIR, 'state_dict.pt'))
    model.load_state_dict(state_dict, strict=False)
    model = MultiHeadWrapper(model, 10, 512, 10)
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
    def evaluate(data_loader, t):
        losses = []
        corrects = []
        model.eval()
        for input, target in data_loader:
            input = input.cuda()
            target = target.cuda()
            output = model(input)[t]
            loss = criterion(output, 15.*F.one_hot(target, num_classes=10).float()).sum(-1)
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

    train_loader_sequence = [make_dataloader(dset, train=True) for dset in train_dataset_sequence]
    train_loader_sequence_cycle = [icycle(x) for x in train_loader_sequence]
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.INIT_LR)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, linear_schedule(len(train_dataset_sequence)*FLAGS.MAX_STEP))

    model.eval()
    for i in (pbar := trange(len(train_dataset_sequence)*FLAGS.MAX_STEP)):
        t = i%10

        optimizer.zero_grad()
        loader = train_loader_sequence_cycle[t]
        input, target = next(loader)
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        mse_loss = criterion(output[t], 15.*F.one_hot(target, num_classes=10).float()).sum(-1).mean()
        loss = mse_loss
        loss.backward()

        # weight_decay(model.module.named_parameters(), FLAGS.WEIGHT_DECAY)
        weight_decay_origin(model.module, FLAGS.WEIGHT_DECAY)

        optimizer.step()
        lr_scheduler.step()

        if global_step%100 == 0:
            tprint(f'[TRAIN][{i}/{len(pbar)}] LR {lr_scheduler.get_last_lr()[-1]:.2e} | {mse_loss.cpu().item():.3f}')
            summary_writer.add_scalar('lr', lr_scheduler.get_last_lr()[-1], global_step=global_step)

        if global_step%500 == 0:
            evaluate_sequence(9)

        global_step += 1

    evaluate_sequence(9)

    if FLAGS.SAVE:
        save_pickle()

    summary_writer.close()


if __name__ == '__main__':
    main()
