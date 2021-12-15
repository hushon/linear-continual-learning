from typing import NamedTuple
import torch
from torch import nn, utils, optim, cuda
import torchvision.transforms as T
import torch.nn.functional as F
import os
from tqdm import tqdm, trange
import numpy as np
import random
from torch.nn.parallel import DataParallel
from torchvision import datasets
import atexit
from PIL import Image
import copy
from torch.utils.tensorboard.writer import SummaryWriter
from utils import MultiEpochsDataLoader
from dataset import TaskIncrementalTenfoldCIFAR100
import shutil
from kfac import KFACRegularizer, EWCRegularizer, get_center_dict, EKFACRegularizer
from models.modules import CustomConv2d, CustomLinear, CustomBatchNorm2d
import torchvision.transforms.functional as VF
from utils import get_log_dir, get_timestamp, SlackWriter
from regularizer import MASRegularizer
from tkfac import TKFACRegularizer


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
    CHECKPOINT_DIR = '/workspace/runs/temp111' # ImageNet32, ResNet18 pretrained
    # CHECKPOINT_DIR = '/workspace/runs/pretrain_tinyimagenet_resnet18' # TinyImageNet, ResNet18 pretrained
    LOG_DIR = '/workspace/runs'
    BATCH_SIZE = 128
    # INIT_LR = 1e-4
    INIT_LR = 1e-3
    WEIGHT_DECAY = 1e-5
    # WEIGHT_DECAY = 0
    MAX_STEP = 50000
    N_WORKERS = 4
    BN_UPDATE_STEPS = 0
    SAVE = True
    METHOD = 'KFAC'
    OPTIM = 'SGD'
    LOSS_FN = 'SCE'
    LINEARIZED = False
    TRACK_BN = False
    WEBHOOKS_URL = "https://hooks.slack.com/services/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"
    LR_SCHEDULE = 'LINEAR'
    CLIP_GRAD = 1.0
    LOG_HISTOGRAM = False
    DAMPING = 1e-4


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
        if type(module) in (nn.Linear, nn.Conv2d):
            module.weight.grad.data.add_(module.weight.data, alpha=lam)
            # if module.bias is not None:
            #     module.bias.grad.data.add_(module.bias.data, alpha=lam)
        elif type(module) in (CustomLinear, CustomConv2d):
            module.weight_tangent.grad.data.add_(module.weight.data + module.weight_tangent.data, alpha=lam)
            # if module.bias_tangent is not None:
            #     module.bias_tangent.grad.data.add_(module.bias.data + module.bias_tangent.data, alpha=lam)


def damping(params_orig, params, lam):
    for p_o, p_p in zip(params_orig, params):
        p_p.grad.data.add_(p_p.data - p_o.data, alpha=lam)


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

    def forward(self, x):
        x = self.module(x)
        return [head(x) for head in self.heads]


class CustomMSELoss(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5*F.mse_loss(input, target, reduction=self.reduction)


class LWF:
    def __init__(self, teacher_model: MultiHeadWrapper, loss: str):
        self.teacher_model = copy.deepcopy(teacher_model)
        self.loss = loss

    def compute_loss(self, input: list, pred: list, t: int):
        with torch.no_grad():
            target = self.teacher_model(input)

        if self.loss == 'kl':
            loss = sum(F.kl_div(pred[i].log_softmax(1), target[i].softmax(1), reduction='none').sum(1).mean(0) for i in range(t))
        elif self.loss == 'mse':
            loss = sum(0.5*F.mse_loss(pred[i], target[i], reduction='none').sum(1).mean(0) for i in range(t))
        else:
            raise ValueError
        return loss

def initialize_model(model: MultiHeadWrapper, sample_input: torch.Tensor):
    output = model(sample_input)
    loss = sum(o.sum() for o in output)
    loss.backward()
    model.zero_grad()


def get_target_transform_fn(num_classes: int = 10, alpha: float = 15.0):
    def transform_fn(target: torch.Tensor) -> torch.Tensor:
        # assert target.dtype == torch.long and target.ndim == 1
        return alpha*F.one_hot(target, num_classes=num_classes).float()
    return transform_fn


def main():
    timestamp = get_timestamp()
    log_dir = os.path.join(FLAGS.LOG_DIR, timestamp)
    summary_writer = SummaryWriter(log_dir=log_dir, max_queue=1)
    print(f"{log_dir=}")

    # if FLAGS.SAVE:
    #     shutil.copytree('./', log_dir, dirs_exist_ok=True)

    slack_writer = SlackWriter(FLAGS.WEBHOOKS_URL)
    slack_writer.write(f"started {timestamp}")
    atexit.register(slack_writer.write, f"exited {timestamp}")


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
    num_classes = 10
    target_transform = get_target_transform_fn(num_classes=num_classes, alpha=15.0)

    train_dataset_sequence = [TaskIncrementalTenfoldCIFAR100(FLAGS.DATA_ROOT, task_id=i, train=True, transform=transform_train) for i in range(10)]
    test_dataset_sequence = [TaskIncrementalTenfoldCIFAR100(FLAGS.DATA_ROOT, task_id=i, train=False, transform=transform_test) for i in range(10)]
    test_loader_sequence = [make_dataloader(dset, train=False) for dset in test_dataset_sequence]

    if FLAGS.LINEARIZED:
        from models.resnet_cifar100_jvplrelu import resnet18, resnet50
    else:
        from models.resnet_cifar100_lrelu import resnet18, resnet50

    model = resnet18(num_classes=0)
    state_dict = torch.load(os.path.join(FLAGS.CHECKPOINT_DIR, 'state_dict.pt'))
    model.load_state_dict(state_dict, strict=False)
    model = MultiHeadWrapper(model, 10, 512, 10)
    model.cuda()

    # freeze_parameters(model.module)

    # initialize grad attributes to zeros
    initialize_model(model, torch.zeros((1, 3, 32, 32), device='cuda'))


    @torch.no_grad()
    def update_batchnorm():
        data_loader = make_dataloader(datasets.CIFAR100(FLAGS.DATA_ROOT, train=True, transform=transform_train), True)
        data_loader_cycle = icycle(data_loader)
        model.train()
        for _ in trange(FLAGS.BN_UPDATE_STEPS, desc="update batchnorm"):
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
            if FLAGS.LOSS_FN == 'SCE':
                loss = F.cross_entropy(output, target, reduction='none')
            elif FLAGS.LOSS_FN == 'MSE':
                loss = 0.5*F.mse_loss(output, target_transform(target), reduction='none').sum(1)
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
        pickle_path = os.path.join(log_dir, f'state_dict.pt')
        torch.save(pickle, pickle_path)
        tprint(f'[SAVE] Saved to {pickle_path}')

    def add_summary_histograms(model: nn.Module):
        for name, param in model.named_parameters():
            summary_writer.add_histogram(f"param/{name}", param.data, global_step=global_step)
            summary_writer.add_histogram(f"grad/{name}", param.grad.data, global_step=global_step)

    global_step = 0

    update_batchnorm()

    max_steps = [2*FLAGS.MAX_STEP] + [FLAGS.MAX_STEP]*9
    init_lrs = [FLAGS.INIT_LR] + [FLAGS.INIT_LR]*9

    for t in range(len(train_dataset_sequence)):
        train_loader = make_dataloader(train_dataset_sequence[t], True)
        train_loader_cycle = icycle(train_loader)

        trainable_params = [p for n, p in model.named_parameters() if 'bn' not in n]
        # trainable_params = list(model.parameters())

        params_orig = copy.deepcopy(trainable_params)

        if FLAGS.OPTIM == 'SGD':
            optimizer = optim.SGD(trainable_params, lr=init_lrs[t], momentum=0.9)
        elif FLAGS.OPTIM == 'ADAM':
            optimizer = optim.Adam(trainable_params, lr=init_lrs[t])
        else:
            raise NotImplementedError

        if FLAGS.LR_SCHEDULE == 'LINEAR':
            lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, linear_schedule(max_steps[t]))
        elif FLAGS.LR_SCHEDULE == 'COS':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_steps[t])
        else:
            raise NotImplementedError

        if FLAGS.METHOD == 'LWF':
            if FLAGS.LOSS_FN == 'SCE':
                regularizer = LWF(model, loss='kl')
            elif FLAGS.LOSS_FN == 'MSE':
                regularizer = LWF(model, loss='mse')

        model.train(FLAGS.TRACK_BN)
        for i in (pbar := trange(max_steps[t])):
            if t == 0:
                msg = model.load_state_dict(torch.load('./checkpoint/tenfold_task/state_dict.pt'), strict=False)
                tprint(msg)
                break

            input, target = next(train_loader_cycle)
            input = input.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output = model(input)
            if FLAGS.LOSS_FN == 'SCE':
                data_loss = F.cross_entropy(output[t], target, reduction='none').mean(0)
            elif FLAGS.LOSS_FN == 'MSE':
                data_loss = 0.5*F.mse_loss(output[t], target_transform(target), reduction='none').sum(1).mean(0)

            if FLAGS.METHOD == 'LWF':
                if t > 0:
                    reg_loss = regularizer.compute_loss(input, output, t)
                else:
                    reg_loss = 0.
                loss = (data_loss + reg_loss)/(t+1)
            elif FLAGS.METHOD == 'EWC':
                if t > 0:
                    reg_loss = regularizer.compute_loss(center_dict) * t
                else:
                    reg_loss = 0.
                loss = (data_loss + reg_loss)/(t+1)
            elif FLAGS.METHOD == 'KFAC':
                if t > 0:
                    reg_loss = regularizer.compute_loss(center_dict) * t
                else:
                    reg_loss = 0.
                loss = (data_loss + reg_loss)/(t+1)
            elif FLAGS.METHOD == 'EKFAC':
                if t > 0:
                    reg_loss = regularizer.compute_loss(center_dict) * t
                else:
                    reg_loss = 0.
                loss = (data_loss + reg_loss)/(t+1)
            elif FLAGS.METHOD == 'TKFAC':
                if t > 0:
                    reg_loss = regularizer.compute_loss(center_dict) * t
                else:
                    reg_loss = 0.
                loss = (data_loss + reg_loss)/(t+1)
            elif FLAGS.METHOD == 'LWF+KFAC':
                lwf_loss = regularizer_lwf.compute_loss(input, output, t)
                reg_loss = sum(r.compute_loss() for r in regularizer_list)
                reg_loss /= 2
                loss = (data_loss + reg_loss)/(t+1)
            elif FLAGS.METHOD == 'LWF+EWC':
                reg_loss = regularizer_lwf.compute_loss(input, output, t) + sum(r.compute_loss() for r in regularizer_list) * 1.0
                reg_loss /= 2
                loss = (data_loss + reg_loss)/(t+1)
            elif FLAGS.METHOD == 'MAS':
                if t>0:
                    reg_loss = regularizer.compute_loss(center_dict) * t
                else:
                    reg_loss = 0.
                loss = (data_loss + reg_loss)/(t+1)
            elif FLAGS.METHOD == None:
                reg_loss = 0.
                loss = data_loss
            else:
                raise NotImplementedError(FLAGS.METHOD)

            loss.backward()
            weight_decay(model.module.named_parameters(), FLAGS.WEIGHT_DECAY)
            # weight_decay_origin(model.module, FLAGS.WEIGHT_DECAY)
            weight_decay(model.heads[t].named_parameters(), FLAGS.WEIGHT_DECAY)

            if FLAGS.METHOD in ('EWC', 'KFAC', 'MAS'):
                damping(params_orig, trainable_params, lam=FLAGS.DAMPING)

            if FLAGS.CLIP_GRAD:
                nn.utils.clip_grad_norm_(trainable_params, FLAGS.CLIP_GRAD)

            optimizer.step()
            lr_scheduler.step()

            if (global_step+1)%200 == 0:
                tprint(f'[TRAIN][{i}/{len(pbar)}] LR {lr_scheduler.get_last_lr()[-1]:.2e} | {data_loss:.3f} | {reg_loss:.3f}')
                summary_writer.add_scalar('train_loss', data_loss, global_step=global_step)
                summary_writer.add_scalar('reg_loss', reg_loss, global_step=global_step)
                summary_writer.add_scalar('lr', lr_scheduler.get_last_lr()[-1], global_step=global_step)
                if FLAGS.LOG_HISTOGRAM:
                    add_summary_histograms(model)

            if (global_step+1)%500 == 0:
                evaluate_sequence(t)
                model.train(FLAGS.TRACK_BN)

            global_step += 1

        # ###
        # if t == 0:
        #     save_pickle()
        #     import sys
        #     sys.exit()
        # ###


        if FLAGS.METHOD == 'EWC':
            if FLAGS.LOSS_FN == 'SCE':
                criterion = lambda logit, target: F.cross_entropy(logit, target, reduction='none')
                pseudo_target_fn = lambda logit: torch.distributions.Categorical(logit.softmax(1)).sample()
            elif FLAGS.LOSS_FN == 'MSE':
                criterion = lambda logit, target: 0.5*F.mse_loss(logit, target, reduction='none').sum(1)
                pseudo_target_fn = torch.normal
            modules_to_regularize = [m for m in model.module.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
            # modules_to_regularize = [m for m in model.module.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d))]
            tprint(modules_to_regularize)
            if t == 0:
                regularizer = EWCRegularizer(model, criterion, modules_to_regularize)
                regularizer.compute_curvature(train_dataset_sequence[t], 1000, t, pseudo_target_fn=pseudo_target_fn)
                center_dict = get_center_dict(modules_to_regularize)
            else:
                new_regularizer = EWCRegularizer(model, criterion, modules_to_regularize)
                new_regularizer.compute_curvature(train_dataset_sequence[t], 1000, t, pseudo_target_fn=pseudo_target_fn)
                center_dict = get_center_dict(modules_to_regularize)
                for old_ewc_state, new_ewc_state in zip(regularizer.ewc_state_dict.values(), new_regularizer.ewc_state_dict.values()):
                    old_ewc_state.G_weight = (old_ewc_state.G_weight*t + new_ewc_state.G_weight)/(t+1)
                    if old_ewc_state.G_bias is not None:
                        old_ewc_state.G_bias = (old_ewc_state.G_bias*t + new_ewc_state.G_bias)/(t+1)

        if FLAGS.METHOD == 'KFAC':
            if FLAGS.LOSS_FN == 'SCE':
                criterion = lambda logit, target: F.cross_entropy(logit, target, reduction='none')
                pseudo_target_fn = lambda logit: torch.distributions.Categorical(logit.softmax(1)).sample()
            elif FLAGS.LOSS_FN == 'MSE':
                criterion = lambda logit, target: 0.5*F.mse_loss(logit, target, reduction='none').sum(1)
                pseudo_target_fn = torch.normal
            modules_to_regularize = [m for m in model.module.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
            # modules_to_regularize = [m for m in model.module.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d))]
            tprint(modules_to_regularize)
            if t == 0:
                regularizer = KFACRegularizer(model, criterion, modules_to_regularize)
                regularizer.compute_curvature(train_dataset_sequence[t], 1000, t, pseudo_target_fn=pseudo_target_fn)
                center_dict = get_center_dict(modules_to_regularize)
            else:
                new_regularizer = KFACRegularizer(model, criterion, modules_to_regularize)
                new_regularizer.compute_curvature(train_dataset_sequence[t], 1000, t, pseudo_target_fn=pseudo_target_fn)
                center_dict = get_center_dict(modules_to_regularize)
                for old_kfac_state, new_kfac_state in zip(regularizer.kfac_state_dict.values(), new_regularizer.kfac_state_dict.values()):
                    old_kfac_state.S = (old_kfac_state.S*t + new_kfac_state.S)/(t+1)
                    old_kfac_state.A = (old_kfac_state.A*t + new_kfac_state.A)/(t+1)


        if FLAGS.METHOD == 'EKFAC':
            if FLAGS.LOSS_FN == 'SCE':
                criterion = lambda logit, target: F.cross_entropy(logit, target, reduction='none')
                pseudo_target_fn = lambda logit: torch.distributions.Categorical(logit.softmax(1)).sample()
            elif FLAGS.LOSS_FN == 'MSE':
                criterion = lambda logit, target: 0.5*F.mse_loss(logit, target, reduction='none').sum(1)
                pseudo_target_fn = torch.normal
            modules_to_regularize = [m for m in model.module.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
            # modules_to_regularize = [m for m in model.module.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d))]
            tprint(modules_to_regularize)
            if t == 0:
                regularizer = EKFACRegularizer(model, criterion, modules_to_regularize)
                regularizer.compute_curvature(train_dataset_sequence[t], 1000, t, batch_size=128, pseudo_target_fn=pseudo_target_fn)
                center_dict = get_center_dict(modules_to_regularize)
            else:
                new_regularizer = EKFACRegularizer(model, criterion, modules_to_regularize)
                new_regularizer.compute_curvature(train_dataset_sequence[t], 1000, t, batch_size=128, pseudo_target_fn=pseudo_target_fn)
                center_dict = get_center_dict(modules_to_regularize)
                for module in regularizer.modules:
                    kfac_state = regularizer.kfac_state_dict[module]
                    ekfac_state = regularizer.ekfac_state_dict[module]
                    new_kfac_state = new_regularizer.kfac_state_dict[module]
                    new_ekfac_state = new_regularizer.ekfac_state_dict[module]
                    # update kfac A, S
                    kfac_state.A = (kfac_state.A*t + new_kfac_state.A)/(t+1)
                    kfac_state.S = (kfac_state.S*t + new_kfac_state.S)/(t+1)
                    # update KFE
                    ekfac_state.Q_A = torch.symeig(kfac_state.A, eigenvectors=True).eigenvectors
                    ekfac_state.Q_S = torch.symeig(kfac_state.S, eigenvectors=True).eigenvectors
                    # update scaling
                    ekfac_state.scale = (ekfac_state.scale*t + new_ekfac_state.scale)/(t+1)
                # del new_regularizer
        if FLAGS.METHOD == 'TKFAC':
            if FLAGS.LOSS_FN == 'SCE':
                criterion = lambda logit, target: F.cross_entropy(logit, target, reduction='none')
                pseudo_target_fn = lambda logit: torch.distributions.Categorical(logit.softmax(1)).sample()
            elif FLAGS.LOSS_FN == 'MSE':
                criterion = lambda logit, target: 0.5*F.mse_loss(logit, target, reduction='none').sum(1)
                pseudo_target_fn = torch.normal
            modules_to_regularize = [m for m in model.module.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
            # modules_to_regularize = [m for m in model.module.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d))]
            tprint(modules_to_regularize)
            if t == 0:
                regularizer = TKFACRegularizer(model, criterion, modules_to_regularize)
                regularizer.compute_curvature(train_dataset_sequence[t], 1000, t, pseudo_target_fn=pseudo_target_fn)
                center_dict = get_center_dict(modules_to_regularize)
            else:
                new_regularizer = TKFACRegularizer(model, criterion, modules_to_regularize)
                new_regularizer.compute_curvature(train_dataset_sequence[t], 1000, t, pseudo_target_fn=pseudo_target_fn)
                center_dict = get_center_dict(modules_to_regularize)
                for old_tkfac_state, new_tkfac_state in zip(regularizer.tkfac_state_dict.values(), new_regularizer.tkfac_state_dict.values()):
                    old_tkfac_state.A = (old_tkfac_state.A*t + new_tkfac_state.A)/(t+1)
                    old_tkfac_state.S = (old_tkfac_state.S*t + new_tkfac_state.S)/(t+1)


        elif FLAGS.METHOD == 'LWF+KFAC':
            regularizer = KFACRegularizer(model, criterion, [m for m in model.modules() if isinstance(m, (CustomLinear, CustomConv2d, CustomBatchNorm2d))])
            regularizer.compute_curvature(train_dataset_sequence[t], 1000, t)
            regularizer_list.append(regularizer)

        elif FLAGS.METHOD == 'LWF+EWC':
            regularizer = EWCRegularizer(model, criterion, [m for m in model.modules() if isinstance(m, (CustomLinear, CustomConv2d, CustomBatchNorm2d))])
            regularizer.compute_curvature(train_dataset_sequence[t], 1000, t)
            regularizer_list.append(regularizer)

        elif FLAGS.METHOD == 'MAS':
            modules_to_regularize = [m for m in model.module.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
            # modules_to_regularize = [m for m in model.module.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d))]
            tprint(modules_to_regularize)
            if t == 0:
                regularizer = MASRegularizer(model, modules_to_regularize)
                regularizer.compute_importance(train_dataset_sequence[t], 1000, t)
                center_dict = get_center_dict(modules_to_regularize)
            else:
                new_regularizer = MASRegularizer(model, modules_to_regularize)
                new_regularizer.compute_importance(train_dataset_sequence[t], 1000, t)
                center_dict = get_center_dict(modules_to_regularize)
                for module in regularizer.modules:
                    mas_state = regularizer.mas_state_dict[module]
                    new_mas_state = new_regularizer.mas_state_dict[module]
                    mas_state.O_weight = (mas_state.O_weight*t + new_mas_state.O_weight) / (t+1)
                    if mas_state.O_bias is not None:
                        mas_state.O_bias = (mas_state.O_bias*t + new_mas_state.O_bias) / (t+1)

    if FLAGS.SAVE:
        save_pickle()

    summary_writer.close()


if __name__ == '__main__':
    main()
