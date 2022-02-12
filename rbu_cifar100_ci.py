from typing import NamedTuple
import torch
from torch import nn, utils, optim, cuda
import torchvision.transforms as T
import torch.nn.functional as F
import os
from tqdm.auto import tqdm, trange
import numpy as np
import random
from torch.nn.parallel import DataParallel
from torchvision import datasets
import atexit
from PIL import Image
import copy
from torch.utils.tensorboard.writer import SummaryWriter
from utils import MultiEpochsDataLoader
from dataset import DataIncrementalTenfoldCIFAR100, DataIncrementalHundredfoldCIFAR100, DataIncrementalFourfoldMIT67, MIT67, ClassIncrementalTenfoldCIFAR100, ClassIncrementalFivefoldCIFAR10
import shutil
from kfac import KFACRegularizer, EWCRegularizer, EKFACRegularizer, get_center_dict
from regularizer import MASRegularizer
from models.modules import CustomConv2d, CustomLinear, CustomBatchNorm2d
import torchvision.transforms.functional as VF
from utils import get_timestamp, image_loader, SlackWriter
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
    CHECKPOINT_PATH = '/workspace/runs/temp111/state_dict.pt' # resnet18 trained on ImageNet32
    # CHECKPOINT_PATH = '/workspace/runs/pretrain_tinyimagenet_resnet18/state_dict.pt' # TinyImageNet, ResNet18 pretrained
    # CHECKPOINT_PATH = './checkpoint/imagenet_resnet18_lrelu_lr0.001/model_best.pt' # resnet18 lrelu trained on ImageNet
    LOG_DIR = '/workspace/runs/'
    BATCH_SIZE = 128
    INIT_LR = 1e-4
    # INIT_LR = 1e-3
    WEIGHT_DECAY = 1e-5
    # WEIGHT_DECAY = 0.
    # MAX_STEP = 100
    # MAX_STEP = 5000
    MAX_STEP = 10000
    # MAX_STEP = 50000
    N_WORKERS = 8
    BN_UPDATE_STEPS = 0
    SAVE = True
    METHOD = 'TKFAC'
    OPTIM = 'ADAM'
    LOSS_FN = 'MSE'
    LINEARIZED = True
    TRACK_BN = False
    LR_SCHEDULE = 'LINEAR'
    DAMPING = 1e-4
    CLIP_GRAD = 1.0
    LOG_HISTOGRAM = False
    WEBHOOKS_URL = "https://hooks.slack.com/services/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"


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


def weight_decay_origin(model: nn.Module, lam: float):
    for module in model.modules():
        try:
            if type(module) in (nn.Linear, nn.Conv2d):
                module.weight.grad.data.add_(module.weight.data, alpha=lam)
                # if module.bias is not None:
                #     module.bias.grad.data.add_(module.bias.data, alpha=lam)
            elif type(module) in (CustomLinear, CustomConv2d):
                module.weight_tangent.grad.data.add_(module.weight.data + module.weight_tangent.data, alpha=lam)
                # if module.bias_tangent is not None:
                #     module.bias_tangent.grad.data.add_(module.bias.data + module.bias_tangent.data, alpha=lam)
        except:
            pass


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


def initialize_model(model: nn.Module, sample_input: torch.Tensor):
    output = model(sample_input)
    output.sum().backward()
    model.zero_grad()


def get_target_transform_fn(num_classes: int = 10, alpha: float = 15.0):
    def transform_fn(target: torch.Tensor) -> torch.Tensor:
        # assert target.dtype == torch.long and target.ndim == 1
        return alpha*F.one_hot(target, num_classes=num_classes).float()
    return transform_fn


def main():
    timestamp = get_timestamp()
    log_dir = os.path.join(FLAGS.LOG_DIR, timestamp)
    # if FLAGS.SAVE:
    #     shutil.copytree('./', log_dir, dirs_exist_ok=True)
    summary_writer = SummaryWriter(log_dir=log_dir, flush_secs=30)
    print(f"{log_dir=}")

    slack_writer = SlackWriter(FLAGS.WEBHOOKS_URL)
    slack_writer.write(f"started {timestamp}")
    atexit.register(slack_writer.write, f"exited {timestamp}")

    # transform_train = T.Compose([
    #     T.RandomCrop(32, padding=4),
    #     T.RandomHorizontalFlip(),
    #     T.ToTensor(),
    #     T.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    #     ])
    # transform_test = T.Compose([
    #     T.ToTensor(),
    #     T.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    #     ])
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])

    # train_dataset_sequence = [ClassIncrementalTenfoldCIFAR100(FLAGS.DATA_ROOT, task_classes=list(range(10*i, 10*(i+1))), train=True, transform=transform_train) for i in range(10)]
    train_dataset_sequence = [ClassIncrementalFivefoldCIFAR10(FLAGS.DATA_ROOT, task_classes=list(range(2*i, 2*(i+1))), train=True, transform=transform_train, download=True) for i in range(5)]

    # test_loader_sequence = [make_dataloader(dset, train=False) for dset in test_dataset_sequence]


    if FLAGS.LINEARIZED:
        # from models.resnet_cifar100_jvplrelu import resnet18, resnet50
        # from models.resnet_imagenet_jvplrelu import resnet18, resnet50
        from models.resnet_cifar100_jvplrelu_bn_merged import resnet18, resnet50
    else:
        from models.resnet_cifar100_lrelu import resnet18, resnet50
        # from models.resnet_imagenet_lrelu import resnet18, resnet50
    # model = resnet18(num_classes=100)
    # model = resnet18(num_classes=10)
    model = resnet18(num_classes=2)
    # missing_keys, unexpected_keys = model.load_state_dict(torch.load(FLAGS.CHECKPOINT_PATH), strict=False)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load("./resnet18_bn_merged.pt"), strict=False)
    model.cuda()

    nn.init.zeros_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)

    tprint(f'{missing_keys=}')
    tprint(f'{unexpected_keys=}')

    # freeze_parameters(model)
    # model.fc.weight.requires_grad_()
    # model.fc.bias.requires_grad_()


    # initialize grad attributes to zeros
    initialize_model(model, torch.zeros((1, 3, 32, 32), device='cuda'))
    # initialize_model(model, torch.zeros((1, 3, 224, 224), device='cuda'))


    @torch.no_grad()
    def update_batchnorm():
        data_loader = make_dataloader(datasets.CIFAR100(FLAGS.DATA_ROOT, train=True, transform=transform_train), True)
        # data_loader = make_dataloader(MIT67(os.path.join(FLAGS.DATA_ROOT, 'MIT67'), train=True, transform=transform_train, loader=image_loader), True)
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
            if FLAGS.LOSS_FN == 'SCE':
                loss = F.cross_entropy(output, target, reduction='none')
            elif FLAGS.LOSS_FN == 'MSE':
                loss = 0.5*F.mse_loss(output, target_transform(target), reduction='none').sum(1)
            losses.append(loss.view(-1))
            corrects.append((target == output.max(-1).indices).view(-1))
            if t>0:
                breakpoint()
        avg_loss = torch.cat(losses).mean().item()
        avg_acc = torch.cat(corrects).float().mean().item()*100
        summary_writer.add_scalar('test_loss', avg_loss, global_step=global_step)
        summary_writer.add_scalar('test_acc', avg_acc, global_step=global_step)
        tprint(f"[TEST] loss {avg_loss:.3f} | T1acc {avg_acc:.2f}")

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

    max_steps = [FLAGS.MAX_STEP] + [FLAGS.MAX_STEP]*99
    init_lrs = [FLAGS.INIT_LR] + [FLAGS.INIT_LR]*99

    for t in trange(len(train_dataset_sequence), desc=f'{timestamp}', dynamic_ncols=True):
        # if t > 0:
        #     # add classification units
        #     weight_temp = torch.cat([model.fc.weight.data, torch.zeros((10, model.fc.weight.data.shape[1]), device='cuda')], dim=0)
        #     bias_temp = torch.cat([model.fc.bias.data, torch.zeros((10,), device='cuda')], dim=0)
        #     model.fc.__init__(model.fc.in_features, model.fc.out_features + 10)
        #     model.fc.cuda()
        #     model.fc.weight.data.copy_(weight_temp)
        #     model.fc.bias.data.copy_(bias_temp)

        #     # modify kfac state
        #     if FLAGS.METHOD == 'KFAC':
        #         kfac_state = list(regularizer.kfac_state_dict.values())[-1]
        #         temp_S = torch.eye(kfac_state.S.shape[0] + 10, device='cuda')
        #         # temp_S[:kfac_state.S.shape[0], :kfac_state.S.shape[0]].copy_(kfac_state.S)
        #         kfac_state.S = temp_S
        #         center_state = list(center_dict.values())[-1]
        #         center_state.weight = torch.cat([center_state.weight, torch.zeros((10, center_state.weight.shape[1]), device='cuda')], dim=0)
        #         center_state.bias = torch.cat([center_state.bias, torch.zeros((10,), device='cuda')], dim=0)
        #     elif FLAGS.METHOD == 'TKFAC':
        #         tkfac_state = list(regularizer.tkfac_state_dict.values())[-1]
        #         temp_S = torch.eye(tkfac_state.S.shape[0] + 10, device='cuda')
        #         # temp_S[:tkfac_state.S.shape[0], :tkfac_state.S.shape[0]].copy_(tkfac_state.S)
        #         tkfac_state.S = temp_S
        #         tkfac_state.trace *= (t+1)/t
        #         center_state = list(center_dict.values())[-1]
        #         center_state.weight = torch.cat([center_state.weight, torch.zeros((10, center_state.weight.shape[1]), device='cuda')], dim=0)
        #         center_state.bias = torch.cat([center_state.bias, torch.zeros((10,), device='cuda')], dim=0)
        if t > 0:
            # add classification units
            weight_temp = torch.cat([model.fc.weight.data, torch.zeros((2, model.fc.weight.data.shape[1]), device='cuda')], dim=0)
            bias_temp = torch.cat([model.fc.bias.data, torch.zeros((2,), device='cuda')], dim=0)
            model.fc.__init__(model.fc.in_features, model.fc.out_features + 2)
            model.fc.cuda()
            model.fc.weight.data.copy_(weight_temp)
            model.fc.bias.data.copy_(bias_temp)


            # modify kfac state
            if FLAGS.METHOD == 'KFAC':
                kfac_state = list(regularizer.kfac_state_dict.values())[-1]
                temp_S = torch.eye(kfac_state.S.shape[0] + 2, device='cuda')
                # temp_S[:kfac_state.S.shape[0], :kfac_state.S.shape[0]].copy_(kfac_state.S)
                kfac_state.S = temp_S
                center_state = list(center_dict.values())[-1]
                center_state.weight = torch.cat([center_state.weight, torch.zeros((2, center_state.weight.shape[1]), device='cuda')], dim=0)
                center_state.bias = torch.cat([center_state.bias, torch.zeros((2,), device='cuda')], dim=0)
            elif FLAGS.METHOD == 'TKFAC':
                tkfac_state = list(regularizer.tkfac_state_dict.values())[-1]
                temp_S = torch.eye(tkfac_state.S.shape[0] + 2, device='cuda')
                # temp_S[:tkfac_state.S.shape[0], :tkfac_state.S.shape[0]].copy_(tkfac_state.S)
                tkfac_state.S = temp_S
                tkfac_state.trace *= (t+1)/t
                center_state = list(center_dict.values())[-1]
                center_state.weight = torch.cat([center_state.weight, torch.zeros((2, center_state.weight.shape[1]), device='cuda')], dim=0)
                center_state.bias = torch.cat([center_state.bias, torch.zeros((2,), device='cuda')], dim=0)


        # if t>0:
        #     tkfac_state = list(regularizer.tkfac_state_dict.values())[-1]
        #     tkfac_state.S.copy_(torch.eye(100, device='cuda'))

        train_loader = make_dataloader(train_dataset_sequence[t], True)
        train_loader_cycle = icycle(train_loader)

        # test_dataset = ClassIncrementalTenfoldCIFAR100(FLAGS.DATA_ROOT, task_classes=list(range(0, 10*(t+1))), train=False, transform=transform_test)
        test_dataset = ClassIncrementalFivefoldCIFAR10(FLAGS.DATA_ROOT, task_classes=list(range(0, 2*(t+1))), train=False, transform=transform_test)
        test_loader = make_dataloader(test_dataset, train=False)

        # test_dataset_sequence = [ClassIncrementalTenfoldCIFAR100(FLAGS.DATA_ROOT, task_classes=list(range(10*j, 10*(j+1))), train=False, transform=transform_test) for j in range(t+1)]
        test_dataset_sequence = [ClassIncrementalFivefoldCIFAR10(FLAGS.DATA_ROOT, task_classes=list(range(2*j, 2*(j+1))), train=False, transform=transform_test) for j in range(t+1)]
        test_loader_sequence = [make_dataloader(dset, train=False) for dset in test_dataset_sequence]

        # target_transform = get_target_transform_fn(num_classes=10*(t+1), alpha=15.0)
        # target_transform = get_target_transform_fn(num_classes=100, alpha=15.0)
        # target_transform = get_target_transform_fn(num_classes=10, alpha=15.0)
        target_transform = get_target_transform_fn(num_classes=2*(t+1), alpha=15.0)


        trainable_params = [p for n, p in model.named_parameters() if 'bn' not in n]
        # trainable_params = list(model.parameters())
        # trainable_params = list(model.fc.parameters())

        params_orig = copy.deepcopy(trainable_params)


        if FLAGS.OPTIM == 'SGD':
            optimizer = optim.SGD(trainable_params, lr=init_lrs[t], momentum=0.9, weight_decay=FLAGS.WEIGHT_DECAY)
        elif FLAGS.OPTIM == 'ADAM':
            optimizer = optim.Adam(trainable_params, lr=init_lrs[t], weight_decay=FLAGS.WEIGHT_DECAY)
        else:
            raise NotImplementedError


        if FLAGS.LR_SCHEDULE == 'LINEAR':
            lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, linear_schedule(max_steps[t]))
        elif FLAGS.LR_SCHEDULE == 'COS':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_steps[t])
        elif FLAGS.LR_SCHEDULE == 'ONECYCLE':
            lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=init_lrs[t], total_steps=max_steps[t])
        else:
            raise NotImplementedError


        model.train(FLAGS.TRACK_BN)
        for i in (pbar := trange(max_steps[t], dynamic_ncols=True)):
            # if t == 0:
            #     # model.load_state_dict(torch.load('./checkpoint/tenfold_init/state_dict.pt'), strict=not FLAGS.LINEARIZED)
            #     model.load_state_dict(torch.load('./checkpoint/hundredfold_init/state_dict.pt'), strict=not FLAGS.LINEARIZED)
            #     # model.load_state_dict(torch.load('./checkpoint/hundredfold-mit67-init/state_dict.pt'), strict=not FLAGS.LINEARIZED)
            #     break

            optimizer.zero_grad()
            input, target = next(train_loader_cycle)
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            if FLAGS.LOSS_FN == 'SCE':
                data_loss = F.cross_entropy(output, target, reduction='none').mean(0)
            elif FLAGS.LOSS_FN == 'MSE':
                data_loss = 0.5*F.mse_loss(output, target_transform(target), reduction='none').sum(1).mean(0)
            else:
                raise ValueError

            if FLAGS.METHOD in ('KFAC', 'TKFAC', 'EKFAC'):
                if t > 0:
                    reg_loss = regularizer.compute_loss(center_dict) * t
                else:
                    reg_loss = 0.
                loss = (data_loss + reg_loss)/(t+1)
            elif FLAGS.METHOD is None:
                reg_loss = 0.
                loss = data_loss
            else:
                raise NotImplementedError(FLAGS.METHOD)

            loss.backward()
            # weight_decay(model.named_parameters(), FLAGS.WEIGHT_DECAY)
            # weight_decay_origin(model, FLAGS.WEIGHT_DECAY)
            # if FLAGS.METHOD in ('EWC', 'KFAC', 'MAS', 'TKFAC'):
            #     damping(params_orig, trainable_params, lam=FLAGS.DAMPING)
            # if FLAGS.CLIP_GRAD:
            #     nn.utils.clip_grad_norm_(trainable_params, FLAGS.CLIP_GRAD)
            optimizer.step()
            lr_scheduler.step()

            if (global_step+1)%100 == 0:
                tprint(f'[TRAIN][{i}] LR {lr_scheduler.get_last_lr()[-1]:.2e} | {data_loss=:.3f} | {reg_loss=:.3f}')
                summary_writer.add_scalar('train_loss', data_loss, global_step=global_step)
                summary_writer.add_scalar('lr', lr_scheduler.get_last_lr()[-1], global_step=global_step)
                if FLAGS.LOG_HISTOGRAM:
                    add_summary_histograms(model)

            if (global_step+1)%200 == 0:
                # evaluate(test_loader)
                for loader in test_loader_sequence:
                    evaluate(loader)
                evaluate(test_loader)
                model.train(FLAGS.TRACK_BN)

            global_step += 1

        # ###
        # if t == 0:
        #     save_pickle()
        #     import sys
        #     sys.exit()
        # ###

        if FLAGS.METHOD == 'KFAC':
            if FLAGS.LOSS_FN == 'SCE':
                criterion = lambda logit, target: F.cross_entropy(logit, target, reduction='none')
                pseudo_target_fn = lambda logit: torch.distributions.Categorical(logit.softmax(1)).sample()
            elif FLAGS.LOSS_FN == 'MSE':
                criterion = lambda logit, target: 0.5*F.mse_loss(logit, target, reduction='none').sum(1)
                pseudo_target_fn = torch.normal
            modules_to_regularize = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
            # modules_to_regularize = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d))]
            tprint(modules_to_regularize)
            if t == 0:
                regularizer = KFACRegularizer(model, criterion, modules_to_regularize)
                regularizer.compute_curvature(train_dataset_sequence[t], 1000, pseudo_target_fn=pseudo_target_fn)
                center_dict = get_center_dict(modules_to_regularize)
            else:
                new_regularizer = KFACRegularizer(model, criterion, modules_to_regularize)
                new_regularizer.compute_curvature(train_dataset_sequence[t], 1000, pseudo_target_fn=pseudo_target_fn)
                center_dict = get_center_dict(modules_to_regularize)
                for old_kfac_state, new_kfac_state in zip(regularizer.kfac_state_dict.values(), new_regularizer.kfac_state_dict.values()):
                    old_kfac_state.A = (old_kfac_state.A*t + new_kfac_state.A)/(t+1)
                    old_kfac_state.S = (old_kfac_state.S*t + new_kfac_state.S)/(t+1)

        elif FLAGS.METHOD == 'TKFAC':
            if FLAGS.LOSS_FN == 'SCE':
                criterion = lambda logit, target: F.cross_entropy(logit, target, reduction='none')
                pseudo_target_fn = lambda logit: torch.distributions.Categorical(logit.softmax(1)).sample()
            elif FLAGS.LOSS_FN == 'MSE':
                criterion = lambda logit, target: 0.5*F.mse_loss(logit, target, reduction='none').sum(1)
                pseudo_target_fn = torch.normal
            modules_to_regularize = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
            # modules_to_regularize = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d))]
            # modules_to_regularize = [model.fc]
            tprint(modules_to_regularize)
            if t == 0:
                regularizer = TKFACRegularizer(model, criterion, modules_to_regularize)
                regularizer.compute_curvature(train_dataset_sequence[t], 1000, pseudo_target_fn=pseudo_target_fn)
                # for key, tkfac_state in regularizer.tkfac_state_dict.items():
                    # if isinstance(key, nn.Conv2d):
                    #     tkfac_state.trace *= 1000
                center_dict = get_center_dict(modules_to_regularize)
            else:
                new_regularizer = TKFACRegularizer(model, criterion, modules_to_regularize)
                new_regularizer.compute_curvature(train_dataset_sequence[t], 1000, pseudo_target_fn=pseudo_target_fn)
                # for key, tkfac_state in new_regularizer.tkfac_state_dict.items():
                    # if isinstance(key, nn.Conv2d):
                    #     tkfac_state.trace *= 1000
                center_dict = get_center_dict(modules_to_regularize)
                for old_tkfac_state, new_tkfac_state in zip(regularizer.tkfac_state_dict.values(), new_regularizer.tkfac_state_dict.values()):
                    old_tkfac_state.A = (old_tkfac_state.A*t + new_tkfac_state.A)/(t+1)
                    old_tkfac_state.S = (old_tkfac_state.S*t + new_tkfac_state.S)/(t+1)
                    old_tkfac_state.trace = (old_tkfac_state.trace*t + new_tkfac_state.trace)/(t+1)

        elif FLAGS.METHOD == 'EKFAC':
            if FLAGS.LOSS_FN == 'SCE':
                criterion = lambda logit, target: F.cross_entropy(logit, target, reduction='none')
                pseudo_target_fn = lambda logit: torch.distributions.Categorical(logit.softmax(1)).sample()
            elif FLAGS.LOSS_FN == 'MSE':
                criterion = lambda logit, target: 0.5*F.mse_loss(logit, target, reduction='none').sum(1)
                pseudo_target_fn = torch.normal
            modules_to_regularize = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
            # modules_to_regularize = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d))]
            # modules_to_regularize = [model.fc]
            tprint(modules_to_regularize)
            if t == 0:
                regularizer = EKFACRegularizer(model, criterion, modules_to_regularize)
                regularizer.compute_curvature(train_dataset_sequence[t], 1000, pseudo_target_fn=pseudo_target_fn)
                center_dict = get_center_dict(modules_to_regularize)
            else:
                new_regularizer = EKFACRegularizer(model, criterion, modules_to_regularize)
                new_regularizer.compute_curvature(train_dataset_sequence[t], 1000, pseudo_target_fn=pseudo_target_fn)
                center_dict = get_center_dict(modules_to_regularize)
                for old_tkfac_state, new_tkfac_state in zip(regularizer.tkfac_state_dict.values(), new_regularizer.tkfac_state_dict.values()):
                    old_tkfac_state.A = (old_tkfac_state.A*t + new_tkfac_state.A)/(t+1)
                    old_tkfac_state.S = (old_tkfac_state.S*t + new_tkfac_state.S)/(t+1)

        else:
            raise NotImplementedError(FLAGS.METHOD)

    if FLAGS.SAVE:
        save_pickle()

    summary_writer.close()


if __name__ == '__main__':
    main()
