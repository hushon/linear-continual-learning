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
import torch_optimizer
from models.modules import CustomLinear, CustomConv2d, CustomBatchNorm2d
from torch.utils.tensorboard import SummaryWriter

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
    # CHECKPOINT_DIR = '/workspace/runs/torch_imagenet32_resnet50_new'
    LOG_ROOT = '/workspace/runs/'
    BATCH_SIZE = 512
    INIT_LR = 1E-2
    WEIGHT_DECAY = 1E-5
    MAX_STEP = 8000
    N_WORKERS = 4
    BN_UPDATE_STEPS = 1000
    SAVE = False




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


class CustomMSELoss(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5*F.mse_loss(input, target, reduction=self.reduction)


def get_target_transform_fn(num_classes: int = 10, alpha: float = 15.0):
    def transform_fn(target: torch.Tensor) -> torch.Tensor:
        # assert target.dtype == torch.long and target.ndim == 1
        return F.one_hot(target, num_classes=num_classes).float().mul_(alpha)
    return transform_fn


def main():
    summary_writer = SummaryWriter(max_queue=1)
    print(f'Log_dir = {summary_writer.log_dir}')

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
    transform_target = get_target_transform_fn(100, 15.)

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

    model = resnet18(num_classes=100).cuda()
    # model = resnet50(num_classes=100).cuda()
    state_dict = torch.load(os.path.join(FLAGS.CHECKPOINT_DIR, 'state_dict.pt'))
    model.load_state_dict(state_dict, strict=False)

    # freeze feature extractor
    # for name, param in model.named_parameters():
    #     if 'fc' not in name:
    #         param.requires_grad_(False)

    criterion = CustomMSELoss(reduction='none')
    # criterion = nn.CrossEntropyLoss(reduction='none')
    # optimizer = optim.Adam(model.parameters(), lr=FLAGS.INIT_LR)
    optimizer = torch_optimizer.Adahessian(model.parameters(), lr=FLAGS.INIT_LR)
    # lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, linear_schedule(FLAGS.MAX_STEP))
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 1500], gamma=0.1)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, FLAGS.MAX_STEP)


    @torch.no_grad()
    def update_batchnorm():
        model.train()
        data_loader = icycle(train_loader)
        for _ in trange(FLAGS.BN_UPDATE_STEPS):
            input, _ = next(data_loader)
            input = input.cuda()
            model(input)


    @torch.no_grad()
    def evaluate():
        losses = []
        corrects = []
        corrects_t5 = []
        model.eval()
        for input, target in test_loader:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, transform_target(target)).sum(1).mean()
            losses.append(loss.view(-1))
            corrects.append((target == output.max(-1).indices).view(-1))
            corrects_t5.append(correct(output, target, 5))
        avg_loss = torch.cat(losses).mean().item()
        avg_acc = torch.cat(corrects).float().mean().item()*100
        avg_acc_t5 = torch.cat(corrects_t5).float().mean().item()*100
        tprint(f'[TEST] loss {avg_loss:.3f} | T1acc {avg_acc:.2f} | T5acc {avg_acc_t5:.2f}')
        summary_writer.add_scalar('t1_acc/test', avg_acc, global_step=i)
        summary_writer.add_scalar('t5_acc/test', avg_acc_t5, global_step=i)
        summary_writer.add_scalar('loss/test', avg_loss, global_step=i)


    def save_pickle():
        pickle = model.state_dict()
        pickle_path = os.path.join(summary_writer.log_dir, f'state_dict.pt')
        torch.save(pickle, pickle_path)
        tprint(f'[SAVE] Saved to {pickle_path}')


    def add_summary_histograms(model: nn.Module):
        for name, param in model.named_parameters():
            summary_writer.add_histogram(f"param/{name}", param.data, global_step=global_step)
            summary_writer.add_histogram(f"grad/{name}", param.grad.data, global_step=global_step)


    update_batchnorm()

    train_loader = icycle(train_loader)
    global_step = 0

    model.eval()
    for i in (pbar := trange(FLAGS.MAX_STEP)):
        optimizer.zero_grad()
        input, target = next(train_loader)
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, transform_target(target)).sum(1).mean()
        loss.backward(create_graph=True)
        # weight_decay(model.named_parameters(), FLAGS.WEIGHT_DECAY)
        weight_decay_origin(model, FLAGS.WEIGHT_DECAY)
        optimizer.step()
        lr_scheduler.step()
        global_step += 1

        if i%100 == 0:
            tprint(f'LR {lr_scheduler.get_last_lr()[-1]:.2e} | {loss:.3f}')
            summary_writer.add_scalar('loss/train', loss, global_step=global_step)
            summary_writer.add_scalar('lr', lr_scheduler.get_last_lr()[-1], global_step=global_step)
            add_summary_histograms(model)

        if i%200 == 0:
            evaluate()


    if FLAGS.SAVE:
        save_pickle()

    summary_writer.close()


if __name__ == '__main__':
    main()
