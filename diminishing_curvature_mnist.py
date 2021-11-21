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
from torch.nn.parallel import DataParallel
from torchvision import datasets
import atexit
from PIL import Image
import torch_optimizer
from models.modules import CustomLinear, CustomConv2d, CustomBatchNorm2d
from kfac import EKFACRegularizer
from torch.utils.tensorboard import SummaryWriter
import dataset
from utils import MultiEpochsDataLoader, get_timestamp, image_loader
import shutil


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
    # CHECKPOINT_PATH = '/workspace/runs/temp111/state_dict.pt'
    # CHECKPOINT_PATH = '/workspace/runs/torch_imagenet32_resnet50_new/state_dict.pt'
    # CHECKPOINT_PATH = '/workspace/runs/imagenet_resnet18_lrelu/model_best.pt'
    # CHECKPOINT_PATH = './checkpoint/imagenet_resnet18_lrelu_lr0.001/model_best.pt'
    LOG_DIR = '/workspace/runs/'
    BATCH_SIZE = 128
    INIT_LR = 1E-2
    WEIGHT_DECAY = 1E-5
    MAX_STEP = 10000
    N_WORKERS = 4
    BN_UPDATE_STEPS = 1000
    SAVE = True
    LOSS_FN = 'SCE'
    OPTIM = 'SGD'
    TRACK_BN = False


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
    return (pred == target).any(1)


def weight_decay(named_parameters, lam):
    for name, param in named_parameters:
        if 'bn' not in name and param.grad is not None:
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


def get_mnist():
    transform_train = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081))
        ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081))
        ])
    train_dataset = datasets.MNIST(FLAGS.DATA_ROOT, True, transform_train, download=True)
    test_dataset = datasets.MNIST(FLAGS.DATA_ROOT, False, transform_test, download=True)
    return train_dataset, test_dataset, 10


def main():
    log_dir = os.path.join(FLAGS.LOG_DIR, get_timestamp())
    # if FLAGS.SAVE:
    #     shutil.copytree('./', log_dir, dirs_exist_ok=True)
    summary_writer = SummaryWriter(log_dir=log_dir, max_queue=1)
    print(f"{log_dir=}")

    train_dataset, test_dataset, n_classes = get_mnist()

    transform_target = get_target_transform_fn(n_classes, 15.)

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

    # from models.resnet_cifar100_jvplrelu import resnet18, resnet50
    from models.resnet_cifar100_lrelu import resnet18, resnet50
    # from models.resnet_imagenet_lrelu import resnet18
    # from models.resnet_imagenet_jvplrelu import resnet18

    # model = resnet18(num_classes=n_classes).cuda()
    # # # model = resnet50(num_classes=n_classes).cuda()
    # state_dict = torch.load(FLAGS.CHECKPOINT_PATH)
    # state_dict.pop('fc.weight')
    # state_dict.pop('fc.bias')
    # model.load_state_dict(state_dict, strict=False)

    import torchvision
    # model = torchvision.models.resnet18(pretrained=True)
    # model.fc = nn.Linear(512, n_classes)
    # model = torchvision.models.resnet50(pretrained=True)
    # model.fc = nn.Linear(2048, n_classes)
    # model.cuda()

    # freeze feature extractor
    # for name, param in model.named_parameters():
    #     if 'fc' not in name:
    #         param.requires_grad_(False)

    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(5408, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    ).cuda()

    if FLAGS.OPTIM == 'SGD':
        optimizer = optim.SGD([p for n, p in model.named_parameters() if 'bn' not in n], lr=FLAGS.INIT_LR, momentum=0.9)
    elif FLAGS.OPTIM == 'ADAM':
        optimizer = optim.Adam([p for n, p in model.named_parameters() if 'bn' not in n], lr=FLAGS.INIT_LR)
    elif FLAGS.OPTIM == 'ADAHESSIAN':
        optimizer = torch_optimizer.Adahessian([p for n, p in model.named_parameters() if 'bn' not in n], lr=FLAGS.INIT_LR)
    else:
        raise NotImplementedError

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, linear_schedule(FLAGS.MAX_STEP))
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 1500], gamma=0.1)
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, FLAGS.MAX_STEP)


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
        model.eval()
        for input, target in test_loader:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            if FLAGS.LOSS_FN == 'SCE':
                loss = F.cross_entropy(output, target, reduction='none')
            elif FLAGS.LOSS_FN == 'MSE':
                loss = 0.5*F.mse_loss(output, 15.*F.one_hot(target, num_classes=n_classes).float(), reduction='none').sum(1)
            losses.append(loss.view(-1))
            corrects.append((target == output.max(-1).indices).view(-1))
        avg_loss = torch.cat(losses).mean().item()
        avg_acc = torch.cat(corrects).float().mean().item()*100
        tprint(f'[TEST] loss {avg_loss:.3f} | T1acc {avg_acc:.2f}')
        summary_writer.add_scalar('t1_acc/test', avg_acc, global_step=i)
        summary_writer.add_scalar('loss/test', avg_loss, global_step=i)


    def save_pickle():
        pickle = model.state_dict()
        pickle_path = os.path.join(log_dir, f'state_dict.pt')
        torch.save(pickle, pickle_path)
        tprint(f'[SAVE] Saved to {pickle_path}')


    def add_summary_histograms(model: nn.Module):
        for name, param in model.named_parameters():
            summary_writer.add_histogram(f"param/{name}", param.data, global_step=global_step)
            summary_writer.add_histogram(f"grad/{name}", param.grad.data, global_step=global_step)


    # update_batchnorm()

    train_loader = icycle(train_loader)
    global_step = 0

    model.train(FLAGS.TRACK_BN)

    for i in (pbar := trange(FLAGS.MAX_STEP)):

        if i == 0:
            if FLAGS.LOSS_FN == 'SCE':
                criterion = lambda logit, target: F.cross_entropy(logit, target, reduction='none')
                pseudo_target_fn = lambda logit: torch.distributions.Categorical(logit.softmax(1)).sample()
            elif FLAGS.LOSS_FN == 'MSE':
                criterion = lambda logit, target: 0.5*F.mse_loss(logit, target, reduction='none').sum(1)
                pseudo_target_fn = torch.normal
            regularizer = EKFACRegularizer(model, criterion, [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))])
            regularizer.compute_curvature(train_dataset, 1000, pseudo_target_fn=pseudo_target_fn)
            eig = torch.cat([state.scale.cpu().view(-1) for state in regularizer.ekfac_state_dict.values()])
            torch.save(eig, f'./figs/diminishing-curvature/eig_mnist_sce/eig_{i}.pt')
            summary_writer.add_histogram(f"eigenspectrum", eig, global_step=global_step)


        optimizer.zero_grad()
        input, target = next(train_loader)
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        if FLAGS.LOSS_FN == 'SCE':
            loss = F.cross_entropy(output, target, reduction='none').mean(0)
        elif FLAGS.LOSS_FN == 'MSE':
            loss = 0.5*F.mse_loss(output, 15.*F.one_hot(target, num_classes=n_classes).float(), reduction='none').sum(1).mean(0)
        loss.backward(create_graph=isinstance(optimizer, torch_optimizer.Adahessian))
        weight_decay(model.named_parameters(), FLAGS.WEIGHT_DECAY)
        # weight_decay_origin(model, FLAGS.WEIGHT_DECAY)
        optimizer.step()
        lr_scheduler.step()
        global_step += 1

        if (i+1)%50 == 0:
            tprint(f'LR {lr_scheduler.get_last_lr()[-1]:.2e} | {loss=:.3f}')
            summary_writer.add_scalar('loss/train', loss, global_step=global_step)
            summary_writer.add_scalar('lr', lr_scheduler.get_last_lr()[-1], global_step=global_step)
            # add_summary_histograms(model)

        if (i+1)%100 == 0:
            evaluate()
            model.train(FLAGS.TRACK_BN)

        if (i+1)%200 == 0:
            if FLAGS.LOSS_FN == 'SCE':
                criterion = lambda logit, target: F.cross_entropy(logit, target, reduction='none')
                pseudo_target_fn = lambda logit: torch.distributions.Categorical(logit.softmax(1)).sample()
            elif FLAGS.LOSS_FN == 'MSE':
                criterion = lambda logit, target: 0.5*F.mse_loss(logit, target, reduction='none').sum(1)
                pseudo_target_fn = torch.normal
            regularizer = EKFACRegularizer(model, criterion, [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))])
            regularizer.compute_curvature(train_dataset, 1000, pseudo_target_fn=pseudo_target_fn)
            eig = torch.cat([state.scale.cpu().view(-1) for state in regularizer.ekfac_state_dict.values()])
            torch.save(eig, f'./figs/diminishing-curvature/eig_mnist_sce/eig_{i}.pt')
            summary_writer.add_histogram(f"eigenspectrum", eig, global_step=global_step)


    # if FLAGS.SAVE:
    #     save_pickle()


    summary_writer.close()


if __name__ == '__main__':
    main()
