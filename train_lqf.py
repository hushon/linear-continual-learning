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
from utils import MultiEpochsDataLoader
import torch_optimizer
from models.modules import CustomLinear, CustomConv2d, CustomBatchNorm2d
from torch.utils.tensorboard import SummaryWriter
import dataset
from utils import get_timestamp, image_loader
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
    # CHECKPOINT_PATH = '/workspace/runs/temp111/state_dict.pt' # CIFAR ResNet18, ImageNet32 pretrained
    # CHECKPOINT_PATH = '/workspace/runs/torch_imagenet32_resnet50_new/state_dict.pt' # CIFAR ResNet50, ImageNet32 pretrained
    # CHECKPOINT_PATH = '/workspace/runs/pretrain_tinyimagenet_resnet18/state_dict.pt' # CIFAR ResNet18, TinyImageNet pretrained
    # CHECKPOINT_PATH = '/workspace/runs/imagenet_resnet18_lrelu/model_best.pt' # ImageNet ResNet18, ImageNet pretrained
    # CHECKPOINT_PATH = './checkpoint/imagenet_resnet18_lrelu_lr0.001/model_best.pt' # ImageNet ResNet18, ImageNet pretrained
    CHECKPOINT_PATH = "./checkpoint/mocov2_resnet50/moco_v2_800ep_pretrain.pth.tar" # ResNet50 MoCo v2 pretrained
    LOG_DIR = '/workspace/runs/'
    BATCH_SIZE = 128
    INIT_LR = 1E-4
    # INIT_LR = 1E-2
    WEIGHT_DECAY = 1E-5
    MAX_STEP = 20000
    N_WORKERS = 8
    # BN_UPDATE_STEPS = 1000
    BN_UPDATE_STEPS = 0
    SAVE = True
    LOSS_FN = 'MSE'
    OPTIM = 'ADAM'
    GAF = True
    TRACK_BN = False
    AMP = True



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


def get_cifar100():
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
    return train_dataset, test_dataset, 100


def get_mit67():
    transform_train = T.Compose([
        # T.RandomResizedCrop(224),
        # T.RandomHorizontalFlip(),
        T.Resize((256,256)),
        T.RandomCrop((224,224)),
        # T.CenterCrop((224,224)),
        T.ToTensor(),
        T.Normalize(dataset.MIT67.MEAN, dataset.MIT67.STD)
        ])
    transform_test = T.Compose([
        T.Resize((256,256)),
        T.CenterCrop((224,224)),
        T.ToTensor(),
        T.Normalize(dataset.MIT67.MEAN, dataset.MIT67.STD)
        ])
    train_dataset = dataset.MIT67(os.path.join(FLAGS.DATA_ROOT, 'MIT67'), transform_train, train=True, loader=image_loader)
    test_dataset = dataset.MIT67(os.path.join(FLAGS.DATA_ROOT, 'MIT67'), transform_test, train=False, loader=image_loader)
    return train_dataset, test_dataset, 67


def get_caltech256():
    # transform_train = T.Compose([
    #     T.Resize((224,224)),
    #     T.RandomHorizontalFlip(),
    #     T.ToTensor(),
    #     T.Normalize(dataset.Caltech256.MEAN, dataset.Caltech256.STD)
    #     ]   )
    # transform_test = T.Compose([
    #     T.Resize((224,224)),
    #     T.ToTensor(),
    #     T.Normalize(dataset.Caltech256.MEAN, dataset.Caltech256.STD)
    #     ])
    # transform_train = T.Compose([
    #     T.Resize((224,224)),
    #     # T.RandomCrop(224, pad_if_needed=True),
    #     # T.Resize(224),
    #     # T.CenterCrop(224),
    #     T.RandomHorizontalFlip(),
    #     # T.RandomAffine(15, (0.3, 0.3), (0.7, 1.3)),
    #     T.ToTensor(),
    #     T.Normalize(dataset.Caltech256.MEAN, dataset.Caltech256.STD)
    #     ]   )
    # transform_test = T.Compose([
    #     T.Resize((224,224)),
    #     # T.CenterCrop(224),
    #     T.ToTensor(),
    #     T.Normalize(dataset.Caltech256.MEAN, dataset.Caltech256.STD)
    #     ])
    from PIL import ImageEnhance
    enhancers = {
        0: lambda image, f: ImageEnhance.Color(image).enhance(f),
        1: lambda image, f: ImageEnhance.Contrast(image).enhance(f),
        2: lambda image, f: ImageEnhance.Brightness(image).enhance(f),
        3: lambda image, f: ImageEnhance.Sharpness(image).enhance(f)
    }

    factors = {
        0: lambda: np.random.normal(1.0, 0.3),
        1: lambda: np.random.normal(1.0, 0.1),
        2: lambda: np.random.normal(1.0, 0.1),
        3: lambda: np.random.normal(1.0, 0.3),
    }
    
    # random enhancers in random order
    def enhance(image):
        order = [0, 1, 2, 3]
        np.random.shuffle(order)
        for i in order:
            f = factors[i]()
            image = enhancers[i](image, f)
        return image
    
    # train data augmentation on the fly
    # transform_train = T.Compose([
    #     T.Scale(384, Image.LANCZOS),
    #     T.RandomCrop(299),
    #     T.RandomHorizontalFlip(),
    #     T.Lambda(enhance),
    #     T.ToTensor(),
    #     T.Normalize(dataset.Caltech256.MEAN, dataset.Caltech256.STD)
    # ])
    
    # # validation data is already resized
    # transform_test = T.Compose([
    #     T.Scale(299, Image.LANCZOS),
    #     T.CenterCrop(299),
    #     T.ToTensor(),
    #     T.Normalize(dataset.Caltech256.MEAN, dataset.Caltech256.STD)
    # ])
    transform_train = T.Compose([
        T.Resize((256,256)),
        T.RandomCrop((224,224)),
        T.ToTensor(),
        T.Normalize(dataset.Caltech256.MEAN, dataset.Caltech256.STD)
        ])
    transform_test = T.Compose([
        T.Resize((256,256)),
        T.CenterCrop((224,224)),
        T.ToTensor(),
        T.Normalize(dataset.Caltech256.MEAN, dataset.Caltech256.STD)
        ])

    train_dataset = dataset.Caltech256(FLAGS.DATA_ROOT, transform_train, train=True, loader=image_loader)
    test_dataset = dataset.Caltech256(FLAGS.DATA_ROOT, transform_test, train=False, loader=image_loader)
    return train_dataset, test_dataset, 256


def main():
    timestamp = get_timestamp()
    log_dir = os.path.join(FLAGS.LOG_DIR, timestamp)
    # if FLAGS.SAVE:
    #     shutil.copytree('./', log_dir, dirs_exist_ok=True)
    summary_writer = SummaryWriter(log_dir=log_dir, max_queue=1)
    print(f"{log_dir=}")

    # train_dataset, test_dataset, n_classes = get_cifar100()
    train_dataset, test_dataset, n_classes = get_caltech256()
    # train_dataset, test_dataset, n_classes = get_mit67()

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


    if FLAGS.GAF:
        # from models.resnet_cifar100_jvplrelu import resnet18, resnet50
        from models.resnet_imagenet_jvplrelu import resnet18, resnet50
    else:
        # from models.resnet_cifar100_lrelu import resnet18, resnet50
        from models.resnet_imagenet_lrelu import resnet18, resnet50

    # from torchvision.models import resnet18, resnet50

    # model = resnet18(num_classes=n_classes).cuda()
    model = resnet50(num_classes=n_classes).cuda()
    state_dict = torch.load(FLAGS.CHECKPOINT_PATH)
    # state_dict.pop('fc.weight')
    # state_dict.pop('fc.bias')

    if "moco" in FLAGS.CHECKPOINT_PATH:
        state_dict = state_dict["state_dict"]
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    response = model.load_state_dict(state_dict, strict=False)
    print(response)

    # model = torchvision.models.resnet18(pretrained=True)
    # model.fc = nn.Linear(512, n_classes)
    # model = torchvision.models.resnet50(pretrained=True)
    # model.fc = nn.Linear(2048, n_classes)
    # model.cuda()

    # freeze feature extractor
    # for name, param in model.named_parameters():
    #     if 'fc' not in name:
    #         param.requires_grad_(False)

    if FLAGS.LOSS_FN == 'MSE':
        criterion = CustomMSELoss(reduction='none')
    elif FLAGS.LOSS_FN == 'SCE':
        criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        raise NotImplementedError

    # if FLAGS.OPTIM == 'SGD':
    #     optimizer = optim.SGD([p for n, p in model.named_parameters() if 'bn' not in n], lr=FLAGS.INIT_LR, momentum=0.9)
    # elif FLAGS.OPTIM == 'ADAM':
    #     optimizer = optim.Adam([p for n, p in model.named_parameters() if 'bn' not in n], lr=FLAGS.INIT_LR)
    # elif FLAGS.OPTIM == 'ADAHESSIAN':
    #     optimizer = torch_optimizer.Adahessian([p for n, p in model.named_parameters() if 'bn' not in n], lr=FLAGS.INIT_LR)
    # else:
    #     raise NotImplementedError

    if FLAGS.OPTIM == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=FLAGS.INIT_LR, momentum=0.9)
    elif FLAGS.OPTIM == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.INIT_LR)
    elif FLAGS.OPTIM == 'ADAHESSIAN':
        optimizer = torch_optimizer.Adahessian(model.parameters(), lr=FLAGS.INIT_LR)
    else:
        raise NotImplementedError


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
            if FLAGS.LOSS_FN == 'MSE':
                loss = criterion(output, transform_target(target)).sum(1).mean()
            elif FLAGS.LOSS_FN == 'SCE':
                loss = criterion(output, target).mean()
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
        pickle_path = os.path.join(log_dir, f'state_dict.pt')
        torch.save(pickle, pickle_path)
        tprint(f'[SAVE] Saved to {pickle_path}')


    def add_summary_histograms(model: nn.Module):
        for name, param in model.named_parameters():
            summary_writer.add_histogram(f"param/{name}", param.data, global_step=global_step)
            summary_writer.add_histogram(f"grad/{name}", param.grad.data, global_step=global_step)


    update_batchnorm()

    train_loader = icycle(train_loader)
    global_step = 0

    grad_scaler = amp.GradScaler(enabled=FLAGS.AMP)

    model.train(FLAGS.TRACK_BN)

    for i in (pbar := trange(FLAGS.MAX_STEP, desc=f'{timestamp}')):
        optimizer.zero_grad()
        input, target = next(train_loader)
        input = input.cuda()
        target = target.cuda()
        with amp.autocast(enabled=FLAGS.AMP):
            output = model(input)
            if FLAGS.LOSS_FN == 'MSE':
                loss = criterion(output, transform_target(target)).sum(1).mean()
            elif FLAGS.LOSS_FN == 'SCE':
                loss = criterion(output, target).mean()
        # loss.backward(create_graph=isinstance(optimizer, torch_optimizer.Adahessian))
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)

        weight_decay(model.named_parameters(), FLAGS.WEIGHT_DECAY)
        # weight_decay_origin(model, FLAGS.WEIGHT_DECAY)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        lr_scheduler.step()
        global_step += 1

        if i%100 == 0:
            tprint(f'LR {lr_scheduler.get_last_lr()[-1]:.2e} | {loss=:.3f}')
            summary_writer.add_scalar('loss/train', loss, global_step=global_step)
            summary_writer.add_scalar('lr', lr_scheduler.get_last_lr()[-1], global_step=global_step)
            # add_summary_histograms(model)

        if i%200 == 0:
            evaluate()
            model.train(FLAGS.TRACK_BN)


    if FLAGS.SAVE:
        save_pickle()


    summary_writer.close()


if __name__ == '__main__':
    main()
