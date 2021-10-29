from typing import Tuple
import numpy as np
import torch
import torch.utils.data
import PIL.Image
import logging
from tqdm import tqdm
import tree
from simplejpeg import decode_jpeg
from PIL import Image
from torch import nn
import torch.nn.functional as F
import socket
from datetime import datetime
import os


def get_log_dir(base_path, comment=''):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(base_path, current_time + '_' + socket.gethostname() + comment)
    return log_dir


def get_logger(log_path, level=logging.INFO):
    # logging.basicConfig(datefmt='%H:%M:%S', level=level)
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(logging.FileHandler(log_path, mode='a'))
    # logger.addHandler(TqdmLoggingHandler())
    return logger

def set_log(log_path):
    fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(message)s')

    fileHandler = logging.FileHandler(log_path+'tjproject.log')
    streamHandler = logging.StreamHandler()

    fileHandler.setFormatter(fomatter)
    streamHandler.setFormatter(fomatter)

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    logger.setLevel(logging.INFO)

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def pil_to_tensor(pic):
    '''convert to HxWxC tensor'''
    assert isinstance(pic, PIL.Image.Image)
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    return img


def icycle(iterable):
    while True:
        for x in iterable:
            yield x


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class CustomMSELoss(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5*F.mse_loss(input, target, reduction=self.reduction)
