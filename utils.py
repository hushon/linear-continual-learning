from typing import Tuple
import numpy as np
import torch
import torch.utils.data
import PIL.Image
import logging
from tqdm import tqdm
import jax
import jax.numpy as jnp
import jax.tree_util
import tree
from simplejpeg import decode_jpeg
from PIL import Image


def uint8_to_float(input: jnp.ndarray, mean: Tuple, std: Tuple) -> jnp.ndarray:
    assert input.ndim == 4
    input = input.astype(jnp.float32) / 255.0
    mean = jnp.array(mean).reshape(1, 1, 1, -1)
    std = jnp.array(std).reshape(1, 1, 1, -1)
    input = (input - mean) / std
    return input


def image_loader(path: str) -> np.ndarray:
    try:
        with open(path, 'rb') as fp:
            image = decode_jpeg(fp.read(), colorspace='RGB')
    except:
        image = Image.open(path).convert('RGB')
        image = np.asarray(image)
    return image

def softmax_cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    logp = jax.nn.log_softmax(logits)
    loss = -jnp.take_along_axis(logp, labels[:, None], axis=-1)
    return loss

def correct_topk(logits: jnp.ndarray, labels: jnp.ndarray, k: int) -> jnp.ndarray:
    labels = labels[..., None]
    preds = jnp.argsort(logits, axis=-1)[..., -k:]
    return jnp.any(preds == labels, axis=-1)


def l2_loss(params) -> jnp.ndarray:
    # l2_params = jax.tree_util.tree_leaves(params)
    l2_params = [p for ((mod_name, _), p) in tree.flatten_with_path(
        params) if 'batchnorm' not in mod_name]
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in l2_params)


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


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class ArrayNormalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        assert isinstance(
            arr, np.ndarray), f'Input should be ndarray. Got {type(arr)}.'
        assert arr.ndim >= 3, f'Expected array to be a image of size (..., H, W, C). Got {arr.shape}.'

        dtype = arr.dtype
        mean = np.asarray(self.mean, dtype=dtype)
        std = np.asarray(self.std, dtype=dtype)
        if (std == 0).any():
            raise ValueError(
                f'std evaluated to zero after conversion to {dtype}, leading to division by zero.')
        if mean.ndim == 1:
            mean = mean.reshape(1, 1, -1)
        if std.ndim == 1:
            std = std.reshape(1, 1, -1)
        arr -= mean
        arr /= std
        return arr


class ToArray(torch.nn.Module):
    '''convert image to float and 0-1 range'''
    dtype = np.float32

    def __call__(self, x: PIL.Image.Image) -> np.ndarray:
        assert isinstance(x, PIL.Image.Image)
        x = np.asarray(x, dtype=self.dtype)
        x /= 255.0
        return x
