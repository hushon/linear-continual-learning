import os
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import datasets
from typing import NamedTuple, Any, Callable, List, Optional, Union, Tuple, Set
from glob import glob
import scipy.io
import random
import sklearn.utils


def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech256(datasets.VisionDataset):
    MEAN = (0.5502, 0.5328, 0.5062)
    STD = (0.3155, 0.3113, 0.3253)

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            train: bool = True,
            loader = pil_loader,
    ) -> None:
        super().__init__(os.path.join(root, 'caltech256'),
                                        transform=transform,
                                        target_transform=target_transform)
        if train:
            csv_file = 'train_metadata.csv'
        else:
            csv_file = 'val_metadata.csv'
        self.dataframe = pd.read_csv(os.path.join(root, 'caltech256', csv_file))
        self.loader = loader

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = self.loader(os.path.join(self.root,
                                      "256_ObjectCategories",
                                      self.dataframe.iloc[index].directory,
                                      self.dataframe.iloc[index].img_name))

        target = self.dataframe.iloc[index].category_number-1

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.dataframe)



class ImageNet32(datasets.VisionDataset):
    train_list = (
        'Imagenet32_train_npz/train_data_batch_1.npz',
        'Imagenet32_train_npz/train_data_batch_2.npz',
        'Imagenet32_train_npz/train_data_batch_3.npz',
        'Imagenet32_train_npz/train_data_batch_4.npz',
        'Imagenet32_train_npz/train_data_batch_5.npz',
        'Imagenet32_train_npz/train_data_batch_6.npz',
        'Imagenet32_train_npz/train_data_batch_7.npz',
        'Imagenet32_train_npz/train_data_batch_8.npz',
        'Imagenet32_train_npz/train_data_batch_9.npz',
        'Imagenet32_train_npz/train_data_batch_10.npz',
    )
    val_list = (
        'Imagenet32_val_npz/val_data.npz',
    )
    train_cache_file = 'train_data.npz'
    val_cache_file = 'val_data.npz'
    MEAN = (0.4811, 0.4575, 0.4079)
    STD = (0.2604, 0.2532, 0.2682)

    def __init__(self, root, transform=None, target_transform=None, train=True):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train

        if not os.path.exists(os.path.join(self.root, self.train_cache_file)):
            self._create_cache(self.train_list, os.path.join(self.root, self.train_cache_file))
        if not os.path.exists(os.path.join(self.root, self.val_cache_file)):
            self._create_cache(self.val_list, os.path.join(self.root, self.val_cache_file))

        entry = np.load(os.path.join(self.root, self.train_cache_file if self.train else self.val_cache_file))
        self.data = entry['data']
        self.targets = entry['labels']

    def _create_cache(self, file_list, cache_filepath):
        data = []
        targets = []
        for file_name in file_list:
            file_path = os.path.join(self.root, file_name)
            entry = np.load(file_path)
            data.append(entry['data'])
            targets.append(entry['labels'])
        data = np.concatenate(data, axis=0).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        targets = np.concatenate(targets, axis=0) - 1
        np.savez(cache_filepath, data=data, labels=targets)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.targets)


class TinyImageNet(datasets.VisionDataset):
    wnids_file = "wnids.txt"
    words_file = "words.txt"
    MEAN = (0.4824, 0.4495, 0.3981)
    STD = (0.2770, 0.2693, 0.2829)

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            train: bool = True,
            loader = pil_loader,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        base_dir = os.path.join(self.root, "tiny-imagenet-200")

        with open(os.path.join(base_dir, self.wnids_file), 'r') as fp:
            self.label_to_wnid = [line.rstrip() for line in fp]

        self.wnid_to_label = {wnid: label for label, wnid in enumerate(self.label_to_wnid)}

        if train:
            self.images = glob(os.path.join(base_dir, "train", "**", "*.JPEG"), recursive=True)
            self.labels = [self.wnid_to_label[os.path.basename(x).split('_')[0]] for x in self.images]
        else:
            self.images = []
            self.labels = []
            with open(os.path.join(base_dir, 'val', 'val_annotations.txt'), 'r') as fp:
                for line in fp:
                    line = line.rstrip().split()
                    file_name, wnid = line[:2]
                    self.images.append(os.path.join(base_dir, 'val', 'images', file_name))
                    self.labels.append(self.wnid_to_label[wnid])

        self.loader = loader

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = self.loader(self.images[index])
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.labels)


class MIT67(datasets.VisionDataset):
    train_meta = 'TrainImages.txt'
    test_meta = 'TestImages.txt'
    code_to_class = [
            'airport_inside', 'artstudio', 'auditorium', 'bakery', 'bar',
            'bathroom', 'bedroom', 'bookstore', 'bowling', 'buffet', 'casino',
            'children_room', 'church_inside', 'classroom', 'cloister',
            'closet', 'clothingstore', 'computerroom', 'concert_hall',
            'corridor', 'deli', 'dentaloffice', 'dining_room', 'elevator',
            'fastfood_restaurant', 'florist', 'gameroom', 'garage', 'greenhouse',
            'grocerystore', 'gym', 'hairsalon', 'hospitalroom', 'inside_bus',
            'inside_subway', 'jewelleryshop', 'kindergarden', 'kitchen',
            'laboratorywet', 'laundromat', 'library', 'livingroom', 'lobby',
            'locker_room', 'mall', 'meeting_room', 'movietheater', 'museum',
            'nursery', 'office', 'operating_room', 'pantry', 'poolinside',
            'prisoncell', 'restaurant', 'restaurant_kitchen', 'shoeshop',
            'stairscase', 'studiomusic', 'subway', 'toystore', 'trainstation',
            'tv_studio', 'videostore', 'waitingroom', 'warehouse', 'winecellar'
        ]
    MEAN = (0.4811, 0.4272, 0.3688)
    STD = (0.2656, 0.2584, 0.2606)

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            train: bool = True,
            loader = pil_loader,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        if train:
            meta_file = os.path.join(self.root, self.train_meta)
        else:
            meta_file = os.path.join(self.root, self.test_meta)

        with open(meta_file, 'r') as fp:
            txt_file = [l.rstrip() for l in fp]

        self.class_to_code = {s: i for i, s in enumerate(self.code_to_class)}

        self.images = [os.path.join(self.root, 'Images', x) for x in txt_file]
        self.labels = [self.class_to_code[x.split('/')[0]] for x in txt_file]

        self.loader = loader

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = self.loader(self.images[index])
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.labels)


class StanfordDogs120(datasets.VisionDataset):
    train_list = 'train_list.mat'
    test_list = 'test_list.mat'
    MEAN = (0.4729, 0.4497, 0.3883)
    STD = (0.2638, 0.2583, 0.2634)

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            train: bool = True,
            loader = pil_loader,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        if train:
            mat_file = scipy.io.loadmat(os.path.join(self.root, self.train_list))
        else:
            mat_file = scipy.io.loadmat(os.path.join(self.root, self.test_list))

        self.images = [os.path.join(self.root, 'Images', x.item()) for x in mat_file['file_list'].squeeze()]
        self.labels = mat_file['labels'].squeeze()

        self.loader = loader

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = self.loader(self.images[index])
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.labels)


class StanfordCars196(datasets.VisionDataset):
    meta_file = 'devkit/cars_meta.mat'
    train_list = 'devkit/cars_train_annos.mat'
    test_list = 'cars_test_annos_withlabels.mat'
    MEAN = (0.4674, 0.4557, 0.4497)
    STD = (0.2955, 0.2941, 0.3024)

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            train: bool = True,
            loader = pil_loader,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        if train:
            mat_file = scipy.io.loadmat(os.path.join(self.root, self.train_list))
        else:
            mat_file = scipy.io.loadmat(os.path.join(self.root, self.test_list))

        meta_mat_file = scipy.io.loadmat(os.path.join(self.root, self.meta_file))
        self.label_to_class = [x.item() for x in meta_mat_file['class_names'].squeeze()]

        self.images = [os.path.join(self.root, 'cars_train', x[5].item()) for x in mat_file['annotations'].squeeze()]
        self.labels = np.array([x[4].item() for x in mat_file['annotations'].squeeze()])

        self.loader = loader

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = self.loader(self.images[index])
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.labels)


# class TaskIncrementalTenfoldCIFAR100(datasets.CIFAR100):
#     task_labels = [
#             [64, 67, 7, 75, 50, 84, 93, 87, 29, 63],
#             [1, 3, 9, 44, 45, 14, 28, 22, 24, 60],
#             [96, 35, 69, 10, 74, 12, 46, 15, 85, 54],
#             [39, 47, 81, 82, 83, 23, 56, 26, 91, 95],
#             [32, 38, 72, 41, 73, 80, 25, 58, 27, 62],
#             [33, 65, 70, 49, 51, 19, 21, 53, 59, 61],
#             [66, 2, 36, 37, 5, 77, 78, 16, 18, 30],
#             [0, 97, 40, 8, 76, 79, 55, 57, 92, 94],
#             [99, 42, 43, 17, 20, 86, 88, 89, 90, 31],
#             [98, 34, 68, 4, 6, 71, 11, 13, 48, 52]
#         ]
#     def __init__(
#             self,
#             root: str,
#             task_id: int,
#             train: bool = True,
#             transform = None,
#             target_transform = None,
#             download: bool = False,
#     ) -> None:
#         super().__init__(root, train=train, transform=transform,
#                         target_transform=target_transform, download=download)
#         label_set = self.task_labels[task_id]
#         lut = {v: i for i, v in enumerate(label_set)}
#         mask = np.isin(self.targets, label_set)

#         self.data = np.array(self.data)[mask]
#         self.targets = np.array(self.targets)[mask]
#         self.targets = np.array([lut[k] for k in self.targets])


class TaskIncrementalTenfoldCIFAR100(datasets.VisionDataset):
    base_folder = 'cifar-100-python-tenfold_ti'
    train_list = ['train'+str(i)+'.npz' for i in range(10)]
    test_list = ['test'+str(i)+'.npz' for i in range(10)]

    def __init__(
            self,
            root: str,
            task_id: int,
            train: bool = True,
            transform = None,
            target_transform = None
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        self.task_id = task_id

        if train:
            filename = self.train_list[self.task_id]
        else:
            filename = self.test_list[self.task_id]
        entry = np.load(os.path.join(self.root, self.base_folder, filename))
        self.data = entry['data']
        self.targets = entry['labels']

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> None:
        return len(self.targets)


# class DataIncrementalTenfoldCIFAR100(datasets.CIFAR100):
#     def __init__(
#             self,
#             root: str,
#             task_id: int,
#             train: bool = True,
#             transform: Optional[Callable] = None,
#             target_transform: Optional[Callable] = None,
#             download: bool = False,
#     ) -> None:
#         super().__init__(root, train=train, transform=transform,
#                         target_transform=target_transform, download=download)
#         assert isinstance(task_id, int) and 0 <= task_id < 10
#         self.data = np.asarray(self.data)
#         self.targets = np.asarray(self.targets)
#         # sklearn.utils.shuffle(self.data, self.targets, random_state=123)
#         # self.data = np.split(self.data, 10)[task_id]
#         # self.targets = np.split(self.targets, 10)[task_id]
#         n = len(self.targets) // 10
#         self.data = self.data[task_id*n : (task_id+1)*n]
#         self.targets = self.targets[task_id*n : (task_id+1)*n]


# class DataIncrementalHundredfoldCIFAR100(datasets.CIFAR100):
#     def __init__(
#             self,
#             root: str,
#             task_id: int,
#             train: bool = True,
#             transform: Optional[Callable] = None,
#             target_transform: Optional[Callable] = None,
#             download: bool = False,
#     ) -> None:
#         super().__init__(root, train=train, transform=transform,
#                         target_transform=target_transform, download=download)
#         assert isinstance(task_id, int) and 0 <= task_id < 100
#         self.data = np.asarray(self.data)
#         self.targets = np.asarray(self.targets)
#         # sklearn.utils.shuffle(self.data, self.targets, random_state=123)
#         # self.data = np.split(self.data, 100)[task_id]
#         # self.targets = np.split(self.targets, 100)[task_id]
#         n = len(self.targets) // 100
#         self.data = self.data[task_id*n : (task_id+1)*n]
#         self.targets = self.targets[task_id*n : (task_id+1)*n]

class DataIncrementalTenfoldCIFAR100(datasets.VisionDataset):
    base_folder = 'cifar-100-python-tenfold_di'
    train_list = ['train'+str(i)+'.npz' for i in range(10)]

    def __init__(
            self,
            root: str,
            task_id: int,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        assert train == True
        self.train = train
        self.task_id = task_id

        filename = self.train_list[self.task_id]
        entry = np.load(os.path.join(self.root, self.base_folder, filename))
        self.data = entry['data']
        self.targets = entry['labels']

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> None:
        return len(self.targets)

class DataIncrementalHundredfoldCIFAR100(datasets.VisionDataset):
    base_folder = 'cifar-100-python-hundredfold_di'
    train_list = ['train'+str(i)+'.npz' for i in range(100)]

    def __init__(
            self,
            root: str,
            task_id: int,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        assert train == True
        self.train = train
        self.task_id = task_id

        filename = self.train_list[self.task_id]
        entry = np.load(os.path.join(self.root, self.base_folder, filename))
        self.data = entry['data']
        self.targets = entry['labels']

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> None:
        return len(self.targets)


class TaskIncrementalTenfoldImageNet(datasets.ImageNet):

    def __init__(self, root: str, split: str = 'train', download: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(root, split, download, **kwargs)

        self.samples
        self.targets


