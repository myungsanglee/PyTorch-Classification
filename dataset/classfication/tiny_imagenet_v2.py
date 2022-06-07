import os
import sys  
import glob
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchvision.transforms as T
from torchvision.transforms import autoaugment
from PIL import Image
import numpy as np
import cv2


class TinyImageNetDatasetV2(Dataset):
    def __init__(self, dataset_dir, transforms, is_train):
        super().__init__()
        self.transforms = transforms
        self.is_train = is_train
        with open(os.path.join(dataset_dir, 'wnids.txt'), 'r') as f:
            self.label_list = f.read().splitlines()

        if is_train:
            self.data = glob.glob(os.path.join(dataset_dir, 'train/*/images/*.JPEG'))
            self.train_list = dict()
            for data in self.data:
                label = data.split(os.sep)[-3]
                self.train_list[data] = self.label_list.index(label)
        else:
            self.data = glob.glob(os.path.join(dataset_dir, 'val/images/*.JPEG'))
            self.val_list = dict()
            with open(os.path.join(dataset_dir, 'val/val_annotations.txt'), 'r') as f:
                val_labels = f.read().splitlines()
                for label in val_labels:
                    f_name, label, _, _, _, _ = label.split('\t')
                    self.val_list[f_name] = self.label_list.index(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file = self.data[index]
        img = Image.open(img_file).convert('RGB')
        if self.is_train:
            label = self.train_list[img_file]
        else:
            label = self.val_list[os.path.basename(img_file)]

        transformed = self.transforms(img)
        return transformed, label


class TinyImageNetV2(pl.LightningDataModule):
    def __init__(self, dataset_dir, workers, batch_size, input_size):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.workers = workers
        self.batch_size = batch_size
        self.input_size = input_size
        
    def setup(self, stage=None):
        train_transforms = T.Compose([
            T.RandomResizedCrop(self.input_size),
            T.RandomHorizontalFlip(),
            autoaugment.AutoAugment(autoaugment.AutoAugmentPolicy.IMAGENET),
            T.ToTensor(),
            T.RandomErasing()
        ])

        valid_transform = T.Compose([
            T.ToTensor()
        ])

        self.train_dataset = TinyImageNetDatasetV2(
            dataset_dir=self.dataset_dir,
            transforms=train_transforms,
            is_train=True
        )

        self.valid_dataset = TinyImageNetDatasetV2(
            dataset_dir=self.dataset_dir,
            transforms=valid_transform,
            is_train=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0
        )


if __name__ == '__main__':
    input_size = 64

    train_transforms = T.Compose([
        T.RandomResizedCrop(input_size),
        T.RandomHorizontalFlip(),
        autoaugment.AutoAugment(autoaugment.AutoAugmentPolicy.IMAGENET),
        T.ToTensor(),
        T.RandomErasing()
    ])

    valid_transform = T.Compose([
        T.ToTensor()
    ])

    train_loader = DataLoader(
        TinyImageNetDatasetV2(
            dataset_dir='/home/fssv2/myungsang/datasets/tiny_imagenet/tiny-imagenet-200',
            transforms=train_transforms,
            is_train=True
        ),
        batch_size=1,
        shuffle=False
    )

    origin_loader = DataLoader(
        TinyImageNetDatasetV2(
            dataset_dir='/home/fssv2/myungsang/datasets/tiny_imagenet/tiny-imagenet-200',
            transforms=valid_transform,
            is_train=True
        ),
        batch_size=1,
        shuffle=False
    )

    label_name_path = '/home/fssv2/myungsang/datasets/tiny_imagenet/tiny-imagenet-200/tiny-imagenet.names'
    with open(label_name_path, 'r') as f:
        label_name_list = f.read().splitlines()

    for train_sample, origin_sample in zip(train_loader, origin_loader):
        train_x, train_y = train_sample

        img = train_x[0].numpy()   
        img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        origin_x, origin_y = origin_sample
        origin_img = origin_x[0].numpy()
        origin_img = (np.transpose(origin_img, (1, 2, 0))*255.).astype(np.uint8).copy()
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)

        # print(batch_x.size())
        print(f'label: {label_name_list[train_y[0]]}\n')

        cv2.imshow('Train', img)
        cv2.imshow('Origin', origin_img)
        key = cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()
