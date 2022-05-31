import os
import sys  
import glob
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


class TinyImageNetDataset(Dataset):
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
        img = cv2.imread(img_file)
        if self.is_train:
            label = self.train_list[img_file]
        else:
            label = self.val_list[os.path.basename(img_file)]

        transformed = self.transforms(image=img)['image']
        return transformed, label


class TinyImageNet(pl.LightningDataModule):
    def __init__(self, dataset_dir, workers, batch_size, input_size):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.workers = workers
        self.batch_size = batch_size
        self.input_size = input_size
        
    def setup(self, stage=None):
        train_transforms = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Blur(),
            A.CLAHE(),
            A.ColorJitter(
                brightness=0.5,
                contrast=0.2,
                saturation=0.5,
                hue=0.1
            ),
            A.ShiftScaleRotate(),
            A.Cutout(max_h_size=int(self.input_size*0.125), max_w_size=int(self.input_size*0.125)),
            A.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
            A.Normalize(0, 1),
            ToTensorV2(),
        ],)

        valid_transform = A.Compose([
            A.Resize(self.input_size, self.input_size, always_apply=True),
            A.Normalize(0, 1),
            ToTensorV2(),
        ],)

        self.train_dataset = TinyImageNetDataset(
            dataset_dir=self.dataset_dir,
            transforms=train_transforms,
            is_train=True
        )

        self.valid_dataset = TinyImageNetDataset(
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

    train_transform = A.Compose([
        A.HorizontalFlip(),
        A.Blur(),
        A.ShiftScaleRotate(),
        A.GaussNoise(),
        A.Cutout(max_h_size=int(input_size*0.125), max_w_size=int(input_size*0.125)),
        # albumentations.ElasticTransform(),
        A.RandomResizedCrop(input_size, input_size, (0.8, 1)),
        A.Normalize(0, 1),
        ToTensorV2(),
    ],)

    origin_transform = A.Compose([
        A.Resize(input_size, input_size, always_apply=True),
        A.Normalize(0, 1),
        ToTensorV2(),
    ],)

    train_loader = DataLoader(
        TinyImageNetDataset(
            dataset_dir='/home/fssv2/myungsang/datasets/tiny_imagenet/tiny-imagenet-200',
            transforms=train_transform,
            is_train=True
        ),
        batch_size=1,
        shuffle=False
    )

    origin_loader = DataLoader(
        TinyImageNetDataset(
            dataset_dir='/home/fssv2/myungsang/datasets/tiny_imagenet/tiny-imagenet-200',
            transforms=origin_transform,
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

        origin_x, origin_y = origin_sample
        origin_img = origin_x[0].numpy()
        origin_img = (np.transpose(origin_img, (1, 2, 0))*255.).astype(np.uint8).copy()

        # print(batch_x.size())
        print(f'label: {label_name_list[train_y[0]]}\n')

        cv2.imshow('Train', img)
        cv2.imshow('Origin', origin_img)
        key = cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()
