import os

import numpy as np
import pandas as pd
import cv2

from PIL import Image, ImageFile
from tqdm.notebook import tqdm
import logging

import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
import albumentations

import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')



class MaskDataset(Dataset):
    """
    Path&Label of MaskDataset for Image Classification
    """
    def __init__(self, transform=None, training=True):
        self.transform = transform
        self.training = training
        self.toTensor = transforms.ToTensor()

        self.train_data_path = '/opt/ml/input/data/train/customized_train.csv'
        self.eval_data_path = '/opt/ml/input/data/eval/info.csv'
        self.train_img_path = '/opt/ml/input/data/train/images'
        self.eval_img_path = '/opt/ml/input/data/eval/images'

        self.train_data = pd.read_csv(self.train_data_path)
        self.eval_data = pd.read_csv(self.eval_data_path)
        self.labels = sorted(self.train_data['label'].unique())

        self.train_images = {'images':[], 'labels':[]}
        self.eval_images = {'images':[], 'labels':[]}

        # 이미지를 읽어와서 PIL Image로 append
        if self.training:
            logging.info('Converting train dataset...')
            for path, label in tqdm(zip(self.train_data['path'], self.train_data['label']), desc=str(len(self.train_data))+'it'):
                path = os.path.join(self.train_img_path, path)
                # print(path)
                # img = cv2.imread(path)
                img = Image.open(path)
                # print(img)
                self.train_images['images'].append(img)
                self.train_images['labels'].append(label)
        else:
            logging.info('Converting test dataset...')
            for path, label in tqdm(zip(self.eval_data['ImageID'], self.eval_data['ans']), desc=str(len(self.eval_data))+'it'):
                path = os.path.join(self.eval_img_path, path)
                # img = cv2.imread(path)
                img = Image.open(path)
                self.eval_images['images'].append(img)
                self.eval_images['labels'].append(label)



    def __len__(self):
        if self.training:
            return len(self.train_data)
        else:
            return len(self.eval_data)


    # PIL Image를 전처리해서 Tensor로 return
    def __getitem__(self, idx):
        if self.training:
            img = self.train_images['images'][idx]
            if self.transform:
                img = self.transform(img)
            else:
                img = self.toTensor(img)
            label = self.train_images['labels'][idx]
            return img, label
        else:
            img = self.eval_images['images'][idx]
            if self.transform:
                img = self.transform(img)
            else:
                img = self.toTensor(img)
            label = self.eval_images['labels'][idx]
            return img, label


# ===================================================================


if __name__=="__main__":
    transform = transforms.Compose([
                    transforms.Resize((224,224)), 
                    transforms.ToTensor(), 
                    transforms.Normalize((0,), (1,))
                ])
    dataset = MaskDataset(transform, training=True)
    print(len(dataset))
    print(dataset[0])

    print(os.path.expanduser('~')) # /opt/ml
    print(os.getcwd())