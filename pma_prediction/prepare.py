import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2
from scipy.misc import imresize
from PIL import Image
from skimage import io, transform
from skimage.color import rgba2rgb
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class RetinaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.rop_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.rop_df)

    def __getitem__(self, idx):
        sample_df = self.rop_df.iloc[[idx]]
        try:
            img_name = os.path.join(self.root_dir, sample_df['imageName'].iloc[0])
            image = io.imread(img_name)
            if image.shape[2] > 3:
                image = rgba2rgb(image)
            pma = sample_df['PMARaw'].iloc[0]
            label_to_int = {'No': 0, 'Pre-Plus': 1, 'Plus': 2}
            plus = label_to_int[sample_df['Golden Reading Plus'].iloc[0]]
            if self.transform:
                image = self.transform(image)
            sample = {'image': image, 'age': pma, 'plus': plus, 'img': img_name}
        except:
            sample = None
        return sample

class Preprocess(object):
    def __init__(self, params):
        self.params = params

    def __call__(self, image):
        resize, crop = self.params['resize'], self.params['crop']
        crop_width = (resize['width'] - crop['width']) / 2
        crop_height = (resize['height'] - crop['height']) / 2
        crop_width = int(crop_width)
        crop_height = int(crop_height)
        interp = resize.get('interp', 'bilinear')
        resized_im = imresize(image, (resize['height'], resize['width']), interp=interp)
        cropped_im = resized_im[crop_height:-crop_height, crop_width:-crop_width]
        image = cropped_im
        return image

def prepare(action, data_dir, csv_file, save_file=None):
    # params = {'resize': {'width': 302, 'height': 226, 'interp': 'bicubic'}, 
    #     'crop': {'width': 224, 'height': 224}} # for vgg
    params = {'resize': {'width': 404, 'height': 302, 'interp': 'bicubic'}, 
        'crop': {'width': 300, 'height': 300}}
    batch_size = 32
    collate_fn=(lambda x: torch.utils.data.dataloader.default_collate(list(filter(lambda y: y is not None, x))))

    if action == 'prepare':
        dataset = RetinaDataset(csv_file=csv_file,
                                root_dir=data_dir,
                                transform=transforms.Compose([
                                   Preprocess(params),
                                    # transforms.RandomResizedCrop(300),
                                    # transforms.RandomVerticalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                ]))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        if save_file is not None:
            with open(save_file, "wb") as f:
                pickle.dump(dataloader, f)
        return True
    elif action == 'train':
        dataset = RetinaDataset(csv_file=csv_file,
                                root_dir=data_dir,
                                transform=transforms.Compose([
                                   Preprocess(params),
                                   # transforms.RandomResizedCrop(300),
                                   # transforms.RandomVerticalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                ]))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        return dataloader, len(dataset)
    elif action == 'eval':
        dataset = RetinaDataset(csv_file=csv_file,
                                root_dir=data_dir,
                                transform=transforms.Compose([
                                   Preprocess(params),
                                   # transforms.CenterCrop(300), 
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                ]))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return dataloader, len(dataset)
    elif action == 'predict':
        # TODO: make dataloader with no labels
        raise NotImplementedError


if __name__ == "__main__":
    action = sys.argv[1] # prepare, train, eval, predict
    data_dir = sys.argv[2] # file of train, test data
    csv_file = sys.argv[3] # csv file of info OR output file name
    dataloader_file = sys.argv[4] # model path to save or load
    prepare(action, data_dir, csv_file, dataloader_file)
