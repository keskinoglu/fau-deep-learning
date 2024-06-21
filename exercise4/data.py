from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import torchvision.transforms as tr
import copy

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data # Pandas DataFrame, stores the information found in the file "data.csv"
        self.mode = mode # string, either "val" for evaluate or "train"
        self._transform = tr.Compose([tr.ToPILImage(), tr.ToTensor(), tr.Normalize(mean=train_mean, std=train_std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_loc = self.data.iloc[index]['filename']
        try:
            img_gray = imread(img_loc)
        except FileNotFoundError:
            import os
            img_loc = os.path.join(Path(__file__).parent, img_loc)
            img_gray = imread(img_loc)
        except:
            raise FileNotFoundError("File not found: " + img_loc)
        img = gray2rgb(img_gray)

        if self.mode == "train":
            train_transform = copy.deepcopy(self._transform)
            
            # add random transformations to the image
            #train_transform.append(tr.RandomHorizontalFlip(p=0.5))
            #train_transform.append(tr.RandomVerticalFlip(p=0.5))
            #train_transform.append(tr.RandomRotation(degrees=90))
            #train_transform.append(tr.RandomRotation(degrees=45))
            #train_transform.append(tr.RandomAffine(degrees=0, shear=45))
            #train_transform.append(tr.RandomInvert(p=0.5))
            #train_transform.appedn(tr.RandomEqualize(p=0.5))

            #train_transform.insert(1, tr.RandomHorizontalFlip(p=0.5))
            #train_transform.insert(1, tr.RandomVerticalFlip(p=0.5))
            #train_transform.insert(1, tr.RandomRotation(degrees=90))

            #train_transform.transforms.insert(1, tr.AutoAugment())

            train_transform.transforms.insert(1, tr.TrivialAugmentWide())

            img = train_transform(img)
        elif self.mode == "val":
            img = self._transform(img)
        else:
            print("unrecognized mode")

        label_crack = self.data.iloc[index]['crack']
        label_inactive = self.data.iloc[index]['inactive']

        label = torch.tensor([label_crack, label_inactive])

        return img, label