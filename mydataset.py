import os
import random
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class OmniglotTrain(Dataset):

    def __init__(self, dataPath, transform=None):
        super(OmniglotTrain, self).__init__()
        np.random.seed(0)
        self.transform = transform
        self.datas, self.num_classes = self.loadToMem(dataPath)

    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        datas = {}
        idx = 0
        for alphaPath in os.listdir(dataPath):
            for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = []
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                    filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                    datas[idx].append(Image.open(filePath).convert('L'))
                idx += 1
        print("finish loading training dataset to memory")
        return datas, idx

    def __len__(self):
        """
        Repeat N times minibatch sampling 

        batch_size: DataLoader method's argument
        N = len / batch_size
        """
        return 99999999999

    def __getitem__(self, idx):
        """
        DataLoader will randomly select a minibatch from the Dataset instance using index 1 ~ 128.
        We want to feed (img1, img2, label) = Dataset[n]; for n = 1:128(batch_size)
        Furthermore, 64 same class image pair & 64 different class image pair

        Outputs
        ----------
        img1 : torch.tensor(128, 128)
        img2 : torch.tensor(128, 128)
        label : 1(two image are in same class) | 0(two image are in different classes)
        """
        label = None
        img1 = None
        img2 = None
        # get image from same class
        if idx % 2 == 1:
            label = 1.0
            img_class = random.randint(0, self.num_classes-1)
            img1 = random.choice(self.datas[img_class])
            img2 = random.choice(self.datas[img_class])
        # get image from different class
        else:
            label = 0.0
            img_class1 = random.randint(0, self.num_classes-1)
            img_class2 = random.randint(0, self.num_classes-1)
            while img_class1 == img_class2:
                img_class2 = random.randint(0, self.num_classes-1)
            img1 = random.choice(self.datas[img_class1])
            img2 = random.choice(self.datas[img_class2])
            
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.FloatTensor([label])

class OmniglotTest(Dataset):

    def __init__(self, dataPath, transform=None, times=200, way=20):
        np.random.seed(1)
        super(OmniglotTest, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None # query
        self.c1 = None
        self.c2_list = []
        self.datas, self.num_classes = self.loadToMem(dataPath)

    def loadToMem(self, dataPath):
        print("begin loading test dataset to memory")
        datas = {}
        idx = 0
        for alphaPath in os.listdir(dataPath):
            for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = []
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                    filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                    datas[idx].append(Image.open(filePath).convert('L'))
                idx += 1
        print("finish loading test dataset to memory")
        return datas, idx

    def __len__(self):
        """
        Repeat N times minibatch sampling
        
        batch_size: DataLoader method's argument
        N = len / batch_size
        """
        return self.times * self.way

    def __getitem__(self, idx):
        """
        DataLoader will randomly select a minibatch from the Dataset instance using index 0 ~ 19(20-ways).
        We want to feed [img1's class] == [img2's class] where (img1, img2) = Dataset[0]
        Furthermore, [img1's class] == [img2's class] where (img1, img2) = Dataset[n] for n = 1:19

        Outputs
        ----------
        img1 : torch.tensor(128, 128)
        img2 : torch.tensor(128, 128)
        """
        label = None
        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            self.img1 = random.choice(self.datas[self.c1])
            img2 = random.choice(self.datas[self.c1])
        else:
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2 or c2 in self.c2_list:
                c2 = random.randint(0, self.num_classes - 1)
            self.c2_list.append(c2)
            img2 = random.choice(self.datas[c2])

        if self.transform:
            self.img1 = self.transform(self.img1)
            img2 = self.transform(img2)

        return self.img1, img2
            
# test
if __name__=='__main__':
    print('=================TRAIN=================')
    omniglotTrain = OmniglotTrain('./omniglot/python/images_background')
    print(omniglotTrain[0])
    print('=================TEST=================')
    omniglotTest = OmniglotTest('./omniglot/python/images_background')
    print(omniglotTest[0])