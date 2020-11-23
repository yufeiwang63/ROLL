import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import random
import math

class FrankaSegmentationDataset(Dataset):

    def __init__(self, dataDir, transform=None):
        self.dataDir = dataDir

        imageFileNames = os.listdir(self.dataDir)
        rgbFileNames = [x for x in imageFileNames if "rgb" in x]
        self.rgbFileIDs = [x.split(".")[0] for x in rgbFileNames]
        self.rgbFileIDs = [x.split("_")[1] for x in self.rgbFileIDs]
        random.shuffle(self.rgbFileIDs)
        self.datasetSize = len(self.rgbFileIDs)

    def __len__(self):
        return self.datasetSize

    def __getitem__(self,idx):
        rgbImage = cv2.imread(self.dataDir + "/rgb_" + str(self.rgbFileIDs[idx]) + ".jpg")
        maskImage = cv2.imread(self.dataDir + "/mask_" + str(self.rgbFileIDs[idx]) + ".jpg", cv2.IMREAD_GRAYSCALE)
        rgbImage = np.rollaxis(rgbImage, 2, 0)

        return rgbImage, maskImage 

