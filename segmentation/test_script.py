import os
import sys
import numpy as np
import cv2
import time
import copy
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

import unet
from DataLoader import FrankaSegmentationDataset

def applyMask(image, mask, threshold):
    tempImage = image.copy()
    tempImage[mask<threshold] = 0.0
    return tempImage

def main(testDir, modelDir):
    testDataset = FrankaSegmentationDataset(testDir)
    testDataLoader = DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=5)

    model = unet.UNet(n_channels=3, n_classes=1)
    model = model.float()
    model = model.cuda()

    criterion = torch.nn.BCELoss()
    model.load_state_dict(torch.load(modelDir))

    backSub = cv2.createBackgroundSubtractorMOG2() 

    print("Testing started")
    counter = 0
    robotThreshold = 2

    ### Very important to set no_grad(), else model might train on testing data ###
    model.eval()
    with torch.no_grad():
        for batch, (image, mask_true) in enumerate(testDataLoader):
            image = image.cuda()
            mask_true = mask_true.cuda().float()
            mask_pred = model(image.float())

            loss = criterion(mask_pred, mask_true)
            print("Batch: %d\tTesting Loss: %f" % (batch, loss.detach().cpu().item()))

            maskBatch = mask_pred.detach().cpu().numpy()
            maskTrueBatch = mask_true.detach().cpu().numpy()

            for i in range(maskBatch.shape[0]):
                mask = maskBatch[i].squeeze()
                maskTrue = maskTrueBatch[i].squeeze()
                tempImage = image[i]
                tempImage = np.rollaxis(tempImage.detach().cpu().numpy(), 0,3)
                fgMask = backSub.apply(tempImage)

                kernel = np.ones((5,5),np.uint8)
                maskOpened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                masked_image_pred = copy.deepcopy(tempImage)
                masked_image_pred_opened = copy.deepcopy(tempImage)
                masked_image_true = copy.deepcopy(tempImage)

                masked_image_pred[mask<0.9] = 0
                masked_image_pred_opened[maskOpened<0.2] = 0
                masked_image_true[maskTrue<0.4] = 0
                stacked_image = np.hstack([tempImage, masked_image_pred, masked_image_pred_opened])
                counter += 1
               
                maskCopy = copy.deepcopy(mask)*255
                maskCopy[maskCopy<robotThreshold] = 0
                maskedImage = applyMask(tempImage, maskCopy, 200)

                cv2.imshow("maskedImage", maskedImage)
                cv2.imshow("rgbImage", tempImage)
                cv2.waitKey()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test_data_dir", required=True)
    args = parser.parse_args()
    
    main(args.test_data_dir, args.model)
