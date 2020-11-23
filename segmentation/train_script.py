import os
import sys
import numpy as np
import cv2
import time
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter

import unet
from DataLoader import FrankaSegmentationDataset

def main(trainDir, valDir, modelDir):
    trainDataset = FrankaSegmentationDataset(trainDir)
    trainDataLoader = DataLoader(trainDataset)

    valDataset = FrankaSegmentationDataset(valDir)
    valDataLoader = DataLoader(valDataset, batch_size=2, shuffle=False, num_workers=5)

    if not os.path.exists(modelDir): 
        os.makedirs(modelDir)

    model = unet.UNet(n_channels=3, n_classes=1)
    model = model.float()
    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.BCELoss()

    trainWriter = SummaryWriter("logs/train")
    valWriter = SummaryWriter("logs/val")

    trainIter = 1
    valIter = 1
    globalIter = 1
    for epoch in range(31):
        epochTrainLoss = 0
        model.train()
        print("training started")
        for batch, (image, mask_true) in enumerate(trainDataLoader):

            image = image.cuda()
            mask_true = mask_true.cuda().float()
            mask_pred = model(image.float())

            loss = criterion(mask_pred, mask_true)
            epochTrainLoss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch: %d\tBatch: %d\tTraining Loss: %f" % (epoch, batch, loss.detach().cpu().item()))
            trainWriter.add_scalar('train_loss', loss.detach().cpu().item(), trainIter)
            trainWriter.add_scalar('train_val_loss_combined', loss.detach().cpu().item(), globalIter)
            trainIter += 1; globalIter += 1


        trainWriter.add_scalar('epoch_train_loss', epochTrainLoss/(batch+1), epoch)
        if(epoch%10 == 0):
            torch.save(model.state_dict(), modelDir + "/pytorchmodel_epoch"+str(epoch) + time.strftime("_%Y%m%d_%H_%M_%S"))

        print("Validation started...")
        epochValLoss = 0
        model.eval()
        with torch.no_grad():
            for batch, (image, mask_true) in enumerate(valDataLoader):

                image = image.cuda()
                mask_true = mask_true.cuda().float()
                mask_pred = model(image.float())

                loss = criterion(mask_pred, mask_true)
                epochValLoss += loss

                print("Epoch: %d\tBatch: %d\tValidation Loss: %f" % (epoch, batch, loss.detach().cpu().item()))
                valWriter.add_scalar('val_loss', loss.detach().cpu().item(), valIter)
                valWriter.add_scalar('train_val_loss_combined', loss.detach().cpu().item(), globalIter)
                valIter += 1; globalIter += 1

            valWriter.add_scalar('epoch_val_loss', epochValLoss/(batch+1), epoch)

        trainWriter.close()
        valWriter.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_dir", required=True)
    parser.add_argument("--train_data_dir", required=True)
    parser.add_argument("--val_data_dir", required=True)
    args = parser.parse_args()
    
    main(args.train_data_dir, args.val_data_dir, args.model_save_dir)
