from segmentation import unet
import numpy as np
import cv2
import torch
import copy

import gym
import multiworld
from multiworld.core.image_env import normalize_image
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in as camera
import os.path as osp

model_hurdle = unet.UNet(n_channels=3, n_classes=1)
model_hurdle = model_hurdle.float()
model_hurdle = model_hurdle.cuda()
model_hurdle.load_state_dict(torch.load("./segmentation/pytorchmodel_sawyer_hurdle"))

model_push = unet.UNet(n_channels=3, n_classes=1)
model_push = model_push.float()
model_push = model_push.cuda()
model_push.load_state_dict(torch.load("./segmentation/pytorchmodel_sawyer_push"))

model_door = unet.UNet(n_channels=3, n_classes=1)
model_door = model_door.float()
model_door = model_door.cuda()
model_door.load_state_dict(torch.load("./segmentation/pytorchmodel_sawyer_door"))

model_pickup = unet.UNet(n_channels=3, n_classes=1)
model_pickup = model_pickup.float()
model_pickup = model_pickup.cuda()
model_pickup.load_state_dict(torch.load("./segmentation/pytorchmodel_sawyer_pickup"))

bgsb = cv2.createBackgroundSubtractorMOG2() 
save_cnt = 0


def unnormalize_image(image, imsize):
    tempImg = copy.deepcopy(image)
    tempImg = tempImg.reshape(3,imsize, imsize).transpose()
    tempImg = tempImg[::-1, :, ::-1]
    tempImg = tempImg * 255 
    tempImg = tempImg.astype(np.uint8)
    return tempImg

def train_bgsb(env, imsize=48, use_presampled_goal=False, train_num=2000):
    print("Training background subtractor using ", env)
    bgsb.setHistory(train_num)
    
    env.reset()
    for frameId in range(train_num):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        image = next_obs['observation']
        if frameId % 1000 == 0:
            print(frameId)

        image = unnormalize_image(image, imsize)

        fgMask = bgsb.apply(image)

    print("training done")


def segment_image_unet(img, imsize=48, normalize=True, show=False,  dilation=False, 
    fg_threshold=120, robot_threshold=0.2, dilation_size=2, env_id=None, save=False,
    save_path=None, segment=True):

    if not segment:
        if normalize:
            return img
        else:
            return (img * 255).astype(np.uint8)

    global save_cnt

    if env_id == 'SawyerPushHurdle-v0'  or 'SawyerPushMutiple-v0':
        model = model_hurdle
    elif env_id == 'SawyerPushHurdleMiddle-v0':
        model = model_hurdle
    elif env_id == 'SawyerPushNIPSEasy-v0' :
        model = model_push
    elif env_id == 'SawyerDoorHookResetFreeEnv-v1':
        model = model_door
    elif env_id == 'SawyerPickupEnvYZEasy-v0':
        model = model_pickup
    else:
        raise NotImplementedError

    if show:
        show_img = copy.deepcopy(img)
        show_img = show_img.reshape(3, imsize, imsize).transpose()
        show_img = show_img[::-1, :, ::-1]
        cv2.imshow("original image", show_img)
        cv2.waitKey()
    
    if save:
        show_img = (img * 255).astype(np.uint8)
        show_img = show_img.reshape(3, imsize, imsize).transpose()
        show_img = show_img[::-1, :, ::-1]
        cv2.imwrite(osp.join(save_path, "original_image_{}.png".format(save_cnt)), show_img)


    tempImg = copy.deepcopy(img)
    tempImg = tempImg.reshape(3, imsize, imsize).transpose()
    tempImg = tempImg[::-1, :, ::-1]
    tempImg = tempImg * 255
    tempImg = tempImg.astype(np.uint8)

    tempImgCopy = copy.deepcopy(tempImg)

    tempImg = np.rollaxis(tempImg, 2,0)
    tempImg = torch.tensor(tempImg).unsqueeze(0)
    tempImg = tempImg.cuda()

    mask_pred = model(tempImg.float())
    mask_pred = mask_pred.detach().cpu().numpy()
    mask_pred = mask_pred.squeeze()

    mask_pred[mask_pred < robot_threshold] = 0
    if dilation:
        kernel = np.ones((dilation_size, dilation_size),np.uint8)
        mask_pred = cv2.dilate(mask_pred, kernel)

    # Predict background mask
    fgMask = bgsb.apply(tempImgCopy, learningRate=0)
    if show:
        cv2.imshow('fgmask', fgMask)
        cv2.waitKey()
        cv2.imshow('robot mask', mask_pred)
        cv2.waitKey()

    if save:
        cv2.imwrite(osp.join(save_path, 'fgmask_{}.png'.format(save_cnt)), fgMask)
        mask_pred_save = (mask_pred * 255).astype(np.uint8)
        cv2.imwrite(osp.join(save_path, 'robot_mask_{}.png'.format(save_cnt)), mask_pred_save)
        obj_mask = np.ones_like(mask_pred) * 255
        obj_mask[fgMask<fg_threshold] = 0
        obj_mask[mask_pred>robot_threshold] = 0
        cv2.imwrite(osp.join(save_path, 'obj_mask_{}.png'.format(save_cnt)), obj_mask)


    # Apply segmentation
    tempImgCopy[fgMask<fg_threshold] = 0 # Removing background
    tempImgCopy[mask_pred>robot_threshold] = 0 # Removing robot

    tempImgCopy = tempImgCopy[::-1, :, ::-1]
    tempImgCopy = tempImgCopy.transpose()
    tempImgCopy = tempImgCopy.reshape([1,-1])

    if show:
        img = tempImgCopy.reshape(3, imsize, imsize).transpose()
        img = img[::-1, :, ::-1]
        cv2.imshow("segmented image", img)
        cv2.waitKey()
    
    if save:
        img = tempImgCopy.reshape(3, imsize, imsize).transpose()
        img = img[::-1, :, ::-1]
        cv2.imwrite(osp.join(save_path, "segmented_image_{}.png".format(save_cnt)), img)
        save_cnt += 1

    if normalize:
        tempImgCopy = normalize_image(tempImgCopy)

    # This is the correct code to send in the segmented image.
    return np.ndarray.flatten(tempImgCopy)
