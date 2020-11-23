import numpy as np
import cv2
import copy
import os

import gym
import multiworld
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in as camera

import torch
import unet

def unnormalize_image(image):
    tempImg = copy.deepcopy(image)
    tempImg = tempImg.reshape(3,imsize, imsize).transpose()
    tempImg = tempImg[::-1, :, ::-1]
    tempImg = tempImg * 255 
    tempImg = tempImg.astype(np.uint8)
    return tempImg

def apply_mask(image, mask, threshold):
    masked_image = copy.deepcopy(image)
    masked_image[mask<threshold] = 0
    return masked_image

def main(training_data_dir, validation_data_dir, test_data_dir, imsize):

    if not os.path.exists(training_data_dir): os.makedirs(training_data_dir)
    if not os.path.exists(validation_data_dir): os.makedirs(validation_data_dir)
    if not os.path.exists(test_data_dir): os.makedirs(test_data_dir)

    backSub = cv2.createBackgroundSubtractorMOG2(history=10) 

    # Registering required nmultiworld environments.
    multiworld.register_all_envs()
    base_env_background = gym.make('SawyerPushHurdlePuckAndRobotInvisible-v0')
    env_background = ImageEnv(base_env_background, imsize=imsize, init_camera=camera, transpose=True, normalize=True)
    env_background.reset()

    base_env_hurdle = gym.make('SawyerPushHurdlePuckInvisible-v0')
    env_hurdle = ImageEnv(base_env_hurdle, imsize=imsize, init_camera=camera, transpose=True, normalize=True)
    env_hurdle.reset()

    # Generating training, validation and test data
    print("Training background subctractor")
    for i in range(10):
        action = env_background.action_space.sample()
        next_obs, reward, done, info = env_background.step(action)
        image = next_obs['observation']
        image = unnormalize_image(image)
        bg_mask = backSub.apply(image, learningRate=-1)

    print("Generating training data")
    for i in range(1000):
        action = env_hurdle.action_space.sample()
        next_obs, reward, done, info = env_hurdle.step(action)
        image = next_obs['observation']
        image = unnormalize_image(image)
        fg_mask = backSub.apply(image, learningRate=0)
        fg_mask = fg_mask/fg_mask.max()
        cv2.imwrite(training_data_dir + "/rgb_" + str(i) + ".jpg", image)
        cv2.imwrite(training_data_dir + "/mask_" + str(i) + ".jpg", fg_mask)


    print("Generating validation data")
    for i in range(200):
        action = env_hurdle.action_space.sample()
        next_obs, reward, done, info = env_hurdle.step(action)
        image = next_obs['observation']
        image = unnormalize_image(image)
        fg_mask = backSub.apply(image, learningRate=0)
        fg_mask = fg_mask/fg_mask.max()
        cv2.imwrite(validation_data_dir + "/rgb_" + str(i) + ".jpg", image)
        cv2.imwrite(validation_data_dir + "/mask_" + str(i) + ".jpg", fg_mask)


    print("Generating testing data")
    for i in range(200):
        action = env_hurdle.action_space.sample()
        next_obs, reward, done, info = env_hurdle.step(action)
        image = next_obs['observation']
        image = unnormalize_image(image)
        fg_mask = backSub.apply(image, learningRate=0)
        fg_mask = fg_mask/fg_mask.max()
        cv2.imwrite(test_data_dir + "/rgb_" + str(i) + ".jpg", image)
        cv2.imwrite(test_data_dir + "/mask_" + str(i) + ".jpg", fg_mask)

    print("Completed generating data at:", training_data_dir)


if __name__ == "__main__":

    training_data_dir = "./data/train/"
    validation_data_dir = "./data/val/"
    test_data_dir = "./data/test/"
    imsize=480

    main(training_data_dir, validation_data_dir, test_data_dir, imsize)
