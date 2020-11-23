import numpy as np
import argparse
from os import path as osp
import torch
import multiworld, gym
from multiworld.core.image_env import ImageEnv, unormalize_image, normalize_image
import rlkit.torch.pytorch_util as ptu
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in, sawyer_init_camera_full
from matplotlib import pyplot as plt
from ROLL.LSTM_model import imsize48_default_architecture
import cv2, copy
import cv2
from matplotlib import pyplot as plt
import os
from ROLL.LSTM_model import ConvLSTM2

plt.rcParams['font.size'] = 18

def show_obs(normalized_img_vec_, imsize=48, name='img'):
    print(name)
    show_img = copy.deepcopy(normalized_img_vec_)
    show_img = show_img.reshape(3, imsize, imsize).transpose()
    show_img = show_img[::-1, :, ::-1]
    cv2.imshow("original image", show_img)
    cv2.waitKey()

def test_lstm_traj(env_id, lstm, save_path=None, save_name=None):
    pjhome = os.environ['PJHOME']
    # optimal
    if not osp.exists(osp.join(pjhome, 'data/local/env/{}-optimal-traj.npy'.format(env_id))):
        return
    imgs = np.load(osp.join(pjhome, 'data/local/env/{}-optimal-traj.npy'.format(env_id)))
    puck_distances = np.load(osp.join(pjhome, 'data/local/env/{}-optimal-traj-puck-distance.npy'.format(env_id)))
    traj_len, batch_size, imlen = imgs.shape

    _, _, vae_latent_distribution_params, lstm_latent_encodings = lstm(ptu.from_numpy(imgs))
    latents = lstm_latent_encodings
    vae_latents = vae_latent_distribution_params[0]

    latents = ptu.get_numpy(latents).reshape(traj_len, -1)
    latent_distances = np.linalg.norm(latents - latents[-1], axis=1)

    vae_latents = ptu.get_numpy(vae_latents).reshape(traj_len, -1)
    vae_latent_distances = np.linalg.norm(vae_latents - vae_latents[-1], axis=1)
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))

    axs = axs.reshape(-1)

    ax = axs[0]
    ax2 = ax.twinx()
    ax.plot(puck_distances, label='puck distance', color='r')
    ax2.plot(latent_distances, label='lstm distance', color='b')
    ax.legend(loc='upper right')
    ax2.legend(loc='center right')
    ax.set_title('optimal traj')

    ax = axs[1]
    ax2 = ax.twinx()
    ax.plot(puck_distances, label='puck distance', color='r')
    ax2.plot(vae_latent_distances, label='vae distance', color='g')
    ax.legend(loc='upper right')
    ax2.legend(loc='center right')
    ax.set_title('optimal traj')

    # local optimal
    imgs = np.load(osp.join(pjhome, 'data/local/env/{}-local-optimal-traj.npy'.format(env_id)))
    goal_image = np.load(osp.join(pjhome, 'data/local/env/{}-local-optimal-traj-goal.npy'.format(env_id)))
    traj_len, batch_size, imlen = imgs.shape
    puck_distances = np.load(osp.join(pjhome, 'data/local/env/{}-local-optimal-traj-puck-distance.npy'.format(env_id)))

    _, _, vae_latent_distribution_params, lstm_latent_encodings = lstm(ptu.from_numpy(imgs))
    latents = lstm_latent_encodings
    vae_latents = vae_latent_distribution_params[0]
    _, _, vae_latent_distribution_params, lstm_latent_encodings = lstm(ptu.from_numpy(goal_image))
    latent_goal = lstm_latent_encodings
    vae_latent_goal = vae_latent_distribution_params[0]

    latents = ptu.get_numpy(latents).reshape(traj_len, -1)
    latent_goal = ptu.get_numpy(latent_goal).flatten()
    latent_distances = np.linalg.norm(latents - latent_goal, axis=1)

    vae_latents = ptu.get_numpy(vae_latents).reshape(traj_len, -1)
    vae_latent_goal = ptu.get_numpy(vae_latent_goal).flatten()
    vae_latent_distances = np.linalg.norm(vae_latents - vae_latent_goal, axis=1)

    ax = axs[2]
    ax2 = ax.twinx()
    ax.plot(puck_distances, label='puck distance', color='r')
    ax2.plot(latent_distances, label='lstm distance', color='b')
    ax.legend(loc='upper right')
    ax2.legend(loc='center right')
    ax.set_title('local optimal traj')

    ax = axs[3]
    ax2 = ax.twinx()
    ax.plot(puck_distances, label='puck distance', color='r')
    ax2.plot(vae_latent_distances, label='vae distance', color='g')
    ax.legend(loc='upper right')
    ax2.legend(loc='center right')
    ax.set_title('local optimal traj')

    plt.savefig(osp.join(save_path, save_name))
    plt.cla()
    plt.clf()
    plt.close('all')
