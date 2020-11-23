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
from ROLL.LSTM_model import ConvLSTM2

plt.rcParams['font.size'] = 18

def show_obs(normalized_img_vec_, imsize=48, name='img'):
    print(name)
    show_img = copy.deepcopy(normalized_img_vec_)
    show_img = show_img.reshape(3, imsize, imsize).transpose()
    show_img = show_img[::-1, :, ::-1]
    cv2.imshow("original image", show_img)
    cv2.waitKey()


def compare_latent_distance(lstm, all_data, obj_states, obj_name='puck', save_dir=None, save_name=None, vae=None):
    batch_size, traj_len, imlen = all_data.shape
    all_data_tensor = np.swapaxes(all_data, 0, 1) # turn to traj_len, batch_size, feature_size
    all_data_tensor = ptu.from_numpy(all_data_tensor)
    # print("all data tensor shape: ", all_data_tensor.shape)
    # print("obj_states shape: ", obj_states.shape)

    _, _, obj_state_dim = obj_states.shape
    obj_states = obj_states.reshape((-1, obj_state_dim))
    obj_distances = np.linalg.norm(obj_states - obj_states[0], axis=1)
    sort_idx = np.argsort(obj_distances)
    obj_distances = obj_distances[sort_idx]

    _, _, vae_latent_distribution_params, lstm_latent_encodings = lstm(all_data_tensor)
    latents = lstm_latent_encodings
    vae_latents = vae_latent_distribution_params[0]

    traj_len, batch_size, feature_size = latents.shape
    latents = ptu.get_numpy(latents)
    latents = np.swapaxes(latents, 0, 1) # batch_size, traj_len, feature_size
    latents = latents.reshape((-1, feature_size))
    latent_distances = np.linalg.norm(latents - latents[0], axis=1)
    latent_distances = latent_distances[sort_idx]

    vae_latents = ptu.get_numpy(vae_latents).reshape((traj_len, batch_size, feature_size))
    vae_latents = vae_latents.swapaxes(0, 1)
    vae_latents = vae_latents.reshape((-1, feature_size))
    vae_latent_distances = np.linalg.norm(vae_latents - vae_latents[0], axis=1)
    vae_latent_distances = vae_latent_distances[sort_idx]  


    fig, axs = plt.subplots(2, 2, figsize=(20, 12))

    axs = axs.reshape(-1)

    obj_label = obj_name + '_distance' if obj_name == 'puck' else obj_name + '_angle_distance'
    ax = axs[0]
    ax2 = ax.twinx()
    ax.plot(obj_distances, label=obj_label, color='r')
    ax2.plot(latent_distances, label='latent_distances', color='b')
    ax.legend(loc='upper left')
    ax2.legend(loc='center left')

    ax = axs[1]
    ax2 = ax.twinx()
    ax.plot(obj_distances, label=obj_label, color='r')
    ax2.plot(vae_latent_distances, label='vae_latent_distances', color='b')
    ax.legend(loc='upper left')
    ax2.legend(loc='center left')

    ax = axs[2]
    ax.plot(obj_distances, latent_distances)
    ax.set_title("lstm latent distance")

    ax = axs[3]
    ax.plot(obj_distances, vae_latent_distances)
    ax.set_title("vae latent distance")
    
    if save_dir is not None and save_name is not None:
        save_path = osp.join(save_dir, save_name)
        plt.savefig(save_path)
    else:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close('all')
