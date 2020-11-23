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
import cv2, copy, os

def compare_latent_distance(vae, all_data, puck_pos, save_dir=None, save_name=None):
    all_data_tensor = ptu.from_numpy(all_data)
    latents, _ = vae.encode(all_data_tensor)
    latents = ptu.get_numpy(latents)

    puck_pos = puck_pos.reshape((-1, 2))
    puck_distances = np.linalg.norm(puck_pos - puck_pos[0], axis=1)
    latent_distances = np.linalg.norm(latents - latents[0], axis=1)
    sort_idx = np.argsort(puck_distances)
    puck_distances = puck_distances[sort_idx]
    latent_distances = latent_distances[sort_idx]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs = axs.reshape(-1)

    ax = axs[0]
    ax2 = ax.twinx()
    ax.plot(puck_distances, label='puck_distances', color='r')
    ax2.plot(latent_distances, label='latent_distances', color='b')
    ax.legend(loc='upper left')
    ax2.legend(loc='center left')

    ax = axs[1]
    ax.plot(puck_distances, latent_distances)
    ax.set_xlabel('puck distance')
    ax.set_ylabel('latent distance')

    if save_dir is not None and save_name is not None:
        save_path = osp.join(save_dir, save_name)
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close('all')

def test_vae_traj(vae, env_id, save_path=None, save_name=None):
    pjhome = os.environ['PJHOME']
    
    # optimal traj
    data_path = osp.join(pjhome, 'data/local/env/{}-optimal-traj.npy'.format(env_id))
    if not osp.exists(data_path):
        return
    imgs = np.load(data_path)
    traj_len, batch_size, imlen = imgs.shape
    imgs = imgs.reshape((-1, imlen))
    latents, _ = vae.encode(ptu.from_numpy(imgs))
    latents = ptu.get_numpy(latents)
    latent_distances = np.linalg.norm(latents - latents[-1], axis=1)
    puck_distances = np.load(osp.join(pjhome, 'data/local/env/{}-optimal-traj-puck-distance.npy'.format(env_id)))

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs = axs.reshape(-1)

    ax = axs[0]
    ax2 = ax.twinx()
    ax.plot(puck_distances, label='puck distance', color='r')
    ax2.plot(latent_distances, label='vae distance', color='b')
    ax.legend(loc='upper right')
    ax2.legend(loc='center right')

    # sub-optimal
    imgs = np.load(osp.join(pjhome, 'data/local/env/{}-local-optimal-traj.npy'.format(env_id)))
    goal_image = np.load(osp.join(pjhome, 'data/local/env/{}-local-optimal-traj-goal.npy'.format(env_id)))
    traj_len, batch_size, imlen = imgs.shape
    puck_distances = np.load(osp.join(pjhome, 'data/local/env/{}-local-optimal-traj-puck-distance.npy'.format(env_id)))

    imgs = imgs.reshape((-1, imlen))
    goal_image = goal_image.reshape((-1, imlen))
    latents = vae.encode(ptu.from_numpy(imgs))[0]
    latent_goal = vae.encode(ptu.from_numpy(goal_image))[0]

    latents = ptu.get_numpy(latents).reshape(traj_len, -1)
    latent_goal = ptu.get_numpy(latent_goal).flatten()
    latent_distances = np.linalg.norm(latents - latent_goal, axis=1)

    ax = axs[1]
    ax2 = ax.twinx()
    ax.plot(puck_distances, label='puck distance', color='r')
    ax2.plot(latent_distances, label='vae distance', color='b')
    ax.legend(loc='upper right')
    ax2.legend(loc='center right')

    plt.savefig(osp.join(save_path, save_name))
    plt.close('all')

def test_masked_traj_vae(env_id, vae, save_dir=None, save_name=None):
    # NOTE: acutally you can also generate the masked data online here
    pjhome = os.environ['PJHOME']
    ori_data = osp.join(pjhome, 'data/local/env/{}-masked-test-traj-ori.npy'.format(env_id))
    masked_data = osp.join(pjhome, 'data/local/env/{}-masked-test-traj-masked.npy'.format(env_id))
    masked_idx = osp.join(pjhome, 'data/local/env/{}-masked-idx.npy'.format(env_id))

    if not osp.exists(ori_data):
        return

    ori_data = np.load(ori_data)
    masked_data = np.load(masked_data)
    masked_idx = np.load(masked_idx)
    batch_size, traj_len, imlen = ori_data.shape
    
    ori_data = ori_data.reshape((-1, imlen))
    masked_data = masked_data.reshape((-1, imlen))

    latents_ori = vae.encode(ptu.from_numpy(ori_data))[0]
    latents_masked = vae.encode(ptu.from_numpy(masked_data))[0]
    latents_ori = ptu.get_numpy(latents_ori).reshape((batch_size, traj_len, -1))
    latents_masked = ptu.get_numpy(latents_masked).reshape((batch_size, traj_len, -1))
    batch_size, traj_len, feature_size = latents_ori.shape

    latents_masked_vectors = latents_masked[np.arange(batch_size), masked_idx] # batch_size x feature_size
    distances = latents_masked_vectors[:, np.newaxis, :] - latents_ori 
    assert distances.shape == (batch_size, traj_len, feature_size)
    distances = np.linalg.norm(distances, axis=-1) # batch_size x traj_len
    closest = np.argmin(distances, axis=-1) # batch size

    plt.plot(range(len(masked_idx)), masked_idx, label='true label')
    plt.plot(range(len(closest)), closest, label='prediction label')
    plt.legend()
    correct = np.sum(closest == masked_idx)
    total = batch_size
    acc = np.sum(correct) / batch_size
    plt.title("vae correct {}/{}, acc: {} ".format(correct, total, acc))
    if save_dir is not None:
        plt.savefig(osp.join(save_dir, save_name))
    else:
        plt.show()

    plt.close('all')
