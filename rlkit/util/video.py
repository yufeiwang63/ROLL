import os
import os.path as osp
import time

import numpy as np
import scipy.misc
import skvideo.io

from rlkit.envs.vae_wrapper import VAEWrappedEnv
from rlkit.envs.vae_wrapper_segmented import VAEWrappedEnvSegmented
from ROLL.LSTM_wrapped_env import LSTMWrappedEnv


def dump_video(
        env,
        policy,
        filename,
        rollout_function,
        rows=3,
        columns=6,
        pad_length=0,
        pad_color=255,
        do_timer=True,
        horizon=100,
        dirname_to_save_images=None,
        subdirname="rollouts",
        imsize=84,
        num_channels=3,
        fps=20,
):
    frames = []
    H = 3 * imsize
    W = imsize
    N = rows * columns
    for i in range(N):
        start = time.time()
        path = rollout_function(
            env,
            policy,
            max_path_length=horizon,
            render=False,
        )
        is_vae_env = isinstance(env, VAEWrappedEnv) 
        is_seg_vae_env = isinstance(env, VAEWrappedEnvSegmented)
        is_lstm_env = isinstance(env, LSTMWrappedEnv)
        l = []
        path_length = len(path['full_observations'])

        if is_lstm_env:
            img_traj = [d['image_observation'] for d in path['full_observations']]
            img_traj = np.asarray(img_traj)
            lstm_reconstructions = env._reconstruct_img_traj(img_traj)

        for d_idx, d in enumerate(path['full_observations']):
            if is_vae_env:
                recon = np.clip(env._reconstruct_img(d['image_observation']), 0,
                                1)
            elif is_seg_vae_env:
                recon_ori = np.clip(env._reconstruct_img(d['image_observation'], env.vae_original), 0,
                                1)
                recon_seg = np.clip(env._reconstruct_img(d['image_observation'], env.vae_segmented), 0,
                                1)
                H = 5 * imsize
            elif is_lstm_env:
                recon_ori = np.clip(env._reconstruct_img(d['image_observation'], env.vae_original), 0,
                                    1)
    
                recon_seg = np.clip(lstm_reconstructions[d_idx], 0,
                                1)
                H = 5 * imsize
            else:
                recon = d['image_observation']

            if is_vae_env:
                l.append(
                    get_image(
                        d['image_desired_goal'],
                        d['image_observation'],
                        recon,
                        pad_length=pad_length,
                        pad_color=pad_color,
                        imsize=imsize,
                    )
                )
            else:
                l.append(
                    get_image_seg(
                        d['image_desired_goal'],
                        d['image_observation'],
                        d['image_observation_segmented'],
                        recon_ori,
                        recon_seg,
                        pad_length=pad_length,
                        pad_color=pad_color,
                        imsize=imsize,
                    )
                )

        frames += l

        if dirname_to_save_images:
            rollout_dir = osp.join(dirname_to_save_images, subdirname, str(i))
            os.makedirs(rollout_dir, exist_ok=True)
            rollout_frames = frames[-101:]
            goal_img = np.flip(rollout_frames[0][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir + "/goal.png", goal_img)
            goal_img = np.flip(rollout_frames[1][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir + "/z_goal.png", goal_img)
            for j in range(0, 101, 1):
                img = np.flip(rollout_frames[j][imsize:, :imsize, :], 0)
                scipy.misc.imsave(rollout_dir + "/" + str(j) + ".png", img)
        if do_timer:
            print(i, time.time() - start)

    frames = np.array(frames, dtype=np.uint8)
    # print(frames.shape)
    # path_length = frames.size // (
    #         N * (H + 2 * pad_length) * (W + 2 * pad_length) * num_channels
    # )
    frames = np.array(frames, dtype=np.uint8).reshape(
        (N, path_length, H + 2 * pad_length, W + 2 * pad_length, num_channels)
    )
    f1 = []
    for k1 in range(columns):
        f2 = []
        for k2 in range(rows):
            k = k1 * rows + k2
            f2.append(frames[k:k + 1, :, :, :, :].reshape(
                (path_length, H + 2 * pad_length, W + 2 * pad_length,
                 num_channels)
            ))
        f1.append(np.concatenate(f2, axis=1))
    outputdata = np.concatenate(f1, axis=2)
    skvideo.io.vwrite(filename, outputdata, inputdict={'-r': str(fps)}, outputdict={'-r': str(fps)})
    print("Saved video to ", filename)


def get_image(goal, obs, recon_obs, imsize=84, pad_length=1, pad_color=255):
    if len(goal.shape) == 1:
        goal = goal.reshape(-1, imsize, imsize).transpose()
        obs = obs.reshape(-1, imsize, imsize).transpose()
        recon_obs = recon_obs.reshape(-1, imsize, imsize).transpose()
        goal = goal[::-1, :, :]
        obs = obs[::-1, :, :]
        recon_obs = recon_obs[::-1, :, :]
    img = np.concatenate((goal, obs, recon_obs))
    img = np.uint8(255 * img)
    if pad_length > 0:
        img = add_border(img, pad_length, pad_color)
    return img

def get_image_seg(goal, obs, obs_seg, recon_obs_ori, recon_obs_seg, imsize=84, pad_length=1, pad_color=255):
    if len(goal.shape) == 1:
        goal = goal.reshape(-1, imsize, imsize).transpose()
        obs = obs.reshape(-1, imsize, imsize).transpose()
        obs_seg = obs_seg.reshape(-1, imsize, imsize).transpose()
        recon_obs_seg = recon_obs_seg.reshape(-1, imsize, imsize).transpose()
        recon_obs_ori = recon_obs_ori.reshape(-1, imsize, imsize).transpose()
        goal = goal[::-1, :, :]
        obs = obs[::-1, :, :]
        obs_seg = obs_seg[::-1, :, :]
        recon_obs_seg = recon_obs_seg[::-1, :, :]
        recon_obs_ori = recon_obs_ori[::-1, :, :]
    img = np.concatenate((goal, obs, obs_seg, recon_obs_ori, recon_obs_seg))
    img = np.uint8(255 * img)
    if pad_length > 0:
        img = add_border_2(img, pad_length, pad_color)
    # print(img.shape)
    return img


def add_border(img, pad_length, pad_color, imsize=84):
    H = 3 * imsize
    W = imsize
    img = img.reshape((3 * imsize, imsize, -1))
    img2 = np.ones((H + 2 * pad_length, W + 2 * pad_length, img.shape[2]),
                   dtype=np.uint8) * pad_color
    img2[pad_length:-pad_length, pad_length:-pad_length, :] = img
    return img2

def add_border_2(img, pad_length, pad_color, imsize=84):
    H = 4 * imsize
    W = imsize
    img = img.reshape((4 * imsize, imsize, -1))
    img2 = np.ones((H + 2 * pad_length, W + 2 * pad_length, img.shape[2]),
                   dtype=np.uint8) * pad_color
    img2[pad_length:-pad_length, pad_length:-pad_length, :] = img
    return img2
