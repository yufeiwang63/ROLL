import argparse
import pickle

from rlkit.core import logger
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch import pytorch_util as ptu
from rlkit.envs.vae_wrapper import VAEWrappedEnv
from ROLL.LSTM_wrapped_env import LSTMWrappedEnv
import torch
import os.path as osp
import json
from segmentation.segment_image import train_bgsb
from multiworld.envs.mujoco.cameras import (
    sawyer_init_camera_full, 
    sawyer_init_camera_zoomed_in, 
    sawyer_pick_and_place_camera,
    sawyer_door_env_camera_v0
)


cameras = {
    'SawyerPushNIPSEasy-v0': sawyer_init_camera_zoomed_in,
    'SawyerPushHurdle-v0': sawyer_init_camera_zoomed_in,
    'SawyerPushHurdleMiddle-v0': sawyer_init_camera_zoomed_in,
    'SawyerDoorHookResetFreeEnv-v1': sawyer_door_env_camera_v0,
    'SawyerPickupEnvYZEasy-v0': sawyer_pick_and_place_camera,
}

def rollout(*args, **kwargs):
    return multitask_rollout(*args, **kwargs, 
                                observation_key='latent_observation',
                                desired_goal_key='latent_desired_goal', )

def train_bg(variant):
    import numpy as np

    invisiable_env_id = {
        'SawyerPushNIPSEasy-v0': "SawyerPushNIPSPuckInvisible-v0",
        'SawyerPushHurdle-v0': 'SawyerPushHurdlePuckInvisible-v0',
        'SawyerPushHurdleMiddle-v0': 'SawyerPushHurdleMiddlePuckInvisible-v0',
        'SawyerDoorHookResetFreeEnv-v1': 'SawyerDoorHookResetFreeEnvDoorInvisible-v0',
        'SawyerPickupEnvYZEasy-v0': 'SawyerPickupResetFreeEnvBallInvisible-v0',
    }

    print("training opencv background model!")
    env_id = variant.get('env_id', None)
    env_id_invis = invisiable_env_id[env_id]
    import gym
    import multiworld
    from multiworld.core.image_env import ImageEnv
    multiworld.register_all_envs()
    obj_invisible_env = gym.make(env_id_invis)
    init_camera = variant.get('init_camera', None)
    
    presampled_goals = None
    if variant.get("presampled_goals_path") is not None:
        presampled_goals = np.load(
                variant['presampled_goals_path'], allow_pickle=True
            ).item()
        print("presampled goal path is: ", variant['presampled_goals_path'])
        # print("presampled goals are: ", presampled_goals)

    obj_invisible_env = ImageEnv(
        obj_invisible_env,
        variant.get('imsize'),
        init_camera=init_camera,
        transpose=True,
        normalize=True,
        presampled_goals=presampled_goals,
    )
    train_bgsb(obj_invisible_env)

def simulate_policy(args):
    data = torch.load(osp.join(args.dir, 'params.pkl'))
    policy = data['evaluation/policy']
    env = data['evaluation/env']

    variant = json.load(open(osp.join(args.dir, 'variant.json')))
    variant = variant['variant']
    bg_variant = {
        'env_id': variant['env_id'],
        'imsize': variant['imsize'],
        'init_camera': cameras[variant['env_id']],
        'presampled_goals_path': variant['skewfit_variant'].get('presampled_goals_path')
    }
    train_bg(bg_variant)
    imsize = variant['imsize']

    env._goal_sampling_mode = 'reset_of_env'
    if args.gpu:
        ptu.set_gpu_mode(True)
        policy.stochastic_policy.to(ptu.device)
    print("Policy and environment loaded")

    env.reset()
    env.reset()
    env.decode_goals = False

    from rlkit.util.video import dump_video
    save_dir = osp.join(args.dir, 'visual-tmp.gif')
    dump_video(
        env, policy, save_dir, rollout, horizon=args.H, imsize=imsize, rows=1, columns=8, fps=30,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=50,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    simulate_policy(args)
