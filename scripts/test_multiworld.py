import multiworld
import gym
import cv2
from multiworld.envs.mujoco.cameras import (
    sawyer_init_camera_full, 
    sawyer_init_camera_zoomed_in, 
    sawyer_pick_and_place_camera,
    sawyer_door_env_camera_v0
)
from multiworld.core.image_env import ImageEnv
import copy
import numpy as np

def show_obs(normalized_img_vec_, imsize=48, name='img'):
    print(name)
    normalized_img_vec = copy.deepcopy(normalized_img_vec_)
    img = (normalized_img_vec * 255).astype(np.uint8)
    img = img.reshape(3, imsize, imsize).transpose()
    img = img[::-1, :, ::-1]
    cv2.imshow(name, img)
    cv2.waitKey()

if __name__ == '__main__':
    # env_name = 'SawyerDoorHookResetFreeEnv-v1'
    env_name = 'SawyerPushHurdle-v0'
    # env_name = 'SawyerPushNIPSFull-v0'
    # env_name = 'SawyerPushNIPSEasy-v0'
    # env_name = 'SawyerPushHurdleResetFreeEnv-v0'
    multiworld.register_all_envs()
    imsize = 48

    # presampled_goals_path = 'data/local/goals/SawyerDoorHookResetFreeEnv-v1-goal.npy'
    # presampled_goals_path = 'data/local/goals/SawyerPickupEnvYZEasy-v0-goal-500.npy'
    # presampled_goals = np.load(presampled_goals_path, allow_pickle=True).item()

    env = ImageEnv(
            env,
            imsize, 
            init_camera=sawyer_init_camera_zoomed_in,
            transpose=True,
            normalize=True,
            presampled_goals=None
        )

    print(env.action_space.low)
    print(env.action_space.high)

    for i in range(50):
        o = env.reset()
        for t in range(50):
            print(t)
            action = env.action_space.sample()
            s, r, _, _ = env.step(action)
            img = s['image_observation']
            show_obs(img, imsize=imsize, name=env_name)
