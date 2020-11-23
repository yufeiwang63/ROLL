import numpy as np
from segmentation.segment_image import segment_image_unet
import copy
import os
from os import path as osp
invisiable_env_id = {
    'SawyerPushNIPSEasy-v0': "SawyerPushNIPSPuckInvisible-v0",
    'SawyerPushHurdle-v0': 'SawyerPushHurdlePuckInvisible-v0',
    'SawyerPushT-v0': 'SawyerPushTPuckInvisible-v0',
}

def generate_sawyerhurdle_dataset(variant, segmented=False, segmentation_method='unet'):
    from multiworld.core.image_env import ImageEnv, unormalize_image

    env_id = variant.get('env_id', None)
    N = variant.get('N', 10000)
    test_p = variant.get('test_p', 0.9)
    imsize = variant.get('imsize', 84)
    num_channels = variant.get('num_channels', 3)
    init_camera = variant.get('init_camera', None)
    segmentation_kwargs = variant.get('segmentation_kwargs', {})

    pjhome = os.environ['PJHOME']
    seg_name = 'seg-' + segmentation_method if segmented else 'no-seg'
    data_file_path = osp.join(pjhome, 'data/local/pre-train-vae', '{}-{}-{}.npy'.format(env_id, seg_name, N))
    puck_pos_path = osp.join(pjhome, 'data/local/pre-train-vae', '{}-{}-{}-puck-pos.npy'.format(env_id, seg_name, N))

    if osp.exists(data_file_path):
        all_data = np.load(data_file_path)
        if len(all_data) >= N:
            print("load stored data at: ", data_file_path)
            n = int(len(all_data) * test_p)
            train_dataset = all_data[:n]
            test_dataset = all_data[n:]
            puck_pos = np.load(puck_pos_path)
            info = {'puck_pos': puck_pos}
            return train_dataset, test_dataset, info

    if segmented:
        print("generating vae dataset with segmented images using method: ", segmentation_method)
        if segmentation_method == 'unet':
            segment_func = segment_image_unet
        else:
            raise NotImplementedError
    else:
        print("generating vae dataset with original images")
    
    assert env_id is not None
    import gym
    import multiworld
    multiworld.register_all_envs()
    env = gym.make(env_id)
    
    if not isinstance(env, ImageEnv):
        env = ImageEnv(
            env,
            imsize,
            init_camera=init_camera,
            transpose=True,
            normalize=True,
        )
    
    info = {}
    env.reset()
    info['env'] = env

    dataset = np.zeros((N, imsize * imsize * num_channels),
                        dtype=np.uint8)
    puck_pos = np.zeros((N, 2), dtype=np.float)

    for i in range(N):
        print("sawyer hurdle custom vae data set generation, number: ", i)
        if env_id == 'SawyerPushHurdle-v0':
            obs, puck_p = _generate_sawyerhurdle_dataset(env, return_puck_pos=True)
        elif env_id == 'SawyerPushHurdleMiddle-v0':
            obs, puck_p = _generate_sawyerhurdlemiddle_dataset(env, return_puck_pos=True)
        else:
            raise NotImplementedError
        img = obs['image_observation'] # NOTE yufei: this is already normalized image, of detype np.float64.
    
        if segmented:
            dataset[i, :] = segment_func(img, normalize=False, **segmentation_kwargs)
        else:
            dataset[i, :] = unormalize_image(img)
        puck_pos[i] = puck_p

    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]

    info['puck_pos'] = puck_pos

    if N >= 2000:
        print('save data to: ', data_file_path)
        all_data = np.concatenate([train_dataset, test_dataset], axis=0)
        np.save(data_file_path, all_data)
        np.save(puck_pos_path, puck_pos)

    return train_dataset, test_dataset, info


def _generate_sawyerhurdle_dataset(env, return_puck_pos=False):
    # y location: [0.54, 0.67]
    # top part y: [0.54, 0.6]
    # lower part y: [0.65, 0.67]

    # x location for the left hurdle part: [0.095, 0.11]
    # x location for the right hurdle part: [-0.03, 0.0]
    
    # left part:
    y_range_left = [0.54, 0.65]
    x_range_left = [0.095, 0.11]

    # right part:
    y_range_right = [0.54, 0.65]
    x_range_right = [-0.03, 0.]

    # lower part:
    y_range_lower = [0.65, 0.67]
    x_range_lower = [-0.03, 0.11]

    # uniformly sample a puck
    p = np.random.uniform()
    if p < 0.33:
        x_range, y_range = x_range_left, y_range_left
    elif p > 0.33 and p < 0.67:
        x_range, y_range = x_range_right, y_range_right
    else:
        x_range, y_range = x_range_lower, y_range_lower

    puck_x = np.random.uniform(x_range[0], x_range[1])
    puck_y = np.random.uniform(y_range[0], y_range[1])

    # uniformly sample a hand
    p = np.random.uniform()
    if p < 0.33:
        x_range, y_range = x_range_left, y_range_left
    elif p > 0.33 and p < 0.67:
        x_range, y_range = x_range_right, y_range_right
    else:
        x_range, y_range = x_range_lower, y_range_lower

    hand_x = np.random.uniform(x_range[0], x_range[1])
    hand_y = np.random.uniform(y_range[0], y_range[1])

    goal = env.sample_goal()
    goal['state_desired_goal'][-2] = puck_x
    goal['state_desired_goal'][-1] = puck_y
    goal['state_desired_goal'][0] = hand_x
    goal['state_desired_goal'][1] = hand_y
    env.set_to_goal(goal)
    obs = env._get_obs()

    if return_puck_pos:
        return obs, np.array([puck_x, puck_y])
    return obs

def _generate_sawyerhurdlemiddle_dataset(env, return_puck_pos=False):
    # y location: [0.52, 0.67]
    # top part y: [0.52, 0.55]
    # lower part y: [0.65, 0.67]

    # x location for the left hurdle part: [0.095, 0.11]
    # x location for the right hurdle part: [-0.025, 0.0]
    
    # left part:
    y_range_left = [0.52, 0.67]
    x_range_left = [0.09, 0.11]

    # right part:
    y_range_right = [0.52, 0.67]
    x_range_right = [-0.025, 0.]

    # upper part:
    y_range_upper = [0.52, 0.55]
    x_range_upper = [-0.025, 0.11]

    # uniformly sample a puck position
    p = np.random.uniform()
    if p < 0.33:
        x_range, y_range = x_range_left, y_range_left
    elif p > 0.33 and p < 0.67:
        x_range, y_range = x_range_right, y_range_right
    else:
        x_range, y_range = x_range_upper, y_range_upper

    puck_x = np.random.uniform(x_range[0], x_range[1])
    puck_y = np.random.uniform(y_range[0], y_range[1])

    # uniformly sample a hand position
    p = np.random.uniform()
    if p < 0.33:
        x_range, y_range = x_range_left, y_range_left
    elif p > 0.33 and p < 0.67:
        x_range, y_range = x_range_right, y_range_right
    else:
        x_range, y_range = x_range_upper, y_range_upper

    hand_x = np.random.uniform(x_range[0], x_range[1])
    hand_y = np.random.uniform(y_range[0], y_range[1])

    goal = env.sample_goal()
    goal['state_desired_goal'][-2] = puck_x
    goal['state_desired_goal'][-1] = puck_y
    goal['state_desired_goal'][0] = hand_x
    goal['state_desired_goal'][1] = hand_y
    env.set_to_goal(goal)
    obs = env._get_obs()

    if return_puck_pos:
        return obs, np.array([puck_x, puck_y])
    return obs




