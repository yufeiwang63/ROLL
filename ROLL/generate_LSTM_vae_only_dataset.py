import numpy as np
from segmentation.segment_image import (
    segment_image_unet, 
    train_bgsb, 
)
import copy
import os
from os import path as osp
invisiable_env_id = {
    'SawyerPushNIPSEasy-v0': "SawyerPushNIPSPuckInvisible-v0",
    'SawyerPushHurdle-v0': 'SawyerPushHurdlePuckInvisible-v0',
    'SawyerPushT-v0': 'SawyerPushTPuckInvisible-v0',
}

def generate_LSTM_vae_only_dataset(variant, segmented=False, segmentation_method='color'):
    from multiworld.core.image_env import ImageEnv, unormalize_image

    env_id = variant.get('env_id', None)
    N = variant.get('N', 500)
    test_p = variant.get('test_p', 0.9)
    imsize = variant.get('imsize', 48)
    num_channels = variant.get('num_channels', 3)
    init_camera = variant.get('init_camera', None)
    occlusion_prob = variant.get('occlusion_prob', 0)
    occlusion_level = variant.get('occlusion_level', 0.5)
    segmentation_kwargs = variant.get('segmentation_kwargs', {})
    if segmentation_kwargs.get('segment') is not None:
        segmented = segmentation_kwargs.get('segment')

    assert env_id is not None, 'you must provide an env id!'

    obj = 'puck-pos'
    if env_id == 'SawyerDoorHookResetFreeEnv-v1':
        obj = 'door-angle'

    pjhome = os.environ['PJHOME']
    if segmented:
        if 'unet' in segmentation_method:
            seg_name = 'seg-unet'
        else:
            seg_name = 'seg-' + segmentation_method
    else:
        seg_name = 'no-seg'

    if env_id == 'SawyerDoorHookResetFreeEnv-v1':
        seg_name += '-2'

    data_file_path = osp.join(pjhome, 'data/local/pre-train-lstm', 'vae-only-{}-{}-{}-{}-{}.npy'.format(
        env_id, seg_name, N, occlusion_prob, occlusion_level))
    obj_state_path = osp.join(pjhome, 'data/local/pre-train-lstm', 'vae-only-{}-{}-{}-{}-{}-{}.npy'.format(
        env_id, seg_name, N, occlusion_prob, occlusion_level, obj))
    
    print(data_file_path)
    if osp.exists(data_file_path):
        all_data = np.load(data_file_path)
        if len(all_data) >= N:
            print("load stored data at: ", data_file_path)
            n = int(len(all_data) * test_p)
            train_dataset = all_data[:n]
            test_dataset = all_data[n:]
            obj_states = np.load(obj_state_path)
            info = {'obj_state': obj_states}
            return train_dataset, test_dataset, info

    if segmented:
        print("generating lstm vae pretrain only dataset with segmented images using method: ", segmentation_method)
        if segmentation_method == 'unet':
            segment_func = segment_image_unet
        else:
            raise NotImplementedError
    else:
        print("generating lstm vae pretrain only dataset with original images")

    info = {}
    dataset = np.zeros((N, imsize * imsize * num_channels), dtype=np.uint8)
    imgs = []
    obj_states = None

    if env_id == 'SawyerDoorHookResetFreeEnv-v1':
        from rlkit.util.io import load_local_or_remote_file
        pjhome = os.environ['PJHOME']
        pre_sampled_goal_path = osp.join(
                pjhome, 
                'data/local/pre-train-vae/door_original_dataset.npy'
            )
        goal_dict = np.load(pre_sampled_goal_path, allow_pickle=True).item()
        imgs = goal_dict['image_desired_goal']
        door_angles = goal_dict['state_desired_goal'][:, -1]
        obj_states = door_angles[:, np.newaxis]
    elif env_id == 'SawyerPickupEnvYZEasy-v0':
        from rlkit.util.io import load_local_or_remote_file
        pjhome = os.environ['PJHOME']
        pre_sampled_goal_path = osp.join(
                pjhome, 
                'data/local/pre-train-vae/pickup-original-dataset.npy'
            )
        goal_dict = load_local_or_remote_file(pre_sampled_goal_path).item()
        imgs = goal_dict['image_desired_goal']
        puck_pos = goal_dict['state_desired_goal'][:, 3:]
        obj_states = puck_pos
    
    else:
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
        env.reset()
        info['env'] = env

        puck_pos = np.zeros((N, 2), dtype=np.float)
        for i in range(N):
            print("lstm vae pretrain only dataset generation, number: ", i)
            if env_id == 'SawyerPushHurdle-v0':
                obs, puck_p = _generate_sawyerhurdle_dataset(env, return_puck_pos=True, segmented=segmented)
            elif env_id == 'SawyerPushHurdleMiddle-v0':
                obs, puck_p = _generate_sawyerhurdlemiddle_dataset(env, return_puck_pos=True)
            elif env_id == 'SawyerPushNIPSEasy-v0':
                obs, puck_p = _generate_sawyerpushnipseasy_dataset(env, return_puck_pos=True)
            elif env_id == 'SawyerPushHurdleResetFreeEnv-v0':
                obs, puck_p = _generate_sawyerhurldeblockresetfree_dataset(env, return_puck_pos=True)
            else: 
                raise NotImplementedError
            img = obs['image_observation'] # NOTE: this is already normalized image, of detype np.float64.
            imgs.append(img)
            puck_pos[i] = puck_p

        obj_states = puck_pos
           
    # now we segment the images
    for i in range(N):
        print("segmenting image ", i)
        img = imgs[i]
        if segmented:
            dataset[i, :] = segment_func(img, normalize=False, **segmentation_kwargs)
            p = np.random.rand() # manually drop some images, so as to make occlusions
            if p < occlusion_prob:
                mask = (np.random.uniform(low=0, high=1, size=(imsize, imsize)) > occlusion_level).astype(np.uint8)
                img = dataset[i].reshape(3, imsize, imsize).transpose()
                img[mask < 1] = 0
                dataset[i] = img.transpose().flatten()

        else:
            dataset[i, :] = unormalize_image(img)

    # add the trajectory dimension
    dataset = dataset[:, np.newaxis, :] # batch_size x traj_len = 1 x imlen
    obj_states = obj_states[:, np.newaxis, :] # batch_size x traj_len = 1 x imlen
    info['obj_state'] = obj_states

    n = int(N * test_p)
    train_dataset = dataset[:n]
    test_dataset = dataset[n:]
    
    if N >= 500:
        print('save data to: ', data_file_path)
        all_data = np.concatenate([train_dataset, test_dataset], axis=0)
        np.save(data_file_path, all_data)
        np.save(obj_state_path, obj_states)

    return train_dataset, test_dataset, info


def _generate_sawyerhurdle_dataset(env, return_puck_pos=False, segmented=True):
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

    # put hand out of scene
    if segmented:
        hand_x, hand_y = -0.2, 0.2
    else:
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

    # put hand out of scene
    hand_x, hand_y = -0.2, 0.2
    # hand_x = np.random.uniform(x_range[0], x_range[1])
    # hand_y = np.random.uniform(y_range[0], y_range[1])

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


def _generate_sawyerpushnipseasy_dataset(env, return_puck_pos=True):
    goal = env.sample_goal()
    env.set_to_goal(goal)
    obs = env._get_obs()

    puck_pos = goal['state_desired_goal'][-2:]
    if return_puck_pos:
        return obs, puck_pos
    return obs

def _generate_sawyerhurldeblockresetfree_dataset(env, return_puck_pos=True):
    goal = env.sample_goal()
    env.set_to_goal(goal)
    obs = env._get_obs()

    puck_pos = goal['state_desired_goal'][-2:]
    if return_puck_pos:
        return obs, puck_pos
    return obs

