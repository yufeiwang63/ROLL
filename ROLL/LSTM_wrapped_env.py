import copy
import random
import warnings

import torch

import cv2
import numpy as np
from gym.spaces import Box, Dict
import rlkit.torch.pytorch_util as ptu
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict
from rlkit.envs.wrappers import ProxyEnv
from segmentation.segment_image import (
    segment_image_unet
)
from ROLL.LSTM_model import ConvLSTM2
from multiworld.core.image_env import normalize_image
import time

class LSTMWrappedEnv(ProxyEnv, MultitaskEnv):
    """This class wraps an image-based environment with a LSTM.
    Assumes you get flattened (channels,84,84) observations from wrapped_env.
    This class adheres to the "Silent Multitask Env" semantics: on reset,
    it resamples a goal.
    """
    def __init__(
        self,
        wrapped_env,
        vae_original, # for unsegmented images
        lstm_segmented, # segmeneted images
        segmentation=True,
        segmentation_method='unet',
        vae_input_key_prefix='image',
        sample_from_true_prior=False,
        decode_goals=False,
        render_goals=False,
        render_rollouts=False,
        reward_params=None,
        goal_sampling_mode="vae_prior",
        imsize=84,
        obs_size=None,
        norm_order=2,
        epsilon=20,
        presampled_goals=None,
        segmentation_kwargs=dict(),
        observation_mode='original_image', 
    ):
        if reward_params is None:
            reward_params = dict()
        super().__init__(wrapped_env)
        self.vae_original = vae_original
        self.lstm_segmented = lstm_segmented
        self.representation_size = self.vae_original.representation_size
        self.input_channels = self.lstm_segmented.input_channels
        self.sample_from_true_prior = sample_from_true_prior
        self._decode_goals = decode_goals
        self.render_goals = render_goals
        self.render_rollouts = render_rollouts
        self.default_kwargs=dict(
            decode_goals=decode_goals,
            render_goals=render_goals,
            render_rollouts=render_rollouts,
        )
        self.imsize = imsize
        self.reward_params = reward_params
        self.reward_type = self.reward_params.get("type", 'latent_distance')
        self.norm_order = self.reward_params.get("norm_order", norm_order)
        self.epsilon = self.reward_params.get("epsilon", epsilon)
        self.reward_min_variance = self.reward_params.get("min_variance", 0)
        
        assert observation_mode == 'original_image'
        if observation_mode == 'original_image':
            latent_space_obs = Box(
                -10 * np.ones(obs_size or self.vae_original.representation_size),
                10 * np.ones(obs_size or self.vae_original.representation_size),
                dtype=np.float32,
            )


        latent_space_goal = Box(
            -10 * np.ones(obs_size or self.lstm_segmented.representation_size),
            10 * np.ones(obs_size or self.lstm_segmented.representation_size),
            dtype=np.float32,
        )
        
        self.segmentation = segmentation
        self.segmentation_method = segmentation_method
        self.segmentation_kwargs = segmentation_kwargs
        if self.segmentation_method == 'unet':
            self.segment_func = segment_image_unet
        else:
            raise NotImplementedError

        assert observation_mode in ['original_image', 'segmentation_proprio_cross_weight', 'segmentation_proprio_conv_concat'], \
                'unspported observation mode!'
        self.observation_mode = observation_mode

        spaces = self.wrapped_env.observation_space.spaces
        
        # NOTE: we only differentiate between the segmented and non-segmented observations;
        # the goals are by default segmented
        spaces['observation'] = latent_space_obs
        spaces['desired_goal'] = latent_space_goal
        spaces['achieved_goal'] = latent_space_goal
        spaces['latent_observation'] = latent_space_obs
        spaces['latent_desired_goal'] = latent_space_goal
        spaces['latent_achieved_goal'] = latent_space_goal
        spaces['image_observation_segmented'] = spaces['image_observation']
        

        self.observation_space = Dict(spaces)
        self._presampled_goals = presampled_goals
        if self._presampled_goals is None:
            self.num_goals_presampled = 0
        else:
            self.num_goals_presampled = presampled_goals[random.choice(list(presampled_goals))].shape[0]

        self.vae_input_key_prefix = vae_input_key_prefix
        assert vae_input_key_prefix in {'image', 'image_proprio'}
        self.vae_input_observation_key = vae_input_key_prefix + '_observation'
        self.lstm_input_observation_key_segmented = vae_input_key_prefix + '_observation' + '_segmented'
        self.vae_input_achieved_goal_key = vae_input_key_prefix + '_achieved_goal'
        self.vae_input_desired_goal_key = vae_input_key_prefix + '_desired_goal'
        self._mode_map = {}
        self.desired_goal = {'latent_desired_goal': latent_space_goal.sample()}
        self._initial_obs = None
        self._custom_goal_sampler = None
        self._goal_sampling_mode = goal_sampling_mode

    def reset(self):
        obs = self.wrapped_env.reset()
        self.lstm_hidden = self.lstm_segmented.init_hidden()
        goal = self.sample_goal()
        self.set_goal(goal)
        self._initial_obs = obs
        return self._update_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        self._update_info(info, new_obs) 
        reward = self.compute_reward(
            action,
            {'latent_achieved_goal': new_obs['latent_achieved_goal'],
             'latent_desired_goal': new_obs['latent_desired_goal']}
        )
        self.try_render(new_obs)
        return new_obs, reward, done, info

    def segment_obs(self, img):
        return self.segment_func(img, normalize=True, **self.segmentation_kwargs)

    def show_obs(self, normalized_img_vec_, name='img'):
        print(name)
        normalized_img_vec = copy.deepcopy(normalized_img_vec_)
        img = (normalized_img_vec * 255).astype(np.uint8)
        img = img.reshape(3, self.imsize, self.imsize).transpose()
        img = img[::-1, :, ::-1]
        cv2.imshow(name, img)
        cv2.waitKey()
    
    def _update_obs(self, obs):
        segmented_obs = self.segment_obs(obs['image_observation'])

        obs['image_observation_segmented'] = segmented_obs # this will be needed for training the segmented vae
        obs['image_achieved_goal'] = segmented_obs # this will be needed when refreshing the replay buffer

        assert self.vae_input_observation_key == 'image_observation'
        assert self.lstm_input_observation_key_segmented == 'image_observation_segmented'

        latent_obs_segmented = self._encode_lstm_one(obs[self.lstm_input_observation_key_segmented], 
            self.lstm_segmented, hidden=self.lstm_hidden, update_hidden=True) 
        obs['latent_achieved_goal'] = latent_obs_segmented
        obs['achieved_goal'] = latent_obs_segmented

        if self.observation_mode == 'original_image':
            latent_obs_ori = self._encode_one(obs[self.vae_input_observation_key], self.vae_original) 
            obs['latent_observation'] = latent_obs_ori
            obs['observation'] = latent_obs_ori
        
        obs = {**obs, **self.desired_goal}
        
        return obs

    def _update_info(self, info, obs):
        pass


    """
    Multitask functions
    """
    def sample_goals(self, batch_size):
        # TODO: make mode a parameter you pass in

        if self._goal_sampling_mode == 'custom_goal_sampler':
            # NOTE LSTM: for now we do not skew the goal distribution in the LSTM version.
            return self.custom_goal_sampler(batch_size, skew=False)
        elif self._goal_sampling_mode == 'presampled':
            idx = np.random.randint(0, self.num_goals_presampled, batch_size)
            sampled_goals = {
                k: v[idx] for k, v in self._presampled_goals.items()
            }
            # ensures goals are encoded using latest lstm
            if 'image_desired_goal' in sampled_goals:                
                segmented_image_desired_goals = np.zeros_like(sampled_goals['image_desired_goal'])
                for idx in range(batch_size):
                    segmented_image_desired_goals[idx] = self.segment_obs(sampled_goals['image_desired_goal'][idx])
                
                sampled_goals['latent_desired_goal'] = self._encode_lstm(
                    segmented_image_desired_goals, 
                    self.lstm_segmented,
                    None,
                    False
                )
            return sampled_goals
        elif self._goal_sampling_mode == 'env':
            goals = self.wrapped_env.sample_goals(batch_size)
            if 'image_desired_goal' in goals:                
                segmented_image_desired_goals = np.zeros_like(goals['image_desired_goal'])
                for idx in range(batch_size):
                    segmented_image_desired_goals[idx] = self.segment_obs(goals['image_desired_goal'][idx])
                
                goals['latent_desired_goal'] = self._encode_lstm(
                    segmented_image_desired_goals, 
                    self.lstm_segmented,
                    None,
                    False
                )        

            latent_goals = goals['latent_desired_goal']
        elif self._goal_sampling_mode == 'reset_of_env':
            assert batch_size == 1
            goal = self.wrapped_env.get_goal()
            if self.segmentation:
                goal_image = goal[self.vae_input_desired_goal_key]
                goal[self.vae_input_desired_goal_key] = self.segment_obs(goal_image)
            
            goals = {k: v[None] for k, v in goal.items()}
            
            latent_goals = self._encode_lstm(
                goals[self.vae_input_desired_goal_key], self.lstm_segmented, None, False
            )
        elif self._goal_sampling_mode == 'vae_prior': # actually lstm prior
            goals = {}
            latent_goals = self._sample_latent_prior(batch_size, self.lstm_segmented)
        else:
            raise RuntimeError("Invalid: {}".format(self._goal_sampling_mode))

        # for latent sampled goals, we need to decode them into images, becasue once the 
        # vae / lstm is trained, the latent space change, and the old latent goals become invalid,
        # and we need to use the updated vae / lstm encode the decoded images again 
        # to obtain the new valid updated latent.
        if self._decode_goals:
            decoded_goals = self._decode(latent_goals, self.lstm_segmented)
        else:
            decoded_goals = None
        image_goals, proprio_goals = self._image_and_proprio_from_decoded(
            decoded_goals
        )

        goals['desired_goal'] = latent_goals
        goals['latent_desired_goal'] = latent_goals
        if proprio_goals is not None:
            goals['proprio_desired_goal'] = proprio_goals
        if image_goals is not None:
            goals['image_desired_goal'] = image_goals
        if decoded_goals is not None:
            goals[self.vae_input_desired_goal_key] = decoded_goals
        return goals

    def get_goal(self):
        return self.desired_goal

    def compute_reward(self, action, obs):
        actions = action[None]
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        return self.compute_rewards(actions, next_obs)[0]

    def compute_rewards(self, actions, obs, show=False):
        # TODO: implement log_prob/mdist
        if self.reward_type == 'latent_distance':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']

            assert achieved_goals.shape[1] == self.lstm_segmented.representation_size
            assert desired_goals.shape[1] == self.lstm_segmented.representation_size

            if show is True:
                achieved_goals_decoded = self._decode(achieved_goals, self.vae_original)
                desired_goals_decoded = self._decode(desired_goals, self.vae_original)

                latent_obs = obs['latent_observation']
                latent_obs_decoded = self._decode(latent_obs, self.vae_original)

                for idx in range(10):
                    self.show_obs(achieved_goals_decoded[idx], "reward achieved_goal")
                    self.show_obs(desired_goals_decoded[idx], "reward desired_goal")
                    self.show_obs(latent_obs_decoded[idx], "reward observation")

            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            return -dist
        elif self.reward_type == 'vectorized_latent_distance':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            return -np.abs(desired_goals - achieved_goals)
        elif self.reward_type == 'latent_sparse':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            reward = 0 if dist < self.epsilon else -1
            return reward
        elif self.reward_type == 'state_distance':
            achieved_goals = obs['state_achieved_goal']
            desired_goals = obs['state_desired_goal']
            return - np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
        elif self.reward_type == 'wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)
        else:
            raise NotImplementedError

    @property
    def goal_dim(self):
        return self.representation_size

    def set_goal(self, goal):
        """
        Assume goal contains both image_desired_goal and any goals required for wrapped envs

        :param goal:
        :return:
        """
        self.desired_goal = goal
        # TODO: fix this hack / document this
        if self._goal_sampling_mode in {'presampled', 'env'}:
            self.wrapped_env.set_goal(goal)

    def get_diagnostics(self, paths, **kwargs):
        statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)
        return statistics

    """
    Other functions
    """
    @property
    def goal_sampling_mode(self):
        return self._goal_sampling_mode

    @goal_sampling_mode.setter
    def goal_sampling_mode(self, mode):
        assert mode in [
            'custom_goal_sampler',
            'presampled',
            'vae_prior',
            'env',
            'reset_of_env'
        ], "Invalid env mode"
        self._goal_sampling_mode = mode
        if mode == 'custom_goal_sampler':
            test_goals = self.custom_goal_sampler(1)
            if test_goals is None:
                self._goal_sampling_mode = 'vae_prior'
                warnings.warn(
                    "self.goal_sampler returned None. " + \
                    "Defaulting to vae_prior goal sampling mode"
                )

    @property
    def custom_goal_sampler(self):
        return self._custom_goal_sampler

    @custom_goal_sampler.setter
    def custom_goal_sampler(self, new_custom_goal_sampler):
        assert self.custom_goal_sampler is None, (
            "Cannot override custom goal setter"
        )
        self._custom_goal_sampler = new_custom_goal_sampler

    @property
    def decode_goals(self):
        return self._decode_goals

    @decode_goals.setter
    def decode_goals(self, _decode_goals):
        self._decode_goals = _decode_goals

    def get_env_update(self):
        """
        For online-parallel. Gets updates to the environment since the last time
        the env was serialized.

        subprocess_env.update_env(**env.get_env_update())
        """
        return dict(
            mode_map=self._mode_map,
            gpu_info=dict(
                use_gpu=ptu._use_gpu,
                gpu_id=ptu._gpu_id,
            ),
            vae_state=self.vae.__getstate__(),
        )

    def update_env(self, mode_map, vae_state, gpu_info):
        self._mode_map = mode_map
        self.vae.__setstate__(vae_state)
        gpu_id = gpu_info['gpu_id']
        use_gpu = gpu_info['use_gpu']
        ptu.device = torch.device("cuda:" + str(gpu_id) if use_gpu else "cpu")
        self.vae.to(ptu.device)

    def enable_render(self):
        self._decode_goals = True
        self.render_goals = True
        self.render_rollouts = True

    def disable_render(self):
        self._decode_goals = False
        self.render_goals = False
        self.render_rollouts = False

    def try_render(self, obs):
        if self.render_rollouts:
            img = obs['image_observation'].reshape(
                self.input_channels,
                self.imsize,
                self.imsize,
            ).transpose()
            cv2.imshow('env', img)
            cv2.waitKey(1)
            reconstruction = self._reconstruct_img(obs['image_observation']).transpose()
            cv2.imshow('env_reconstruction', reconstruction)
            cv2.waitKey(1)
            init_img = self._initial_obs['image_observation'].reshape(
                self.input_channels,
                self.imsize,
                self.imsize,
            ).transpose()
            cv2.imshow('initial_state', init_img)
            cv2.waitKey(1)
            init_reconstruction = self._reconstruct_img(
                self._initial_obs['image_observation']
            ).transpose()
            cv2.imshow('init_reconstruction', init_reconstruction)
            cv2.waitKey(1)

        if self.render_goals:
            goal = obs['image_desired_goal'].reshape(
                self.input_channels,
                self.imsize,
                self.imsize,
            ).transpose()
            cv2.imshow('goal', goal)
            cv2.waitKey(1)

    def _sample_latent_prior(self, batch_size, model=None):
        if model is None:
            model = self.lstm_segmented
        if self.sample_from_true_prior:
            mu, sigma = 0, 1  # sample from prior
        else: # LSTM NOTE: not used
            mu, sigma = model.dist_mu, model.dist_std
        n = np.random.randn(batch_size, model.representation_size)
        latents = sigma * n + mu

        lstm_latents = self.lstm_segmented.from_vae_latents_to_lstm_latents(ptu.from_numpy(latents))
        return ptu.get_numpy(lstm_latents)

    def _decode(self, latents, model=None):
        if model is None:
            model = self.vae_original
        reconstructions, _ = model.decode(ptu.from_numpy(latents))
        decoded = ptu.get_numpy(reconstructions)
        return decoded

    def _encode_one(self, img, vae=None):
        return self._encode(img[None], vae)[0]

    def _encode(self, imgs, model=None):
        if model is None:
            model = self.vae_original
        latent_distribution_params = model.encode(ptu.from_numpy(imgs))
        return ptu.get_numpy(latent_distribution_params[0])

    def _encode_lstm_one(self, img, model=None, hidden=None, update_hidden=False):
        return self._encode_lstm(img[None], model, hidden, update_hidden)[0]

    def _encode_lstm(self, imgs, model=None, hidden=None, update_hidden=False):
        '''
        assume images are in shape batch_size x imlen
        '''
        batch_size, imlen = imgs.shape
        imgs = imgs.reshape(1, batch_size, imlen)
        if model is None:
            model = self.lstm_segmented

        latent_distribution_params, hidden = model.encode(ptu.from_numpy(imgs), hidden, return_hidden=True)
        traj_len, batch_size, feature_size = latent_distribution_params[0].shape
        if update_hidden:
            self.lstm_hidden = hidden
        return ptu.get_numpy(latent_distribution_params[0]).reshape((-1, feature_size)) # remove the frist traj dim

    def _reconstruct_img(self, flat_img, model=None):
        '''
        reconstruct one image
        '''
        if model is None:
            model = self.vae_original
        if model == self.lstm_segmented:
            flat_img = self.segment_obs(flat_img)
            flat_img = flat_img.reshape(1, 1, -1)
        else:
            flat_img = flat_img.reshape(1, -1)

        latent_distribution_params = model.encode(ptu.from_numpy(flat_img))
        reconstructions, _ = model.decode(latent_distribution_params[0])
        imgs = ptu.get_numpy(reconstructions)
        imgs = imgs.reshape(
            1, self.input_channels, self.imsize, self.imsize
        )
        return imgs[0]

    def _reconstruct_img_traj(self, flat_img_traj):
        '''
        flat_img_traj: traj_len x imlen
        '''
        traj_len, imlen = flat_img_traj.shape
        seg_img = np.zeros_like(flat_img_traj)
        for idx, img in enumerate(flat_img_traj):
            seg_img[idx] = self.segment_obs(img)
        
        seg_img = seg_img.reshape((traj_len, 1, imlen))
        latent_distribution_params = self.lstm_segmented.encode(ptu.from_numpy(seg_img))
        reconstructions, _ = self.lstm_segmented.decode(latent_distribution_params[0])
        imgs = ptu.get_numpy(reconstructions)
        imgs = imgs.reshape(
            -1, self.input_channels, self.imsize, self.imsize
        )
        return imgs

    def _image_and_proprio_from_decoded(self, decoded):
        if decoded is None:
            return None, None
        if self.vae_input_key_prefix == 'image_proprio':
            images = decoded[:, :self.image_length]
            proprio = decoded[:, self.image_length:]
            return images, proprio
        elif self.vae_input_key_prefix == 'image':
            return decoded, None
        else:
            raise AssertionError("Bad prefix for the vae input key.")

    def __getstate__(self):
        state = super().__getstate__()
        state = copy.copy(state)
        state['_custom_goal_sampler'] = None
        warnings.warn('VAEWrapperEnv.custom_goal_sampler is not saved.')
        return state

    def __setstate__(self, state):
        warnings.warn('VAEWrapperEnv.custom_goal_sampler was not loaded.')
        super().__setstate__(state)


def temporary_mode(env, mode, func, args=None, kwargs=None):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    cur_mode = env.cur_mode
    env.mode(env._mode_map[mode])
    return_val = func(*args, **kwargs)
    env.mode(cur_mode)
    return return_val
