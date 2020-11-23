import gtimer as gt
from rlkit.core import logger
from ROLL.online_LSTM_replay_buffer import OnlineLSTMRelabelingBuffer
import rlkit.torch.vae.vae_schedules as vae_schedules
import ROLL.LSTM_schedule as lstm_schedules
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
)
import rlkit.torch.pytorch_util as ptu
from torch.multiprocessing import Process, Pipe
from threading import Thread

from test_latent_space.test_LSTM import compare_latent_distance
from test_latent_space.test_LSTM2 import test_lstm_traj
from test_latent_space.test_masked_traj import test_masked_traj_lstm

import os
import os.path as osp
import numpy as np
from multiworld.core.image_env import unormalize_image, normalize_image

class OnlineLSTMAlgorithm(TorchBatchRLAlgorithm):

    def __init__(
            self,
            env_id,
            vae_original,
            lstm_segmented,
            vae_trainer_original,
            lstm_trainer_segmented,
            *base_args,
            vae_save_period=1,
            lstm_save_period=1,
            vae_training_schedule=vae_schedules.never_train,
            lstm_training_schedule=lstm_schedules.never_train,
            lstm_test_N=500,
            lstm_segmentation_method='color',
            oracle_data=False,
            parallel_vae_train=False,
            vae_min_num_steps_before_training=0,
            uniform_dataset=None,
            keep_train_segmentation_lstm=False,
            keep_train_original_vae=True,
            **base_kwargs
    ):
        super().__init__(*base_args, **base_kwargs)
        assert isinstance(self.replay_buffer, OnlineLSTMRelabelingBuffer)
        self.vae_original = vae_original
        self.lstm_segmented = lstm_segmented
        self.vae_trainer_original = vae_trainer_original
        self.lstm_trainer_segmented = lstm_trainer_segmented
        self.vae_trainer_original.model = self.vae_original
        self.lstm_trainer_segmented.model = self.lstm_segmented

        self.vae_save_period = vae_save_period
        self.lstm_save_period = lstm_save_period
        self.vae_training_schedule = vae_training_schedule
        self.lstm_training_schedule = lstm_training_schedule
        self.oracle_data = oracle_data

        self.parallel_vae_train = parallel_vae_train
        self.vae_min_num_steps_before_training = vae_min_num_steps_before_training
        self.uniform_dataset = uniform_dataset

        self._vae_training_process = None
        self._update_subprocess_vae_thread = None
        self._vae_conn_pipe = None

        self.keep_train_segmentation_lstm = keep_train_segmentation_lstm
        self.keep_train_original_vae = keep_train_original_vae
        
        # below is just used for testing the segmentation vae.
        self.env_id = env_id
        self.lstm_test_N = lstm_test_N
        self.lstm_segmentation_method = lstm_segmentation_method


    def _train(self):
        super()._train()
        self._cleanup()

    def _end_epoch(self, epoch):
        # self.check_replay_buffer()
        self._train_vae(epoch)
        gt.stamp('vae training')
        super()._end_epoch(epoch)

    def _log_stats(self, epoch):
        self._log_vae_stats()
        super()._log_stats(epoch)

    def to(self, device):
        self.vae_original.to(device)
        self.lstm_segmented.to(device)
        super().to(device)

    def _get_snapshot(self):
        snapshot = super()._get_snapshot()
        assert 'vae' not in snapshot
        snapshot['vae_original'] = self.vae_original
        snapshot['lstm_segmented'] = self.lstm_segmented
        return snapshot

    """
    debug code
    """
    def check_replay_buffer(self):
        batch = self.replay_buffer.random_batch(
                        self.batch_size)
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goals = batch['resampled_goals']


        print("obs: ", type(obs))
        print("obs shape: ", obs.shape)
        decoded_obs = self.eval_env._decode(obs, self.eval_env.vae_original)
        for idx in range(10):
            self.eval_env.show_obs(decoded_obs[idx], "sac policy obs")

        print("next_obs: ", type(next_obs))
        print("next obs shape: ", next_obs.shape)
        decoded_next_obs = self.eval_env._decode(next_obs, self.eval_env.vae_original)
        for idx in range(10):
            self.eval_env.show_obs(decoded_next_obs[idx], "sac policy next_obs")

        decoded_goal = self.eval_env._decode(goals, self.eval_env.lstm_segmented)
        for idx in range(10):
            self.eval_env.show_obs(decoded_goal[idx], "sac policy goal")


    """
    VAE-specific Code
    """
    def _train_vae(self, epoch):
        if self.parallel_vae_train and self._vae_training_process is None:
            self.init_vae_training_subprocess()
        should_train, amount_to_train = self.vae_training_schedule(epoch)
        _, lstm_amount_to_train = self.lstm_training_schedule(epoch)
        rl_start_epoch = int(self.min_num_steps_before_training / (
                self.num_expl_steps_per_train_loop * self.num_train_loops_per_epoch
        ))
        print(" _train_vae called, should_train, amount_to_train", should_train, amount_to_train)
        if should_train or epoch <= (rl_start_epoch - 1):
            if self.parallel_vae_train:
                assert self._vae_training_process.is_alive()
                # Make sure the last vae update has finished before starting
                # another one
                if self._update_subprocess_vae_thread is not None:
                    self._update_subprocess_vae_thread.join()
                self._update_subprocess_vae_thread = Thread(
                    target=OnlineVaeAlgorithmSegmented.update_vae_in_training_subprocess,
                    args=(self, epoch, ptu.device)
                )
                self._update_subprocess_vae_thread.start()
                self._vae_conn_pipe.send((amount_to_train, epoch))
            else:
                if self.keep_train_original_vae:
                    _train_vae(
                        self.vae_trainer_original,
                        self.replay_buffer,
                        epoch,
                        amount_to_train,
                        key='image_observation'
                    )

                    _test_vae(
                        self.vae_trainer_original,
                        epoch,
                        self.replay_buffer,
                        vae_save_period=self.vae_save_period,
                        uniform_dataset=self.uniform_dataset,
                        save_prefix='r_original_'
                    )

                if self.keep_train_segmentation_lstm:
                    _train_lstm(
                        lstm_trainer=self.lstm_trainer_segmented,
                        replay_buffer=self.replay_buffer,
                        epoch=epoch,
                        batches=lstm_amount_to_train,
                        oracle_data=False,
                        key='image_observation_segmented'
                    )

                    _test_lstm(
                        lstm_trainer=self.lstm_trainer_segmented,
                        epoch=epoch,
                        replay_buffer=self.replay_buffer,
                        env_id=self.env_id,
                        lstm_save_period=self.lstm_save_period,
                        uniform_dataset=None,
                        save_prefix='r_lstm_' ,
                        lstm_test_N=self.lstm_test_N,
                        lstm_segmentation_method=self.lstm_segmentation_method
                    )

                # we only refresh goals if the segmentation lstm (used for goal sampling) has changed
                self.replay_buffer.refresh_latents(epoch, refresh_goals=self.keep_train_segmentation_lstm)

    def _log_vae_stats(self):
        logger.record_dict(
            self.vae_trainer_original.get_diagnostics(),
            prefix='vae_trainer_original/',
        )
        logger.record_dict(
            self.lstm_trainer_segmented.get_diagnostics(),
            prefix='lstm_trainer_segmented/',
        )

    def _cleanup(self):
        if self.parallel_vae_train:
            self._vae_conn_pipe.close()
            self._vae_training_process.terminate()

    def init_vae_training_subprocess(self):

        self._vae_conn_pipe, process_pipe = Pipe()
        self._vae_training_process = Process(
            target=subprocess_train_vae_loop,
            args=(
                process_pipe,
                self.vae,
                self.vae.state_dict(),
                self.replay_buffer,
                self.replay_buffer.get_mp_info(),
                ptu.device,
            )
        )
        self._vae_training_process.start()
        self._vae_conn_pipe.send(self.vae_trainer)

    def update_vae_in_training_subprocess(self, epoch, device):
        self.vae.__setstate__(self._vae_conn_pipe.recv())
        self.vae.to(device)
        _test_vae(
            self.vae_trainer,
            epoch,
            self.replay_buffer,
            vae_save_period=self.vae_save_period,
            uniform_dataset=self.uniform_dataset,
        )


def _train_vae(vae_trainer, replay_buffer, epoch, batches=50, oracle_data=False, key='image_observation'):
    batch_sampler = replay_buffer.random_vae_training_data
    if oracle_data:
        batch_sampler = None
    vae_trainer.train_epoch(
        epoch,
        sample_batch=batch_sampler,
        batches=batches,
        from_rl=True,
        key=key,
    )

def _train_lstm(lstm_trainer, replay_buffer, epoch, batches=50, oracle_data=False, key='image_observation_segmented'):
    batch_sampler = replay_buffer.random_lstm_training_data
    if oracle_data:
        batch_sampler = None
    lstm_trainer.train_epoch(
        epoch,
        sample_batch=batch_sampler,
        batches=batches,
        from_rl=True,
        key=key,
    )


def _test_vae(vae_trainer, epoch, replay_buffer, vae_save_period=1, uniform_dataset=None, save_prefix='r'):
    save_imgs = epoch % vae_save_period == 0
    log_fit_skew_stats = replay_buffer._prioritize_vae_samples and uniform_dataset is not None
    if uniform_dataset is not None:
        replay_buffer.log_loss_under_uniform(uniform_dataset, vae_trainer.batch_size, rl_logger=vae_trainer.vae_logger_stats_for_rl)
    vae_trainer.test_epoch(
        epoch,
        from_rl=True,
        save_reconstruction=save_imgs,
        save_prefix=save_prefix
    )
    if save_imgs:
        sample_save_prefix = save_prefix.replace('r', 's')
        vae_trainer.dump_samples(epoch, save_prefix=sample_save_prefix)
        if log_fit_skew_stats:
            replay_buffer.dump_best_reconstruction(epoch)
            replay_buffer.dump_worst_reconstruction(epoch)
            replay_buffer.dump_sampling_histogram(epoch, batch_size=vae_trainer.batch_size)
        if uniform_dataset is not None:
            replay_buffer.dump_uniform_imgs_and_reconstructions(dataset=uniform_dataset, epoch=epoch)

def _test_lstm(lstm_trainer, epoch, replay_buffer, env_id, lstm_save_period=1, uniform_dataset=None, 
    save_prefix='r', lstm_segmentation_method='color', lstm_test_N=500, key='image_observation_segmented'):
    
    batch_sampler = replay_buffer.random_lstm_training_data
    
    save_imgs = epoch % lstm_save_period == 0
    log_fit_skew_stats = replay_buffer._prioritize_vae_samples and uniform_dataset is not None
    if uniform_dataset is not None:
        replay_buffer.log_loss_under_uniform(uniform_dataset, lstm_trainer.batch_size, rl_logger=lstm_trainer.vae_logger_stats_for_rl)
    lstm_trainer.test_epoch(
        epoch,
        from_rl=True,
        key=key,
        sample_batch=batch_sampler,
        save_reconstruction=save_imgs,
        save_prefix=save_prefix
    )
    if save_imgs:
        sample_save_prefix = save_prefix.replace('r', 's')
        lstm_trainer.dump_samples(epoch, save_prefix=sample_save_prefix)
        if log_fit_skew_stats:
            replay_buffer.dump_best_reconstruction(epoch)
            replay_buffer.dump_worst_reconstruction(epoch)
            replay_buffer.dump_sampling_histogram(epoch, batch_size=lstm_trainer.batch_size)
        if uniform_dataset is not None:
            replay_buffer.dump_uniform_imgs_and_reconstructions(dataset=uniform_dataset, epoch=epoch)

        m = lstm_trainer.model
        pjhome = os.environ['PJHOME']
        seg_name = 'seg-' + 'color' 

        if env_id in ['SawyerPushNIPSEasy-v0', 'SawyerPushHurdle-v0', 'SawyerPushHurdleMiddle-v0']:   
            N = 500
            data_file_path = osp.join(pjhome, 'data/local/pre-train-lstm', '{}-{}-{}-0.3-0.5.npy'.format(env_id, seg_name, N))
            puck_pos_path = osp.join(pjhome, 'data/local/pre-train-lstm', '{}-{}-{}-0.3-0.5-puck-pos.npy'.format(env_id, seg_name, N))
            if osp.exists(data_file_path):
                all_data = np.load(data_file_path)
                puck_pos = np.load(puck_pos_path)
                all_data = normalize_image(all_data.copy())
                compare_latent_distance(m, all_data, puck_pos, save_dir=logger.get_snapshot_dir(), obj_name='puck',
                    save_name='online_lstm_latent_distance_{}.png'.format(epoch))
        elif env_id == 'SawyerDoorHookResetFreeEnv-v1':
            N = 1000
            seg_name = 'seg-' + 'unet' 
            data_file_path = osp.join(pjhome, 'data/local/pre-train-lstm', 'vae-only-{}-{}-{}-0-0.npy'.format(env_id, seg_name, N))
            door_angle_path = osp.join(pjhome, 'data/local/pre-train-lstm', 'vae-only-{}-{}-{}-0-0-door-angle.npy'.format(env_id, seg_name, N))
            if osp.exists(data_file_path):
                all_data = np.load(data_file_path)
                door_angle = np.load(door_angle_path)
                all_data = normalize_image(all_data.copy())
                compare_latent_distance(m, all_data, door_angle, save_dir=logger.get_snapshot_dir(), obj_name='door',
                    save_name='online_lstm_latent_distance_{}.png'.format(epoch))
        elif env_id == 'SawyerPushHurdleResetFreeEnv-v0':
            N = 2000
            data_file_path = osp.join(pjhome, 'data/local/pre-train-lstm', 'vae-only-{}-{}-{}-0.3-0.5.npy'.format(env_id, seg_name, N))
            puck_pos_path = osp.join(pjhome, 'data/local/pre-train-lstm', 'vae-only-{}-{}-{}-0.3-0.5-puck-pos.npy'.format(env_id, seg_name, N))
            if osp.exists(data_file_path):
                all_data = np.load(data_file_path)
                puck_pos = np.load(puck_pos_path)
                all_data = normalize_image(all_data.copy())
                compare_latent_distance(m, all_data, puck_pos, save_dir=logger.get_snapshot_dir(), obj_name='puck',
                    save_name='online_lstm_latent_distance_{}.png'.format(epoch))

        test_lstm_traj(env_id, m, save_path=logger.get_snapshot_dir(), 
            save_name='online_lstm_test_traj_{}.png'.format(epoch))
        test_masked_traj_lstm(env_id, m, save_dir=logger.get_snapshot_dir(), 
            save_name='online_masked_test_{}.png'.format(epoch))


def subprocess_train_vae_loop(
        conn_pipe,
        vae,
        vae_params,
        replay_buffer,
        mp_info,
        device,
):
    """
    The observations and next_observations of the replay buffer are stored in
    shared memory. This loop waits until the parent signals to start vae
    training, trains and sends the vae back, and then refreshes the latents.
    Refreshing latents in the subprocess reflects in the main process as well
    since the latents are in shared memory. Since this is does asynchronously,
    it is possible for the main process to see half the latents updated and half
    not.
    """
    ptu.device = device
    vae_trainer = conn_pipe.recv()
    vae.load_state_dict(vae_params)
    vae.to(device)
    vae_trainer.set_vae(vae)
    replay_buffer.init_from_mp_info(mp_info)
    replay_buffer.env.vae = vae
    while True:
        amount_to_train, epoch = conn_pipe.recv()
        _train_vae(vae_trainer, replay_buffer, epoch, amount_to_train)
        conn_pipe.send(vae_trainer.model.__getstate__())
        replay_buffer.refresh_latents(epoch)
