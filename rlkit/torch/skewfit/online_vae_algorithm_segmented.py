import gtimer as gt
from rlkit.core import logger
# from rlkit.data_management.online_vae_replay_buffer import OnlineVaeRelabelingBuffer
from rlkit.data_management.online_vae_replay_buffer_segmented import OnlineVaeRelabelingBufferSegmented
from rlkit.data_management.shared_obs_dict_replay_buffer import SharedObsDictRelabelingBuffer
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
)
import rlkit.torch.pytorch_util as ptu
from torch.multiprocessing import Process, Pipe
from threading import Thread


class OnlineVaeAlgorithmSegmented(TorchBatchRLAlgorithm):

    def __init__(
            self,
            vae_original,
            vae_segmented,
            vae_trainer_original,
            vae_trainer_segmented,
            *base_args,
            vae_save_period=1,
            vae_training_schedule=vae_schedules.never_train,
            oracle_data=False,
            parallel_vae_train=True,
            vae_min_num_steps_before_training=0,
            uniform_dataset=None,
            keep_train_segmentation_vae=False,
            **base_kwargs
    ):
        super().__init__(*base_args, **base_kwargs)
        assert isinstance(self.replay_buffer, OnlineVaeRelabelingBufferSegmented)
        self.vae_original = vae_original
        self.vae_segmented = vae_segmented
        self.vae_trainer_original = vae_trainer_original
        self.vae_trainer_segmented = vae_trainer_segmented
        self.vae_trainer_original.model = self.vae_original
        self.vae_trainer_segmented.model = self.vae_segmented

        self.vae_save_period = vae_save_period
        self.vae_training_schedule = vae_training_schedule
        self.oracle_data = oracle_data

        self.parallel_vae_train = parallel_vae_train
        self.vae_min_num_steps_before_training = vae_min_num_steps_before_training
        self.uniform_dataset = uniform_dataset

        self._vae_training_process = None
        self._update_subprocess_vae_thread = None
        self._vae_conn_pipe = None

        self.keep_train_segmentation_vae = keep_train_segmentation_vae


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
        self.vae_segmented.to(device)
        super().to(device)

    def _get_snapshot(self):
        snapshot = super()._get_snapshot()
        assert 'vae' not in snapshot
        snapshot['vae_original'] = self.vae_original
        snapshot['vae_segmented'] = self.vae_segmented
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

        decoded_goal = self.eval_env._decode(goals, self.eval_env.vae_segmented)
        for idx in range(10):
            self.eval_env.show_obs(decoded_goal[idx], "sac policy goal")


    """
    VAE-specific Code
    """
    def _train_vae(self, epoch):
        if self.parallel_vae_train and self._vae_training_process is None:
            self.init_vae_training_subprocess()
        should_train, amount_to_train = self.vae_training_schedule(epoch)
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
                _train_vae(
                    self.vae_trainer_original,
                    self.replay_buffer,
                    epoch,
                    amount_to_train,
                    key='image_observation'
                )

                # train segmentation vae using both oracle data and newly collected data
                # train using newly collected data
                if self.keep_train_segmentation_vae:
                    _train_vae(
                        self.vae_trainer_segmented,
                        self.replay_buffer,
                        epoch,
                        amount_to_train // 3 * 2,
                        key='image_observation_segmented'
                    )

                    # train using pre-collected oracle data
                    _train_vae(
                        self.vae_trainer_segmented,
                        self.replay_buffer,
                        epoch,
                        amount_to_train // 3,
                        key='image_observation_segmented',
                        oracle_data=True
                    )

                self.replay_buffer.refresh_latents(epoch)

                _test_vae(
                    self.vae_trainer_original,
                    epoch,
                    self.replay_buffer,
                    vae_save_period=self.vae_save_period,
                    uniform_dataset=self.uniform_dataset,
                    save_prefix='r_original_'
                )
                _test_vae(
                    self.vae_trainer_segmented,
                    epoch,
                    self.replay_buffer,
                    vae_save_period=self.vae_save_period,
                    uniform_dataset=self.uniform_dataset,
                    save_prefix='r_segmented_'
                )

    def _log_vae_stats(self):
        logger.record_dict(
            self.vae_trainer_original.get_diagnostics(),
            prefix='vae_trainer_original/',
        )
        logger.record_dict(
            self.vae_trainer_segmented.get_diagnostics(),
            prefix='vae_trainer_segmented/',
        )

    def _cleanup(self):
        if self.parallel_vae_train:
            self._vae_conn_pipe.close()
            self._vae_training_process.terminate()

    def init_vae_training_subprocess(self):
        assert isinstance(self.replay_buffer, SharedObsDictRelabelingBuffer)

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