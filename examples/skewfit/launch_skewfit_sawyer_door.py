import rlkit.util.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v0
from rlkit.launchers.launcher_util import run_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.launchers.skewfit_experiments import run_task as run_task_original
from rlkit.torch.vae.conv_vae import imsize48_default_architecture, imsize48_default_architecture_with_more_hidden_layers
from ROLL.generate_vae_dataset import generate_sawyerhurdle_dataset
from chester.run_exp import run_experiment_lite, VariantGenerator
import time
import click
import os.path as osp

@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    vg = VariantGenerator()

    vg.add("variant", [
            dict(
            algorithm='Skew-Fit-SAC',
            double_algo=False,
            online_vae_exploration=False,
            imsize=48,
            env_id='SawyerDoorHookResetFreeEnv-v1',
            init_camera=sawyer_door_env_camera_v0,
            skewfit_variant=dict(
                save_video=True,
                segmentation=False,
                segmentation_method='color', # or 'unet'.
                keep_train_segmentation_vae=False,
                custom_goal_sampler='replay_buffer',
                online_vae_trainer_kwargs=dict(
                    beta=20,
                    lr=1e-3,
                ),
                save_video_period=50,
                qf_kwargs=dict(
                    hidden_sizes=[400, 300],
                ),
                policy_kwargs=dict(
                    hidden_sizes=[400, 300],
                ),
                twin_sac_trainer_kwargs=dict(
                    reward_scale=1,
                    discount=0.99,
                    soft_target_tau=1e-3,
                    target_update_period=1,
                    use_automatic_entropy_tuning=True,
                ),
                max_path_length=100,
                algo_kwargs=dict(
                    batch_size=1024,
                    num_epochs=500,
                    num_eval_steps_per_epoch=500,
                    num_expl_steps_per_train_loop=500,
                    num_trains_per_train_loop=1000,
                    min_num_steps_before_training=10000,
                    vae_training_schedule=vae_schedules.custom_schedule,
                    oracle_data=False,
                    vae_save_period=50,
                    parallel_vae_train=False,
                ),
                replay_buffer_kwargs=dict(
                    start_skew_epoch=10,
                    max_size=int(100000),
                    fraction_goals_rollout_goals=0.2,
                    fraction_goals_env_goals=0.5,
                    exploration_rewards_type='None',
                    vae_priority_type='vae_prob',
                    priority_function_kwargs=dict(
                        sampling_method='importance_sampling',
                        decoder_distribution='gaussian_identity_variance',
                        num_latents_to_sample=10,
                    ),
                    power=-0.5,
                    relabeling_goal_sampling_mode='custom_goal_sampler',
                ),
                exploration_goal_sampling_mode='custom_goal_sampler',
                evaluation_goal_sampling_mode='presampled',
                training_mode='train',
                testing_mode='test',
                reward_params=dict(
                    type='latent_distance',
                ),
                observation_key='latent_observation',
                desired_goal_key='latent_desired_goal',
                presampled_goals_path='data/local/goals/SawyerDoorHookResetFreeEnv-v1-goal.npy',
                presample_goals=True,
                vae_wrapped_env_kwargs=dict(
                    sample_from_true_prior=True,
                ),
            ),
            train_vae_variant=dict(
                representation_size=16,
                beta=20,
                num_epochs=2000,
                dump_skew_debug_plots=False,
                seg_pretrain=True, # if pretrain the segmentation vae
                ori_pretrain=True, # if pretrain the original vae
                decoder_activation='gaussian',
                generate_vae_dataset_kwargs=dict(
                    dataset_path='data/local/pre-train-vae/door_original_dataset.npy',
                    # dataset_path='data/local/pre-train-vae/door_original_dataset_2000.npy',
                    N=1000,
                    test_p=.9,
                    use_cached=True,
                    show=False,
                    oracle_dataset=False,
                    n_random_steps=1,
                    non_presampled_goal_img_is_garbage=True,
                ),
                vae_kwargs=dict(
                    decoder_distribution='gaussian_identity_variance',
                    input_channels=3,
                    architecture=imsize48_default_architecture,
                ),
                algo_kwargs=dict(
                    lr=1e-3,
                ),
                save_period=25,
            ),
        )
        ]
    )

    if not debug and mode == 'seuss':
        vg.add('seed', [[100], [200], [400]])
    else:
        vg.add('seed', [[100]])

    # exp_prefix = '8-6-door-baseline-separate'
    exp_prefix = '10-21-door-baseline-rerun'

    if debug: # use very small parameters to make sure code at least compiles and can run
        exp_prefix = 'debug'
        for vv in vg.variants():
            vv['variant']['skewfit_variant']['algo_kwargs']['batch_size'] = 32
            vv['variant']['skewfit_variant']['algo_kwargs']['num_trains_per_train_loop'] = 10
            vv['variant']['skewfit_variant']['algo_kwargs']['min_num_steps_before_training'] = 200
            vv['variant']['skewfit_variant']['replay_buffer_kwargs']['max_size'] = 200
            vv['variant']['train_vae_variant']['seg_pretrain'] = False
            vv['variant']['train_vae_variant']['ori_pretrain'] = False
            vv['variant']['train_vae_variant']['num_epochs'] = 0
            vv['variant']['train_vae_variant']['generate_vae_dataset_kwargs']['N'] = 50

    print("there are {} variants to run".format(len(vg.variants())))
    sub_process_popens = []
    for idx, vv in enumerate(vg.variants()):
        while len(sub_process_popens) >= 1:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
        compile_script = wait_compile = None

        run_task = run_task_original

        cur_popen = run_experiment_lite(
            stub_method_call=run_task,
            variant=vv,
            mode=mode,
            dry=dry,
            use_gpu=True,
            exp_prefix=exp_prefix,
            wait_subprocess=debug,
            compile_script=compile_script,
            wait_compile=wait_compile,
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break

if __name__ == '__main__':
    main()
