import rlkit.util.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
from rlkit.launchers.launcher_util import run_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
import ROLL.LSTM_schedule as LSTM_schedules
from ROLL.skewfit_full_experiments_LSTM import run_task
from ROLL.generate_LSTM_vae_only_dataset import generate_LSTM_vae_only_dataset
from rlkit.torch.vae.conv_vae import imsize48_default_architecture as vae_48_default_architecture
from ROLL.LSTM_model import imsize48_default_architecture as lstm_48_default_architecture
from ROLL.generate_vae_dataset import generate_sawyerhurdle_dataset
from chester.run_exp import run_experiment_lite, VariantGenerator
import time
import click

@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    vg = VariantGenerator()

    variant=dict(
        algorithm='Skew-Fit',
        imsize=48,
        double_algo=False,
        env_id="SawyerPickupEnvYZEasy-v0",
        skewfit_variant=dict(
            sample_goals_from_buffer=True,
            presample_goals=True,
            presampled_goals_path='data/local/goals/SawyerPickupEnvYZEasy-v0-goal-500.npy',
            observation_mode='original_image', # if segmentation_proprio, use segmentation_latent concated with eef position
            segmentation=True,
            segmentation_method='unet', # or 'unet'
            segmentation_kwargs=dict(
                    dilation=False,
                    dilation_size=2,
                    show=False,
                    robot_threshold=0.02,
                    fg_threshold=120,
                ),
            keep_train_segmentation_lstm=True,
            save_video=True,
            custom_goal_sampler='replay_buffer',
            online_vae_trainer_kwargs=dict(
                beta=30,
                lr=1e-3,
            ),
            online_lstm_trainer_kwargs=dict(
                    beta=0,
                    recon_loss_coef=0,
                    triplet_loss_coef=[],
                    triplet_loss_type=[],
                    triplet_loss_margin=1,
                    matching_loss_coef=50,
                    vae_matching_loss_coef=0,
                    ae_loss_coef=0.5,
                    lstm_kl_loss_coef=0,
                    contrastive_loss_coef=0,
                    adaptive_margin=0,
                    negative_range=15,
                    batch_size=16,
                ),
            save_video_period=50,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            vf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            max_path_length=50,
            algo_kwargs=dict(
                batch_size=1024,
                num_epochs=800,
                num_eval_steps_per_epoch=500,
                num_expl_steps_per_train_loop=500,
                num_trains_per_train_loop=1000,
                min_num_steps_before_training=10000,
                vae_training_schedule=vae_schedules.custom_schedule,
                lstm_training_schedule=LSTM_schedules.custom_schedule_2,
                oracle_data=False,
                vae_save_period=50,
                lstm_save_period=25,
                parallel_vae_train=False,
            ),
            twin_sac_trainer_kwargs=dict(
                discount=0.99,
                reward_scale=1,
                soft_target_tau=1e-3,
                target_update_period=1,
                use_automatic_entropy_tuning=True,
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
                power=-1, # Use Skewfit
                relabeling_goal_sampling_mode='custom_goal_sampler',
            ),
            exploration_goal_sampling_mode='vae_prior',
            evaluation_goal_sampling_mode='env',
            normalize=False,
            render=False,
            exploration_noise=0.0,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            ),
        ),
        train_vae_variant=dict(
            lstm_path=None,
            only_train_lstm=False,
            lstm_version=2,
            vae_representation_size=16,
            lstm_representation_size=6,
            lstm_beta=20,
            vae_beta=30,
            num_vae_epochs=2000, # pretrain vae epochs
            num_lstm_epochs=2000, # pretrain lstm epochs
            dump_skew_debug_plots=False,
            decoder_activation='gaussian', # will be later replaced by identity. only sigmoid or identity.
            seg_pretrain=True, # if pretrain the segmentation lstm
            ori_pretrain=True, # if pretrain the original vae
            lstm_pretrain_vae_only=True, # if true, will only use random sampled images (not trajs) to train the first vae part, no training of the ROLL.
            generate_lstm_data_fctn=generate_LSTM_vae_only_dataset,
            vae_kwargs=dict(
                input_channels=3,
                architecture=vae_48_default_architecture,
                decoder_distribution='gaussian_identity_variance',
            ),
            generate_lstm_dataset_kwargs=dict( 
                N=2000, # pretrain lstm dataset size
                test_p=.9,
                show=False,
                occlusion_prob=0,
                occlusion_level=0,
            ),
            generate_vae_dataset_kwargs=dict(
                dataset_path='data/local/pre-train-vae/pickup-original-dataset.npy',
                N=2000,
                oracle_dataset=True,
                use_cached=False,
                num_channels=3,
            ),
            lstm_kwargs=dict(
                    input_channels=3,
                    architecture=lstm_48_default_architecture,
                    decoder_distribution='gaussian_identity_variance',
                    detach_vae_output=True,
            ),


            algo_kwargs=dict(
                    start_skew_epoch=5000,
                    is_auto_encoder=False,
                    batch_size=128,
                    lr=1e-3,
                    skew_config=dict(
                        method='vae_prob',
                        power=-1,
                    ),
                    recon_loss_coef=1,
                    triplet_loss_coef=[],
                    triplet_loss_type=[],
                    triplet_loss_margin=1,
                    matching_loss_coef=0,
                    vae_matching_loss_coef=50,
                    ae_loss_coef=0.5,
                    lstm_kl_loss_coef=0,
                    contrastive_loss_coef=0,
                    matching_loss_one_side=False,
                    adaptive_margin=0,
                    negative_range=15,
                    skew_dataset=False,
                    priority_function_kwargs=dict(
                        decoder_distribution='gaussian_identity_variance',
                        sampling_method='importance_sampling',
                        num_latents_to_sample=10,
                    ),
                    use_parallel_dataloading=False,
                ),
                save_period=25,
            ),
            init_camera=sawyer_pick_and_place_camera,
        )

    if not debug:
        vg.add('seed', [[100], [200], [300], [400], [500], [600]])
    else:
        vg.add('seed', [[100]])

    # exp_prefix = 'debug-8-11-reduce-hp-pickup-2000-latent-6-beta-20-vae'
    exp_prefix = '11-20-ROLL-release-pickup-2'

    if variant['train_vae_variant']['lstm_version'] == 3 or variant['train_vae_variant']['lstm_version'] == 2:
        lstm_48_default_architecture['LSTM_args']['input_size'] = variant['train_vae_variant']['lstm_representation_size']
        lstm_48_default_architecture['conv_args']['output_size'] = 6
        variant['train_vae_variant']['lstm_kwargs']['architecture'] = lstm_48_default_architecture

    if debug: # use very small parameters to make sure code at least compiles and can run
        exp_prefix = 'debug'
        vg.add('variant', [variant])
        for vv in vg.variants():
            vv['variant']['skewfit_variant']['algo_kwargs']['batch_size'] = 32
            vv['variant']['skewfit_variant']['algo_kwargs']['num_trains_per_train_loop'] = 10
            vv['variant']['skewfit_variant']['algo_kwargs']['min_num_steps_before_training'] = 100
            vv['variant']['skewfit_variant']['replay_buffer_kwargs']['max_size'] = 1000
            vv['variant']['train_vae_variant']['seg_pretrain'] = True
            vv['variant']['train_vae_variant']['ori_pretrain'] = True
            vv['variant']['train_vae_variant']['num_vae_epochs'] = 2
            vv['variant']['train_vae_variant']['num_lstm_epochs'] = 1
            vv['variant']['train_vae_variant']['save_period'] = 1
            vv['variant']['train_vae_variant']['generate_vae_dataset_kwargs']['N'] = 2000
            vv['variant']['train_vae_variant']['generate_lstm_dataset_kwargs']['N'] = 2000
    else:
        vg.add('variant', [variant])

    print("there are {} variants to run".format(len(vg.variants())))
    sub_process_popens = []
    for idx, vv in enumerate(vg.variants()):
        while len(sub_process_popens) >= 1:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
        compile_script = wait_compile = None

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
