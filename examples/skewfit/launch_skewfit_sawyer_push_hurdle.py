import rlkit.util.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from rlkit.launchers.launcher_util import run_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.launchers.skewfit_experiments import run_task as run_task_original
from ROLL.generate_vae_dataset import generate_sawyerhurdle_dataset
from rlkit.torch.vae.conv_vae import imsize48_default_architecture
from chester.run_exp import run_experiment_lite, VariantGenerator
import time
import click
import copy

@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    vg = VariantGenerator()

    vg.add("variant", [
        dict(
            algorithm='Skew-Fit',
            double_algo=False,
            online_vae_exploration=False,
            imsize=48,
            init_camera=sawyer_init_camera_zoomed_in,
            env_id='SawyerPushHurdle-v0',    
            skewfit_variant=dict(
                # vae_path=[
                #     "./data/05-09-test-color-thresholding-vae/05-09-test_color_thresholding_vae_2020_05_09_19_20_04_0000--s-35695/vae_ori_pretrain.pkl",
                #     "./data/05-09-test-color-thresholding-vae/05-09-test_color_thresholding_vae_2020_05_09_19_20_04_0000--s-35695/vae_seg_pretrain.pkl",
                # ],
                segmentation=True, # if true, using segmentation, otherwise not.
                segmentation_method='unet', # or 'unet'.
                segmentation_kwargs=dict(
                    dilation=False,
                    show=False,
                    robot_threshold=0.05,
                    fg_threshold=120,
                ),
                keep_train_segmentation_vae=False,
                save_video=True,
                custom_goal_sampler='replay_buffer',
                online_vae_trainer_kwargs=dict(
                    beta=20,
                    lr=1e-3,
                ),
                save_video_period=100,
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
                    batch_size=1024, #32,
                    num_epochs=800,
                    num_eval_steps_per_epoch=500,
                    num_expl_steps_per_train_loop=500,
                    num_trains_per_train_loop=1000, #10,
                    min_num_steps_before_training=10000, #100,
                    vae_training_schedule=vae_schedules.custom_schedule_2,
                    oracle_data=False,
                    vae_save_period=50,
                    parallel_vae_train=False,
                ),
                twin_sac_trainer_kwargs=dict(
                    discount=0.99,
                    reward_scale=1,
                    soft_target_tau=1e-3,
                    target_update_period=1,  # 1
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
                    power=-1,
                    relabeling_goal_sampling_mode='vae_prior',
                ),
                exploration_goal_sampling_mode='vae_prior',
                evaluation_goal_sampling_mode='reset_of_env',
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
                representation_size=6,
                beta=20,
                num_epochs=2000, # pretrain vae epochs
                dump_skew_debug_plots=False,
                decoder_activation='gaussian',
                seg_pretrain=True, # if pretrain the segmentation vae
                ori_pretrain=True, # if pretrain the original vae
                generate_vae_data_fctn=generate_sawyerhurdle_dataset, # use a custom vae dataset generate function
                generate_vae_dataset_kwargs=dict( 
                    N=2000, # pretrain vae dataset size
                    test_p=.9,
                    use_cached=False,
                    show=False,
                    oracle_dataset=True,
                    oracle_dataset_using_set_to_goal=True,
                    n_random_steps=100,
                    non_presampled_goal_img_is_garbage=True,
                ),
                vae_kwargs=dict(
                    input_channels=3,
                    architecture=imsize48_default_architecture,
                    decoder_distribution='gaussian_identity_variance',
                ),
                seg_vae_kwargs=dict(
                    input_channels=3,
                    architecture=imsize48_default_architecture,
                    decoder_distribution='gaussian_identity_variance',
                ),
                # TODO: why the redundancy?
                algo_kwargs=dict(
                    start_skew_epoch=5000,
                    is_auto_encoder=False,
                    batch_size=64,
                    lr=1e-3,
                    skew_config=dict(
                        method='vae_prob',
                        power=-1,
                    ),
                    matching_loss_coef=0,
                    skew_dataset=True,
                    priority_function_kwargs=dict(
                        decoder_distribution='gaussian_identity_variance',
                        sampling_method='importance_sampling',
                        num_latents_to_sample=10,
                    ),
                    use_parallel_dataloading=False,
                ),

                save_period=25,
            ),
        )
        ]
    )

    if not debug:
        vg.add('seed', [[400], [500], [600]])
    else:
        vg.add('seed', [[100]])

    # exp_prefix = '6-5-skewfit-hurdlemiddle-segcolorpretrain'
    exp_prefix = '7-17-CoRL-sawyerhurlde-segvae-ablation'

    for vv in vg.variants():
        seg_imsize48_arch = copy.deepcopy(imsize48_default_architecture)
        seg_imsize48_arch['conv_args']['output_size'] = 6
        vv['variant']['train_vae_variant']['seg_vae_kwargs']['architecture'] = seg_imsize48_arch

    if debug: # use very small parameters to make sure code at least compiles and can run
        print(debug)
        exp_prefix = 'debug'
        for vv in vg.variants():
            vv['variant']['skewfit_variant']['algo_kwargs']['batch_size'] = 32
            vv['variant']['skewfit_variant']['algo_kwargs']['num_trains_per_train_loop'] = 10
            vv['variant']['skewfit_variant']['algo_kwargs']['min_num_steps_before_training'] = 100
            vv['variant']['skewfit_variant']['replay_buffer_kwargs']['max_size'] = 1000
            vv['variant']['train_vae_variant']['seg_pretrain'] = False
            vv['variant']['train_vae_variant']['ori_pretrain'] = False
            vv['variant']['train_vae_variant']['num_epochs'] = 3
            vv['variant']['train_vae_variant']['generate_vae_dataset_kwargs']['N'] = 5


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
