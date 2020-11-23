import time
from multiworld.core.image_env import ImageEnv
from rlkit.core import logger
from rlkit.envs.vae_wrapper import temporary_mode

import cv2
import numpy as np
import os.path as osp
import pickle, os

from test_latent_space.test_LSTM import compare_latent_distance
from test_latent_space.test_LSTM2 import test_lstm_traj
from test_latent_space.test_masked_traj import test_masked_traj_lstm

from multiworld.core.image_env import ImageEnv, unormalize_image, normalize_image
from rlkit.samplers.data_collector.vae_env import (
    VAEWrappedEnvPathCollector,
)
from ROLL.LSTM_path_collector import LSTMWrappedEnvPathCollector
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.skewfit.online_vae_algorithm_segmented import OnlineVaeAlgorithmSegmented
from ROLL.onlineLSTMalgorithm import OnlineLSTMAlgorithm
from rlkit.util.io import load_local_or_remote_file
from rlkit.util.video import dump_video
import __main__ as main

import copy
import torch
from segmentation import unet
from segmentation.segment_image import (
    segment_image_unet, train_bgsb
)
from rlkit.launchers.launcher_util import reset_execution_environment, set_gpu_mode, set_seed, setup_logger, save_experiment_data
from torch.multiprocessing import Pool, set_start_method

invisiable_env_id = {
    'SawyerPushNIPSEasy-v0': "SawyerPushNIPSPuckInvisible-v0",
    'SawyerPushHurdle-v0': 'SawyerPushHurdlePuckInvisible-v0',
    'SawyerPushHurdleMiddle-v0': 'SawyerPushHurdleMiddlePuckInvisible-v0',
    'SawyerPushT-v0': 'SawyerPushTPuckInvisible-v0',
    'SawyerDoorHookResetFreeEnv-v1': 'SawyerDoorHookResetFreeEnvDoorInvisible-v0',
    'SawyerPickupEnvYZEasy-v0': 'SawyerPickupResetFreeEnvBallInvisible-v0',
}

def run_task(variant, log_dir, exp_prefix):
    print("log_dir: ", log_dir)
    exp_prefix = time.strftime("%m-%d") + "-" + exp_prefix

    variants = []    
    log_dirs = []
    exp_prefixs = []
    seeds = variant['seed']
    for seed in seeds:
        tmp_vv = copy.deepcopy(variant)
        tmp_vv['seed'] = seed
        variants.append(tmp_vv)
        
        seed_log_dir = log_dir + '/' + str(seed)
        print("seed_log_dir: ", seed_log_dir)
        log_dirs.append(seed_log_dir)
        exp_prefixs.append(exp_prefix)

    for i in range(len(seeds)):
        skewfit_full_experiment_chester([variants[i], log_dirs[i], exp_prefixs[i]])

def skewfit_full_experiment_chester(args):
    variant, log_dir, exp_prefix = args
    base_log_dir = log_dir
    logger.reset()
    script_name = main.__file__
    seed = variant['seed']
    actual_log_dir = setup_logger(
        exp_prefix=exp_prefix,
        variant=variant,
        exp_id=0,
        seed=seed,
        snapshot_mode='gap_and_last',
        snapshot_gap=25,
        log_dir=base_log_dir,
        script_name=script_name,
    )

    variant = variant['variant']

    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False

    set_gpu_mode(use_gpu)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
 
    variant['skewfit_variant']['save_vae_data'] = True
    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)
    skewfit_experiment(variant['skewfit_variant'])

def skewfit_full_experiment(variant):
    variant['skewfit_variant']['save_vae_data'] = True
    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)
    skewfit_experiment(variant['skewfit_variant'])

def train_vae_experiment(variant):
    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)

def full_experiment_variant_preprocess(variant):
    train_vae_variant = variant['train_vae_variant']
    skewfit_variant = variant['skewfit_variant']
    if 'env_id' in variant:
        assert 'env_class' not in variant
        env_id = variant['env_id']
        skewfit_variant['env_id'] = env_id
        train_vae_variant['generate_vae_dataset_kwargs']['env_id'] = env_id
        train_vae_variant['generate_lstm_dataset_kwargs']['env_id'] = env_id
    else:
        env_class = variant['env_class']
        env_kwargs = variant['env_kwargs']
        train_vae_variant['generate_vae_dataset_kwargs']['env_class'] = (
            env_class
        )
        train_vae_variant['generate_vae_dataset_kwargs']['env_kwargs'] = (
            env_kwargs
        )
        skewfit_variant['env_class'] = env_class
        skewfit_variant['env_kwargs'] = env_kwargs
    init_camera = variant.get('init_camera', None)
    imsize = variant.get('imsize', 84)
    train_vae_variant['generate_vae_dataset_kwargs']['init_camera'] = init_camera
    train_vae_variant['generate_lstm_dataset_kwargs']['init_camera'] = init_camera
    train_vae_variant['generate_vae_dataset_kwargs']['imsize'] = imsize
    train_vae_variant['generate_lstm_dataset_kwargs']['imsize'] = imsize
    train_vae_variant['imsize'] = imsize
    skewfit_variant['imsize'] = imsize
    skewfit_variant['init_camera'] = init_camera

    # for online training LSTM
    skewfit_variant['replay_buffer_kwargs']['max_path_length'] = skewfit_variant['max_path_length']

    # for online testing LSTM
    skewfit_variant['algo_kwargs']['lstm_test_N'] = train_vae_variant['generate_lstm_dataset_kwargs']['N']
    skewfit_variant['algo_kwargs']['env_id'] = variant['env_id']
    skewfit_variant['algo_kwargs']['lstm_segmentation_method'] = skewfit_variant['segmentation_method']

    # for loading door and pick up presampled goals to train original vae 
    if skewfit_variant.get('presampled_goals_path') is not None:
        if train_vae_variant['generate_vae_dataset_kwargs'].get('dataset_path') is None:
            train_vae_variant['generate_vae_dataset_kwargs']['dataset_path'] = skewfit_variant['presampled_goals_path']

    # pass in the segmentation method kwargs to generate lstm dataset function
    skewfit_variant['segmentation_kwargs']['env_id'] = env_id
    train_vae_variant['generate_lstm_dataset_kwargs']['segmentation_kwargs'] = skewfit_variant['segmentation_kwargs']

    # handles the case of using segmentation + proprio information for observation
    if skewfit_variant['observation_mode'] == 'segmentation_proprio':
        skewfit_variant['algo_kwargs']['keep_train_original_vae'] = False
        train_vae_variant['num_vae_epochs'] = 0 
    skewfit_variant['replay_buffer_kwargs']['observation_mode'] = skewfit_variant['observation_mode'] 


def train_vae_and_update_variant(variant): # actually pretrain vae and ROLL.
    skewfit_variant = variant['skewfit_variant']
    train_vae_variant = variant['train_vae_variant']

    # prepare the background subtractor needed to perform segmentation
    if 'unet' in skewfit_variant['segmentation_method']:
        print("training opencv background model!")
        v = train_vae_variant['generate_lstm_dataset_kwargs']
        env_id = v.get('env_id', None)
        env_id_invis = invisiable_env_id[env_id]
        import gym
        import multiworld
        multiworld.register_all_envs()
        obj_invisible_env = gym.make(env_id_invis)
        init_camera = v.get('init_camera', None)
        
        presampled_goals = None
        if skewfit_variant.get("presampled_goals_path") is not None:
            presampled_goals = load_local_or_remote_file(
                    skewfit_variant['presampled_goals_path']
                ).item()
            print("presampled goal path is: ", skewfit_variant['presampled_goals_path'])

        obj_invisible_env = ImageEnv(
            obj_invisible_env,
            v.get('imsize'),
            init_camera=init_camera,
            transpose=True,
            normalize=True,
            presampled_goals=presampled_goals,
        )

        train_num = 2000 if 'Push' in env_id else 4000
        train_bgsb(obj_invisible_env, train_num=train_num)


    if skewfit_variant.get('vae_path', None) is None: # train new vaes
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )

        vaes, vae_train_datas, vae_test_datas = train_vae(train_vae_variant, skewfit_variant=skewfit_variant,
                                                       return_data=True) # one original vae, one segmented ROLL.
        if skewfit_variant.get('save_vae_data', False):
            skewfit_variant['vae_train_data'] = vae_train_datas
            skewfit_variant['vae_test_data'] = vae_test_datas
            
        logger.add_tabular_output(
            'progress.csv',
            relative_to_snapshot_dir=True,
        )
        skewfit_variant['vae_path'] = vaes  # just pass the VAE directly
    else: # load pre-trained vaes
        print("load pretrain scene-/objce-VAE from: {}".format(skewfit_variant['vae_path']))
        data = torch.load(osp.join(skewfit_variant['vae_path'], 'params.pkl'))
        vae_original = data['vae_original']
        vae_segmented = data['lstm_segmented']
        skewfit_variant['vae_path'] = [vae_segmented, vae_original]

        generate_vae_dataset_fctn = train_vae_variant.get('generate_vae_data_fctn',
                                            generate_vae_dataset)
        generate_lstm_dataset_fctn = train_vae_variant.get('generate_lstm_data_fctn')
        assert generate_lstm_dataset_fctn is not None, "Must provide a custom generate lstm pretraining dataset function!"

        train_data_lstm, test_data_lstm, info_lstm = generate_lstm_dataset_fctn(
            train_vae_variant['generate_lstm_dataset_kwargs'], segmented=True,
            segmentation_method=skewfit_variant['segmentation_method']
        )

        train_data_ori, test_data_ori, info_ori = generate_vae_dataset_fctn(
            train_vae_variant['generate_vae_dataset_kwargs']
        )

        train_datas = [train_data_lstm, train_data_ori]    
        test_datas = [test_data_lstm, test_data_ori]

        if skewfit_variant.get('save_vae_data', False):
            skewfit_variant['vae_train_data'] = train_datas
            skewfit_variant['vae_test_data'] = test_datas
        

def train_vae(variant, return_data=False, skewfit_variant=None): # acutally train both the vae and the lstm
    from rlkit.util.ml_util import PiecewiseLinearSchedule
    from rlkit.torch.vae.conv_vae import (
        ConvVAE,
    )
    import rlkit.torch.vae.conv_vae as conv_vae
    import ROLL.LSTM_model as LSTM_model
    from ROLL.LSTM_model import ConvLSTM2
    from ROLL.LSTM_trainer import ConvLSTMTrainer
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    import rlkit.torch.pytorch_util as ptu
    from rlkit.pythonplusplus import identity
    import torch
    seg_pretrain = variant['seg_pretrain']
    ori_pretrain = variant['ori_pretrain']
    generate_vae_dataset_fctn = variant.get('generate_vae_data_fctn',
                                            generate_vae_dataset)
    generate_lstm_dataset_fctn = variant.get('generate_lstm_data_fctn')
    assert generate_lstm_dataset_fctn is not None, "Must provide a custom generate lstm pretraining dataset function!"

    train_data_lstm, test_data_lstm, info_lstm = generate_lstm_dataset_fctn(
        variant['generate_lstm_dataset_kwargs'], segmented=True,
        segmentation_method=skewfit_variant['segmentation_method']
    )

    train_data_ori, test_data_ori, info_ori = generate_vae_dataset_fctn(
        variant['generate_vae_dataset_kwargs']
    )

    if 'beta_schedule_kwargs' in variant:
        beta_schedule = PiecewiseLinearSchedule(
            **variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None
    if variant.get('decoder_activation', None) == 'sigmoid':
        decoder_activation = torch.nn.Sigmoid()
    else:
        decoder_activation = identity

    architecture = variant['vae_kwargs'].get('architecture', None)
    if not architecture and variant.get('imsize') == 84:
        architecture = conv_vae.imsize84_default_architecture
    elif not architecture and variant.get('imsize') == 48:
        architecture = conv_vae.imsize48_default_architecture
    variant['vae_kwargs']['architecture'] = architecture
    variant['vae_kwargs']['imsize'] = variant.get('imsize')

    architecture = variant['lstm_kwargs'].get('architecture', None)
    if not architecture and variant.get('imsize') == 84:
        architecture = None # TODO LSTM: wrap a 84 lstm architecutre
    elif not architecture and variant.get('imsize') == 48:
        architecture =  LSTM_model.imsize48_default_architecture
    variant['lstm_kwargs']['architecture'] = architecture
    variant['lstm_kwargs']['imsize'] = variant.get('imsize')


    train_datas = [train_data_lstm, train_data_ori, ]    
    test_datas = [test_data_lstm, test_data_ori, ]
    names = ['lstm_seg_pretrain', 'vae_ori_pretrain', ]
    vaes = []
    env_id = variant['generate_lstm_dataset_kwargs'].get('env_id')
    assert env_id is not None
    lstm_pretrain_vae_only = variant.get('lstm_pretrain_vae_only', False)

    for idx in range(2):
        train_data, test_data, name = train_datas[idx], test_datas[idx], names[idx]    
        
        logger.add_tabular_output(
            '{}_progress.csv'.format(name), relative_to_snapshot_dir=True
        )

        if idx == 1: # train the original vae
            representation_size = variant.get("vae_representation_size", variant.get('representation_size'))
            beta = variant.get('vae_beta', variant.get('beta'))
            m = ConvVAE(
                representation_size,
                decoder_output_activation=decoder_activation,
                **variant['vae_kwargs']
            )
            t = ConvVAETrainer(train_data, test_data, m, beta=beta,
                            beta_schedule=beta_schedule, **variant['algo_kwargs'])
        else: # train the segmentation lstm
            lstm_version = variant.get('lstm_version', 2)
            if lstm_version == 2:
                lstm_class = ConvLSTM2
 
            representation_size = variant.get("lstm_representation_size", variant.get('representation_size'))
            beta = variant.get('lstm_beta', variant.get('beta'))
            m = lstm_class(
                representation_size,
                decoder_output_activation=decoder_activation,
                **variant['lstm_kwargs']
            )
            t = ConvLSTMTrainer(train_data, test_data, m, beta=beta,
                            beta_schedule=beta_schedule, **variant['algo_kwargs'])

        m.to(ptu.device)
        
        vaes.append(m)

        print("test data len: ", len(test_data))
        print("train data len: ", len(train_data))
        
        save_period = variant['save_period']

        pjhome = os.environ['PJHOME']
        if env_id == 'SawyerPushHurdle-v0' and osp.exists(
            osp.join(pjhome, 'data/local/pre-train-lstm', '{}-{}-{}-0.3-0.5.npy'.format(
                'SawyerPushHurdle-v0', 'seg-color', '500'))
        ):
            data_file_path = osp.join(pjhome, 'data/local/pre-train-lstm', '{}-{}-{}-0.3-0.5.npy'.format(env_id, 'seg-color', 500))
            puck_pos_path = osp.join(pjhome, 'data/local/pre-train-lstm', '{}-{}-{}-0.3-0.5-puck-pos.npy'.format(env_id, 'seg-color', 500))
            all_data = np.load(data_file_path)
            puck_pos = np.load(puck_pos_path)
            all_data = normalize_image(all_data.copy())
            obj_states = puck_pos
        else:
            all_data = np.concatenate([train_data_lstm, test_data_lstm], axis=0)
            all_data = normalize_image(all_data.copy())
            obj_states = info_lstm['obj_state']

        obj = 'door' if 'Door' in env_id else 'puck'

        num_epochs = variant['num_lstm_epochs'] if idx == 0 else variant['num_vae_epochs']

        if (idx == 0 and seg_pretrain) or (idx == 1 and ori_pretrain):
            for epoch in range(num_epochs):
                should_save_imgs = (epoch % save_period == 0)
                if idx == 0: # only LSTM trainer has 'only_train_vae' argument
                    t.train_epoch(epoch, only_train_vae=lstm_pretrain_vae_only)
                    t.test_epoch(
                        epoch,
                        save_reconstruction=should_save_imgs,
                        save_prefix='r_' + name,
                        only_train_vae=lstm_pretrain_vae_only
                    )
                else:
                    t.train_epoch(epoch)
                    t.test_epoch(
                        epoch,
                        save_reconstruction=should_save_imgs,
                        save_prefix='r_' + name,
                    )
                
                if should_save_imgs:
                    t.dump_samples(epoch, save_prefix='s_' + name)

                    if idx == 0:
                        compare_latent_distance(m, all_data, obj_states, obj_name=obj, save_dir=logger.get_snapshot_dir(),
                            save_name='lstm_latent_distance_{}.png'.format(epoch))
                        test_lstm_traj(env_id, m, save_path=logger.get_snapshot_dir(), 
                            save_name='lstm_test_traj_{}.png'.format(epoch))
                        test_masked_traj_lstm(env_id, m, save_dir=logger.get_snapshot_dir(), 
                            save_name='masked_test_{}.png'.format(epoch))

                t.update_train_weights()

            logger.save_extra_data(m, '{}.pkl'.format(name), mode='pickle')

        logger.remove_tabular_output(
            '{}_progress.csv'.format(name), relative_to_snapshot_dir=True
        )

        if idx == 0 and variant.get("only_train_lstm", False):
            exit()

    if return_data:
        return vaes, train_datas, test_datas
    return m


def generate_vae_dataset(variant):
    """
    If not provided a pre-train vae dataset generation function, this function will be used to collect
    the dataset for training vae.
    """
    import rlkit.torch.pytorch_util as ptu
    import gym
    import multiworld
    multiworld.register_all_envs()

    print("generating vae dataset with original images")

    env_class = variant.get('env_class', None)
    env_kwargs = variant.get('env_kwargs', None)
    env_id = variant.get('env_id', None)
    N = variant.get('N', 10000)
    test_p = variant.get('test_p', 0.9)
    use_cached = variant.get('use_cached', True)
    imsize = variant.get('imsize', 84)
    num_channels = variant.get('num_channels', 3)
    show = variant.get('show', False)
    init_camera = variant.get('init_camera', None)
    dataset_path = variant.get('dataset_path', None)
    oracle_dataset_using_set_to_goal = variant.get(
        'oracle_dataset_using_set_to_goal', False)
    random_rollout_data = variant.get('random_rollout_data', False)
    random_and_oracle_policy_data = variant.get('random_and_oracle_policy_data',
                                                False)
    random_and_oracle_policy_data_split = variant.get(
        'random_and_oracle_policy_data_split', 0)
    policy_file = variant.get('policy_file', None)
    n_random_steps = variant.get('n_random_steps', 100)
    vae_dataset_specific_env_kwargs = variant.get(
        'vae_dataset_specific_env_kwargs', None)
    save_file_prefix = variant.get('save_file_prefix', None)
    non_presampled_goal_img_is_garbage = variant.get(
        'non_presampled_goal_img_is_garbage', None)
    tag = variant.get('tag', '')

    info = {}
    if dataset_path is not None:
        print('load vae training dataset from: ', dataset_path)
        pjhome = os.environ['PJHOME']
        dataset = np.load(osp.join(pjhome, dataset_path), allow_pickle=True).item()
        if isinstance(dataset, dict):
            dataset = dataset['image_desired_goal']
        dataset = unormalize_image(dataset)
        N = dataset.shape[0]
    else:
        if env_kwargs is None:
            env_kwargs = {}
        if save_file_prefix is None:
            save_file_prefix = env_id
        if save_file_prefix is None:
            save_file_prefix = env_class.__name__
        filename = "/tmp/{}_N{}_{}_imsize{}_random_oracle_split_{}{}.npy".format(
            save_file_prefix,
            str(N),
            init_camera.__name__ if init_camera else '',
            imsize,
            random_and_oracle_policy_data_split,
            tag,
        )
        if use_cached and osp.isfile(filename):
            dataset = np.load(filename)
            print("loaded data from saved file", filename)
        else:
            now = time.time()

            if env_id is not None:
                import gym
                import multiworld
                multiworld.register_all_envs()
                env = gym.make(env_id)
            else:
                if vae_dataset_specific_env_kwargs is None:
                    vae_dataset_specific_env_kwargs = {}
                for key, val in env_kwargs.items():
                    if key not in vae_dataset_specific_env_kwargs:
                        vae_dataset_specific_env_kwargs[key] = val
                env = env_class(**vae_dataset_specific_env_kwargs)
            if not isinstance(env, ImageEnv):
                env = ImageEnv(
                    env,
                    imsize,
                    init_camera=init_camera,
                    transpose=True,
                    normalize=True,
                    non_presampled_goal_img_is_garbage=non_presampled_goal_img_is_garbage,
                )
            else:
                imsize = env.imsize
                env.non_presampled_goal_img_is_garbage = non_presampled_goal_img_is_garbage
            env.reset()
            info['env'] = env
            if random_and_oracle_policy_data:
                policy_file = load_local_or_remote_file(policy_file)
                policy = policy_file['policy']
                policy.to(ptu.device)
            if random_rollout_data:
                from rlkit.exploration_strategies.ou_strategy import OUStrategy
                policy = OUStrategy(env.action_space)

            dataset = np.zeros((N, imsize * imsize * num_channels),
                               dtype=np.uint8)

            for i in range(N):
                if random_and_oracle_policy_data:
                    num_random_steps = int(
                        N * random_and_oracle_policy_data_split)
                    if i < num_random_steps:
                        env.reset()
                        for _ in range(n_random_steps):
                            obs = env.step(env.action_space.sample())[0]
                    else:
                        obs = env.reset()
                        policy.reset()
                        for _ in range(n_random_steps):
                            policy_obs = np.hstack((
                                obs['state_observation'],
                                obs['state_desired_goal'],
                            ))
                            action, _ = policy.get_action(policy_obs)
                            obs, _, _, _ = env.step(action)
                elif oracle_dataset_using_set_to_goal:
                    print(i)
                    goal = env.sample_goal()
                    env.set_to_goal(goal)
                    obs = env._get_obs()

                elif random_rollout_data:
                    if i % n_random_steps == 0:
                        g = dict(
                            state_desired_goal=env.sample_goal_for_rollout())
                        env.set_to_goal(g)
                        policy.reset()
                        # env.reset()
                    u = policy.get_action_from_raw_action(
                        env.action_space.sample())
                    obs = env.step(u)[0]
                else:
                    print("using totally random rollouts")
                    for _ in range(n_random_steps):
                        obs = env.step(env.action_space.sample())[0]

                img = obs['image_observation'] # NOTE yufei: this is already normalized image, of detype np.float64.

                dataset[i, :] = unormalize_image(img)

            np.save(filename, dataset)

    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset, info


def get_envs(variant):
    from multiworld.core.image_env import ImageEnv
    from rlkit.envs.vae_wrapper import VAEWrappedEnv
    from rlkit.envs.vae_wrapper_segmented import VAEWrappedEnvSegmented
    from ROLL.LSTM_wrapped_env import LSTMWrappedEnv
    from rlkit.util.io import load_local_or_remote_file

    render = variant.get('render', False)
    vae_path = variant.get("vae_path", None)
    reward_params = variant.get("reward_params", dict())
    init_camera = variant.get("init_camera", None)
    do_state_exp = variant.get("do_state_exp", False)
    presample_goals = variant.get('presample_goals', False)
    presample_image_goals_only = variant.get('presample_image_goals_only',
                                             False)
    presampled_goals_path = variant.get('presampled_goals_path', None)

    vaes = load_local_or_remote_file(vae_path) if type(
        vae_path) is str else vae_path
    vae = vaes[1] # this is the vae using original images

    env_kwargs = variant.get('env_kwargs', {})
    if 'env_id' in variant:
        import gym
        import multiworld
        multiworld.register_all_envs()
        print('env_kwargs is: ', env_kwargs)
        env = gym.make(variant['env_id'], **env_kwargs)
    else:
        env = variant["env_class"](**variant['env_kwargs'])

    segmentation_kwargs = variant.get('segmentation_kwargs', {})
    observation_mode = variant.get('observation_mode', 'original_image')
    if not do_state_exp:
        if isinstance(env, ImageEnv):
            image_env = env
        else:
            image_env = ImageEnv(
                env,
                variant.get('imsize'),
                init_camera=init_camera,
                transpose=True,
                normalize=True,
            )
        if presample_goals:
            """
            This will fail for online-parallel as presampled_goals will not be
            serialized. Also don't use this for online-vae.
            """
            if presampled_goals_path is None:
                image_env.non_presampled_goal_img_is_garbage = True
                vae_env = VAEWrappedEnv(
                    image_env,
                    vae,
                    imsize=image_env.imsize,
                    decode_goals=render,
                    render_goals=render,
                    render_rollouts=render,
                    reward_params=reward_params,
                    **variant.get('vae_wrapped_env_kwargs', {})
                )
                presampled_goals = variant['generate_goal_dataset_fctn'](
                    env=vae_env,
                    env_id=variant.get('env_id', None),
                    **variant['goal_generation_kwargs']
                )
                del vae_env
            else:
                presampled_goals = load_local_or_remote_file(
                    presampled_goals_path
                ).item()
            del image_env
            image_env = ImageEnv(
                env,
                variant.get('imsize'),
                init_camera=init_camera,
                transpose=True,
                normalize=True,
                presampled_goals=presampled_goals,
                **variant.get('image_env_kwargs', {})
            )
            vae_env = LSTMWrappedEnv(
                image_env,
                vae_original=vaes[1],
                lstm_segmented=vaes[0],
                segmentation=variant['segmentation'],
                segmentation_method=variant['segmentation_method'],
                imsize=image_env.imsize,
                decode_goals=render,
                render_goals=render,
                render_rollouts=render,
                presampled_goals=presampled_goals,
                reward_params=reward_params,
                segmentation_kwargs=segmentation_kwargs,
                observation_mode=observation_mode,
                **variant.get('vae_wrapped_env_kwargs', {})
            )
            print("Presampling all goals only")
        else:
            vae_env = LSTMWrappedEnv(
                image_env,
                vae_original=vaes[1],
                lstm_segmented=vaes[0],
                segmentation=variant['segmentation'],
                segmentation_method=variant['segmentation_method'],
                imsize=image_env.imsize,
                decode_goals=render,
                render_goals=render,
                render_rollouts=render,
                reward_params=reward_params,
                segmentation_kwargs=segmentation_kwargs,
                observation_mode=observation_mode,
                **variant.get('vae_wrapped_env_kwargs', {})
            )
            if presample_image_goals_only:
                presampled_goals = variant['generate_goal_dataset_fctn'](
                    image_env=vae_env.wrapped_env,
                    **variant['goal_generation_kwargs']
                )
                image_env.set_presampled_goals(presampled_goals)
                print("Presampling image goals only")
            else:
                print("Not using presampled goals")

        env = vae_env

    return env

def skewfit_preprocess_variant(variant):
    if variant.get("do_state_exp", False):
        variant['observation_key'] = 'state_observation'
        variant['desired_goal_key'] = 'state_desired_goal'
        variant['achieved_goal_key'] = 'state_acheived_goal'


def skewfit_experiment(variant):
    import rlkit.torch.pytorch_util as ptu
    from ROLL.online_LSTM_replay_buffer import OnlineLSTMRelabelingBuffer
    from rlkit.torch.networks import FlattenMlp, FlattenPreprocessMlp
    from rlkit.torch.sac.policies import TanhGaussianPolicy, TanhGaussianPreprocessPolicy
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    from ROLL.LSTM_trainer import ConvLSTMTrainer

    skewfit_preprocess_variant(variant)
    env = get_envs(variant)

    uniform_dataset_fn = variant.get('generate_uniform_dataset_fn', None)
    if uniform_dataset_fn:
        uniform_dataset = uniform_dataset_fn(
            **variant['generate_uniform_dataset_kwargs']
        )
    else:
        uniform_dataset = None

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
    )

    action_dim = env.action_space.low.size
    hidden_sizes = variant.get('hidden_sizes', [400, 300])

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    )

    vae = env.vae_original
    vae_original = env.vae_original
    lstm_segmented = env.lstm_segmented

    replay_buffer = OnlineLSTMRelabelingBuffer(
        vae_ori=vae_original,
        lstm_seg=lstm_segmented,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    vae_trainer_original = ConvVAETrainer(
        variant['vae_train_data'][1],
        variant['vae_test_data'][1],
        env.vae_original,
        **variant['online_vae_trainer_kwargs']
    )

    lstm_trainer_segmented = ConvLSTMTrainer(
        variant['vae_train_data'][0],
        variant['vae_test_data'][0],
        env.lstm_segmented,
        **variant['online_lstm_trainer_kwargs']
    )

    assert 'vae_training_schedule' not in variant, "Just put it in algo_kwargs"
    max_path_length = variant['max_path_length']

    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['twin_sac_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = LSTMWrappedEnvPathCollector(
        variant['evaluation_goal_sampling_mode'],
        env,
        MakeDeterministic(policy),
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = LSTMWrappedEnvPathCollector(
        variant['exploration_goal_sampling_mode'],
        env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )

    keep_train_segmentation_lstm = variant.get('keep_train_segmentation_lstm', False)
    algorithm = OnlineLSTMAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        vae_original=vae_original,
        lstm_segmented=lstm_segmented,
        vae_trainer_original=vae_trainer_original,
        lstm_trainer_segmented=lstm_trainer_segmented,
        uniform_dataset=uniform_dataset,
        max_path_length=max_path_length,
        keep_train_segmentation_lstm=keep_train_segmentation_lstm,
        **variant['algo_kwargs']
    )

    if variant['custom_goal_sampler'] == 'replay_buffer':
        env.custom_goal_sampler = replay_buffer.sample_buffer_goals

    algorithm.to(ptu.device)
    vae.to(ptu.device)
    algorithm.train()
