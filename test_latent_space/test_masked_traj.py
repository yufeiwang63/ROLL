import numpy as np
import os
import os.path as osp
import rlkit.torch.pytorch_util as ptu
from matplotlib import pyplot as plt
import torch
from ROLL.LSTM_model import ConvLSTM2

def test_masked_traj_lstm(env_id, lstm, save_dir=None, save_name=None):
    # NOTE: acutally you can also generate the masked data online here
    pjhome = os.environ['PJHOME']
    ori_data = osp.join(pjhome, 'data/local/env/{}-masked-test-traj-ori.npy'.format(env_id))
    masked_data = osp.join(pjhome, 'data/local/env/{}-masked-test-traj-masked.npy'.format(env_id))
    masked_idx = osp.join(pjhome, 'data/local/env/{}-masked-idx.npy'.format(env_id))
    if not osp.exists(ori_data):
        return
    ori_data = np.load(ori_data)
    masked_data = np.load(masked_data)
    masked_idx = np.load(masked_idx)
    batch_size, traj_len, imlen = ori_data.shape

    latents_ori = ROLL.encode(ptu.from_numpy(ori_data.swapaxes(0, 1)))[0]
    latents_masked = ROLL.encode(ptu.from_numpy(masked_data.swapaxes(0, 1)))[0]
    latents_ori = ptu.get_numpy(latents_ori).swapaxes(0, 1)
    latents_masked = ptu.get_numpy(latents_masked).swapaxes(0, 1)
    batch_size, traj_len, feature_size = latents_ori.shape

    latents_masked_vectors = latents_masked[np.arange(batch_size), masked_idx] # batch_size x feature_size
    distances = latents_masked_vectors[:, np.newaxis, :] - latents_ori 
    assert distances.shape == (batch_size, traj_len, feature_size)
    distances = np.linalg.norm(distances, axis=-1) # batch_size x traj_len
    closest = np.argmin(distances, axis=-1) # batch size

    plt.plot(range(len(masked_idx)), masked_idx, label='true label')
    plt.plot(range(len(closest)), closest, label='prediction label')
    plt.legend()
    correct = np.sum(closest == masked_idx)
    total = batch_size
    acc = np.sum(correct) / batch_size
    plt.title("lstm correct {}/{}, acc: {} ".format(correct, total, acc))
    if save_dir is not None:
        plt.savefig(osp.join(save_dir, save_name))
    else:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close('all')

def test_masked_traj_vae(env_id, vae, save_dir=None, save_name=None):
    # NOTE: acutally you can also generate the masked data online here
    pjhome = os.environ['PJHOME']
    ori_data = osp.join(pjhome, 'data/local/env/{}-masked-test-traj-ori.npy'.format(env_id))
    masked_data = osp.join(pjhome, 'data/local/env/{}-masked-test-traj-masked.npy'.format(env_id))
    masked_idx = osp.join(pjhome, 'data/local/env/{}-masked-idx.npy'.format(env_id))

    ori_data = np.load(ori_data)
    masked_data = np.load(masked_data)
    masked_idx = np.load(masked_idx)
    batch_size, traj_len, imlen = ori_data.shape
    
    ori_data = ori_data.reshape((-1, imlen))
    masked_data = masked_data.reshape((-1, imlen))

    latents_ori = vae.encode(ptu.from_numpy(ori_data))[0]
    latents_masked = vae.encode(ptu.from_numpy(masked_data))[0]
    latents_ori = ptu.get_numpy(latents_ori).reshape((batch_size, traj_len, -1))
    latents_masked = ptu.get_numpy(latents_masked).reshape((batch_size, traj_len, -1))
    batch_size, traj_len, feature_size = latents_ori.shape

    latents_masked_vectors = latents_masked[np.arange(batch_size), masked_idx] # batch_size x feature_size
    distances = latents_masked_vectors[:, np.newaxis, :] - latents_ori 
    assert distances.shape == (batch_size, traj_len, feature_size)
    distances = np.linalg.norm(distances, axis=-1) # batch_size x traj_len
    closest = np.argmin(distances, axis=-1) # batch size

    plt.plot(range(len(masked_idx)), masked_idx, label='true label')
    plt.plot(range(len(closest)), closest, label='prediction label')
    plt.legend()
    correct = np.sum(closest == masked_idx)
    total = batch_size
    acc = np.sum(correct) / batch_size
    plt.title("vae correct {}/{}, acc: {} ".format(correct, total, acc))
    if save_dir is not None:
        plt.savefig(osp.join(save_dir, save_name))
    else:
        plt.show()


if __name__ == '__main__':

    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False
    ptu.set_gpu_mode(use_gpu)
    
    lstm = torch.load('./data/local/debuglstm/6-10-lstm2-2/params.pkl')
    data = torch.load('./data/seuss/6-10-skewfit-sawyerhurdle-lstm2-ae-loss-0.5/6-10-skewfit-sawyerhurdle-lstm2-ae-loss-0.5/'
        '6-10-skewfit-sawyerhurdle-lstm2-ae-loss-0.5_2020_06_10_02_27_34_0002/200/params.pkl')
    lstm = data['lstm_segmented']

    # a random LSTM
    # from ROLL.LSTM_model import imsize48_default_architecture
    # LSTM_kwargs=dict(
    #     representation_size=6,
    # )
    # lstm = ConvLSTM2(
    #     architecture=imsize48_default_architecture,
    #     **LSTM_kwargs
    # )
    # ROLL.to(ptu.device)
    
    test_masked_traj_lstm('SawyerPushHurdle-v0', lstm)

    # test vae
    # vae_path = './data/seuss/6-5-skewfit-hurdle-segcolorpretrain/6-5-skewfit-hurdle-segcolorpretrain/6-5-skewfit-hurdle-segcolorpretrain_2020_06_06_02_48_49_0001/100'
    # vae_path = osp.join(vae_path, 'params.pkl')
    # data = torch.load(vae_path)
    # vae_segmented = data['vae_segmented']
    # test_masked_traj_vae('SawyerPushHurdle-v0', vae_segmented)



    
    

    


