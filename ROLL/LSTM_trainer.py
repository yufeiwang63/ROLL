from collections import OrderedDict
from os import path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from multiworld.core.image_env import normalize_image
from rlkit.core import logger
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.data import (
    ImageDataset,
    InfiniteWeightedRandomSampler,
    InfiniteRandomSampler,
)
from rlkit.util.ml_util import ConstantSchedule
from ROLL.LSTM_model import ConvLSTM2


def relative_probs_from_log_probs(log_probs):
    """
    Returns relative probability from the log probabilities. They're not exactly
    equal to the probability, but relative scalings between them are all maintained.

    For correctness, all log_probs must be passed in at the same time.
    """
    probs = np.exp(log_probs - log_probs.mean())
    assert not np.any(probs <= 0), 'choose a smaller power'
    return probs

def compute_log_p_log_q_log_d(
    model,
    data,
    decoder_distribution='bernoulli',
    num_latents_to_sample=1,
    sampling_method='importance_sampling'
):
    assert data.dtype == np.float64, 'images should be normalized'
    imgs = ptu.from_numpy(data)
    latent_distribution_params = model.encode(imgs)
    batch_size = data.shape[0]
    representation_size = model.representation_size
    log_p, log_q, log_d = ptu.zeros((batch_size, num_latents_to_sample)), ptu.zeros(
        (batch_size, num_latents_to_sample)), ptu.zeros((batch_size, num_latents_to_sample))
    true_prior = Normal(ptu.zeros((batch_size, representation_size)),
                        ptu.ones((batch_size, representation_size)))
    mus, logvars = latent_distribution_params
    for i in range(num_latents_to_sample):
        if sampling_method == 'importance_sampling':
            latents = model.rsample(latent_distribution_params)
        elif sampling_method == 'biased_sampling':
            latents = model.rsample(latent_distribution_params)
        elif sampling_method == 'true_prior_sampling':
            latents = true_prior.rsample()
        else:
            raise EnvironmentError('Invalid Sampling Method Provided')

        stds = logvars.exp().pow(.5)
        vae_dist = Normal(mus, stds)
        log_p_z = true_prior.log_prob(latents).sum(dim=1)
        log_q_z_given_x = vae_dist.log_prob(latents).sum(dim=1)
        if decoder_distribution == 'bernoulli':
            decoded = model.decode(latents)[0]
            log_d_x_given_z = torch.log(imgs * decoded + (1 - imgs) * (1 - decoded) + 1e-8).sum(dim=1)
        elif decoder_distribution == 'gaussian_identity_variance':
            _, obs_distribution_params = model.decode(latents)
            dec_mu, dec_logvar = obs_distribution_params
            dec_var = dec_logvar.exp()
            decoder_dist = Normal(dec_mu, dec_var.pow(.5))
            log_d_x_given_z = decoder_dist.log_prob(imgs).sum(dim=1)
        else:
            raise EnvironmentError('Invalid Decoder Distribution Provided')

        log_p[:, i] = log_p_z
        log_q[:, i] = log_q_z_given_x
        log_d[:, i] = log_d_x_given_z
    return log_p, log_q, log_d

def compute_p_x_np_to_np(
    model,
    data,
    power,
    decoder_distribution='bernoulli',
    num_latents_to_sample=1,
    sampling_method='importance_sampling'
):
    assert data.dtype == np.float64, 'images should be normalized'
    assert power >= -1 and power <= 0, 'power for skew-fit should belong to [-1, 0]'

    log_p, log_q, log_d = compute_log_p_log_q_log_d(
        model,
        data,
        decoder_distribution,
        num_latents_to_sample,
        sampling_method
    )

    if sampling_method == 'importance_sampling':
        log_p_x = (log_p - log_q + log_d).mean(dim=1)
    elif sampling_method == 'biased_sampling' or sampling_method == 'true_prior_sampling':
        log_p_x = log_d.mean(dim=1)
    else:
        raise EnvironmentError('Invalid Sampling Method Provided')
    log_p_x_skewed = power * log_p_x
    return ptu.get_numpy(log_p_x_skewed)


class ConvLSTMTrainer(object):
    def __init__(
            self,
            train_dataset,
            test_dataset,
            model,
            positive_range=2,
            negative_range=10,
            triplet_sample_num=8,
            triplet_loss_margin=0.5,
            batch_size=128,
            log_interval=0,
            recon_loss_coef=1,
            triplet_loss_coef=[],
            triplet_loss_type=[],
            ae_loss_coef=1,
            matching_loss_coef=1,
            vae_matching_loss_coef=1,
            matching_loss_one_side=False,
            contrastive_loss_coef=0,
            lstm_kl_loss_coef=0,
            adaptive_margin=0,
            beta=0.5,
            beta_schedule=None,
            lr=None,
            do_scatterplot=False,
            normalize=False,
            mse_weight=0.1,
            is_auto_encoder=False,
            background_subtract=False,
            use_parallel_dataloading=False,
            train_data_workers=2,
            skew_dataset=False,
            skew_config=None,
            priority_function_kwargs=None,
            start_skew_epoch=0,
            weight_decay=0,
    ):

        print("In LSTM trainer, ae_loss_coef is: ", ae_loss_coef)
        print("In LSTM trainer, matching_loss_coef is: ", matching_loss_coef)
        print("In LSTM trainer, vae_matching_loss_coef is: ", vae_matching_loss_coef)

        if skew_config is None:
            skew_config = {}
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.beta = beta
        if is_auto_encoder:
            self.beta = 0
        if lr is None:
            if is_auto_encoder:
                lr = 1e-2
            else:
                lr = 1e-3
        self.beta_schedule = beta_schedule
        if self.beta_schedule is None or is_auto_encoder:
            self.beta_schedule = ConstantSchedule(self.beta)
        self.imsize = model.imsize
        self.do_scatterplot = do_scatterplot

        self.recon_loss_coef = recon_loss_coef
        self.triplet_loss_coef = triplet_loss_coef
        self.ae_loss_coef = ae_loss_coef
        self.matching_loss_coef = matching_loss_coef
        self.vae_matching_loss_coef = vae_matching_loss_coef
        self.contrastive_loss_coef = contrastive_loss_coef
        self.lstm_kl_loss_coef = lstm_kl_loss_coef
        self.matching_loss_one_side = matching_loss_one_side

        # triplet loss range
        self.positve_range = positive_range
        self.negative_range = negative_range
        self.triplet_sample_num = triplet_sample_num
        self.triplet_loss_margin = triplet_loss_margin
        self.triplet_loss_type = triplet_loss_type
        self.adaptive_margin = adaptive_margin

        model.to(ptu.device)

        self.model = model
        self.representation_size = model.representation_size
        self.input_channels = model.input_channels
        self.imlength = model.imlength

        self.lr = lr
        params = list(self.model.parameters())
        self.optimizer = optim.Adam(params,
            lr=self.lr,
            weight_decay=weight_decay,
        )
        self.train_dataset, self.test_dataset = train_dataset, test_dataset
        assert self.train_dataset.dtype == np.uint8
        assert self.test_dataset.dtype == np.uint8

        self.batch_size = batch_size
        self.use_parallel_dataloading = use_parallel_dataloading
        self.train_data_workers = train_data_workers
        self.skew_dataset = skew_dataset
        self.skew_config = skew_config
        self.start_skew_epoch = start_skew_epoch
        if priority_function_kwargs is None:
            self.priority_function_kwargs = dict()
        else:
            self.priority_function_kwargs = priority_function_kwargs

        if self.skew_dataset:
            self._train_weights = self._compute_train_weights()
        else:
            self._train_weights = None

        if use_parallel_dataloading:
            self.train_dataset_pt = ImageDataset(
                train_dataset,
                should_normalize=True
            )
            self.test_dataset_pt = ImageDataset(
                test_dataset,
                should_normalize=True
            )

            if self.skew_dataset:
                base_sampler = InfiniteWeightedRandomSampler(
                    self.train_dataset, self._train_weights
                )
            else:
                base_sampler = InfiniteRandomSampler(self.train_dataset)
            self.train_dataloader = DataLoader(
                self.train_dataset_pt,
                sampler=InfiniteRandomSampler(self.train_dataset),
                batch_size=batch_size,
                drop_last=False,
                num_workers=train_data_workers,
                pin_memory=True,
            )
            self.test_dataloader = DataLoader(
                self.test_dataset_pt,
                sampler=InfiniteRandomSampler(self.test_dataset),
                batch_size=batch_size,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
            )
            self.train_dataloader = iter(self.train_dataloader)
            self.test_dataloader = iter(self.test_dataloader)

        self.normalize = normalize
        self.mse_weight = mse_weight
        self.background_subtract = background_subtract

        if self.normalize or self.background_subtract:
            self.train_data_mean = np.mean(self.train_dataset, axis=0)
            self.train_data_mean = normalize_image(
                np.uint8(self.train_data_mean)
            )
        self.eval_statistics = OrderedDict()
        self._extra_stats_to_log = None

    def get_dataset_stats(self, data):
        torch_input = ptu.from_numpy(normalize_image(data))
        mus, log_vars = self.model.encode(torch_input)
        mus = ptu.get_numpy(mus)
        mean = np.mean(mus, axis=0)
        std = np.std(mus, axis=0)
        return mus, mean, std

    def update_train_weights(self):
        if self.skew_dataset:
            self._train_weights = self._compute_train_weights()
            if self.use_parallel_dataloading:
                self.train_dataloader = DataLoader(
                    self.train_dataset_pt,
                    sampler=InfiniteWeightedRandomSampler(self.train_dataset, self._train_weights),
                    batch_size=self.batch_size,
                    drop_last=False,
                    num_workers=self.train_data_workers,
                    pin_memory=True,
                )
                self.train_dataloader = iter(self.train_dataloader)

    def _compute_train_weights(self):
        method = self.skew_config.get('method', 'squared_error')
        power = self.skew_config.get('power', 1)
        batch_size = 512
        size = self.train_dataset.shape[0]
        next_idx = min(batch_size, size)
        cur_idx = 0
        weights = np.zeros(size)
        while cur_idx < self.train_dataset.shape[0]:
            idxs = np.arange(cur_idx, next_idx)
            data = self.train_dataset[idxs, :]
            if method == 'vae_prob':
                data = normalize_image(data)
                weights[idxs] = compute_p_x_np_to_np(self.model, data, power=power, **self.priority_function_kwargs)
            else:
                raise NotImplementedError('Method {} not supported'.format(method))
            cur_idx = next_idx
            next_idx += batch_size
            next_idx = min(next_idx, size)

        if method == 'vae_prob':
            weights = relative_probs_from_log_probs(weights)
        return weights

    def set_vae(self, vae):
        self.model = vae
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def get_batch(self, train=True, epoch=None):
        if self.use_parallel_dataloading:
            if not train:
                dataloader = self.test_dataloader
            else:
                dataloader = self.train_dataloader
            samples = next(dataloader).to(ptu.device)
            return samples

        dataset = self.train_dataset if train else self.test_dataset
        skew = False
        if epoch is not None:
            skew = (self.start_skew_epoch < epoch)
        if train and self.skew_dataset and skew:
            probs = self._train_weights / np.sum(self._train_weights)
            ind = np.random.choice(
                len(probs),
                self.batch_size,
                p=probs,
            )
        else:
            ind = np.random.randint(0, len(dataset), self.batch_size)
        samples = normalize_image(dataset[ind, :]) # this should be a batch of trajectories
        if self.normalize:
            samples = ((samples - self.train_data_mean) + 1) / 2
        if self.background_subtract:
            samples = samples - self.train_data_mean

        samples = np.swapaxes(samples, 0, 1) # turn to trajectory, batch_size, feature_size
        return ptu.from_numpy(samples)

    def get_debug_batch(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        X, Y = dataset
        ind = np.random.randint(0, Y.shape[0], self.batch_size)
        X = X[ind, :]
        Y = Y[ind, :]
        return ptu.from_numpy(X), ptu.from_numpy(Y)

    def matching_loss_vae(self, traj_torch):
        _, _, vae_latent_distribution_params, _ = self.model(traj_torch)
        vae_latents_ori = vae_latent_distribution_params[0]

        # new way of correct masking
        masked_traj = traj_torch.detach().clone()
        traj_len, batch_size, imlen = masked_traj.shape
        mask = (np.random.uniform(size=(self.imsize, self.imsize)) > 0.5).astype(np.float)
        mask = np.stack([mask, mask, mask], axis=0).flatten()
        masked_traj = masked_traj * ptu.from_numpy(mask) # mask all images for training vae latent space

        _, _, vae_latent_distribution_params, _ = self.model(masked_traj)
        vae_latents_masked = vae_latent_distribution_params[0]
        
        if self.matching_loss_one_side:
            loss = F.mse_loss(vae_latents_ori.detach(), vae_latents_masked)
        else:
            loss = F.mse_loss(vae_latents_ori, vae_latents_masked)

        return loss

    def contrastive_loss(self, traj):
        masked_traj = traj.detach().clone()
        traj_len, batch_size, imlen = traj.shape
        masked_idx = np.random.randint(low=10, high=traj_len, size=batch_size)
        for i in range(batch_size):
            mask = (np.random.uniform(size=(self.imsize, self.imsize)) > 0.5).astype(np.float)
            mask = np.stack([mask, mask, mask], axis=0).flatten()
            masked_traj[masked_idx[i]][i] *= ptu.from_numpy(mask)

        latents_ori = self.model.encode(traj)[0]
        latents_masked = self.model.encode(masked_traj)[0]
        
        loss = ptu.zeros(1)
        for j in range(batch_size):
            latents_ori_vec = latents_ori[masked_idx[j]:, j]
            latents_masked_vec = latents_masked[masked_idx[j]:, j]
            loss += F.mse_loss(latents_masked_vec, latents_ori_vec)

        postive_loss = loss / batch_size

        encodings = latents_ori
        seq_len, batch_size, _ = encodings.shape

        anchors, negatives, margins = [], [], []

        for t in range(seq_len):
            neg_range_prev_end, neg_range_after_beg = max(0, t - self.negative_range), min(seq_len - 1, t + self.negative_range)
            for _ in range(self.triplet_sample_num):
                neg_idices = np.array([x for x in range(neg_range_prev_end)] + [x for x in range(neg_range_after_beg + 1, seq_len)], dtype=np.int32)
                neg_idx = np.random.randint(0 , len(neg_idices), batch_size)
                neg_idx = neg_idices[neg_idx] # batch_size

                if self.adaptive_margin > 0:
                    time_differences = np.abs(neg_idx - t) # batch_size 
                    adaptive_margins = self.adaptive_margin * time_differences
                    margins.append(ptu.from_numpy(adaptive_margins))
                else:
                    margins.append(ptu.from_numpy(np.array([self.triplet_loss_margin for _ in range(batch_size)])))

                anchor_samples = encodings[t] # batch_size, feature_size
                negative_samples = encodings[neg_idx, np.arange(batch_size)]

                anchors.append(anchor_samples) 
                negatives.append(negative_samples)

        anchors = torch.cat(anchors, dim=0)
        negatives = torch.cat(negatives, dim=0)
        margins = torch.cat(margins)

        negative_distances = (anchors - negatives).pow(2).sum(dim=1)
        losses = F.relu(margins - negative_distances, 0)
        negative_loss = losses.mean()

        return postive_loss + negative_loss

    def matching_loss(self, traj_torch):
        masked_traj = traj_torch.detach().clone()
        traj_len, batch_size, imlen = traj_torch.shape
        masked_idx = np.random.randint(low=10, high=traj_len, size=batch_size)
        for i in range(batch_size):
            mask = (np.random.uniform(size=(self.imsize, self.imsize)) > 0.5).astype(np.float)
            mask = np.stack([mask, mask, mask], axis=0).flatten()

            masked_traj[masked_idx[i]][i] *= ptu.from_numpy(mask)

        latents_ori = self.model.encode(traj_torch)[0]
        latents_masked = self.model.encode(masked_traj)[0]
        
        loss = ptu.zeros(1)
        for j in range(batch_size):
            latents_ori_vec = latents_ori[masked_idx[j]:, j]
            latents_masked_vec = latents_masked[masked_idx[j]:, j]
            if self.matching_loss_one_side:
                loss += F.mse_loss(latents_masked_vec, latents_ori_vec.detach())
            else:
                loss += F.mse_loss(latents_masked_vec, latents_ori_vec)
        return loss / batch_size

    def triplet_loss_3(self, traj_torch):
        warm_len = 10
        seq_len, batch_size, imlen = traj_torch.shape
        traj = traj_torch.clone().detach()
        traj = traj[:, :batch_size // 2, :]
        seq_len, batch_size, imlen = traj.shape
        anchors, positives, negatives = [], [], []

        for t in range(seq_len):
            # print(t)
            neg_range_prev_end, neg_range_after_beg = max(0, t - self.negative_range), min(seq_len - 1, t + self.negative_range)
            for _ in range(self.triplet_sample_num):
                neg_idices = np.array([x for x in range(neg_range_prev_end)] + [x for x in range(neg_range_after_beg + 1, seq_len)], dtype=np.int32)
                neg_idx = np.random.randint(0 , len(neg_idices), batch_size)
                neg_idx = neg_idices[neg_idx]
                
                # get the anchor encodings
                anchor_traj = ptu.zeros((warm_len, batch_size, imlen))
                if t - warm_len + 1 >= 0:
                    anchor_traj = traj[t - warm_len + 1: t+1, :, :]
                else:
                    anchor_traj[-(t+1):] = traj[:t+1]
                    for _ in range(warm_len - t - 1):
                        anchor_traj[_] = traj[0]
                anchor_encodings = self.model.encode(anchor_traj)[0] # always assmue we use the mean as encoding

                # get the positive encodings: mask out part of the anchor samples
                pos_traj = anchor_traj.clone().detach()
                mask = (np.random.uniform(size=(self.imsize, self.imsize)) > 0.5).astype(np.float)
                mask = np.stack([mask, mask, mask], axis=0).flatten()
                pos_traj[-1, :, :] *= ptu.from_numpy(mask)
                pos_encodings = self.model.encode(pos_traj)[0]

                # get the negative encodings
                neg_traj = ptu.zeros((warm_len, batch_size, imlen))
                for b_idx in range(batch_size):
                    n_idx = neg_idx[b_idx]
                    if n_idx - warm_len + 1 >= 0:
                        neg_traj[:, b_idx, :] = traj[n_idx - warm_len + 1: n_idx+1, b_idx, :]
                    else:
                        neg_traj[-(n_idx+1):, b_idx, :] = traj[:n_idx+1, b_idx, :]
                        for _ in range(warm_len - n_idx - 1):
                            neg_traj[_, b_idx, :] = traj[0, b_idx, :]
                neg_encodings = self.model.encode(neg_traj)[0]

                anchors.append(anchor_encodings)
                positives.append(pos_encodings)
                negatives.append(neg_encodings)

        anchors = torch.cat(anchors, dim=0)
        positives = torch.cat(positives, dim=0)
        negatives = torch.cat(negatives, dim=0)

        positive_distances = (anchors - positives).pow(2).sum(dim=1)
        negative_distances = (anchors - negatives).pow(2).sum(dim=1)
        losses = F.relu(positive_distances - negative_distances + self.triplet_loss_margin, 0)
        return losses.mean()

    def triplet_loss_2(self, traj_torch):
        '''
        use the same len of images to warm up the lstm encoding.
        '''
        warm_len = 10
        seq_len, batch_size, imlen = traj_torch.shape
        # traj = traj_torch.clone().detach()
        traj = traj_torch
        seq_len, batch_size, imlen = traj.shape
        anchors, positives, negatives = [], [], []

        for t in range(seq_len):
            # print(t)
            pos_range_beg, pos_range_end = max(0, t - self.positve_range), min(seq_len - 1, t + self.positve_range)
            neg_range_prev_end, neg_range_after_beg = max(0, t - self.negative_range), min(seq_len - 1, t + self.negative_range)
            for _ in range(self.triplet_sample_num):
                pos_indices = np.array([x for x in range(pos_range_beg, t)] + [x for x in range(t+1, pos_range_end + 1)], dtype=np.int32)
                pos_idx = np.random.randint(0, len(pos_indices), batch_size)
                pos_idx = pos_indices[pos_idx]

                neg_idices = np.array([x for x in range(neg_range_prev_end)] + [x for x in range(neg_range_after_beg + 1, seq_len)], dtype=np.int32)
                neg_idx = np.random.randint(0 , len(neg_idices), batch_size)
                neg_idx = neg_idices[neg_idx]
                
                # get the anchor encodings
                anchor_traj = ptu.zeros((warm_len, batch_size, imlen))
                if t - warm_len + 1 >= 0:
                    anchor_traj = traj[t - warm_len + 1: t+1, :, :]
                else:
                    anchor_traj[-(t+1):] = traj[:t+1]
                    for _ in range(warm_len - t - 1):
                        anchor_traj[_] = traj[0]
                anchor_encodings = self.model.encode(anchor_traj)[0] # always assmue we use the mean as encoding

                # get the positive encodings
                pos_traj = ptu.zeros((warm_len, batch_size, imlen))
                for b_idx in range(batch_size):
                    p_idx = pos_idx[b_idx]
                    if p_idx - warm_len + 1 >= 0:
                        pos_traj[:, b_idx, :] = traj[p_idx - warm_len + 1: p_idx+1, b_idx, :]
                    else:
                        pos_traj[-(p_idx+1):, b_idx, :] = traj[:p_idx+1, b_idx, :]
                        for _ in range(warm_len - p_idx - 1):
                            pos_traj[_, b_idx, :] = traj[0, b_idx, :]
                pos_encodings = self.model.encode(pos_traj)[0]

                # get the negative encodings
                neg_traj = ptu.zeros((warm_len, batch_size, imlen))
                for b_idx in range(batch_size):
                    n_idx = neg_idx[b_idx]
                    if n_idx - warm_len + 1 >= 0:
                        neg_traj[:, b_idx, :] = traj[n_idx - warm_len + 1: n_idx+1, b_idx, :]
                    else:
                        neg_traj[-(n_idx+1):, b_idx, :] = traj[:n_idx+1, b_idx, :]
                        for _ in range(warm_len - n_idx - 1):
                            neg_traj[_, b_idx, :] = traj[0, b_idx, :]
                neg_encodings = self.model.encode(neg_traj)[0]

                anchors.append(anchor_encodings)
                positives.append(pos_encodings)
                negatives.append(neg_encodings)

        anchors = torch.cat(anchors, dim=0)
        positives = torch.cat(positives, dim=0)
        negatives = torch.cat(negatives, dim=0)

        positive_distances = (anchors - positives).pow(2).sum(dim=1)
        negative_distances = (anchors - negatives).pow(2).sum(dim=1)
        losses = F.relu(positive_distances - negative_distances + self.triplet_loss_margin, 0)
        return losses.mean()

    def triplet_loss(self, encodings):
        '''
        encodings: [seq_len, batch_size, feature_size]
        '''
        seq_len, batch_size, feature_size = encodings.shape

        anchors, positives, negatives = [], [], []

        for t in range(seq_len):
            pos_range_beg, pos_range_end = max(0, t - self.positve_range), min(seq_len - 1, t + self.positve_range)
            neg_range_prev_end, neg_range_after_beg = max(0, t - self.negative_range), min(seq_len - 1, t + self.negative_range)
            for _ in range(self.triplet_sample_num):
                pos_indices = np.array([x for x in range(pos_range_beg, t)] + [x for x in range(t+1, pos_range_end + 1)], dtype=np.int32)
                pos_idx = np.random.randint(0, len(pos_indices), batch_size)
                pos_idx = pos_indices[pos_idx]

                neg_idices = np.array([x for x in range(neg_range_prev_end)] + [x for x in range(neg_range_after_beg + 1, seq_len)], dtype=np.int32)
                neg_idx = np.random.randint(0 , len(neg_idices), batch_size)
                neg_idx = neg_idices[neg_idx]
                
                anchor_samples = encodings[t] # batch_size, feature_size
                positive_samples = encodings[pos_idx, np.arange(batch_size)]
                negative_samples = encodings[neg_idx, np.arange(batch_size)]

                anchors.append(anchor_samples)
                positives.append(positive_samples)
                negatives.append(negative_samples)

        anchors = torch.cat(anchors, dim=0)
        positives = torch.cat(positives, dim=0)
        negatives = torch.cat(negatives, dim=0)

        positive_distances = (anchors - positives).pow(2).sum(dim=1)
        negative_distances = (anchors - negatives).pow(2).sum(dim=1)
        losses = F.relu(positive_distances - negative_distances + self.triplet_loss_margin, 0)
        return losses.mean()


    def train_epoch(self, epoch, sample_batch=None, batches=25, from_rl=False, key=None, only_train_vae=False):
        self.model.train()
        losses = []
        log_probs = []
        triplet_losses = []
        kles = []
        ae_losses = []
        matching_losses = []
        vae_matching_losses = []
        contrastive_losses = []
        lstm_kles = []
        # zs = []
        beta = float(self.beta_schedule.get_value(epoch))
        for batch_idx in range(batches):
            if sample_batch is not None:
                data = sample_batch(self.batch_size, key=key)
                next_obs = data['next_obs']
            else:
                next_obs = self.get_batch(epoch=epoch)

            self.optimizer.zero_grad()

            reconstructions, obs_distribution_params, vae_latent_distribution_params, lstm_latent_encodings = self.model(next_obs)
            latent_encodings = lstm_latent_encodings
            vae_mu = vae_latent_distribution_params[0]
            latent_distribution_params = vae_latent_distribution_params
           
            triplet_loss = ptu.zeros(1)            
            for tri_idx, triplet_type in enumerate(self.triplet_loss_type):
                if triplet_type == 1 and not only_train_vae:
                    triplet_loss += self.triplet_loss_coef[tri_idx] * self.triplet_loss(latent_encodings)
                elif triplet_type == 2 and not only_train_vae:
                    triplet_loss += self.triplet_loss_coef[tri_idx] * self.triplet_loss_2(next_obs)
                elif triplet_type == 3 and not only_train_vae:
                    triplet_loss += self.triplet_loss_coef[tri_idx] * self.triplet_loss_3(next_obs)

            if self.matching_loss_coef > 0 and not only_train_vae:
                matching_loss = self.matching_loss(next_obs)
            else:
                matching_loss = ptu.zeros(1)

            if self.vae_matching_loss_coef > 0:
                matching_loss_vae = self.matching_loss_vae(next_obs)
            else:
                matching_loss_vae = ptu.zeros(1)

            if self.contrastive_loss_coef > 0 and not only_train_vae:
                contrastive_loss = self.contrastive_loss(next_obs)
            else:
                contrastive_loss = ptu.zeros(1)

            log_prob = self.model.logprob(next_obs, obs_distribution_params)
            kle = self.model.kl_divergence(latent_distribution_params)
            lstm_kle = ptu.zeros(1)

            ae_loss = F.mse_loss(latent_encodings.view((-1, self.model.representation_size)), vae_mu.detach())
            ae_losses.append(ae_loss.item())

            loss = -self.recon_loss_coef * log_prob + \
                    beta * kle + \
                    self.matching_loss_coef * matching_loss + \
                    self.ae_loss_coef * ae_loss + \
                    triplet_loss + \
                    self.vae_matching_loss_coef * matching_loss_vae + \
                    self.contrastive_loss_coef * contrastive_loss

            self.optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            log_probs.append(log_prob.item())
            kles.append(kle.item())
            lstm_kles.append(lstm_kle.item())
            triplet_losses.append(triplet_loss.item())
            matching_losses.append(matching_loss.item())
            vae_matching_losses.append(matching_loss_vae.item())
            contrastive_losses.append(contrastive_loss.item())
            self.optimizer.step()
             
            if self.log_interval and batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(next_obs)))

            # dump a batch of training images for debugging
            # if batch_idx == 0 and epoch % 25 == 0:
            #     n = min(next_obs.size(0), 8)
            #     comparison = torch.cat([
            #         next_obs[:n].narrow(start=0, length=self.imlength, dim=1)
            #             .contiguous().view(
            #             -1, self.input_channels, self.imsize, self.imsize
            #         ).transpose(2, 3),
            #         reconstructions.view(
            #             self.batch_size,
            #             self.input_channels,
            #             self.imsize,
            #             self.imsize,
            #         )[:n].transpose(2, 3)
            #     ])
            #     save_dir = osp.join(logger.get_snapshot_dir(),
            #                         'vae_train_{}_{}.png'.format(key, epoch))
            #     save_image(comparison.data.cpu(), save_dir, nrow=n)

        # if not from_rl:
        #     zs = np.array(zs)
        #     self.model.dist_mu = zs.mean(axis=0)
        #     self.model.dist_std = zs.std(axis=0)

        self.eval_statistics['train/log prob'] = np.mean(log_probs)
        self.eval_statistics['train/triplet loss'] = np.mean(triplet_losses)
        self.eval_statistics['train/matching loss'] = np.mean(matching_losses)
        self.eval_statistics['train/vae matching loss'] = np.mean(vae_matching_losses)
        self.eval_statistics['train/KL'] = np.mean(kles)
        self.eval_statistics['train/lstm KL'] = np.mean(lstm_kles)
        self.eval_statistics['train/loss'] = np.mean(losses)
        self.eval_statistics['train/contrastive loss'] = np.mean(contrastive_losses)
        self.eval_statistics['train/ae loss'] = np.mean(ae_losses)

        torch.cuda.empty_cache()

    def get_diagnostics(self):
        return self.eval_statistics

    def test_epoch(
            self,
            epoch,
            sample_batch=None,
            key=None,
            save_reconstruction=True,
            save_vae=True,
            from_rl=False,
            save_prefix='r',
            only_train_vae=False,
    ):
        self.model.eval()
        losses = []
        log_probs = []
        triplet_losses = []
        matching_losses = []
        vae_matching_losses = []
        kles = []
        lstm_kles = []
        ae_losses = []
        contrastive_losses = []
        beta = float(self.beta_schedule.get_value(epoch))
        for batch_idx in range(10):
            # print(batch_idx)
            if sample_batch is not None:
                data = sample_batch(self.batch_size, key=key)
                next_obs = data['next_obs']
            else:
                next_obs = self.get_batch(epoch=epoch)

            reconstructions, obs_distribution_params, vae_latent_distribution_params, lstm_latent_encodings = self.model(next_obs)
            latent_encodings = lstm_latent_encodings
            vae_mu = vae_latent_distribution_params[0] # this is lstm inputs
            latent_distribution_params = vae_latent_distribution_params

            triplet_loss = ptu.zeros(1)            
            for tri_idx, triplet_type in enumerate(self.triplet_loss_type):
                if triplet_type == 1 and not only_train_vae:
                    triplet_loss += self.triplet_loss_coef[tri_idx] * self.triplet_loss(latent_encodings)
                elif triplet_type == 2 and not only_train_vae:
                    triplet_loss += self.triplet_loss_coef[tri_idx] * self.triplet_loss_2(next_obs)
                elif triplet_type == 3 and not only_train_vae:
                    triplet_loss += self.triplet_loss_coef[tri_idx] * self.triplet_loss_3(next_obs)

            if self.matching_loss_coef > 0 and not only_train_vae:
                matching_loss = self.matching_loss(next_obs)
            else:
                matching_loss = ptu.zeros(1)

            if self.vae_matching_loss_coef > 0:
                matching_loss_vae = self.matching_loss_vae(next_obs)
            else:
                matching_loss_vae = ptu.zeros(1)

            if self.contrastive_loss_coef > 0 and not only_train_vae:
                contrastive_loss = self.contrastive_loss(next_obs)
            else:
                contrastive_loss = ptu.zeros(1)

            log_prob = self.model.logprob(next_obs, obs_distribution_params)
            kle = self.model.kl_divergence(latent_distribution_params)
            lstm_kle = ptu.zeros(1)
            
            ae_loss = F.mse_loss(latent_encodings.view((-1, self.model.representation_size)), vae_mu.detach())
            ae_losses.append(ae_loss.item())

            loss = -self.recon_loss_coef * log_prob + beta * kle + \
                        self.matching_loss_coef * matching_loss + self.ae_loss_coef * ae_loss + triplet_loss + \
                            self.vae_matching_loss_coef * matching_loss_vae + self.contrastive_loss_coef * contrastive_loss

            losses.append(loss.item())
            log_probs.append(log_prob.item())
            triplet_losses.append(triplet_loss.item())
            matching_losses.append(matching_loss.item())
            vae_matching_losses.append(matching_loss_vae.item())
            kles.append(kle.item())
            lstm_kles.append(lstm_kle.item())
            contrastive_losses.append(contrastive_loss.item())

            if batch_idx == 0 and save_reconstruction:
                seq_len, batch_size, feature_size = next_obs.shape
                show_obs = next_obs[0][:8]
                reconstructions = reconstructions.view((seq_len, batch_size, feature_size))[0][:8]
                comparison = torch.cat([
                    show_obs.narrow(start=0, length=self.imlength, dim=1)
                        .contiguous().view(
                        -1, self.input_channels, self.imsize, self.imsize
                    ).transpose(2, 3),
                    reconstructions.view(
                        -1,
                        self.input_channels,
                        self.imsize,
                        self.imsize,
                    ).transpose(2, 3)
                ])
                save_dir = osp.join(logger.get_snapshot_dir(),
                                    '{}{}.png'.format(save_prefix, epoch))
                save_image(comparison.data.cpu(), save_dir, nrow=8)

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['test/log prob'] = np.mean(log_probs)
        self.eval_statistics['test/triplet loss'] = np.mean(triplet_losses)
        self.eval_statistics['test/vae matching loss'] = np.mean(vae_matching_losses)
        self.eval_statistics['test/matching loss'] = np.mean(matching_losses)
        self.eval_statistics['test/KL'] = np.mean(kles)
        self.eval_statistics['test/lstm KL'] = np.mean(lstm_kles)
        self.eval_statistics['test/loss'] = np.mean(losses)
        self.eval_statistics['test/contrastive loss'] = np.mean(contrastive_losses)
        self.eval_statistics['beta'] = beta
        self.eval_statistics['test/ae loss'] = np.mean(ae_losses)

        if not from_rl:
            for k, v in self.eval_statistics.items():
                logger.record_tabular(k, v)
            logger.dump_tabular()
            if save_vae:
                logger.save_itr_params(epoch, self.model)

        torch.cuda.empty_cache()

    def debug_statistics(self):
        """
        Given an image $$x$$, samples a bunch of latents from the prior
        $$z_i$$ and decode them $$\hat x_i$$.
        Compare this to $$\hat x$$, the reconstruction of $$x$$.
        Ideally
         - All the $$\hat x_i$$s do worse than $$\hat x$$ (makes sure VAE
           isn’t ignoring the latent)
         - Some $$\hat x_i$$ do better than other $$\hat x_i$$ (tests for
           coverage)
        """
        debug_batch_size = 64
        data = self.get_batch(train=False)
        reconstructions, _, _ = self.model(data)
        img = data[0]
        recon_mse = ((reconstructions[0] - img) ** 2).mean().view(-1)
        img_repeated = img.expand((debug_batch_size, img.shape[0]))

        samples = ptu.randn(debug_batch_size, self.representation_size)
        random_imgs, _ = self.model.decode(samples)
        random_mses = (random_imgs - img_repeated) ** 2
        mse_improvement = ptu.get_numpy(random_mses.mean(dim=1) - recon_mse)
        stats = create_stats_ordered_dict(
            'debug/MSE improvement over random',
            mse_improvement,
        )
        stats.update(create_stats_ordered_dict(
            'debug/MSE of random decoding',
            ptu.get_numpy(random_mses),
        ))
        stats['debug/MSE of reconstruction'] = ptu.get_numpy(
            recon_mse
        )[0]
        if self.skew_dataset:
            stats.update(create_stats_ordered_dict(
                'train weight',
                self._train_weights
            ))
        return stats

    def dump_samples(self, epoch, save_prefix='s'):
        self.model.eval()
        sample = ptu.randn(64, self.representation_size)
        sample = self.model.decode(sample)[0].cpu()
        save_dir = osp.join(logger.get_snapshot_dir(), '{}{}.png'.format(save_prefix, epoch))
        save_image(
            sample.data.view(64, self.input_channels, self.imsize, self.imsize).transpose(2, 3),
            save_dir
        )

    def _dump_imgs_and_reconstructions(self, idxs, filename):
        imgs = []
        recons = []
        for i in idxs:
            img_np = self.train_dataset[i]
            img_torch = ptu.from_numpy(normalize_image(img_np))
            recon, *_ = self.model(img_torch.view(1, -1))

            img = img_torch.view(self.input_channels, self.imsize, self.imsize).transpose(1, 2)
            rimg = recon.view(self.input_channels, self.imsize, self.imsize).transpose(1, 2)
            imgs.append(img)
            recons.append(rimg)
        all_imgs = torch.stack(imgs + recons)
        save_file = osp.join(logger.get_snapshot_dir(), filename)
        save_image(
            all_imgs.data,
            save_file,
            nrow=len(idxs),
        )

    def log_loss_under_uniform(self, model, data, priority_function_kwargs):
        import torch.nn.functional as F
        log_probs_prior = []
        log_probs_biased = []
        log_probs_importance = []
        kles = []
        mses = []
        for i in range(0, data.shape[0], self.batch_size):
            img = normalize_image(data[i:min(data.shape[0], i + self.batch_size), :])
            torch_img = ptu.from_numpy(img)
            reconstructions, obs_distribution_params, latent_distribution_params = self.model(torch_img)

            priority_function_kwargs['sampling_method'] = 'true_prior_sampling'
            log_p, log_q, log_d = compute_log_p_log_q_log_d(model, img, **priority_function_kwargs)
            log_prob_prior = log_d.mean()

            priority_function_kwargs['sampling_method'] = 'biased_sampling'
            log_p, log_q, log_d = compute_log_p_log_q_log_d(model, img, **priority_function_kwargs)
            log_prob_biased = log_d.mean()

            priority_function_kwargs['sampling_method'] = 'importance_sampling'
            log_p, log_q, log_d = compute_log_p_log_q_log_d(model, img, **priority_function_kwargs)
            log_prob_importance = (log_p - log_q + log_d).mean()

            kle = model.kl_divergence(latent_distribution_params)
            mse = F.mse_loss(torch_img, reconstructions, reduction='elementwise_mean')
            mses.append(mse.item())
            kles.append(kle.item())
            log_probs_prior.append(log_prob_prior.item())
            log_probs_biased.append(log_prob_biased.item())
            log_probs_importance.append(log_prob_importance.item())

        logger.record_tabular("Uniform Data Log Prob (True Prior)", np.mean(log_probs_prior))
        logger.record_tabular("Uniform Data Log Prob (Biased)", np.mean(log_probs_biased))
        logger.record_tabular("Uniform Data Log Prob (Importance)", np.mean(log_probs_importance))
        logger.record_tabular("Uniform Data KL", np.mean(kles))
        logger.record_tabular("Uniform Data MSE", np.mean(mses))

    def dump_uniform_imgs_and_reconstructions(self, dataset, epoch):
        idxs = np.random.choice(range(dataset.shape[0]), 4)
        filename = 'uniform{}.png'.format(epoch)
        imgs = []
        recons = []
        for i in idxs:
            img_np = dataset[i]
            img_torch = ptu.from_numpy(normalize_image(img_np))
            recon, *_ = self.model(img_torch.view(1, -1))

            img = img_torch.view(self.input_channels, self.imsize, self.imsize).transpose(1, 2)
            rimg = recon.view(self.input_channels, self.imsize, self.imsize).transpose(1, 2)
            imgs.append(img)
            recons.append(rimg)
        all_imgs = torch.stack(imgs + recons)
        save_file = osp.join(logger.get_snapshot_dir(), filename)
        save_image(
            all_imgs.data,
            save_file,
            nrow=4,
        )
