import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
import numpy as np
from rlkit.torch.conv_networks import CNN, DCNN
from rlkit.torch.vae.vae_base import GaussianLatentVAE

imsize48_default_architecture = dict(
    conv_args=dict( # conv layers
        kernel_sizes=[5, 3, 3], 
        n_channels=[16, 32, 64],
        strides=[3, 2, 2],
        output_size=6,
    ),
    conv_kwargs=dict(
        hidden_sizes=[], # linear layers after conv
        batch_norm_conv=False,
        batch_norm_fc=False,
    ),

    LSTM_args=dict(
        input_size=6,
        hidden_size=128,
    ),
    LSTM_kwargs=dict(
        num_layers=2,
    ),

    deconv_args=dict(
        hidden_sizes=[],

        deconv_input_width=3,
        deconv_input_height=3,
        deconv_input_channels=64,

        deconv_output_kernel_size=6,
        deconv_output_strides=3,
        deconv_output_channels=3,

        kernel_sizes=[3, 3],
        n_channels=[32, 16],
        strides=[2, 2],
    ),
    deconv_kwargs=dict(
        batch_norm_deconv=False,
        batch_norm_fc=False,
    )
)


class ConvLSTM2(nn.Module):
    def __init__(
        self,
        representation_size,
        architecture,

        encoder_class=CNN,
        decoder_class=DCNN,
        decoder_output_activation=identity,
        decoder_distribution='gaussian_identity_variance',

        input_channels=3,
        imsize=48,
        init_w=1e-3,
        min_variance=1e-3,
        hidden_init=ptu.fanin_init,
        detach_vae_output=True,
    ):
        super(ConvLSTM2, self).__init__() 

        self.representation_size = representation_size
        # record the empirical statistics of latents, when not sample from true prior, sample from them. 
        self.dist_mu = np.zeros(self.representation_size) 
        self.dist_std = np.ones(self.representation_size)

        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize * self.imsize * self.input_channels
        self.detach_vae_output = detach_vae_output

        conv_args, conv_kwargs, deconv_args, deconv_kwargs = \
            architecture['conv_args'], architecture['conv_kwargs'], \
            architecture['deconv_args'], architecture['deconv_kwargs']

        self.encoder = encoder_class(
            **conv_args,
            paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
            input_height=self.imsize,
            input_width=self.imsize,
            input_channels=self.input_channels,
            init_w=init_w,
            hidden_init=hidden_init,
            **conv_kwargs)

        self.lstm_args, self.lstm_kwargs = architecture['LSTM_args'], architecture['LSTM_kwargs']
        self.lstm = nn.LSTM(**self.lstm_args, **self.lstm_kwargs)
        self.lstm_num_layers = self.lstm_kwargs['num_layers']
        self.lstm_hidden_size = self.lstm_args['hidden_size']

        assert representation_size == self.lstm_args['input_size'], "lstm input is vae latent, \
            so lstm input size should be equal to representation_size!"

        self.vae_fc1 = nn.Linear(conv_args['output_size'], representation_size)
        self.vae_fc2 = nn.Linear(conv_args['output_size'], representation_size)

        self.vae_fc1.weight.data.uniform_(-init_w, init_w)
        self.vae_fc1.bias.data.uniform_(-init_w, init_w)

        self.vae_fc2.weight.data.uniform_(-init_w, init_w)
        self.vae_fc2.bias.data.uniform_(-init_w, init_w)

        self.lstm_fc = nn.Linear(self.lstm_hidden_size, representation_size)
        self.lstm_fc.weight.data.uniform_(-init_w, init_w)
        self.lstm_fc.bias.data.uniform_(-init_w, init_w)

        self.decoder = decoder_class(
            **deconv_args,
            fc_input_size=representation_size,
            init_w=init_w,
            output_activation=decoder_output_activation,
            paddings=np.zeros(len(deconv_args['kernel_sizes']), dtype=np.int64),
            hidden_init=hidden_init,
            **deconv_kwargs)
        self.decoder_distribution = decoder_distribution

    def from_vae_latents_to_lstm_latents(self, latents, lstm_hidden=None):
        batch_size, feature_size = latents.shape
        # print(latents.shape)
        lstm_input = latents
        lstm_input = lstm_input.view((1, batch_size, -1))

        if lstm_hidden is None:
            lstm_hidden = (ptu.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size), \
                     ptu.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size))
        
        h, hidden = self.lstm(lstm_input, lstm_hidden) # [seq_len, batch_size, lstm_hidden_size]

        lstm_latent = self.lstm_fc(h)
        lstm_latent = lstm_latent.view((batch_size, -1))
        return lstm_latent

    def encode(self, input, lstm_hidden=None, return_hidden=False, return_vae_latent=False): 
        '''
        input: [seq_len x batch x flatten_img_dim] of flattened images
        lstm_hidden: [lstm_layers x batch x lstm_hidden_size] 
        mark: change depends on how latent distribution parameters are used
        '''
        seq_len, batch_size, feature_size  = input.shape
        # print("in lstm encode: ", seq_len, batch_size, feature_size)
        input = input.reshape((-1, feature_size))
        feature = self.encoder(input) # [seq_len x batch x conv_output_size]

        vae_mu = self.vae_fc1(feature)
        if self.log_min_variance is None:
            vae_logvar = self.vae_fc2(feature)
        else:
            vae_logvar = self.log_min_variance + torch.abs(self.vae_fc2(feature))

        # lstm_input = self.rsample((vae_mu, vae_logvar))
        # if self.detach_vae_output:
        #     lstm_input = lstm_input.detach()
        if self.detach_vae_output:
            lstm_input = vae_mu.detach().clone()
        else:
            lstm_input = vae_mu
        lstm_input = lstm_input.view((seq_len, batch_size, -1))
        # if self.detach_vae_output:
        #     lstm_input = lstm_input.detach()

        if lstm_hidden is None:
            lstm_hidden = (ptu.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size), \
                     ptu.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size))
        
        h, hidden = self.lstm(lstm_input, lstm_hidden) # [seq_len, batch_size, lstm_hidden_size]

        lstm_latent = self.lstm_fc(h)

        ret = (lstm_latent, ptu.ones_like(lstm_latent))
        if return_vae_latent:
            ret += (vae_mu, vae_logvar)

        if return_hidden:
            return ret, hidden
        return ret #, lstm_input # [seq_len, batch_size, representation_size]
        
    def forward(self, input, lstm_hidden=None, return_hidden=False):
        """
        :param input:
        :return: reconstructed input, obs_distribution_params, latent_distribution_params
        mark: change to return the feature latents and the lstm latents
        """
        if return_hidden:
            latent_distribution_params, hidden = self.encode(input, lstm_hidden, return_hidden=True, return_vae_latent=True) # seq_len, batch_size, representation_size
        else:
            latent_distribution_params = self.encode(input, lstm_hidden, return_hidden=False, return_vae_latent=True)
        
        vae_latent_distribution_params = latent_distribution_params[2:]
        lstm_latent_encodings = latent_distribution_params[0]
        vae_latents = self.reparameterize(vae_latent_distribution_params) 
        reconstructions, obs_distribution_params = self.decode(vae_latents) # [seq_len * batch_size, representation_size]
        if return_hidden:
            return reconstructions, obs_distribution_params, vae_latent_distribution_params, lstm_latent_encodings, hidden
        return reconstructions, obs_distribution_params, vae_latent_distribution_params, lstm_latent_encodings

    def reparameterize(self, latent_distribution_params):
        if self.training:
            return self.rsample(latent_distribution_params)
        else:
            return latent_distribution_params[0]

    def kl_divergence(self, latent_distribution_params):
        mu, logvar = latent_distribution_params
        mu = mu.view((-1, self.representation_size)) # fold the possible seq_len dim
        logvar = logvar.view((-1, self.representation_size))
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def get_encoding_from_latent_distribution_params(self, latent_distribution_params):
        return latent_distribution_params[0].cpu()

    def rsample(self, latent_distribution_params):
        mu, logvar = latent_distribution_params
        stds = (0.5 * logvar).exp()
        epsilon = ptu.randn(*mu.size())
        latents = epsilon * stds + mu
        return latents

    def decode(self, latents):
        decoded = self.decoder(latents).view(-1,
                                             self.imsize * self.imsize * self.input_channels)
        if self.decoder_distribution == 'bernoulli':
            return decoded, [decoded]
        elif self.decoder_distribution == 'gaussian_identity_variance':
            return torch.clamp(decoded, 0, 1), [torch.clamp(decoded, 0, 1),
                                                torch.ones_like(decoded)]
        else:
            raise NotImplementedError('Distribution {} not supported'.format(
                self.decoder_distribution))

    def logprob(self, inputs, obs_distribution_params):
        seq_len, batch_size, feature_size = inputs.shape
        inputs = inputs.view((-1, feature_size))
        if self.decoder_distribution == 'bernoulli':
            inputs = inputs.narrow(start=0, length=self.imlength,
                                   dim=1).contiguous().view(-1, self.imlength)
            # obs_distribution_params[0] = obs_distribution_params[0].view((-1, feature_size))
            log_prob = - F.binary_cross_entropy(
                obs_distribution_params[0],
                inputs,
                reduction='elementwise_mean'
            ) * self.imlength
            return log_prob
        if self.decoder_distribution == 'gaussian_identity_variance':
            # obs_distribution_params[0] = obs_distribution_params[0].view((-1, feature_size))
            inputs = inputs.narrow(start=0, length=self.imlength,
                                   dim=1).contiguous().view(-1, self.imlength)
            log_prob = -1 * F.mse_loss(inputs, obs_distribution_params[0],
                                       reduction='elementwise_mean')
            return log_prob
        else:
            raise NotImplementedError('Distribution {} not supported'.format(
                self.decoder_distribution))

    def init_hidden(self, batch_size=1):
        lstm_hidden = (ptu.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size), \
                     ptu.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size))
        return lstm_hidden