import numpy as np
import flax
import jax
from jax import numpy as jnp
from functools import partial
from jax import random, jit
from model import GenerativeModel
from networks import VAE
from copy import deepcopy
from jax.scipy.stats import multivariate_normal
import itertools
# from ipdb import set_trace as db


class VAEModel(GenerativeModel):
    def __init__(self,
                 dirname,
                 num_batches,
                 num_epochs,
                 batch_size,
                 learning_rate,
                 layer_sizes,
                 encoder_layer_sizes,
                 state_dict,
                 data_fn,
                 epsilon,
                 tqdm,
                 dataset,
                 latent_dimension,
                 tunable_decoder_var = False,
                 warm_start = False,
                 dataset_name = None,
                 latent_off_dimension = 0):

        super().__init__(
                dirname=dirname,
                num_batches=num_batches,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                latent_distribution='gaussian',
                latent_dimension=latent_dimension,
                dataset=dataset,
                state_dict=state_dict,
                data_fn=data_fn,
                tqdm=tqdm,
                )
        self.epsilon = epsilon
        self.current_epsilon = epsilon
        self.latent_dimension = latent_dimension
        data_size = np.product(dataset.shape)

        encoder_layer_sizes = ([int(size) for size in encoder_layer_sizes.split('|')] if encoder_layer_sizes != "" else []) + [latent_dimension]
        decoder_layer_sizes = ([int(size) for size in layer_sizes.split('|')] if layer_sizes != "" else []) + [data_size]

        vae_key, self.key = random.split(self.key)
        vae_module = VAE.partial(epsilon = epsilon, encoder_layer_sizes=encoder_layer_sizes, \
                                 decoder_layer_sizes=decoder_layer_sizes, tunable_decoder_var=tunable_decoder_var,
                                 dataset_name = dataset_name)
        _, initial_params = vae_module.init_by_shape(vae_key, [(data_size,), (latent_dimension,), (data_size,)])
        initial_params['epsilon'] = initial_params['epsilon'] * self.epsilon
        if warm_start:
            if dataset_name == "sigmoid":
                assert self.latent_dimension == dataset.dimension
                eye_matrix = jnp.eye(self.latent_dimension)
                zeros = jnp.zeros((self.latent_dimension))

                decoder = jax.ops.index_update(eye_matrix, jax.ops.index[dataset.dim+1:, dataset.dim+1:], 0.)
                encoder_var = jax.ops.index_update(zeros, jax.ops.index[dataset.dim+1:], -3.)
                encoder = jax.ops.index_update(eye_matrix, jax.ops.index[dataset.dim+1:, dataset.dim+1:], 0.)

                decoder_perturb = random.normal(self.key, (self.latent_dimension, dataset.dimension)) * 0.1
                initial_params['Decoder']['FC0']['kernel'] = decoder + decoder_perturb
                sig_decoder = jnp.zeros((self.latent_dimension, dataset.dimension))
                sig_decoder_perturb = random.normal(self.key, (self.latent_dimension, dataset.dimension)) * 0.1
                initial_params['SigDecoder']['FC0']['kernel'] = sig_decoder + sig_decoder_perturb
                encoder_var_perturb = random.normal(self.key, (self.latent_dimension,)) * 0.1
                initial_params['epsilon_p'] = encoder_var + encoder_var_perturb
                encoder_perturb = random.normal(self.key, (dataset.dimension, self.latent_dimension)) * 0.1
                initial_params['Encoder']['FC0']['kernel'] = encoder + encoder_perturb

            if dataset_name == "linear_gaussian":
                assert dataset.dim + latent_off_dimension < self.latent_dimension
                extra_dim = random.normal(self.key, (dataset.dim, latent_off_dimension))
                zero_dim = jnp.zeros((dataset.dim, self.latent_dimension - dataset.dim - latent_off_dimension))
                zero_padding_dim = jnp.zeros((data_size - dataset.dim, self.latent_dimension))
                dec_const = jnp.concatenate((deepcopy(dataset.A), extra_dim, zero_dim), axis=1)
                dec_const = jnp.concatenate((dec_const, zero_padding_dim), axis=0)
                dec_const = dec_const + random.normal(self.key, (dataset.dimension, self.latent_dimension)) * 0.01
                print (dec_const)
                initial_params['Decoder']['FC0']['kernel'] = dec_const.T

                enc_const = jnp.linalg.pinv(deepcopy(dataset.A))
                zero_dim = jnp.zeros((self.latent_dimension - dataset.dim, dataset.dim))
                zero_padding_dim = jnp.zeros((self.latent_dimension, data_size - dataset.dim))
                enc_const = jnp.concatenate((enc_const, zero_dim), axis=0)
                enc_const = jnp.concatenate((enc_const, zero_padding_dim), axis=1)
                enc_const = enc_const + random.normal(self.key, (self.latent_dimension, dataset.dimension)) * 0.01
                print (dec_const)
                initial_params["Encoder"]['FC0']['kernel'] = enc_const.T

                zeros = jnp.zeros((self.latent_dimension))
                latent_dim_tild = dataset.A.shape[1]
                encoder_var = jax.ops.index_update(zeros, jax.ops.index[:latent_dim_tild+latent_off_dimension], -3)
                encoder_var_perturb = random.normal(self.key, (self.latent_dimension,)) * 0.1
                initial_params['epsilon_p'] = encoder_var + encoder_var_perturb
                print (initial_params['epsilon_p'])




        self.model = flax.nn.Model(vae_module, initial_params)
        self.optimizer = flax.optim.Adam(learning_rate=self.learning_rate).create(self.model)
        self.vae_losses = []
        self.var_enc = []
        self.var_dec = []
        self.gt_eigen = []
        self.ht_eigen = []
        self.params_and_gradients = []
        self.correlation_ratios = []


    def train_one_batch(self, batch):
        batch = batch.reshape((batch.shape[0], -1))
        latent_batch_key, self.key = random.split(self.key)
        z = self.sample_latent(latent_batch_key, self.batch_size)
        z1 = z[..., :self.latent_dimension]
        z2 = z[..., self.latent_dimension:]
        self.optimizer, self.model, vae_loss = VAE.train_step(self.optimizer, batch, z1, z2, self.epsilon)
        self.vae_losses.append(vae_loss)

    def compute_model_stats(self, real_batch, fake_batch, latents):
        z1 = latents[..., :self.latent_dimension]
        z2 = latents[..., self.latent_dimension:]
        vae_loss, dkl, mse, logvar_e, epsilon = VAE.loss(self.model, real_batch, z1, z2, self.epsilon)
        self.vae_losses.append(vae_loss)
        self.var_enc.append(logvar_e)
        self.var_dec.append(epsilon)
        self.current_epsilon = epsilon
        data = {"VAE Loss": vae_loss, "KL divergence": dkl, "mse": mse} # "Decoder Variance": epslon, "Encoder Varian}
        return data

    def compute_correlation_ratio(self, params, dloss_dparams):
        inner_product = 0.
        squared_norm = 0.
        opt_model = self.model
        # this is all a hack, could do with jax pytree operations probably
        # need to do this for each parameter of the linear VAE (very annoying)
        opt_dec_bias = opt_model.params['Decoder']['FC0']['bias']
        dec_bias_displacement = opt_dec_bias - params['Decoder']['FC0']['bias']
        inner_product += -dloss_dparams.params['Decoder']['FC0']['bias'].T @ dec_bias_displacement
        squared_norm += jnp.linalg.norm(dec_bias_displacement) ** 2

        opt_dec_kernel = opt_model.params['Decoder']['FC0']['kernel']
        dec_kernel_displacement = opt_dec_kernel - params['Decoder']['FC0']['kernel']
        inner_product += -dloss_dparams.params['Decoder']['FC0']['kernel'].flatten().T @ dec_kernel_displacement.flatten()
        squared_norm += jnp.linalg.norm(dec_kernel_displacement.flatten()) ** 2

        opt_enc_bias = opt_model.params['Encoder']['FC0']['bias']
        enc_bias_displacement = opt_enc_bias - params['Encoder']['FC0']['bias']
        inner_product += -dloss_dparams.params['Encoder']['FC0']['bias'].T @ enc_bias_displacement
        squared_norm += jnp.linalg.norm(enc_bias_displacement) ** 2

        opt_enc_kernel = opt_model.params['Encoder']['FC0']['kernel']
        enc_kernel_displacement = opt_enc_kernel - params['Encoder']['FC0']['kernel']
        inner_product += -dloss_dparams.params['Encoder']['FC0']['kernel'].flatten().T @ enc_kernel_displacement.flatten()
        squared_norm += jnp.linalg.norm(enc_kernel_displacement.flatten()) ** 2

        gt_epsilon = opt_model.params['epsilon']
        eps_displacement = gt_epsilon - params['epsilon']
        inner_product += dloss_dparams.params['epsilon'] * eps_displacement
        squared_norm += eps_displacement ** 2

        gt_eps_p = opt_model.params['epsilon_p']
        epsilon_p_displacement = gt_eps_p - params['epsilon_p']
        inner_product += -dloss_dparams.params['epsilon_p'] @ epsilon_p_displacement
        squared_norm += jnp.linalg.norm(epsilon_p_displacement) ** 2
        ratio = inner_product / squared_norm
        return ratio

    """
    def sample_batch(self, key, batch_size, latents=None):
        if latents is not None:
            z = latents
        else:
            z = self.sample_latent(key, batch_size)
        sampling_model = jax.jit(partial(self.model, sampling=True))
        return sampling_model(None, z)[0], z
    """

    def sample_batch(self, key, batch_size, latents=None):

        if latents is not None:
            z = latents
        else:
            z = self.sample_latent(key, batch_size)
        z1 = z[..., :self.latent_dimension]
        z2 = z[..., self.latent_dimension:]
        sampling_model = jax.jit(partial(self.model, sampling=True, epsilon=self.current_epsilon))
        x_hat, mu, logvar_e, epsilon = sampling_model(None, z1, z2)
        return x_hat, z

    def model_save_data(self, final=False):
        data = {"VAE Loss": self.vae_losses, "Decoder Variance": self.var_dec, "Encoder Variance": self.var_enc,
                "EigenValues": (self.ht_eigen, self.gt_eigen)}
        if final:
            self.correlation_ratios = [self.compute_correlation_ratio(params, gradients) for params, gradients in self.params_and_gradients]
            data['Correlation Ratio'] = self.correlation_ratios
        return data
