import numpy as np
import flax
import jax
from jax import numpy as jnp
from functools import partial
from jax import random, jit
from model import GenerativeModel
from networks import VAE, ACVAE, gammaVAE, MLP_ACVAE, SqVAE
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
        if self.warm_start_linear:
            def pure_loss(model):
                return VAE.loss(model, real_batch, z1, z2, self.epsilon)[0]
            dloss_dparams = jax.grad(pure_loss)(self.model)
            self.params_and_gradients.append((deepcopy(self.model.params), dloss_dparams))
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

class gammaVAEModel(GenerativeModel):
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
                 tqdm,
                 dataset,
                 latent_dimension):
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
        data_size = np.product(dataset.shape)
        encoder_layer_sizes = [int(size) for size in encoder_layer_sizes.split('|')] + [latent_dimension * 2]
        decoder_layer_sizes = [int(size) for size in layer_sizes.split('|')] + [data_size]
        vae_key, self.key = random.split(self.key)
        vae_module = gammaVAE.partial(encoder_layer_sizes=encoder_layer_sizes, decoder_layer_sizes=decoder_layer_sizes)
        _, initial_params = vae_module.init_by_shape(vae_key, [(data_size,), (latent_dimension,)])
        self.model = flax.nn.Model(vae_module, initial_params)
        self.optimizer = flax.optim.Adam(learning_rate=self.learning_rate).create(self.model)
        self.vae_losses = []

    def train_one_batch(self, batch):
        batch = batch.reshape((batch.shape[0], -1))
        latent_batch_key, self.key = random.split(self.key)
        z = self.sample_latent(latent_batch_key, self.batch_size)
        self.optimizer, self.model, vae_loss = gammaVAE.train_step(self.optimizer, batch, z)
        x_hat, loggamma, mu, logvar = self.model(batch, z)
        #print (loggamma)
        self.vae_losses.append(vae_loss)

    def compute_model_stats(self, real_batch, fake_batch, latents):
        vae_loss = gammaVAE.loss(self.model, real_batch, latents)
        return {"VAE Loss": vae_loss}

    def sample_batch(self, key, batch_size, latents=None):
        if latents is not None:
            z = latents
        else:
            z = self.sample_latent(key, batch_size)
        sampling_model = jax.jit(partial(self.model, sampling=True))
        return sampling_model(None, z)[0], z

    def model_save_data(self):
        return {"VAE Loss": self.vae_losses}

class ACVAEModel(GenerativeModel):
    def __init__(self,
                 dirname,
                 num_batches,
                 num_epochs,
                 batch_size,
                 learning_rate,
                 layer_sizes,
                 n_passes,
                 state_dict,
                 data_fn,
                 tqdm,
                 epsilon,
                 dataset,
                 copy_flow_dataset,
                 initialize_inverse,
                 use_fred_covariance,
                 ):
        latent_dimension = data_size = dataset.dimension
        super().__init__(
                dirname=dirname,
                num_batches=num_batches,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                latent_distribution='gaussian',
                latent_dimension=data_size * 2,
                dataset=dataset,
                state_dict=state_dict,
                data_fn=data_fn,
                tqdm=tqdm,
                )
        self.layer_sizes = [int(size) for size in layer_sizes.split('|')] + [data_size]
        self.n_passes = n_passes
        self.epsilon = epsilon
        vae_key, self.key = random.split(self.key)
        vae_module = ACVAE.partial(layer_sizes=self.layer_sizes, n_passes=self.n_passes, epsilon=self.epsilon,
                                   use_fred_covariance=use_fred_covariance)
        _, initial_params = vae_module.init_by_shape(vae_key, [(1, data_size), (1, latent_dimension), (1, latent_dimension)])

        if copy_flow_dataset:
            initial_params['Decoder'] = deepcopy(dataset.params)
        if initialize_inverse:
            initial_params['Encoder'] = deepcopy(initial_params['Decoder'])

        self.model = flax.nn.Model(vae_module, initial_params)
        self.optimizer = flax.optim.Adam(learning_rate=self.learning_rate).create(self.model)
        self.use_fred_covariance = use_fred_covariance

        # optimizer
        deco = flax.optim.ModelParamTraversal(lambda path, _: 'Decoder' in path)
        enco = flax.optim.ModelParamTraversal(lambda path, _: 'Encoder' in path)
        var = flax.optim.ModelParamTraversal(lambda path, _: 'epsilon_p' in path)
        if copy_flow_dataset:
            deco_opt = flax.optim.Adam(learning_rate=0)
        else:
            deco_opt = flax.optim.Adam(learning_rate=self.learning_rate)
        enco_opt = flax.optim.Adam(learning_rate=self.learning_rate)
        var_opt = flax.optim.Adam(learning_rate=self.learning_rate)
        opt_def = flax.optim.MultiOptimizer((deco, deco_opt), (enco, enco_opt), (var, var_opt))
        self.optimizer = opt_def.create(self.model)

        self.vae_losses = []
        self.likelihoods = []
        self.naive_likelihoods = []
        self.reconstruction_mses = []
        self.norm_recon_mses = []

    def train_one_batch(self, batch):
        batch = batch.reshape((batch.shape[0], -1))
        latent_batch_key, self.key = random.split(self.key)
        z = self.sample_latent(latent_batch_key, self.batch_size)
        z1 = z[..., :self.dataset.dimension]
        z2 = z[..., self.dataset.dimension:]
        self.optimizer, self.model, vae_loss = jit(ACVAE.train_step, static_argnums=[7])(self.optimizer, batch, z1, z2, self.batchnum, self.epsilon, self.learning_rate, self.use_fred_covariance)
        self.vae_losses.append(vae_loss)

    def compute_model_stats(self, real_batch, fake_batch, latents):
        z1 = latents[..., :self.dataset.dimension]
        z2 = latents[..., self.dataset.dimension:]
        vae_loss = ACVAE.loss(self.model, real_batch, z1, z2, epsilon=self.epsilon,
                                        use_fred_covariance=self.use_fred_covariance)
        key = self.get_key()
        likelihood = jit(self.model.average_log_likelihood)(real_batch)
        self.likelihoods.append(likelihood)
        naive_flow_likelihood = self.model.naive_flow_log_likelihood(real_batch)
        self.naive_likelihoods.append(naive_flow_likelihood)
        noise = jnp.zeros_like(real_batch)
        reconstructed_real_batch = self.model(real_batch, noise, noise, initialize=False)[0]
        reconstruction_mse = jnp.sum(jnp.square(real_batch - reconstructed_real_batch), axis=-1).mean()
        norm_recon_mse = jnp.mean(jnp.sum(jnp.square(real_batch - reconstructed_real_batch), axis=-1) / jnp.sum(jnp.square(real_batch), axis=-1))
        self.reconstruction_mses.append(reconstruction_mse)
        self.norm_recon_mses.append(norm_recon_mse)
        return {"VAE Loss": vae_loss, "Decoder Likelihood": likelihood, "Naive Flow Likelihood": naive_flow_likelihood,
                "Reconstruction MSE": reconstruction_mse, "Norm Recon MSE": norm_recon_mse}

    def sample_batch(self, key, batch_size, latents=None):

        if latents is not None:
            z = latents
        else:
            z = self.sample_latent(key, batch_size)
        z1 = z[..., :self.dataset.dimension]
        z2 = z[..., self.dataset.dimension:]
        sampling_model = jax.jit(partial(self.model, sampling=True, initialize=False))
        return sampling_model(None, z1, z2)[0], z

    def model_save_data(self):
        return {"VAE Loss": self.vae_losses, "Decoder Likelihood": self.likelihoods, "Naive Flow Likelihood": self.naive_likelihoods,
                "Reconstruction MSE": self.reconstruction_mses, "Normalized Reconstruction MSE": self.norm_recon_mses}
        #return {"VAE Loss": self.vae_losses}


class SqVAEModel(GenerativeModel):
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
                 tqdm,
                 epsilon,
                 dataset,
                 warm_start,
                 train_bias_only,
                 latent_dimension,
                 if_grid
                 ):
        data_size = dataset.dimension

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

        self.encoder_layer_sizes = [int(size) for size in encoder_layer_sizes.split('|')] + [latent_dimension]
        self.decoder_layer_sizes = [int(size) for size in layer_sizes.split('|')] + [data_size]

        self.latent_dimension = self.encoder_layer_sizes[-1]
        self.dataset = dataset
        self.epsilon = epsilon
        self.if_grid = if_grid
        vae_key, self.key = random.split(self.key)
        vae_module = SqVAE.partial(layer_sizes=self.decoder_layer_sizes, encoder_layer_sizes=self.encoder_layer_sizes,
                                   epsilon=self.epsilon, warm_start=warm_start, train_bias_only = train_bias_only)
        _, initial_params = vae_module.init_by_shape(vae_key, [(1, data_size), (1, self.latent_dimension), (1, data_size)])
        if warm_start:
            decoder_params = initial_params['Decoder']
            for k, v in decoder_params.items():
                if 'kernel' not in v:
                    continue
                dim = v['kernel'].shape[0]
                assert dim == v['kernel'].shape[1]
                v['kernel'] = v['kernel'] + jnp.eye(dim)
            encoder_params = initial_params['Encoder']
            for k, v in encoder_params.items():
                if 'kernel' not in v:
                    continue
                dim = v['kernel'].shape[0]
                assert dim == v['kernel'].shape[1]
                v['kernel'] = v['kernel'] + jnp.eye(dim)

        self.model = flax.nn.Model(vae_module, initial_params)
        self.optimizer = flax.optim.Adam(learning_rate=self.learning_rate).create(self.model)

        # optimizer
        deco_bias = flax.optim.ModelParamTraversal(lambda path, _: 'decoder_bias' in path)
        enco_bias = flax.optim.ModelParamTraversal(lambda path, _: 'encoder_bias' in path)
        deco = flax.optim.ModelParamTraversal(lambda path, _: 'Decoder' in path)
        enco = flax.optim.ModelParamTraversal(lambda path, _: 'Encoder' in path)
        var = flax.optim.ModelParamTraversal(lambda path, _: 'epsilon_p' in path)

        deco_bias_opt = flax.optim.Adam(learning_rate=self.learning_rate)
        enco_bias_opt = flax.optim.Adam(learning_rate=self.learning_rate)
        deco_opt = flax.optim.Adam(learning_rate=self.learning_rate)
        enco_opt = flax.optim.Adam(learning_rate=self.learning_rate)
        var_opt = flax.optim.Adam(learning_rate=self.learning_rate)
        if train_bias_only:
            deco_opt = flax.optim.Adam(learning_rate=0)
            enco_opt = flax.optim.Adam(learning_rate=0)
            var_opt = flax.optim.Adam(learning_rate=0)

        opt_def = flax.optim.MultiOptimizer((deco_bias, deco_bias_opt), (enco_bias, enco_bias_opt), (deco, deco_opt),(enco, enco_opt), (var, var_opt))
        self.optimizer = opt_def.create(self.model)


        self.vae_losses = []
        self.likelihoods = []
        self.naive_likelihoods = []
        self.reconstruction_mses = []
        self.decoder_mses = []
        self.encoder_mses = []
        self.norm_recon_mses = []
        self.decoder_batches = []
        self.grids = []
        self.Xs = []
        self.KLs = []
        self.mses = []
        self.encoder_batches = []

    def train_one_batch(self, batch):
        batch = batch.reshape((batch.shape[0], -1))
        latent_batch_key, self.key = random.split(self.key)
        z = self.sample_latent(latent_batch_key, self.batch_size)

        z1 = z[..., :self.latent_dimension]
        z2 = z[..., self.latent_dimension:self.latent_dimension + self.decoder_layer_sizes[-1]]
        self.optimizer, self.model, vae_loss = jit(SqVAE.train_step)(self.optimizer, batch, z1, z2, self.batchnum, self.epsilon, self.learning_rate)
        self.vae_losses.append(vae_loss)

    def compute_model_stats(self, real_batch, fake_batch, latents):
        latent_batch_key, self.key = random.split(self.key)
        z1 = latents[..., :self.latent_dimension]
        z2 = latents[..., self.latent_dimension:]

        vae_loss, Dkl, mse = SqVAE.loss(self.model, real_batch, z1, z2, epsilon=self.epsilon)
        self.KLs.append(Dkl)
        self.mses.append(mse)
        key = self.get_key()
        noise = jnp.zeros_like(real_batch)
        reconstructed_real_batch, mu, logvar_e = self.model(real_batch, noise, noise)
        reconstruction_mse = jnp.sum(jnp.square(real_batch - reconstructed_real_batch), axis=-1).mean()
        norm_recon_mse = jnp.mean(jnp.sum(jnp.square(real_batch - reconstructed_real_batch), axis=-1) / jnp.sum(jnp.square(real_batch), axis=-1))
        self.norm_recon_mses.append(norm_recon_mse)
        latents = random.normal(key, (real_batch.shape[0], self.latent_dimension))
        Decoder_batch = self.model(real_batch, latents, noise, Decoder_Only = True)[0]
        size = Decoder_batch.shape[0]
        if self.latent_dimension == self.dataset.latent_dimension:
            Generator_batch = self.dataset.get_batch(size, latents = latents)
            Decoder_mse = jnp.sum(jnp.square(Generator_batch - Decoder_batch), axis=-1).mean()
        else:
            Decoder_mse = np.infty
        if self.if_grid:
            X_input = real_batch[:5, :] if len(self.Xs) == 0 else self.Xs[0]
            p_grid = self.grid_estimation(X_input)
            self.grids.append(p_grid)
            if len(self.Xs) == 0:
                self.Xs.append(real_batch[:5, :])
        #Encoder_batch = self.model(real_batch, noise, noise, Encoder_only=True)[1]
        self.decoder_batches.append(Decoder_batch[:100, ...])
        self.encoder_batches.append(mu[:100, ...])
        self.decoder_mses.append(Decoder_mse)
        self.reconstruction_mses.append(reconstruction_mse)
        return {"VAE Loss": vae_loss, "Reconstruction MSE": reconstruction_mse, "Norm Recon MSE": norm_recon_mse,
                "Decoder MSE": Decoder_mse, "Reconstruction Error": mse, "KL Divergence": Dkl} # "Likelihood": likelihood}

    def grid_estimation(self, X):
        pgrid = []
        latent_batch_key, self.key = random.split(self.key)
        #latents = jnp.array([[x * 0.1, y * 0.1] for x in range(-50, 50, 2) for y in range(-50, 50, 2)])
        coordiates = [p for p in itertools.product(range(-50, 50, 2), repeat=self.latent_dimension)]
        latents = np.array([0.1*np.array(item) for item in coordiates])
        stdlogpdf = partial(multivariate_normal.logpdf, mean=jnp.array([0 for i in range(self.latent_dimension)]), cov=jnp.eye(self.latent_dimension))
        logpz = jax.vmap(stdlogpdf)(latents)
        point_size = latents.shape[0]
        z2 = random.normal(latent_batch_key, shape=(point_size, self.dataset.dimension))
        batch = jnp.zeros((point_size, self.dataset.dimension))
        MUs = self.model(batch, latents, z2, Decoder_Only=True)[0]
        def pxz(Mu, logpzi):
            logpz_mu = jnp.array([logpzi for j in range(X.shape[0])])
            logpdf = partial(multivariate_normal.logpdf, mean=Mu, cov=jnp.eye(X.shape[1]) * jnp.exp(self.epsilon))
            logpdf_pxlz = jax.vmap(logpdf)(X)
            logpxz = logpdf_pxlz + logpz_mu
            return jnp.exp(logpxz)
        pgrid = jax.vmap(pxz)(MUs, logpz)
        return pgrid


    def sample_batch(self, key, batch_size, latents=None):
        if latents is not None:
            z = latents
        else:
            z = self.sample_latent(key, batch_size)
        z1 = z[..., :self.latent_dimension]
        z2 = z[..., self.latent_dimension:self.latent_dimension + self.decoder_layer_sizes[-1]]
        sampling_model = jax.jit(partial(self.model, sampling=True))
        return sampling_model(None, z1, z2)[0], z

    def model_save_data(self):
        return {"VAE Loss": self.vae_losses,  "Reconstruction MSE": self.reconstruction_mses,
                "Decoder MSE": self.decoder_mses, "Normalized Reconstruction MSE": self.norm_recon_mses,
                "Decoder Batches": self.decoder_batches, "P(x|z) Grid": self.grids, "X's": self.Xs,
                "KLs": self.KLs, "MSEs": self.mses, "Encoder Batches": self.encoder_batches}

class MLP_ACVAEModel(GenerativeModel):
    def __init__(self,
                 dirname,
                 num_batches,
                 num_epochs,
                 batch_size,
                 learning_rate,
                 layer_sizes,
                 n_passes,
                 state_dict,
                 data_fn,
                 tqdm,
                 dataset,
                 ):
        latent_dimension = data_size = dataset.dimension
        super().__init__(
                dirname=dirname,
                num_batches=num_batches,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                latent_distribution='gaussian',
                latent_dimension=data_size * 2,
                dataset=dataset,
                state_dict=state_dict,
                data_fn=data_fn,
                tqdm=tqdm,
                )
        self.layer_sizes = [int(size) for size in layer_sizes.split('|')] + [data_size]
        self.n_passes = n_passes
        self.epsilon = 0
        vae_key, self.key = random.split(self.key)
        vae_module = MLP_ACVAE.partial(layer_sizes=self.layer_sizes, n_passes=self.n_passes, epsilon=self.epsilon)
        _, initial_params = vae_module.init_by_shape(vae_key, [(data_size,), (latent_dimension,), (latent_dimension,)])
        self.model = flax.nn.Model(vae_module, initial_params)
        self.optimizer = flax.optim.Adam(learning_rate=self.learning_rate).create(self.model)

        # optimizer
        deco = flax.optim.ModelParamTraversal(lambda path, _: 'Decoder' in path)
        enco = flax.optim.ModelParamTraversal(lambda path, _: 'Encoder' in path)
        var = flax.optim.ModelParamTraversal(lambda path, _: 'epsilon_p' in path)
        deco_opt = flax.optim.Adam(learning_rate=self.learning_rate)
        enco_opt = flax.optim.Adam(learning_rate=self.learning_rate)
        var_opt = flax.optim.Adam(learning_rate=self.learning_rate)
        opt_def = flax.optim.MultiOptimizer((deco, deco_opt), (enco, enco_opt), (var, var_opt))
        self.optimizer = opt_def.create(self.model)
        self.train_step = jit(partial(ACVAE.train_step, epsilon=self.epsilon, batch_num=self.batchnum,
                                      lr=self.learning_rate, use_fred_covariance=self.use_fred_covariance))

        self.vae_losses = []
        self.likelihoods = []
        self.naive_likelihoods = []

    def train_one_batch(self, batch):
        batch = batch.reshape((batch.shape[0], -1))
        latent_batch_key, self.key = random.split(self.key)
        z = self.sample_latent(latent_batch_key, self.batch_size)
        z1 = z[..., :self.dataset.dimension]
        z2 = z[..., self.dataset.dimension:]
        self.optimizer, self.model, vae_loss = self.train_step(self.optimizer, batch, z1, z2)
        self.vae_losses.append(vae_loss)

    def compute_model_stats(self, real_batch, fake_batch, latents):
        z1 = latents[..., :self.dataset.dimension]
        z2 = latents[..., self.dataset.dimension:]
        vae_loss = MLP_ACVAE.loss(self.model, real_batch, z1, z2, epsilon=self.epsilon)
        likelihood = self.model.average_log_likelihood(real_batch, z1, z2).mean()
        self.likelihoods.append(likelihood)
        naive_flow_likelihood = self.model.naive_flow_log_likelihood(real_batch)
        self.naive_likelihoods.append(naive_flow_likelihood)
        return {"VAE Loss": vae_loss, "Decoder Likelihood": likelihood, "Naive Flow Likelihood": naive_flow_likelihood}

    def sample_batch(self, key, batch_size, latents=None):
        if latents is not None:
            z = latents
        else:
            z = self.sample_latent(key, batch_size)
        z1 = z[..., :self.dataset.dimension]
        z2 = z[..., self.dataset.dimension:]
        sampling_model = jax.jit(partial(self.model, sampling=True, initialize=False))
        return sampling_model(None, z1, z2)[0], z

    def model_save_data(self):
        return {"VAE Loss": self.vae_losses, "Decoder Likelihood": self.likelihoods, "Naive Flow Likelihood": self.naive_likelihoods}
        #return {"VAE Loss": self.vae_losses}

class VAEModel_2Stage(GenerativeModel):
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
                 tqdm,
                 dataset,
                 latent_dimension,
                 stop_point):
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
        data_size = np.product(dataset.shape)
        encoder1_layer_sizes = [int(size) for size in encoder_layer_sizes.split('|')] + [latent_dimension * 2]
        decoder1_layer_sizes = [int(size) for size in layer_sizes.split('|')] + [data_size]
        encoder2_layer_sizes = [int(size) for size in encoder_layer_sizes.split('|')] + [latent_dimension * 2]
        decoder2_layer_sizes = [int(size) for size in layer_sizes.split('|')] + [latent_dimension]
        vae_key, self.key = random.split(self.key)
        vae_module1 = gammaVAE.partial(encoder_layer_sizes=encoder1_layer_sizes, decoder_layer_sizes=decoder1_layer_sizes)
        _, initial_params1 = vae_module1.init_by_shape(vae_key, [(data_size,), (latent_dimension,)])
        self.model1 = flax.nn.Model(vae_module1, initial_params1)
        vae_key, self.key = random.split(self.key)
        vae_module2 = gammaVAE.partial(encoder_layer_sizes=encoder2_layer_sizes,
                                        decoder_layer_sizes=decoder2_layer_sizes)
        _, initial_params2 = vae_module2.init_by_shape(vae_key, [(latent_dimension,), (latent_dimension,)])
        self.model2 = flax.nn.Model(vae_module2, initial_params2)
        self.optimizer1 = flax.optim.Adam(learning_rate=self.learning_rate).create(self.model1)
        self.optimizer2 = flax.optim.Adam(learning_rate=self.learning_rate).create(self.model2)
        self.vae_losses1 = []
        self.vae_losses2 = []
        self.stage = 1
        self.stop_point = stop_point

    def train_one_batch(self, batch):
        if self.batchnum == self.stop_point:
            self.stage = 2
        if self.stage == 2:
            _, _, mu, logvar = self.extract_posterior(batch)
            batch = self.stage2_generator(mu, logvar, self.batch_size)
        batch = batch.reshape((batch.shape[0], -1))
        latent_batch_key, self.key = random.split(self.key)
        z = self.sample_latent(latent_batch_key, self.batch_size)
        if self.stage == 1:
            self.optimizer1, self.model1, vae_loss = gammaVAE.train_step(self.optimizer1, batch, z)
            self.vae_losses1.append(vae_loss)
        elif self.stage == 2:
            self.optimizer2, self.model2, vae_loss = gammaVAE.train_step(self.optimizer2, batch, z)
            self.vae_losses2.append(vae_loss)

    def extract_posterior(self, batch):
        batch = batch.reshape((batch.shape[0], -1))
        latent_batch_key, self.key = random.split(self.key)
        z = self.sample_latent(latent_batch_key, batch.shape[0])
        return self.model1(batch, z)

    def stage2_generator(self, mu, logvar, size):
        dim = mu.shape[1]
        stdevs = np.exp(logvar / 2)
        latent_key, self.key = random.split(self.key)
        z = random.normal(latent_key, (size, dim))
        samples = mu + stdevs * z
        return samples

    def compute_model_stats(self, real_batch, fake_batch, latents):
        if self.stage == 1:
            vae_loss = gammaVAE.loss(self.model1, real_batch, latents)
        elif self.stage == 2:
            _, _, mu, logvar = self.extract_posterior(real_batch)
            batch_size = real_batch.shape[0]
            real_batch_hat = self.stage2_generator(mu, logvar, batch_size)
            vae_loss = gammaVAE.loss(self.model2, real_batch_hat, latents)
        return {"VAE Loss": vae_loss}

    def sample_batch(self, key, batch_size, latents=None):
        if latents is not None:
            z = latents
        else:
            z = self.sample_latent(key, batch_size)
        sampling_model1 = jax.jit(partial(self.model1, sampling=True))
        sampling_model2 = jax.jit(partial(self.model2, sampling=True))
        return sampling_model1(None, sampling_model2(None, z)[0])[0], z

    def model_save_data(self):
        self.vae_losses = self.vae_losses1 + self.vae_losses2
        return {"VAE Loss": self.vae_losses}

