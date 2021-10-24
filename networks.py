import jax.numpy as jnp
from jax.scipy.stats import norm, logistic
from jax.nn.initializers import normal
import jax
import flax
from functools import partial
from jax import random

from utils import cross_entropy_loss, Constants, leaky_relu, inv_leaky_relu, inv_dense, InvertibleBatchNorm
from utils import relu, get_mask, squeeze_2x2
from utils import inv_batch_norm


EPS = 1e-8

@jax.vmap
def binary_cross_entropy(probs, labels):
    return - jnp.sum(labels * jnp.log(probs + EPS) + (1 - labels) * jnp.log(1 - probs + EPS))

def fill_diagonal(a, val):
    assert a.ndim >= 2
    i, j = jnp.diag_indices(min(a.shape[-2:]))
    return a.at[..., i, j].set(val)


class FullyConnectedNetwork(flax.nn.Module):
    def apply(self, x, layer_sizes, batch_norm=False, leaky=False, coefficient = True, datasets = False, if_sigmoid = False):
        for i, size in enumerate(layer_sizes):
            name = self.get_layer_name(i)
            if coefficient:
                if datasets:
                    x = flax.nn.Dense(x, features=size, name=name, kernel_init=normal(1))
                else:
                    x = flax.nn.Dense(x, features=size, name=name)
            if i + 1 < len(layer_sizes):
                if leaky:
                    x = leaky_relu(x)
                else:
                    x = relu(x)
                if batch_norm:
                    x = flax.nn.BatchNorm(x)
        if if_sigmoid:
            x = flax.nn.sigmoid(x)
        return x

    def get_layer_name(self, i):
        return f"FC{i}"

    @staticmethod
    def train_step(**args):
        raise NotImplementedError()

    @staticmethod
    @jax.jit
    def evaluate(model, data):
        return model(data)

class SquareActivationNetwork(flax.nn.Module):
    def apply(self, x, layer_sizes, warm_start=False, bias = None):

        for i, size in enumerate(layer_sizes):
            name = self.get_layer_name(i)

            if warm_start:
                x = flax.nn.Dense(x, features=size, name=name, kernel_init=normal(1e-6))
            else:
                x = flax.nn.Dense(x, features=size, name=name)
            if i + 1 < len(layer_sizes):
                if bias is None:
                    x = jnp.square(x)
                else:
                    x = jnp.square(x - bias)

        return x

    def get_layer_name(self, i):
        return f"FC{i}"

    @staticmethod
    def train_step(**args):
        raise NotImplementedError()

    @staticmethod
    @jax.jit
    def evaluate(model, data):
        return model(data)


class SqrtActivationNetwork(flax.nn.Module):
    def apply(self, x, layer_sizes, warm_start=False):

        x = jnp.sqrt(jnp.clip(x, a_min=0))
        size = layer_sizes[0]
        if warm_start:
            x = flax.nn.Dense(x, features=size, name=f"FC0", kernel_init=normal(1e-6))
        else:
            x = flax.nn.Dense(x, features=size, name="FC0")
        return x

    def get_layer_name(self, i):
        return f"FC{i}"

    @staticmethod
    def train_step(**args):
        raise NotImplementedError()

    @staticmethod
    @jax.jit
    def evaluate(model, data):
        return model(data)

class SqrtActivationNetwork(flax.nn.Module):
    def apply(self, x, layer_sizes, warm_start=False, bias = None):
        x = jnp.sqrt(jnp.clip(x, a_min=0))
        if bias is not None:
            x = x + bias
        size = layer_sizes[0]
        if warm_start:
            x = flax.nn.Dense(x, features=size, name=f"FC0", kernel_init=normal(1e-6))
        else:
            x = flax.nn.Dense(x, features=size, name="FC0")
        return x

    def get_layer_name(self, i):
        return f"FC{i}"

    @staticmethod
    def train_step(**args):
        raise NotImplementedError()

    @staticmethod
    @jax.jit
    def evaluate(model, data):
        return model(data)

class EndoConvNet(flax.nn.Module):
    def apply(self, x, channel_sizes, kernel_size=3):
        for i, size in enumerate(channel_sizes):
            name = self.get_layer_name(i)
            x = flax.nn.Conv(x, features=size, kernel_size=(kernel_size, kernel_size), name=name)
            if i + 1 < len(channel_sizes):
                x = relu(x)
        return x

    def get_layer_name(self, i):
        return f"Conv{i}"

    @staticmethod
    def train_step(**args):
        raise NotImplementedError()

    @staticmethod
    @jax.jit
    def evaluate(model, data):
        return model(data)


class EndoResNet(flax.nn.Module):
    def apply(self, x, channel_sizes, kernel_size=3):
        res = jnp.concatenate((x, x), axis=-1)
        for i, size in enumerate(channel_sizes):
            name = self.get_layer_name(i)
            x = flax.nn.Conv(x, features=size, kernel_size=(kernel_size, kernel_size), name=name)
            if i + 1 < len(channel_sizes):
                x = relu(x)
        return res + x

    def get_layer_name(self, i):
        return f"Conv{i}"

    @staticmethod
    def train_step(**args):
        raise NotImplementedError()

    @staticmethod
    @jax.jit
    def evaluate(model, data):
        return model(data)


class MLPRegressor(FullyConnectedNetwork):

    @staticmethod
    @jax.jit
    def train_step(optimizer, X, Y):
        def loss_fn(model):
            Y_hat = model(X)
            return jnp.mean(jnp.square(Y - Y_hat))
        l2_loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, l2_loss


class MLPClassifier(FullyConnectedNetwork):

    def apply(self, x, layer_sizes):
        return flax.nn.log_softmax(super().apply(x, layer_sizes))

    @staticmethod
    @jax.jit
    def train_step(optimizer, batch, labels):
        def loss_fn(model):
            logits = model(batch)
            return jnp.mean(cross_entropy_loss(logits, labels))
        ce_loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, ce_loss


class VAE(flax.nn.Module):

    def apply(self, x, z1, z2, epsilon, encoder_layer_sizes, decoder_layer_sizes, sampling=False, if_sigmoid = False, tunable_decoder_var = False, dataset_name = None):
        if sampling:
            mu = 0
            logvar_e = 0
            #z2 = 0 * z2
        else:
            enc_out = FullyConnectedNetwork(x, layer_sizes=encoder_layer_sizes, name="Encoder")
            mu = enc_out
            epsilon_p = self.param('epsilon_p', (z1.shape[-1],), jax.nn.initializers.ones)
            if tunable_decoder_var:
                epsilon = self.param('epsilon', (1,), jax.nn.initializers.ones) * epsilon
            logvar_e = epsilon_p
        stdevs = jnp.exp(logvar_e / 2)
        samples = mu + stdevs * z1
        if dataset_name == "sigmoid":
            x_hat_0 = FullyConnectedNetwork(samples, layer_sizes=decoder_layer_sizes, if_sigmoid=True, name="SigDecoder")
            x_hat_1 = FullyConnectedNetwork(samples, layer_sizes=decoder_layer_sizes, if_sigmoid=False, name="Decoder")
            x_hat = x_hat_0 + x_hat_1
        else:
            x_hat = FullyConnectedNetwork(samples, layer_sizes=decoder_layer_sizes, if_sigmoid = if_sigmoid, name="Decoder")
        stdev = jnp.exp(epsilon / 2.)
        noise = z2 * stdev
        x_hat = x_hat  + noise
        return x_hat, mu, logvar_e, epsilon


    @staticmethod
    @jax.jit
    def train_step(optimizer, batch, z1, z2, epsilon):
        def loss_fn(model):

            x_hat, mu, logvar_e, epsilon = model(batch, z1, z2)
            #bce = binary_cross_entropy(x_hat, batch)
            Dkl = -0.5 * jnp.sum((1 + logvar_e - jnp.exp(logvar_e) - jnp.square(mu)), axis=-1)
            var_d = jnp.exp(epsilon)
            mse = (0.5 * jnp.square(x_hat - batch)/var_d + 0.5 * (jnp.log(2. * jnp.pi) + epsilon)).sum(axis=-1)
            loss = Dkl + mse
            return loss.mean()
        vae_loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, vae_loss

    @staticmethod
    @jax.jit
    def loss(model, batch, z1, z2, epsilon):

        x_hat, mu, logvar_e, epsilon = model(batch, z1, z2)
        Dkl = -0.5 * jnp.sum((1 + logvar_e - jnp.exp(logvar_e) - jnp.square(mu)), axis=-1)
        var_d = jnp.exp(epsilon)
        #bce = binary_cross_entropy(x_hat, batch)
        mse = (0.5 * jnp.square(x_hat - batch) / var_d + 0.5 * (jnp.log(2. * jnp.pi) + epsilon)).sum(axis=-1)
        loss = Dkl + mse
        return loss.mean(), Dkl.mean(), mse.mean(), logvar_e, epsilon



class gammaVAE(flax.nn.Module):

    def apply(self, x, z, encoder_layer_sizes, decoder_layer_sizes, sampling=False):
        if sampling:
            mu = 0
            logvar = 0
        else:
            enc_out = FullyConnectedNetwork(x, layer_sizes=encoder_layer_sizes, name="Encoder")
            latent_dim = enc_out.shape[-1] // 2
            mu = enc_out[..., :latent_dim]
            logvar = enc_out[..., latent_dim:]
        stdevs = jnp.exp(logvar / 2)
        samples = mu + stdevs * z
        dec_out = FullyConnectedNetwork(samples, layer_sizes=decoder_layer_sizes, name="Decoder")
        x_hat = dec_out
        loggamma = self.param('loggamma', (1,), jax.nn.initializers.ones)
        return x_hat, loggamma, mu, logvar

    @staticmethod
    @jax.jit
    def train_step(optimizer, batch, z):
        def loss_fn(model):
            HALF_LOG_TWO_PI = 0.91893
            x_hat, loggamma, mu, logvar = model(batch, z)
            gamma = jnp.exp(loggamma)
            Dkl = -0.5 * jnp.sum((1 + logvar - jnp.exp(logvar) - jnp.square(mu)), axis=-1)
            mse = 0.5 * jnp.square(x_hat - batch).sum(axis=-1)/gamma + loggamma + HALF_LOG_TWO_PI
            loss = Dkl + mse
            return loss.mean()
        vae_loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, vae_loss

    @staticmethod
    @jax.jit
    def loss(model, batch, z):
        HALF_LOG_TWO_PI = 0.91893
        x_hat, loggamma, mu, logvar = model(batch, z)
        gamma = jnp.exp(loggamma)
        Dkl = -0.5 * jnp.sum((1 + logvar - jnp.exp(logvar) - jnp.square(mu)), axis=-1)
        mse = 0.5 * jnp.square(x_hat - batch).sum(axis=-1)/gamma + loggamma + HALF_LOG_TWO_PI
        loss = Dkl + mse
        return loss.mean()


class ACVAE(flax.nn.Module):

    def apply(self, x, z1, z2, layer_sizes, n_passes, epsilon, use_fred_covariance=False, sampling=False, initialize=True):
        if sampling:
            mu = 0
            logvar_e = 0
        else:
            enc_out = RealNVP2Network(x, layer_sizes=layer_sizes, n_passes=n_passes, name="Encoder")
            mu = enc_out
            epsilon_p = self.param('epsilon_p', (x.shape[-1],), jax.nn.initializers.ones)
            logvar_e = epsilon_p * epsilon
        decoder_nvp = RealNVP2Network.shared(layer_sizes=layer_sizes, n_passes=n_passes, name="Decoder")
        if use_fred_covariance and not sampling:
            def get_fred_cov_sample(x, z, mu):
                z_0 = decoder_nvp(x)
                inv = partial(decoder_nvp.apply_inverse, layer_sizes=layer_sizes, n_passes=n_passes)
                Jf = jax.jacfwd(inv)(mu)
                JftJf = Jf.T @ Jf
                Sigma_z = jnp.linalg.inv(JftJf / jnp.exp(epsilon) + jnp.eye(x.shape[-1]))
                chol_sigma = jnp.linalg.cholesky(Sigma_z)
                sample = chol_sigma @ z + mu
                return Sigma_z, sample
            # Sigma_Z, samples = jax.vmap(get_fred_cov_sample)(x, z1, mu)
            Sigma_Z, samples = jax.vmap(get_fred_cov_sample)(x, z1, mu)
        else:
            stdevs = jnp.exp(logvar_e / 2.)
            samples = mu + stdevs * z1
        if initialize:
            x_hat = decoder_nvp(samples)
        else:
            x_hat = decoder_nvp.apply_inverse(samples)
        stdev = jnp.exp(epsilon/2.)
        noise = z2 * stdev
        x_hat = x_hat + noise
        if use_fred_covariance and not sampling:
            return x_hat, mu, Sigma_Z
        else:
            return x_hat, mu, logvar_e

    @flax.nn.module_method
    def approx_log_likelihood(self, data, z1, z2, layer_sizes, n_passes, epsilon, **kwargs):
        # likelihood = RealNVP2Network.average_log_likelihood(data, layer_sizes=layer_sizes, n_passes=n_passes, name="Decoder")
        nvp = RealNVP2Network.shared(layer_sizes=layer_sizes, n_passes=n_passes, name="Decoder")
        nvp_params = self.get_param("Decoder")
        mu = RealNVP2Network.call(nvp_params, data, layer_sizes=layer_sizes, n_passes=n_passes)
        epsilon_p = self.get_param('epsilon_p')
        logvar = epsilon_p  # * epsilon
        stdevs = jnp.exp(logvar/2.)
        var = jnp.exp(epsilon_p)  #### CHECK
        samples = mu + stdevs * z1
        data_hat = nvp.apply_inverse(samples)
        #stdev = jnp.exp(epsilon_p/2.)
        #noise = z2 * stdev
        #data_hat = data_hat + noise
        reconstruction = -0.5 * (jnp.sum(logvar) + jnp.square(data - data_hat) / var + data.shape[-1] * jnp.log(2 * jnp.pi))
        Dkl = -0.5 * jnp.sum((1 + logvar - jnp.exp(logvar) - jnp.square(mu)), axis=-1)
        likelihood = -Dkl + jnp.sum(reconstruction, axis=-1) # CHANGE Dkl -> -Dkl

        return likelihood.mean()

    @flax.nn.module_method
    def naive_flow_log_likelihood(self, data, layer_sizes, n_passes, **kwargs):
        likelihood = RealNVP2Network.average_log_likelihood(data, layer_sizes=layer_sizes, n_passes=n_passes, name="Decoder")
        return likelihood.mean()

    @flax.nn.module_method
    def average_log_likelihood(self, data, layer_sizes, n_passes, epsilon, **kwargs):
        nvp = RealNVP2Network.shared(layer_sizes=layer_sizes, n_passes=n_passes, name="Decoder")
        def score_example(example):
            d = example.shape[-1]
            z_0 = nvp(example)
            inv = partial(nvp.apply_inverse, layer_sizes=layer_sizes, n_passes=n_passes)
            Jf = jax.jacfwd(inv)(z_0)
            JftJf = Jf.T @ Jf
            JftJf_inv = jnp.linalg.inv(JftJf + jnp.eye(d) * jnp.exp(epsilon))
            Sigma_z = jnp.linalg.inv(JftJf / jnp.exp(epsilon) + jnp.eye(d))
            Entropy = -0.5 * (d * (jnp.log(2 * jnp.pi) + epsilon) - jnp.linalg.slogdet(Sigma_z)[1])
            return Entropy - jnp.trace(Sigma_z) / 2 - jnp.sum(jnp.square(z_0)) / 2
        score = jax.vmap(score_example)(data)
        return score.mean()

    @staticmethod
    def train_step(optimizer, batch, z1, z2, batch_num, epsilon, lr, use_fred_covariance=False):
        def loss_fn(model):
            if use_fred_covariance:
                x_hat, mu, sigma_z = model(batch, z1, z2, initialize=False)
                Dkl = -0.5 * (x_hat.shape[-1] + jax.vmap(jnp.linalg.slogdet)(sigma_z)[1] -
                              jax.vmap(jnp.trace)(sigma_z) - jnp.sum(jnp.square(mu), axis=-1))

            else:
                x_hat, mu, logvar_e = model(batch, z1, z2, initialize=False)
                Dkl = -0.5 * jnp.sum((1 + logvar_e - jnp.exp(logvar_e) - jnp.square(mu)), axis=-1)
            var_d = jnp.exp(epsilon)
            mse = (0.5 * jnp.square(x_hat - batch)/var_d + 0.5 * (jnp.log(2. * jnp.pi) + epsilon)).sum(axis=-1)
            loss = Dkl + mse
            return loss.mean()
        vae_loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        #updated_lr = jnp.where(batch_num >= 10000, jnp.zeros(1), lr*jnp.ones(1))
        #optimizer=optimizer.apply_gradient(grad, hyper_params=[
        #                          optimizer.optimizer_def.hyper_params[0].replace(learning_rate=lr),
        #                          optimizer.optimizer_def.hyper_params[1].replace(learning_rate=lr),
        #                          optimizer.optimizer_def.hyper_params[2].replace(learning_rate=lr)
        #])
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, vae_loss

    @staticmethod
    # @jax.jit(static_argnums=[5])
    def loss(model, batch, z1, z2, epsilon, use_fred_covariance):

        if use_fred_covariance:
            x_hat, mu, sigma_z = model(batch, z1, z2, initialize=False)
            Dkl = -0.5 * (x_hat.shape[-1] + jax.vmap(jnp.linalg.slogdet)(sigma_z)[1] -
                          jax.vmap(jnp.trace)(sigma_z) - jnp.sum(jnp.square(mu), axis=-1))

        else:
            x_hat, mu, logvar_e = model(batch, z1, z2, initialize=False)
            Dkl = -0.5 * jnp.sum((1 + logvar_e - jnp.exp(logvar_e) - jnp.square(mu)), axis=-1)
        var_d = jnp.exp(epsilon)
        mse = (0.5 * jnp.square(x_hat - batch) / var_d + 0.5 * (jnp.log(2. * jnp.pi) + epsilon)).sum(axis=-1)
        loss = Dkl + mse
        return loss.mean()



class SqVAE(flax.nn.Module):

    def apply(self, x, z1, z2, layer_sizes, encoder_layer_sizes, epsilon, sampling=False, Decoder_Only = False, Encoder_Only = False,
              warm_start=False, train_bias_only = False, latent_batch_key = None):
        
        if train_bias_only:
            decoder_bias = self.param('decoder_bias', (z2.shape[-1],), jax.nn.initializers.zeros)
            encoder_bias = self.param('encoder_bias', (z1.shape[-1],), jax.nn.initializers.zeros)
        else:
            decoder_bias = None
            encoder_bias = None
        if sampling:
            mu = 0
            logvar_e = 0
        else:

            if warm_start:
                enc_out = SqrtActivationNetwork(x, layer_sizes=encoder_layer_sizes, name="Encoder", warm_start=warm_start, bias = encoder_bias)
            else:
                enc_out = FullyConnectedNetwork(x, layer_sizes=encoder_layer_sizes, name="Encoder")
            mu = enc_out
            epsilon_p = self.param('epsilon_p', (encoder_layer_sizes[-1],), jax.nn.initializers.ones)
            logvar_e = epsilon_p * epsilon

        if not Decoder_Only:
            stdevs = jnp.exp(logvar_e / 2.)
            samples = mu + stdevs * z1
        else:
            samples = z1
        if not Encoder_Only:
            if warm_start:
                x_hat = SquareActivationNetwork(samples, layer_sizes=layer_sizes, name="Decoder", warm_start=warm_start, bias = decoder_bias)
            else:
                x_hat = SquareActivationNetwork(samples, layer_sizes=layer_sizes, name="Decoder", bias = decoder_bias)
            stdev = jnp.exp(epsilon/2.)
            noise = z2 * stdev

            x_hat = x_hat + noise
            
        return x_hat, mu, logvar_e

    @staticmethod
    def train_step(optimizer, batch, z1, z2, batch_num, epsilon, lr):
        def loss_fn(model):

            x_hat, mu, logvar_e = model(batch, z1, z2)
            Dkl = -0.5 * jnp.sum((1 + logvar_e - jnp.exp(logvar_e) - jnp.square(mu)), axis=-1)
            var_d = jnp.exp(epsilon)
            mse = (0.5 * jnp.square(x_hat - batch)/var_d + 0.5 * (jnp.log(2. * jnp.pi) + epsilon)).sum(axis=-1)
            loss = Dkl + mse
            return loss.mean()
        vae_loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        #updated_lr = jnp.where(batch_num >= 10000, jnp.zeros(1), lr*jnp.ones(1))
        #optimizer=optimizer.apply_gradient(grad, hyper_params=[
        #                          optimizer.optimizer_def.hyper_params[0].replace(learning_rate=lr),
        #                          optimizer.optimizer_def.hyper_params[1].replace(learning_rate=lr),
        #                          optimizer.optimizer_def.hyper_params[2].replace(learning_rate=lr)
        #])
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, vae_loss

    @staticmethod
    # @jax.jit(static_argnums=[5])
    def loss(model, batch, z1, z2, epsilon):

        x_hat, mu, logvar_e = model(batch, z1, z2)
        Dkl = -0.5 * jnp.sum((1 + logvar_e - jnp.exp(logvar_e) - jnp.square(mu)), axis=-1)
        var_d = jnp.exp(epsilon)
        mse = (0.5 * jnp.square(x_hat - batch) / var_d + 0.5 * (jnp.log(2. * jnp.pi) + epsilon)).sum(axis=-1)
        loss = Dkl + mse
        return loss.mean(), Dkl.mean(), mse.mean()
    """
    @flax.nn.module_method
    def average_log_likelihood(self, data, latent_dim, epsilon, **kwargs):

        def score_example(example):
            
            mu_z = jnp.zeros(latent_dim)
            Sigma_z = jnp.eye(latent_dim)
            z = random.multivariate_normal(key, mu_z, Sigma_z, [500])
            logpdf = partial(multivariate_normal.logpdf, mean=jnp.zeros(self.latent_dimension),
                             cov=jnp.eye(self.latent_dimension))
            logpz = jax.vmap(logpdf)(z)
            logpxzpdf = partial(multivariate_normal.logpdf, mean=z**2,
                             cov=jnp.eye(self.latent_dimension)*epsilon)
            logpxz = jax.vmap(logpxzpdf)(example)
            
            for x in range(-5, 5, 0.5):
                for y in range(-5, 5, 0.5):

            return (logpxz + logpz).mean()

        score = jax.vmap(score_example)(data)
        return score.mean()
    """
class MLP_ACVAE(flax.nn.Module):

    def apply(self, x, z1, z2, layer_sizes, n_passes, epsilon, sampling=False, initialize=True):
        if sampling:
            mu = 0
            logvar_e = 0
        else:
            enc_out = FullyConnectedNetwork(x, layer_sizes=layer_sizes, name="Encoder")
            mu = enc_out
            epsilon_p = self.param('epsilon_p', (x.shape[-1],), jax.nn.initializers.ones)
            logvar_e = epsilon_p # * epsilon
        stdevs = jnp.exp(logvar_e / 2.)
        samples = mu + stdevs * z1
        if initialize:
            x_hat = RealNVP2Network(samples, layer_sizes=layer_sizes, n_passes=n_passes, name="Decoder")
        else:
            x_hat = RealNVP2Network.apply_inverse(samples, layer_sizes=layer_sizes, n_passes=n_passes, name="Decoder")
        stdev = jnp.exp(epsilon/2.)
        noise = z2 * stdev
        x_hat = x_hat + noise
        return x_hat, mu, logvar_e

    @flax.nn.module_method
    def average_log_likelihood(self, data, z1, z2, layer_sizes, n_passes, epsilon, **kwargs):
        # likelihood = RealNVP2Network.average_log_likelihood(data, layer_sizes=layer_sizes, n_passes=n_passes, name="Decoder")
        nvp_params = self.get_param("Decoder")
        mu = RealNVP2Network.call(nvp_params, data, layer_sizes=layer_sizes, n_passes=n_passes)
        epsilon_p = self.get_param('epsilon_p')
        logvar = epsilon_p # * epsilon
        stdevs = jnp.exp(logvar / 2)
        var = jnp.exp(logvar)
        samples = mu + stdevs * z1
        data_hat = RealNVP2Network.apply_inverse(samples, layer_sizes=layer_sizes, n_passes=n_passes, name="Decoder")
        stdev = jnp.exp(epsilon_p / 2.)
        noise = z2 * stdev
        data_hat = data_hat + noise
        reconstruction = -0.5 * (jnp.sum(logvar) + jnp.square(data - data_hat) / var + data.shape[-1] * jnp.log(2 * jnp.pi))
        Dkl = -0.5 * jnp.sum((1 + logvar - jnp.exp(logvar) - jnp.square(mu)), axis=-1)
        likelihood = -Dkl + jnp.sum(reconstruction, axis=-1)

        return likelihood.mean()


    @flax.nn.module_method
    def naive_flow_log_likelihood(self, data, layer_sizes, n_passes, **kwargs):
        likelihood = RealNVP2Network.average_log_likelihood(data, layer_sizes=layer_sizes, n_passes=n_passes, name="Decoder")
        return likelihood.mean()

    @staticmethod
    @jax.jit
    def train_step(optimizer, batch, z1, z2, epsilon, batch_num, lr):
        def loss_fn(model):
            x_hat, mu, logvar_e = model(batch, z1, z2, initialize=False)
            var_d = jnp.exp(epsilon)
            obs_dim = x_hat.shape[1]
            Dkl = -0.5 * jnp.sum((1 + logvar_e - jnp.exp(logvar_e) - jnp.square(mu)), axis=-1) * var_d
            mse = (0.5 * jnp.square(x_hat - batch) + var_d * 0.5 * obs_dim * (jnp.log(2 * jnp.pi) + epsilon)).sum(axis=-1)
            loss = Dkl + mse
            return loss.mean()
        vae_loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        updated_lr = jnp.where(batch_num  >= 10000, jnp.zeros(1), lr*jnp.ones(1))
        optimizer=optimizer.apply_gradient(grad, hyper_params=[
                                  optimizer.optimizer_def.hyper_params[0].replace(learning_rate=updated_lr),
                                  optimizer.optimizer_def.hyper_params[1].replace(learning_rate=lr),
                                  optimizer.optimizer_def.hyper_params[2].replace(learning_rate=lr)
        ])
        #optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, vae_loss

    @staticmethod
    @jax.jit
    def loss(model, batch, z1, z2, epsilon):
        x_hat, mu, logvar_e = model(batch, z1, z2)
        var_d = jnp.exp(epsilon)
        obs_dim = x_hat.shape[1]
        Dkl = -0.5 * jnp.sum((1 + logvar_e - jnp.exp(logvar_e) - jnp.square(mu)), axis=-1) * var_d
        mse = (0.5 * jnp.square(x_hat - batch) + var_d * 0.5 * obs_dim * (jnp.log(2 * jnp.pi) + epsilon)).sum(axis=-1)
        loss = Dkl + mse
        return loss.mean()


class Generator(FullyConnectedNetwork):

    @staticmethod
    @jax.jit
    def train_step(optimizer, batch, critic):
        def loss_fn(model):
            fake_data = model(batch)
            logits = critic(fake_data)
            # we want the generator to maximize the output of the critic
            loss = -jnp.mean(logits)
            return loss
        gen_loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, gen_loss


class Critic(FullyConnectedNetwork):

    @staticmethod
    @jax.jit
    def train_step(optimizer, real_batch, fake_batch, epsilon):
        def loss_fn(model):
            real_logits = model(real_batch)
            fake_logits = model(fake_batch)
            real_loss = jnp.mean(real_logits)
            fake_loss = jnp.mean(fake_logits)
            hat_batch = epsilon * real_batch + (1 - epsilon) * fake_batch

            def eval_model(x):
                return model(x)[0]
            grad_norm = jnp.linalg.norm(jax.vmap(jax.grad(eval_model))(hat_batch), axis=1)
            grad_loss = jnp.sum((grad_norm - 1) ** 2)
            loss = fake_loss - real_loss + Constants.lambd * grad_loss
            return loss
        disc_loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        real_loss = -jnp.mean(optimizer.target(real_batch))
        return optimizer, optimizer.target, disc_loss, real_loss


class GIN(FullyConnectedNetwork):
    '''
    Haven't come up with a nice way to enforce this, but the layer sizes should all be the same.
    Right now, if that's not the case, the Jacobian should be nonsingular. I'll print a warning
    if the singular value is small.

    Here we compute the following:
    - we have a batch of samples {y_i} from a putative distribution y ~ g(Y)
    - we pass them through our network, which is H^-1, getting "samples" {x_i = H^-1(y_i)} from our prior
        distribution p(X).
        (the prior distribution should be one of a Gaussian or Logistic according to NICE, who prefer Logistic.
        I'm doing Gaussian for now.
    - then we compute and optimize the log likelihood according to:
            log(g(y)) = log(f(inv(H(y)))) + log(abs(det(inv(jac(y)(z)))))
    - sampling will require computing z ~ P(Z) and then y = H^{-1}(z)
    '''

    @staticmethod
    def train_step(optimizer, batch, latent_distribution):
        def loss_fn(H_inv):
            z = H_inv(batch)

            @jax.vmap
            def log_likelihood(z):
                if latent_distribution == 'gaussian':
                    log_pdf = jnp.sum(norm.logpdf(z))
                elif latent_distribution == 'logistic':
                    log_pdf = jnp.sum(logistic.logpdf(z))
                else:
                    raise NotImplementedError()
                jac = jax.jacfwd(H_inv)(z)
                _, log_det_jac = jnp.linalg.slogdet(jac)
                return log_pdf + log_det_jac
            nll = -log_likelihood(z)
            loss = jnp.sum(nll)
            return loss
        gen_loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, gen_loss

    @flax.nn.module_method
    def apply_inverse(self, x, layer_sizes, leaky=True):
        for i in reversed(range(len(layer_sizes))):
            name = self.get_layer_name(i)
            layer = self.get_param(name)
            kernel = layer['kernel']
            bias = layer['bias']
            if i + 1 < len(layer_sizes):
                x = inv_leaky_relu(x)
            x = inv_dense(x, kernel, bias)
        return x

    @flax.nn.module_method
    def average_log_likelihood(self, x, layer_sizes, leaky=True):
        def inverse(x, layer_sizes, leaky=True):
            for i in reversed(range(len(layer_sizes))):
                name = self.get_layer_name(i)
                layer = self.get_param(name)
                kernel = layer['kernel']
                bias = layer['bias']
                if i + 1 < len(layer_sizes):
                    x = inv_leaky_relu(x)
                x = inv_dense(x, kernel, bias)
            z = x
            return z
        inv = partial(inverse, layer_sizes=layer_sizes, leaky=leaky)
        z = inv(x)
        jac = jax.jacfwd(inv)(x)
        print(z)
        return norm.logpdf(z).sum() + jnp.linalg.slogdet(jac)[1]


class RealNVPNetwork(flax.nn.Module):
    '''
    A Real NVP network
    '''

    def apply(self, z, n_passes, layer_sizes, batch_norm=False, mult_layer=False):
        half = z.shape[-1] // 2
        x = z[..., :half]
        y = z[..., half:]
        for i in range(n_passes):
            if i % 2 == 0:
                x = x
                y = y * jnp.exp(jnp.tanh(FullyConnectedNetwork(x, layer_sizes=layer_sizes, name=f"S_{i}"))) + \
                    FullyConnectedNetwork(x, layer_sizes=layer_sizes, name=f"T_{i}")
            else:
                x = x * jnp.exp(jnp.tanh(FullyConnectedNetwork(y, layer_sizes=layer_sizes, name=f"S_{i}"))) + \
                    FullyConnectedNetwork(y, layer_sizes=layer_sizes, name=f"T_{i}")
                y = y
            if mult_layer:
                diag_x = self.param(f"diag_x_{i}", (1, half), jax.nn.initializers.ones)
                diag_y = self.param(f"diag_y_{i}", (1, half), jax.nn.initializers.ones)
                x = x * diag_x
                y = y * diag_y
                x = leaky_relu(x)
                y = leaky_relu(y)
            if i + 1 < n_passes and batch_norm:
                x = InvertibleBatchNorm(x, axis=-1, name=f"BN_x_{i}")
                y = InvertibleBatchNorm(y, axis=-1, name=f"BN_y_{i}")
        output = jnp.concatenate((x, y), axis=-1)
        return output

    @staticmethod
    def train_step(optimizer, x, state, n_passes, latent_distribution):

        def loss_fn(model):
            @jax.vmap
            def slow_log_likelihood(x):
                z = model(x)
                if latent_distribution == 'gaussian':
                    log_pdf = jnp.sum(norm.logpdf(z))
                elif latent_distribution == 'logistic':
                    log_pdf = jnp.sum(logistic.logpdf(z))
                else:
                    raise NotImplementedError()
                jac = jax.jacfwd(model)(z)
                _, log_det_jac = jnp.linalg.slogdet(jac)
                return log_pdf + log_det_jac

            @jax.vmap
            def fast_log_likelihood(x):
                z = model(x)
                if latent_distribution == 'gaussian':
                    log_pdf = jnp.sum(norm.logpdf(z))
                elif latent_distribution == 'logistic':
                    log_pdf = jnp.sum(logistic.logpdf(z))
                else:
                    raise NotImplementedError()
                x_ = x
                log_det_jac = 0
                for i in range(n_passes):
                    jac = jax.jacfwd(model.apply_layer)(x_, i)
                    x_ = model.apply_layer(x_, i)
                    log_det_jac += jnp.sum(jnp.log(jnp.abs(jnp.diag(jac))))
                return log_pdf + log_det_jac
            with flax.nn.stateful(state) as new_state:
                nll = -fast_log_likelihood(x)
                # nll = -slow_log_likelihood(x)
            loss = jnp.mean(nll)
            return loss, new_state
        (nice_loss, new_state), grad = jax.value_and_grad(loss_fn, has_aux=True)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, nice_loss, new_state

    @staticmethod
    @jax.jit
    def supervised_train_step(optimizer, X, Y, state, latent_distribution):
        def loss_fn(model):
            with flax.nn.stateful(state) as new_state:
                Y_hat = model(X)
            return jnp.mean(jnp.square(Y - Y_hat)), new_state
        (mse_loss, new_state), grad = jax.value_and_grad(loss_fn, has_aux=True)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, mse_loss, new_state

    @staticmethod
    @jax.jit
    def wasserstein_train_step(optimizer, z, critic):
        def loss_fn(model):
            fake_data = model(z)
            logits = critic(fake_data)
            # we want the generator to maximize the output of the critic
            loss = -jnp.mean(logits)
            return loss
        gen_loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, gen_loss

    @flax.nn.module_method
    def apply_layer(self, data, i, n_passes, layer_sizes, batch_norm=False, mult_layer=False):
        half = data.shape[-1] // 2
        y = data[..., half:]
        x = data[..., :half]
        s_name = f"S_{i}"
        t_name = f"T_{i}"
        s_params = self.get_param(s_name)
        t_params = self.get_param(t_name)
        s_net = FullyConnectedNetwork.partial(layer_sizes=layer_sizes)
        t_net = FullyConnectedNetwork.partial(layer_sizes=layer_sizes)
        if i % 2 == 0:
            x = x
            y = y * jnp.exp(jnp.tanh(s_net.call(s_params, x))) + t_net.call(t_params, x)
            # y = y * sigmoid(s_net.call(s_params, x) + 2) + t_net.call(t_params, x)
        else:
            x = x * jnp.exp(jnp.tanh(s_net.call(s_params, y))) + t_net.call(t_params, y)
            y = y
        if mult_layer:
            diag_x = self.param(f"diag_x_{i}", (1, half), jax.nn.initializers.ones)
            diag_y = self.param(f"diag_y_{i}", (1, half), jax.nn.initializers.ones)
            x = x * diag_x
            y = y * diag_y
            x = leaky_relu(x)
            y = leaky_relu(y)
        if i + 1 < n_passes and batch_norm:
            x = InvertibleBatchNorm(x, axis=-1, name=f"BN_x_{i}")
            y = InvertibleBatchNorm(y, axis=-1, name=f"BN_y_{i}")

    @flax.nn.module_method
    def apply_inverse_layer(self, data, i, n_passes, layer_sizes, batch_norm=False, mult_layer=False):
        half = data.shape[-1] // 2
        y = data[..., half:]
        x = data[..., :half]
        collection = flax.nn.get_state().as_dict()
        if i + 1 < n_passes and batch_norm:
            x_name = f"BN_x_{i}"
            y_name = f"BN_y_{i}"
            x_bn_params = self.get_param(x_name)
            y_bn_params = self.get_param(y_name)
            x = inv_batch_norm(x, x_bn_params, x_collection)
            y = inv_batch_norm(y, y_bn_params, y_collection)
        if mult_layer:
            x = inv_leaky_relu(x)
            y = inv_leaky_relu(y)
            x_name = f"diag_epsilonx_{i}"
            y_name = f"diag_y_{i}"
            diag_x = self.get_param(x_name)
            diag_y = self.get_param(y_name)
            x = x / diag_x
            y = y / diag_y
        s_name = f"S_{i}"
        t_name = f"T_{i}"
        s_params = self.get_param(s_name)
        t_params = self.get_param(t_name)
        s_net = FullyConnectedNetwork.partial(layer_sizes=layer_sizes)
        t_net = FullyConnectedNetwork.partial(layer_sizes=layer_sizes)
        if i % 2 == 0:
            x = x
            y = (y - t_net.call(t_params, x)) * jnp.exp(-jnp.tanh(s_net.call(s_params, x)))
        else:
            y = y
            x = (x - t_net.call(t_params, y)) * jnp.exp(-jnp.tanh(s_net.call(s_params, y)))
        out = jnp.concatenate((x, y), axis=-1).flatten()
        return out

    @flax.nn.module_method
    def apply_inverse(self, x, n_passes, layer_sizes, batch_norm=False, mult_layer=False):
        half = x.shape[-1] // 2
        y = x[..., half:]
        x = x[..., :half]
        for i in reversed(range(n_passes)):
            if i + 1 < n_passes and batch_norm:
                collection = flax.nn.get_state().as_dict()
                x_name = f"BN_x_{i}"
                y_name = f"BN_y_{i}"
                x_bn_params = self.get_param(x_name)
                y_bn_params = self.get_param(y_name)
                x_collection = collection["/" + x_name]
                y_collection = collection["/" + y_name]
                x = inv_batch_norm(x, x_bn_params, x_collection)
                y = inv_batch_norm(y, y_bn_params, y_collection)
            if mult_layer:
                x = inv_leaky_relu(x)
                y = inv_leaky_relu(y)
                x_name = f"diag_x_{i}"
                y_name = f"diag_y_{i}"
                diag_x = self.get_param(x_name)
                diag_y = self.get_param(y_name)
                x = x / diag_x
                y = y / diag_y
            s_name = f"S_{i}"
            t_name = f"T_{i}"
            s_params = self.get_param(s_name)
            t_params = self.get_param(t_name)
            s_net = FullyConnectedNetwork.partial(layer_sizes=layer_sizes)
            t_net = FullyConnectedNetwork.partial(layer_sizes=layer_sizes)
            if i % 2 == 0:
                x = x
                y = (y - t_net.call(t_params, x)) * jnp.exp(-jnp.tanh(s_net.call(s_params, x)))
            else:
                y = y
                x = (x - t_net.call(t_params, y)) * jnp.exp(-jnp.tanh(s_net.call(s_params, y)))

        return jnp.concatenate((x, y), axis=-1)


class RealNVP2Network(flax.nn.Module):
    def apply(self, data, n_passes, layer_sizes, batch_norm=False, nonlinearity=False):
        half = data.shape[-1] // 2
        x = data[..., :half]
        y = data[..., half:]
        for i in range(n_passes):
            if i % 2 == 0:
                x = x
                scale_shift = FullyConnectedNetwork(x, layer_sizes=layer_sizes, batch_norm=batch_norm, name=f"ST_{i}")
                # scale = sigmoid(scale_shift[..., :half] + 10.)
                scale = jnp.exp(jnp.tanh(scale_shift[..., :half]))
                shift = scale_shift[..., half:]
                y = y * scale + shift
                if nonlinearity and i + 2 < n_passes:
                    y = leaky_relu(y)
            else:
                scale_shift = FullyConnectedNetwork(y, layer_sizes=layer_sizes, batch_norm=batch_norm, name=f"ST_{i}")
                scale = jnp.exp(jnp.tanh(scale_shift[..., :half]))
                shift = scale_shift[..., half:]
                x = x * scale + shift
                y = y
                if nonlinearity and i + 2 < n_passes:
                    x = leaky_relu(x)
            if i + 1 < n_passes and batch_norm:
            # if False:
                # x = InvertibleBatchNorm(x, axis=-1, name=f"BN_x_{i}")
                # y = InvertibleBatchNorm(y, axis=-1, name=f"BN_y_{i}")
                x = flax.nn.BatchNorm(x, name=f"BN_x_{i}")
                y = flax.nn.BatchNorm(y, name=f"BN_y_{i}")
        output = jnp.concatenate((x, y), axis=-1)
        return output

    @flax.nn.module_method
    def average_log_likelihood(self, data, n_passes, layer_sizes, batch_norm=False, nonlinearity=False):
        @jax.vmap
        def fast_log_likelihood(x):
            x_ = x
            log_det_jac = 0

            def apply_layer(data_in, i):
                half = data_in.shape[-1] // 2
                x = data_in[..., :half]
                y = data_in[..., half:]
                st_name = f"ST_{i}"
                st_params = self.get_param(st_name)
                st_net = FullyConnectedNetwork.partial(layer_sizes=layer_sizes)
                if i % 2 == 0:
                    x = x
                    scale_shift = st_net.call(st_params, x)
                    scale = jnp.exp(jnp.tanh(scale_shift[..., :half]))
                    shift = scale_shift[..., half:]
                    y = y * scale + shift
                    if nonlinearity and i + 2 < n_passes:
                        y = leaky_relu(y)
                else:
                    y = y
                    scale_shift = st_net.call(st_params, y)
                    scale = jnp.exp(jnp.tanh(scale_shift[..., :half]))
                    shift = scale_shift[..., half:]
                    x = x * scale + shift
                    if nonlinearity and i + 2 < n_passes:
                        x = leaky_relu(x)
                if i + 1 < n_passes and batch_norm:
                    x_name = f"BN_x_{i}"
                    y_name = f"BN_y_{i}"
                    x_bn_params = self.get_param(x_name)
                    y_bn_params = self.get_param(y_name)
                    # x_collection = collection["/RealNVP/" + x_name]
                    # y_collection = collection["/RealNVP/" + y_name]
                    x = flax.nn.BatchNorm.call(x_bn_params, x)
                    y = flax.nn.BatchNorm.call(y_bn_params, y)
                # if False:
                    # x = InvertibleBatchNorm(x, axis=-1, name=f"BN_x_{i}")
                    # y = InvertibleBatchNorm(y, axis=-1, name=f"BN_y_{i}")
                output = jnp.concatenate((x, y), axis=-1)
                return output

            for i in range(n_passes):
                jac = jax.jacfwd(apply_layer)(x_, i)
                x_ = apply_layer(x_, i)
                log_det_jac += jnp.sum(jnp.log(jnp.abs(jnp.diag(jac))))
            z = x_
            log_pdf = jnp.sum(norm.logpdf(z))
            return log_pdf + log_det_jac

        ll = fast_log_likelihood(data)
        return ll.mean()

    @flax.nn.module_method
    def apply_layer(self, data, i, n_passes, layer_sizes, batch_norm=False, nonlinearity=False):
        half = data.shape[-1] // 2
        x = data[..., :half]
        y = data[..., half:]
        st_name = f"ST_{i}"
        st_params = self.get_param(st_name)
        st_net = FullyConnectedNetwork.partial(layer_sizes=layer_sizes)
        if i % 2 == 0:
            x = x
            scale_shift = st_net.call(st_params, x)
            scale = jnp.exp(jnp.tanh(scale_shift[..., :half]))
            shift = scale_shift[..., half:]
            y = y * scale + shift
            if nonlinearity and i + 2 < n_passes:
                y = leaky_relu(y)
        else:
            y = y
            scale_shift = st_net.call(st_params, y)
            scale = jnp.exp(jnp.tanh(scale_shift[..., :half]))
            shift = scale_shift[..., half:]
            x = x * scale + shift
            if nonlinearity and i + 2 < n_passes:
                x = leaky_relu(x)
        if i + 1 < n_passes and batch_norm:
            x_name = f"BN_x_{i}"
            y_name = f"BN_y_{i}"
            x_bn_params = self.get_param(x_name)
            y_bn_params = self.get_param(y_name)
            # x_collection = collection["/RealNVP/" + x_name]
            # y_collection = collection["/RealNVP/" + y_name]
            x = flax.nn.BatchNorm.call(x_bn_params, x)
            y = flax.nn.BatchNorm.call(y_bn_params, y)
            # x = flax.nn.BatchNorm(x)
            # y = flax.nn.BatchNorm(y)
        # if i + 1 < n_passes and batch_norm:
            # x = InvertibleBatchNorm(x, axis=-1, name=f"BN_x_{i}")
            # y = InvertibleBatchNorm(y, axis=-1, name=f"BN_y_{i}")
        output = jnp.concatenate((x, y), axis=-1)
        return output

    @flax.nn.module_method
    def apply_inverse(self, z, n_passes, layer_sizes, batch_norm=False, nonlinearity=False):
        half = z.shape[-1] // 2
        y = z[..., half:]
        x = z[..., :half]
        for i in reversed(range(n_passes)):
            if i + 1 < n_passes and batch_norm:
                collection = flax.nn.get_state().as_dict()
                x_name = f"BN_x_{i}"
                y_name = f"BN_y_{i}"
                x_bn_params = self.get_param(x_name)
                y_bn_params = self.get_param(y_name)
                x_collection = collection["/RealNVP/" + x_name]
                y_collection = collection["/RealNVP/" + y_name]
                x = inv_batch_norm(x, x_bn_params, x_collection)
                y = inv_batch_norm(y, y_bn_params, y_collection)
            st_name = f"ST_{i}"
            st_params = self.get_param(st_name)
            st_net = FullyConnectedNetwork.partial(layer_sizes=layer_sizes)
            if i % 2 == 0:
                if nonlinearity and i + 2 < n_passes:
                    y = inv_leaky_relu(y)
                scale_shift = st_net.call(st_params, x)
                scale = jnp.exp(-jnp.tanh(scale_shift[..., :half]))
                shift = scale_shift[..., half:]
                x = x
                y = (y - shift) * scale
            else:
                if nonlinearity and i + 2 < n_passes:
                    x = inv_leaky_relu(x)
                scale_shift = st_net.call(st_params, y)
                scale = jnp.exp(-jnp.tanh(scale_shift[..., :half]))
                shift = scale_shift[..., half:]
                y = y
                x = (x - shift) * scale

        return jnp.concatenate((x, y), axis=-1)

    @staticmethod
    def train_step(optimizer, x, state, n_passes, latent_distribution):

        def loss_fn(model):
            @jax.vmap
            def slow_log_likelihood(x):
                z = model(x)
                if latent_distribution == 'gaussian':
                    log_pdf = jnp.sum(norm.logpdf(z))
                elif latent_distribution == 'logistic':
                    log_pdf = jnp.sum(logistic.logpdf(z))
                else:
                    raise NotImplementedError()
                jac = jax.jacfwd(model)(x)
                _, log_det_jac = jnp.linalg.slogdet(jac)
                return log_pdf + log_det_jac

            @jax.vmap
            def fast_log_likelihood(x):
                x_ = x
                log_det_jac = 0
                for i in range(n_passes):
                    jac = jax.jacfwd(model.apply_layer)(x_, i)
                    with flax.nn.stateful(state) as new_state:
                        x_ = model.apply_layer(x_, i)
                    log_det_jac += jnp.sum(jnp.log(jnp.abs(jnp.diag(jac))))
                z = x_
                if latent_distribution == 'gaussian':
                    log_pdf = jnp.sum(norm.logpdf(z))
                elif latent_distribution == 'logistic':
                    log_pdf = jnp.sum(logistic.logpdf(z))
                else:
                    raise NotImplementedError()
                return log_pdf + log_det_jac, new_state

            ll, new_state = fast_log_likelihood(x)
            nll = -ll
            # nll = -slow_log_likelihood(x)
            loss = jnp.mean(nll)
            return loss, new_state
        (nice_loss, new_state), grad = jax.value_and_grad(loss_fn, has_aux=True)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, nice_loss, new_state

    @staticmethod
    @jax.jit
    def supervised_train_step(optimizer, X, Y, state, latent_distribution):
        def loss_fn(model):
            with flax.nn.stateful(state) as new_state:
                X_hat = model(Y)
            return jnp.mean(jnp.square(X - X_hat)), new_state
        (mse_loss, new_state), grad = jax.value_and_grad(loss_fn, has_aux=True)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, mse_loss, new_state


class RealNVPImageNetwork(flax.nn.Module):
    def apply(self, x, n_passes, n_conv):
        assert x.ndim == 3 or x.ndim == 4, f"x has {x.ndim} dimensions and shape {x.shape}, wtf bro"
        even_mask = get_mask(x.shape, False)
        odd_mask = get_mask(x.shape, True)
        for i in range(n_passes):
            mask = even_mask if i % 2 == 0 else odd_mask
            channels = x.shape[-1]
            st = EndoConvNet(x * mask, channel_sizes=[channels * 2] * n_conv, name=f"ST_{i}")
            scale = jnp.exp(jnp.tanh(st[..., :channels]))
            translation = st[..., channels:]
            x = x * mask + (1 - mask) * (x * scale + translation)
        return x

    @staticmethod
    def train_step(optimizer, x, n_passes):

        def loss_fn(model):

            @jax.vmap
            def fast_log_likelihood(x):
                x_ = x
                log_det_jac = 0
                for i in range(n_passes):
                    x_, log_det_jac_layer = model.apply_layer(x_, i)
                    log_det_jac += log_det_jac_layer
                z = x_
                log_pdf = jnp.sum(norm.logpdf(z))
                return log_pdf + log_det_jac
            ll = fast_log_likelihood(x)
            nll = -ll
            loss = jnp.mean(nll)
            return loss
        nll, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, nll

    @flax.nn.module_method
    def apply_layer(self, x, i, n_passes, n_conv):
        channels = x.shape[-1]
        if x.ndim == 3:
            x = x[jnp.newaxis, ...]
        mask = get_mask(x.shape, i % 2 == 1)
        st_name = f"ST_{i}"
        st_params = self.get_param(st_name)
        st_net = EndoConvNet.partial(channel_sizes=[channels * 2] * n_conv)
        st = st_net.call(st_params, x * mask)
        scale = jnp.tanh(st[..., :channels])
        expscale = jnp.exp(scale)
        translation = st[..., channels:]
        x = x * mask + (1 - mask) * (x * expscale + translation)
        log_det_jac = jnp.sum(scale[mask == 0])
        return x, log_det_jac

    @flax.nn.module_method
    def apply_inverse(self, z, n_passes, n_conv):
        assert z.ndim == 4
        channels = z.shape[-1]
        even_mask = get_mask(z.shape, False)
        odd_mask = get_mask(z.shape, True)
        for i in reversed(range(n_passes)):
            mask = even_mask if i % 2 == 0 else odd_mask
            st_name = f"ST_{i}"
            st_params = self.get_param(st_name)
            st_net = EndoConvNet.partial(channel_sizes=[channels * 2] * n_conv)
            st = st_net.call(st_params, z * mask)
            scale = jnp.exp(-jnp.tanh(st[..., :channels]))
            translation = st[..., channels:]
            z = z * mask + (1 - mask) * ((z - translation) * scale)
        return z


class RealNVPImageNetwork2(flax.nn.Module):
    def apply(self, x, n_conv):
        if x.ndim == 3:
            x = x[jnp.newaxis, ...]
        # original_shape = x.shape
        assert x.ndim == 4, f"x has {x.ndim} dimensions and shape {x.shape}, wtf bro"
        batch = x.shape[0]
        n_squeezes = 2
        couplings_per_squeeze = 3
        zs = []
        for i in range(n_squeezes):
            even_mask = get_mask(x.shape, False)
            odd_mask = get_mask(x.shape, True)
            channels = x.shape[-1]
            # first 3 alternating
            for j in range(couplings_per_squeeze):
                mask = even_mask if j % 2 == 0 else odd_mask
                st = EndoResNet(x * mask, channel_sizes=[channels * 2] * n_conv, name=f"ST_{i}_{j}")
                scale = jnp.exp(jnp.tanh(st[..., :channels]))
                translation = st[..., channels:]
                x = x * mask + (1 - mask) * (x * scale + translation)
            x = squeeze_2x2(x)
            even_mask = get_mask(x.shape, False, use_checkerboard=False)
            odd_mask = get_mask(x.shape, True, use_checkerboard=False)
            channels = x.shape[-1]
            for j in range(couplings_per_squeeze):
                mask = even_mask if j % 2 == 0 else odd_mask
                st = EndoResNet(x * mask, channel_sizes=[channels * 2] * n_conv, name=f"ST2_{i}_{j}")
                scale = jnp.exp(jnp.tanh(st[..., :channels]))
                translation = st[..., channels:]
                x = x * mask + (1 - mask) * (x * scale + translation)
            channels = x.shape[-1]
            assert channels % 2 == 0
            half_channels = channels // 2
            zs.append(x[..., half_channels:])
            x = x[..., :half_channels]
        zs.append(x)
        flat_zs = [z.reshape((batch, -1)) for z in zs]
        z = jnp.concatenate(flat_zs, axis=-1)
        return z

    @staticmethod
    def train_step(optimizer, x):
        n_squeezes = 2
        couplings_per_squeeze = 3

        def loss_fn(model):

            @jax.vmap
            def fast_log_likelihood(x):
                x_ = x
                log_det_jac = 0
                zs = []
                for i in range(n_squeezes):
                    for j in range(couplings_per_squeeze):
                        x_, log_det_jac_layer = model.apply_layer(x_, i, j)
                        log_det_jac += log_det_jac_layer
                    x_ = squeeze_2x2(x_)
                    for j in range(couplings_per_squeeze):
                        x_, log_det_jac_layer = model.apply_layer(x_, i, j, checkerboard=False)
                        log_det_jac += log_det_jac_layer
                    half_channels = x_.shape[-1] // 2
                    zs.append(x_[..., half_channels:])
                    x_ = x_[..., :half_channels]
                zs.append(x_)
                flat_zs = [z.flatten() for z in zs]
                z = jnp.concatenate(flat_zs)
                # z = x_
                log_pdf = jnp.sum(norm.logpdf(z))
                return log_pdf + log_det_jac
            ll = fast_log_likelihood(x)
            nll = -ll
            loss = jnp.mean(nll)
            return loss
        nll, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, nll

    @flax.nn.module_method
    def apply_layer(self, x, i, j, n_conv, checkerboard=True):
        channels = x.shape[-1]
        if x.ndim == 3:
            x = x[jnp.newaxis, ...]
        mask = get_mask(x.shape, i % 2 == 1, use_checkerboard=checkerboard)
        st_name = f"ST_{i}_{j}" if checkerboard else f"ST2_{i}_{j}"
        st_params = self.get_param(st_name)
        st_net = EndoResNet.partial(channel_sizes=[channels * 2] * n_conv)
        st = st_net.call(st_params, x * mask)
        scale = jnp.tanh(st[..., :channels])
        expscale = jnp.exp(scale)
        translation = st[..., channels:]
        x = x * mask + (1 - mask) * (x * expscale + translation)
        log_det_jac = jnp.sum(scale[mask == 0])
        return x, log_det_jac

    @flax.nn.module_method
    def apply_inverse(self, z, n_conv):
        assert z.ndim == 4
        n_squeezes = 2
        couplings_per_squeeze = 3
        squeeze_shape = data_shape = z.shape
        squeeze_shapes = [data_shape]
        for i in range(n_squeezes):
            squeeze_shape = (squeeze_shape[0], squeeze_shape[1] // 2, squeeze_shape[2] // 2, squeeze_shape[3] * 2)
            squeeze_shapes.append(squeeze_shape)

        batch = z.shape[0]
        last_squeeze_shape = squeeze_shapes[-1]
        z = z.reshape((batch, last_squeeze_shape[1], last_squeeze_shape[2], -1))
        x = z[..., :last_squeeze_shape[3]]
        z = z[..., last_squeeze_shape[3]:]
        for i in reversed(range(n_squeezes)):
            z = z.reshape((batch, x.shape[1], x.shape[2], -1))
            new_x = z[..., :x.shape[-1]]
            z = z[..., x.shape[-1]:]
            x = jnp.concatenate((x, new_x), axis=-1)
            channels = x.shape[-1]
            even_mask = get_mask(x.shape, False, use_checkerboard=False)
            odd_mask = get_mask(x.shape, True, use_checkerboard=False)
            for j in reversed(range(couplings_per_squeeze)):
                mask = even_mask if i % 2 == 0 else odd_mask
                st_name = f"ST2_{i}_{j}"
                st_params = self.get_param(st_name)
                st_net = EndoConvNet.partial(channel_sizes=[channels * 2] * n_conv)
                st = st_net.call(st_params, x * mask)
                scale = jnp.exp(-jnp.tanh(st[..., :channels]))
                translation = st[..., channels:]
                x = x * mask + (1 - mask) * ((x - translation) * scale)
            x = squeeze_2x2(x, reverse=True)
            even_mask = get_mask(x.shape, False)
            odd_mask = get_mask(x.shape, True)
            channels = x.shape[-1]
            for j in reversed(range(couplings_per_squeeze)):
                mask = even_mask if i % 2 == 0 else odd_mask
                st_name = f"ST_{i}_{j}"
                st_params = self.get_param(st_name)
                st_net = EndoConvNet.partial(channel_sizes=[channels * 2] * n_conv)
                st = st_net.call(st_params, x * mask)
                scale = jnp.exp(-jnp.tanh(st[..., :channels]))
                translation = st[..., channels:]
                x = x * mask + (1 - mask) * ((x - translation) * scale)
        return x


class ConvNVPNetwork(RealNVPNetwork):
    '''
    A Real NVP Network with convolutional architecture
    '''

    def apply(self, z, n_passes, channel_sizes, batch_norm=False, mult_layer=False):
        assert z.shape[-1] == 784 * 2, f"Data has shape {z.shape}, this network currently only supports padded MNIST and 1568d data"  # NOQA
        half = z.shape[-1] // 2
        x = z[..., :half]
        y = z[..., half:]
        for i in range(n_passes):
            x = x.reshape((-1, 28, 28, 1))
            y = y.reshape((-1, 28, 28, 1))
            if i % 2 == 0:
                x = x
                y = y * jnp.exp(jnp.tanh(EndoResNet(x, channel_sizes=channel_sizes, name=f"S_{i}"))) + \
                    EndoResNet(x, channel_sizes=channel_sizes, name=f"T_{i}")
            else:
                x = x * jnp.exp(jnp.tanh(EndoResNet(y, channel_sizes=channel_sizes, name=f"S_{i}"))) + \
                    EndoResNet(y, channel_sizes=channel_sizes, name=f"T_{i}")
                y = y
            if mult_layer:
                diag_x = self.param(f"diag_x_{i}", (1, half), jax.nn.initializers.ones)
                diag_y = self.param(f"diag_y_{i}", (1, half), jax.nn.initializers.ones)
                x = x.reshape((-1, half)) * diag_x
                y = y.reshape((-1, half)) * diag_y
                x = leaky_relu(x)
                y = leaky_relu(y)
            if i + 1 < n_passes and batch_norm:
                x = InvertibleBatchNorm(x.reshape((-1, half)), axis=-1, name=f"BN_x_{i}")
                y = InvertibleBatchNorm(y.reshape((-1, half)), axis=-1, name=f"BN_y_{i}")
        output = jnp.concatenate((x.reshape((-1, half)), y.reshape((-1, half))), axis=-1)
        return output

    @flax.nn.module_method
    def apply_inverse(self, x, n_passes, channel_sizes, batch_norm=False, mult_layer=False):
        half = x.shape[-1] // 2
        y = x[..., half:]
        x = x[..., :half]
        for i in reversed(range(n_passes)):
            if i + i < n_passes and batch_norm:
                collection = flax.nn.get_state().as_dict()
                x_name = f"BN_x_{i}"
                y_name = f"BN_y_{i}"
                x_bn_params = self.get_param(x_name)
                y_bn_params = self.get_param(y_name)
                x_collection = collection["/RealNVP/" + x_name]
                y_collection = collection["/RealNVP/" + y_name]
                x = inv_batch_norm(x, x_bn_params, x_collection)
                y = inv_batch_norm(y, y_bn_params, y_collection)
            if mult_layer:
                x = inv_leaky_relu(x)
                y = inv_leaky_relu(y)
                x_name = f"diag_x_{i}"
                y_name = f"diag_y_{i}"
                diag_x = self.get_param(x_name)
                diag_y = self.get_param(y_name)
                x = x / diag_x
                y = y / diag_y
            s_name = f"S_{i}"
            t_name = f"T_{i}"
            s_params = self.get_param(s_name)
            t_params = self.get_param(t_name)
            s_net = EndoResNet.partial(channel_sizes=channel_sizes)
            t_net = EndoResNet.partial(channel_sizes=channel_sizes)
            x = x.reshape((-1, 28, 28, 1))
            y = y.reshape((-1, 28, 28, 1))
            if i % 2 == 0:
                x = x
                y = (y - t_net.call(t_params, x)) * jnp.exp(-jnp.tanh(s_net.call(s_params, x)))
            else:
                y = y
                x = (x - t_net.call(t_params, y)) * jnp.exp(-jnp.tanh(s_net.call(s_params, y)))
            x = x.reshape((-1, half))
            y = y.reshape((-1, half))
        return jnp.concatenate((x, y), axis=-1)

    @flax.nn.module_method
    def apply_inverse_layer(self, data, i, n_passes, channel_sizes, batch_norm=False, mult_layer=False):
        half = data.shape[-1] // 2
        y = data[..., half:]
        x = data[..., :half]
        collection = flax.nn.get_state().as_dict()
        if i + 1 < n_passes and batch_norm:
            x_name = f"BN_x_{i}"
            y_name = f"BN_y_{i}"
            x_bn_params = self.get_param(x_name)
            y_bn_params = self.get_param(y_name)
            x_collection = collection["/RealNVP/" + x_name]
            y_collection = collection["/RealNVP/" + y_name]
            x = inv_batch_norm(x, x_bn_params, x_collection)
            y = inv_batch_norm(y, y_bn_params, y_collection)
        if mult_layer:
            x = inv_leaky_relu(x)
            y = inv_leaky_relu(y)
            x_name = f"diag_x_{i}"
            y_name = f"diag_y_{i}"
            diag_x = self.get_param(x_name)
            diag_y = self.get_param(y_name)
            x = x / diag_x
            y = y / diag_y
        s_name = f"S_{i}"
        t_name = f"T_{i}"
        s_params = self.get_param(s_name)
        t_params = self.get_param(t_name)
        s_net = EndoResNet.partial(channel_sizes=channel_sizes)
        t_net = EndoResNet.partial(channel_sizes=channel_sizes)
        x = x.reshape((-1, 28, 28, 1))
        y = y.reshape((-1, 28, 28, 1))
        if i % 2 == 0:
            x = x
            y = (y - t_net.call(t_params, x)) * jnp.exp(-jnp.tanh(s_net.call(s_params, x)))
        else:
            y = y
            x = (x - t_net.call(t_params, y)) * jnp.exp(-jnp.tanh(s_net.call(s_params, y)))
        x = x.reshape((-1, half))
        y = y.reshape((-1, half))
        out = jnp.concatenate((x, y), axis=-1).flatten()
        return out


class NICENetwork(flax.nn.Module):
    '''
    A NICE network
    '''

    def apply(self, z, n_passes, layer_sizes, epsilon):
        half = z.shape[-1] // 2
        x = z[..., :half]
        y = z[..., half:]
        for i in range(n_passes):
            y = epsilon * y + FullyConnectedNetwork(x, layer_sizes=layer_sizes, name=f"MLP_{i}")
            x = epsilon * x
        return x, y

    @staticmethod
    def train_step(optimizer, x1, y1, latent_distribution):
        def loss_fn(model):
            x0, y0 = model.apply_inverse(x1, y1)

            @jax.vmap
            def log_likelihood(x0, y0):
                z = jnp.concatenate((x0, y0))
                if latent_distribution == 'gaussian':
                    log_pdf = jnp.sum(norm.logpdf(z))
                elif latent_distribution == 'logistic':
                    log_pdf = jnp.sum(logistic.logpdf(z))
                else:
                    raise NotImplementedError()
                jac = jax.jacfwd(model)(z)
                jac = jnp.concatenate(jac)
                _, log_det_jac = jnp.linalg.slogdet(jac)
                return log_pdf - log_det_jac
            nll = -log_likelihood(x0, y0)
            loss = jnp.sum(nll)
            return loss
        nice_loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, nice_loss

    @flax.nn.module_method
    def apply_inverse(self, x, y, n_passes, layer_sizes, epsilon):
        for i in range(n_passes):
            x = x / epsilon
            name = f"MLP_{i}"
            net_params = self.get_param(name)
            net = FullyConnectedNetwork.partial(layer_sizes=layer_sizes)
            y = (y - net.call(net_params, x)) / epsilon
        return x, y


class PartitionedLinearNetwork(flax.nn.Module):

    def apply(self, batch, n_layers):
        half = batch.shape[-1] // 2
        x = batch[..., :half]
        y = batch[..., half:]
        for i in range(n_layers):
            B = self.param(f'B_{i}', (half,), jax.nn.initializers.ones)
            C = self.param(f'C_{i}', (half,), jax.nn.initializers.ones)
            x1 = x
            y1 = flax.nn.Dense(x, features=half, bias=False, name=f'A_{i}',
                               kernel_init=jax.nn.initializers.normal(stddev=1e-5)) + B * y
            x2 = flax.nn.Dense(y1, features=half, bias=False, name=f'D_{i}',
                               kernel_init=jax.nn.initializers.normal(stddev=1e-5)) + C * x1
            y2 = y1
            E = self.param(f"E_{i}", (half,), jax.nn.initializers.ones)
            F = self.param(f"F_{i}", (half,), jax.nn.initializers.ones)
            x = x2 * E
            y = y2 * F
        return jnp.concatenate((x, y), axis=-1)

    @flax.nn.module_method
    def apply_inverse(self, batch, n_layers):
        half = batch.shape[-1] // 2
        x = batch[..., :half]
        y = batch[..., half:]
        for i in reversed(range(n_layers)):
            x2 = x / self.get_param(f"E_{i}")
            y2 = y / self.get_param(f"F_{i}")
            y1 = y2
            x1 = (x2 - y1 @ self.get_param(f"D_{i}")['kernel']) / self.get_param(f"C_{i}")
            x = x1
            y = (y1 - x @ self.get_param(f"A_{i}")['kernel']) / self.get_param(f"B_{i}")

        return jnp.concatenate((x, y), axis=-1)

    @flax.nn.module_method
    def get_matrix(self, n_layers, layer=None):
        dim = self.get_param("A_0")['kernel'].shape[0]
        if layer is not None:
            i = layer
            block_0 = jnp.block([[jnp.eye(dim), jnp.zeros((dim, dim))],
                                 [self.get_param(f"A_{i}")['kernel'].T, jnp.diag(self.get_param(f"B_{i}"))]])
            block_1 = jnp.block([[jnp.diag(self.get_param(f"C_{i}")), self.get_param(f"D_{i}")['kernel'].T],
                                 [jnp.zeros((dim, dim)), jnp.eye(dim)]])
            diag = jnp.diag(jnp.concatenate((self.get_param(f"E_{i}"), self.get_param(f"F_{i}"))))
            return diag @ block_1 @ block_0
        matrix = jnp.eye(dim * 2)
        for i in range(n_layers):
            block_0 = jnp.block([[jnp.eye(dim), jnp.zeros((dim, dim))],
                                 [self.get_param(f"A_{i}")['kernel'].T, jnp.diag(self.get_param(f"B_{i}"))]])
            block_1 = jnp.block([[jnp.diag(self.get_param(f"C_{i}")), self.get_param(f"D_{i}")['kernel'].T],
                                 [jnp.zeros((dim, dim)), jnp.eye(dim)]])
            diag = jnp.diag(jnp.concatenate((self.get_param(f"E_{i}"), self.get_param(f"F_{i}"))))
            matrix = diag @ block_1 @ block_0 @ matrix
        return matrix

    @staticmethod
    @jax.jit
    def supervised_train_step(optimizer, X, Y):
        def loss_fn(model):
            Y_hat = model(X)
            return jnp.sum(jnp.square(Y - Y_hat))
        mse_loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, mse_loss

    @staticmethod
    def train_step(optimizer, batch, train=True):
        def loss_fn(model):
            # invert model to get values in latent space
            z_batch = model.apply_inverse(batch)

            def log_likelihood(z):
                # compute log likelihood in latent space
                log_pdf = jnp.sum(norm.logpdf(z), axis=1)
                # compute the jacobian of the (forward) model at z
                jac_fn = jax.jacfwd(model)

                jac = jnp.sum(jac_fn(z), axis=2)
                # compute the logdet of the jacobian
                _, log_det_jac = jnp.linalg.slogdet(jac)
                # subtract because we wanted the logdet of the inverse
                return log_pdf - log_det_jac
            nll = -log_likelihood(z_batch)
            loss = jnp.mean(nll)
            return loss
        if not train:
            return optimizer, optimizer.target, loss_fn(optimizer.target)
        nll_loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, nll_loss

class NICENetwork(flax.nn.Module):
    '''
    A NICE network
    '''

    def apply(self, z, n_passes, layer_sizes, epsilon):
        half = z.shape[-1] // 2
        x = z[..., :half]
        y = z[..., half:]
        for i in range(n_passes):
            y = epsilon * y + FullyConnectedNetwork(x, layer_sizes=layer_sizes, name=f"MLP_{i}")
            x = epsilon * x
        return x, y

    @staticmethod
    def train_step(optimizer, x1, y1, latent_distribution):
        def loss_fn(model):
            x0, y0 = model.apply_inverse(x1, y1)

            @jax.vmap
            def log_likelihood(x0, y0):
                z = jnp.concatenate((x0, y0))
                if latent_distribution == 'gaussian':
                    log_pdf = jnp.sum(norm.logpdf(z))
                elif latent_distribution == 'logistic':
                    log_pdf = jnp.sum(logistic.logpdf(z))
                else:
                    raise NotImplementedError()
                jac = jax.jacfwd(model)(z)
                jac = jnp.concatenate(jac)
                _, log_det_jac = jnp.linalg.slogdet(jac)
                return log_pdf - log_det_jac
            nll = -log_likelihood(x0, y0)
            loss = jnp.sum(nll)
            return loss
        nice_loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, nice_loss

    @flax.nn.module_method
    def apply_inverse(self, x, y, n_passes, layer_sizes, epsilon):
        for i in range(n_passes):
            x = x / epsilon
            name = f"MLP_{i}"
            net_params = self.get_param(name)
            net = FullyConnectedNetwork.partial(layer_sizes=layer_sizes)
            y = (y - net.call(net_params, x)) / epsilon
        return x, y


class PartitionedLinearNetwork(flax.nn.Module):

    def apply(self, batch, n_layers):
        half = batch.shape[-1] // 2
        x = batch[..., :half]
        y = batch[..., half:]
        for i in range(n_layers):
            B = self.param(f'B_{i}', (half,), jax.nn.initializers.ones)
            C = self.param(f'C_{i}', (half,), jax.nn.initializers.ones)
            x1 = x
            y1 = flax.nn.Dense(x, features=half, bias=False, name=f'A_{i}',
                               kernel_init=jax.nn.initializers.normal(stddev=1e-5)) + B * y
            x2 = flax.nn.Dense(y1, features=half, bias=False, name=f'D_{i}',
                               kernel_init=jax.nn.initializers.normal(stddev=1e-5)) + C * x1
            y2 = y1
            E = self.param(f"E_{i}", (half,), jax.nn.initializers.ones)
            F = self.param(f"F_{i}", (half,), jax.nn.initializers.ones)
            x = x2 * E
            y = y2 * F
        return jnp.concatenate((x, y), axis=-1)

    @flax.nn.module_method
    def apply_inverse(self, batch, n_layers):
        half = batch.shape[-1] // 2
        x = batch[..., :half]
        y = batch[..., half:]
        for i in reversed(range(n_layers)):
            x2 = x / self.get_param(f"E_{i}")
            y2 = y / self.get_param(f"F_{i}")
            y1 = y2
            x1 = (x2 - y1 @ self.get_param(f"D_{i}")['kernel']) / self.get_param(f"C_{i}")
            x = x1
            y = (y1 - x @ self.get_param(f"A_{i}")['kernel']) / self.get_param(f"B_{i}")

        return jnp.concatenate((x, y), axis=-1)

    @flax.nn.module_method
    def get_matrix(self, n_layers, layer=None):
        dim = self.get_param("A_0")['kernel'].shape[0]
        if layer is not None:
            i = layer
            block_0 = jnp.block([[jnp.eye(dim), jnp.zeros((dim, dim))],
                                 [self.get_param(f"A_{i}")['kernel'].T, jnp.diag(self.get_param(f"B_{i}"))]])
            block_1 = jnp.block([[jnp.diag(self.get_param(f"C_{i}")), self.get_param(f"D_{i}")['kernel'].T],
                                 [jnp.zeros((dim, dim)), jnp.eye(dim)]])
            diag = jnp.diag(jnp.concatenate((self.get_param(f"E_{i}"), self.get_param(f"F_{i}"))))
            return diag @ block_1 @ block_0
        matrix = jnp.eye(dim * 2)
        for i in range(n_layers):
            block_0 = jnp.block([[jnp.eye(dim), jnp.zeros((dim, dim))],
                                 [self.get_param(f"A_{i}")['kernel'].T, jnp.diag(self.get_param(f"B_{i}"))]])
            block_1 = jnp.block([[jnp.diag(self.get_param(f"C_{i}")), self.get_param(f"D_{i}")['kernel'].T],
                                 [jnp.zeros((dim, dim)), jnp.eye(dim)]])
            diag = jnp.diag(jnp.concatenate((self.get_param(f"E_{i}"), self.get_param(f"F_{i}"))))
            matrix = diag @ block_1 @ block_0 @ matrix
        return matrix

    @staticmethod
    @jax.jit
    def supervised_train_step(optimizer, X, Y):
        def loss_fn(model):
            Y_hat = model(X)
            return jnp.sum(jnp.square(Y - Y_hat))
        mse_loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, mse_loss

    @staticmethod
    def train_step(optimizer, batch, train=True):
        def loss_fn(model):
            # invert model to get values in latent space
            z_batch = model.apply_inverse(batch)

            def log_likelihood(z):
                # compute log likelihood in latent space
                log_pdf = jnp.sum(norm.logpdf(z), axis=1)
                # compute the jacobian of the (forward) model at z
                jac_fn = jax.jacfwd(model)

                jac = jnp.sum(jac_fn(z), axis=2)
                # compute the logdet of the jacobian
                _, log_det_jac = jnp.linalg.slogdet(jac)
                # subtract because we wanted the logdet of the inverse
                return log_pdf - log_det_jac
            nll = -log_likelihood(z_batch)
            loss = jnp.mean(nll)
            return loss
        if not train:
            return optimizer, optimizer.target, loss_fn(optimizer.target)
        nll_loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, optimizer.target, nll_loss