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
                epsilon = self.param('epsilon', (1,), jax.nn.initializers.ones) 
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
