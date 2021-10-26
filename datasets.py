from abc import ABC, abstractmethod
import numpy as np
from jax import random, numpy as jnp, jit
import jax
from jax.scipy.stats import norm, multivariate_normal
import pickle as pkl
import matplotlib.pyplot as plt
from networks import FullyConnectedNetwork
from jax.nn.initializers import normal


class Dataset(ABC):

    @property
    def is_epochs(self):
        raise NotImplementedError()

    @abstractmethod
    def plot_batch(self, batch, fn=None):
        pass

    @property
    def shape(self):
        raise NotImplementedError()

    @property
    def dimension(self):
        return np.product(self.shape)

    @abstractmethod
    def save(self, fn):
        pass
class DistributionDataset(Dataset):
    def __init__(self, seed):
        self.key = random.PRNGKey(seed)

    @property
    def is_epochs(self):
        return False

    @property
    @abstractmethod
    def get_batch(self, size):
        pass

    @abstractmethod
    def score_batch(self, batch):
        pass

    def get_key(self):
        self.key, key = random.split(self.key)
        return key


class SphereDataset(DistributionDataset):
    def __init__(self, seed, dimension=3, padding_dimension=0):
        super().__init__(seed)
        self.R = 1
        self.dim = dimension
        self.padding_dim = padding_dimension
        self.ndim = dimension + padding_dimension

    @property
    def shape(self):
        return (self.ndim,)

    def score_batch(self, batch):
        real = batch[:, :self.dim]
        padding = batch[:, self.dim:]
        score = (jnp.linalg.norm(real, axis=1) - 1)**2
        padding_score = (jnp.linalg.norm(padding, axis=1))**2
        return {"Sphere Error": score.mean(),
                "Padding Error": padding_score.mean()}

    def get_batch(self, size, return_latents=False):
        norm_samps = random.normal(self.get_key(), (size, self.dim))
        radiuses = jnp.linalg.norm(norm_samps, axis=1, keepdims=True)
        samps = norm_samps / radiuses
        padding = jnp.zeros((size, self.padding_dim))
        samps = jnp.concatenate([samps, padding], axis=1)

        if return_latents:
            return samps, None
        return samps

    def plot_batch(self, batch, fn):

        if self.dim == 2:
            plt.scatter(batch[:, 0], batch[:, 1])
            plt.title("Sphere with dimension " + str(self.dim) + " and padding " + str(self.padding_dim))
            plt.savefig(fn)
            plt.close()

            #elif self.dim == 3:
            #    fig = plt.figure()
            #    ax = fig.add_subplot(projection='3d')
            #    ax.scatter(batch[:, 0], batch[:, 1], batch[:, 2])
            #    ax.set_title("Sphere with dimension " + str(self.dim) + " and padding " + str(self.padding_dim))
            #    fig.savefig(fn)
        """
        else:
            plt.plot(np.sort(np.linalg.norm(batch, axis=1)))
            plt.ylabel('Norm of points')
            plt.title("Sphere with dimension " + str(self.dim) + " and padding " + str(self.padding_dim))
            plt.savefig(fn)
            plt.close()
        #plt.show(block=False)
        """

    def save(self, fn):
        pass

    def load(self, fn):
        pass


class GaussianDataset(DistributionDataset):
    def __init__(self, seed, dimension=3, padding_dimension=0, noise_level = 0.01):
        super().__init__(seed)
        self.dim = dimension
        self.padding_dim = padding_dimension
        self.ndim = dimension + padding_dimension
        self.noise_level = noise_level

    @property
    def shape(self):
        return (self.ndim,)

    def score_batch(self, batch):
        samps = batch[:, :self.dim]
        padding= batch[:, self.dim:]
        mse = jnp.mean(jnp.sum(jnp.square(padding), axis=1))
        cov_hat = jnp.cov(batch.T)
        real_batch = self.get_batch(batch.shape[0])
        cov_gt = jnp.cov(real_batch.T)
        w_ht, v_ht = jnp.linalg.eigh(cov_hat)
        # w_gt, v_gt = jnp.linalg.eigh(cov_gt)
        w_gt = np.ones_like(w_ht)

        return {"Squared Norm of padding dimensions": mse, "ground truth eigenvalue":w_gt,
                "learnt eigenvalue":w_ht}

    def get_batch(self, size, return_latents=False):
        norm_samps = random.normal(self.get_key(), (size, self.dim))
        noise_mean = jnp.zeros((self.padding_dim,))
        noise_cov = jnp.eye(self.padding_dim) * self.noise_level
        if self.noise_level > 0:
            padding = random.multivariate_normal(self.get_key(), mean=noise_mean,
                                             cov=noise_cov, shape=(size,))
        else:
            padding = jnp.zeros((size, self.padding_dim))
        samps = jnp.concatenate([norm_samps, padding], axis=1)
        if return_latents:
            return samps, None
        return samps

    def plot_batch(self, batch, fn):
        if self.dim == 2:
            plt.scatter(batch[:, 0], batch[:, 1])
            #elif self.dim == 3:
            #fig = plt.figure()
            #ax = fig.add_subplot(projection='3d')
            #ax.scatter(batch[:, 0], batch[:, 1], batch[:, 2])
        else:
            plt.plot(np.sort(np.linalg.norm(batch, axis=1)))
            plt.ylabel('Norm of points')
        plt.title("Gaussian with dimension {self.dim} and padding {self.padding_dim}")
        plt.savefig(fn)
        plt.show(block=False)
        plt.close()

    def save(self, fn):
        pass

    def load(self, fn):
        pass


class LinearGaussianDataset(DistributionDataset):
    def __init__(self, seed, dimension=3, intrinsic_dimension=3, padding_dimension=0, var_added=0.):
        super().__init__(seed)
        self.dim = dimension
        self.intrinsic_dim = intrinsic_dimension
        self.padding_dim = padding_dimension
        self.ndim = dimension + padding_dimension
        self.var_added = var_added
        det_eps = 1e-4
        mat_key = self.get_key()
        mat = random.normal(mat_key, (dimension, intrinsic_dimension))
        rank = jnp.linalg.matrix_rank(mat)
        while rank != min(self.dim, self.intrinsic_dim):
            mat_key = self.get_key()
            mat = random.normal(mat_key, (dimension, intrinsic_dimension))
            rank = jnp.linalg.matrix_rank(mat)

        self.A = mat
        self.transformed_cov = self.A @ self.A.T

    def get_batch(self, size, return_latents=False):
        x_key = self.get_key()
        X = random.normal(x_key, (size, self.intrinsic_dim))
        Y = (self.A @ X.T).T
        padding = jnp.zeros((size, self.padding_dim))
        Y = jnp.concatenate([Y, padding], axis=1)
        if self.var_added > 0:
            noise_key = self.get_key()
            noise = random.normal(noise_key, (size, self.ndim)) * jnp.sqrt(self.var_added)
            Y += noise
        if return_latents:
            return Y, None
        return Y

    @property
    def shape(self):
        return (self.ndim,)

    def score_batch(self, batch):
        samps = batch[:, :self.dim]
        padding= batch[:, self.dim:]
        # cov_hat = jnp.cov(samps.T)
        mse = jnp.mean(jnp.sum(jnp.square(padding), axis=1))
        return {"Squared Norm of padding dimensions": mse}

    def plot_batch(self, batch, fn):

        if self.dim == 2:
            plt.scatter(batch[:, 0], batch[:, 1])
            #elif self.dim == 3:
            #fig = plt.figure()
            #ax = fig.add_subplot(projection='3d')
            #ax.scatter(batch[:, 0], batch[:, 1], batch[:, 2])
        else:
            plt.plot(np.sort(np.linalg.norm(batch, axis=1)))
            plt.ylabel('Norm of points')
        plt.title("Gaussian with dimension {self.dim} and padding {self.padding_dim}")
        plt.savefig(fn)
        plt.show(block=False)
        plt.close()

    def save(self, fn):
        pass

    def load(self, fn):
        pass

class SigmoidDataset(DistributionDataset):
    def __init__(self, seed, dimension=3, padding_dimension=0):
        super().__init__(seed)
        self.dim = dimension
        self.padding_dim = padding_dimension
        self.ndim = dimension + padding_dimension + 1
        mat_key = self.get_key()
        mat = random.normal(mat_key, (self.dim, 1))
        self.A = mat

    def get_batch(self, size, return_latents=False):
        x_key = self.get_key()
        z = random.normal(x_key, (size, self.dim))
        sig = jax.nn.sigmoid(jnp.dot(z, self.A))
        padding = jnp.zeros((size, self.padding_dim))
        Y = jnp.concatenate([z, sig, padding], axis=1)

        if return_latents:
            return Y, None
        return Y

    @property
    def shape(self):
        return (self.ndim,)

    def score_batch(self, batch):
        codomain_hat = batch[:, self.dim]
        codomain = jnp.dot(batch[:, :self.dim], self.A)
        manifold_error = jnp.mean(jnp.square((codomain_hat - codomain)))
        padding= batch[:, self.dim + 1:]
        mse = jnp.mean(jnp.sum(jnp.square(padding), axis=1))
        return {"Squared Norm of Padding Dimensions": mse, "Squared Norm of Manifold Dimension": manifold_error}

    def plot_batch(self, batch, fn):
        pass

    def save(self, fn):
        pass

    def load(self, fn):
        pass
