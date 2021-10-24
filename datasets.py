from abc import ABC, abstractmethod
import numpy as np
from jax import random, numpy as jnp, jit
import jax
from jax.scipy.stats import norm, multivariate_normal
from scipy.linalg import toeplitz
import flax
import torch
import pickle as pkl
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
from scipy.stats import ortho_group
import sklearn.datasets
from utils import img_tile, split_layer_sizes, sin_theta_distance
from networks import FullyConnectedNetwork, RealNVP2Network, GIN, SquareActivationNetwork
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


class EpochDataset(Dataset):

    @property
    def is_epochs(self):
        return True

    @property
    @abstractmethod
    def train_dataloader(self):
        pass

    @property
    @abstractmethod
    def test_dataloader(self):
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


class SwissRollDataset(DistributionDataset):
    def __init__(self, seed):
        np.random.seed(seed)

    def get_batch(self, size, return_latents=False):
        data = sklearn.datasets.make_swiss_roll(n_samples=size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        if return_latents:
            return data, None
        return data

    def score_batch(self, batch):
        return None

    def plot_batch(self, batch, fn=None):
        plt.hist2d(x=batch[:, 0], y=batch[:, 1], bins=50, range=[[-3, 3], [-3, 3]])
        plt.savefig(fn)
        plt.show(block=False)
        return

    @property
    def shape(self):
        return (2,)

    def save(self, fn):
        pass

    def load(self, fn):
        pass


class TwoMoonsDataset(DistributionDataset):

    def __init__(self, seed, d, noise_level=0.1):
        np.random.seed(seed)
        self.noise_level = noise_level
        self.padding_dimension = d

    def get_batch(self, size, return_latents=False):
        data = sklearn.datasets.make_moons(n_samples=size, noise=self.noise_level)[0]
        data = data.astype("float32")
        data *= 2
        data += np.array([-1, -0.2])
        if self.padding_dimension > 0:
            paddings = jnp.zeros((size, self.padding_dimension - 2))
            data = jnp.concatenate((data, paddings), axis=1)
        if return_latents:
            return data, None
        return data

    def score_batch(self, batch):
        return None

    def plot_batch(self, batch, fn=None):
        plt.hist2d(x=batch[:, 0], y=batch[:, 1], bins=50, range=[[-5, 5], [-5, 5]])
        plt.title("2moon with padding dim {self.padding_dimension}, noise level {self.noise_level}")
        plt.savefig(fn)
        plt.show(block=False)
        return

    @property
    def shape(self):
        return (self.padding_dimension + 2,)

    def save(self, fn):
        pass

    def load(self, fn):
        pass


class CheckerboardDataset(DistributionDataset):
    def __init__(self, seed):
        np.random.seed(seed)

    def get_batch(self, size, return_latents=False):
        x1 = np.random.rand(size) * 4 - 2
        x2_ = np.random.rand(size) - np.random.randint(0, 2, size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        data = np.concatenate([x1[:, None], x2[:, None]], 1) * 2
        if return_latents:
            return data, None
        return data

    def score_batch(self, batch):
        return None

    def plot_batch(self, batch, fn=None):
        plt.hist2d(x=batch[:, 0], y=batch[:, 1], bins=50, range=[[-5, 5], [-5, 5]])
        plt.savefig(fn)
        plt.show(block=False)
        return

    @property
    def shape(self):
        return (2,)

    def save(self, fn):
        pass

    def load(self, fn):
        pass


class SmallGANDataset(DistributionDataset):
    def __init__(self, seed, dimension):
        super().__init__(seed)
        self.latent_dimension = dimension
        self.output_dimension = dimension
        layer_sizes = [dimension * 2, dimension // 2, dimension]
        fc_module = FullyConnectedNetwork.partial(layer_sizes=layer_sizes)
        init_key, self.key = random.split(self.key)
        _, initial_params = fc_module.init_by_shape(init_key, [(self.latent_dimension,)])
        self.model = flax.nn.Model(fc_module, initial_params)

    def score_batch(self, batch):
        return None

    def get_batch(self, size, return_latents=False):
        latent_key, self.key = random.split(self.key)
        latents = random.normal(latent_key, (size, self.latent_dimension))
        samples = self.model(latents)
        if return_latents:
            return samples, latents
        return samples

    def plot_batch(self, batch, fn):
        pass

    @property
    def shape(self):
        return (self.output_dimension,)

    def save(self, fn):
        data = flax.serialization.to_state_dict(self.model)
        with open(fn, "wb") as f:
            pkl.dump(data, f)

    def load(self, fn):
        with open(fn, "rb") as f:
            data = pkl.load(f)
        self.model = flax.serialization.from_state_dict(self.model, data)


class SquareDataset(DistributionDataset):
    def __init__(self, seed, dimension, layer_sizes, epsilon, coefficient = True):
        super().__init__(seed)
        layer_sizes = split_layer_sizes(layer_sizes)
        self.latent_dimension = dimension
        print("data latent dimension ", self.latent_dimension)
        self.output_dimension = layer_sizes[-1]
        self.layer_sizes = layer_sizes
        init_key, self.key = random.split(self.key)
        self.epsilon = epsilon
        print("coefficient: ", coefficient)
        fc_module = FullyConnectedNetwork.partial(layer_sizes=self.layer_sizes, coefficient=coefficient, datasets=True)
        init_key, self.key = random.split(self.key)
        _, initial_params = fc_module.init_by_shape(init_key, [(self.latent_dimension,)])
        self.model = flax.nn.Model(fc_module, initial_params)

    def score_batch(self, batch):
        return None  #jax.vmap(self.model.average_log_likelihood)(batch).mean()

    def get_batch(self, size, return_latents=False, latents = None):
        if latents is None:
            latent_key, self.key = random.split(self.key)
            latents = random.normal(latent_key, (size, self.latent_dimension))

        # samples = self.model(latents)
        #samples = latents ** 2
        noise_key = self.get_key()
        stdev = jnp.exp(self.epsilon / 2)
        noise = random.normal(noise_key, (size, self.output_dimension)) * stdev
        samples = self.model(latents**2) + noise #self.model(samples)
        if return_latents:
            return samples, None
        return samples

    def plot_batch(self, batch, fn):
        if self.output_dimension == 2:
            plt.hist2d(x=batch[:, 0], y=batch[:, 1], bins=50)
            plt.savefig(fn)
            plt.show(block=False)

    @property
    def shape(self):
        return (self.output_dimension,)

    def save(self, fn):
        data = flax.serialization.to_state_dict(self.model)
        with open(fn, "wb") as f:
            pkl.dump(data, f)

    def load(self, fn):
        with open(fn, "rb") as f:
            data = pkl.load(f)
        self.model = flax.serialization.from_state_dict(self.model, data)

class GeneratorDataset(DistributionDataset):
    def __init__(self, seed, dimension, layer_sizes):
        super().__init__(seed)
        layer_sizes = split_layer_sizes(layer_sizes)
        self.latent_dimension = dimension
        self.output_dimension = layer_sizes[-1]
        # fc_module = FullyConnectedNetwork.partial(layer_sizes=layer_sizes, leaky=True)
        fc_module = GIN.partial(layer_sizes=layer_sizes, leaky=True)

        init_key, self.key = random.split(self.key)
        _, initial_params = fc_module.init_by_shape(init_key, [(self.latent_dimension,)])
        self.model = flax.nn.Model(fc_module, initial_params)

    def score_batch(self, batch):
        return jax.vmap(self.model.average_log_likelihood)(batch).mean()

    def get_batch(self, size, return_latents=False):
        latent_key, self.key = random.split(self.key)
        latents = random.normal(latent_key, (size, self.latent_dimension))
        samples = self.model(latents)
        if return_latents:
            return samples, latents
        return samples

    def plot_batch(self, batch, fn):
        pass

    @property
    def shape(self):
        return (self.output_dimension,)

    def save(self, fn):
        data = flax.serialization.to_state_dict(self.model)
        with open(fn, "wb") as f:
            pkl.dump(data, f)

    def load(self, fn):
        with open(fn, "rb") as f:
            data = pkl.load(f)
        self.model = flax.serialization.from_state_dict(self.model, data)

class FlowDataset(DistributionDataset):
    def __init__(self, seed, dimension, n_passes, layer_sizes):
        super().__init__(seed)
        layer_sizes = split_layer_sizes(layer_sizes) + [dimension]
        self.latent_dimension = dimension
        self.output_dimension = dimension
        flow_module = RealNVP2Network.partial(layer_sizes=layer_sizes, n_passes=n_passes)
        init_key = self.get_key()
        _, initial_params = flow_module.init_by_shape(init_key, [(self.latent_dimension,)])
        self.model = flax.nn.Model(flow_module, initial_params)
        self.call = jit(self.model.apply_inverse)

    def score_batch(self, batch):
        return self.model.average_log_likelihood(batch)

    def get_batch(self, size, return_latents=False):
        latent_key = self.get_key()
        latents = random.normal(latent_key, (size, self.latent_dimension))
        samples = self.call(latents)
        if return_latents:
            return samples, latents
        return samples

    def plot_batch(self, batch, fn):
        pass

    @property
    def shape(self):
        return (self.output_dimension,)

    def save(self, fn):
        data = flax.serialization.to_state_dict(self.model)
        with open(fn, "wb") as f:
            pkl.dump(data, f)

    def load(self, fn):
        with open(fn, "rb") as f:
            data = pkl.load(f)
        self.model = flax.serialization.from_state_dict(self.model, data)


class NoisyFlowDataset(DistributionDataset):
    def __init__(self, seed, dimension, n_passes, layer_sizes, epsilon, nsamples=1000):
        super().__init__(seed)
        layer_sizes = split_layer_sizes(layer_sizes) + [dimension]
        self.latent_dimension = dimension
        self.output_dimension = dimension
        self.epsilon = epsilon
        self.nsamples = nsamples
        flow_module = RealNVP2Network.partial(layer_sizes=layer_sizes, n_passes=n_passes)
        init_key = self.get_key()
        _, initial_params = flow_module.init_by_shape(init_key, [(self.latent_dimension,)])
        self.params = initial_params
        self.model = flax.nn.Model(flow_module, initial_params)

    def score_batch(self, batch):
        def score_example(example):
            d = example.shape[-1]
            z_0 = self.model(example)
            Jf = jax.jacfwd(self.model.apply_inverse)(z_0)
            JftJf = Jf.T @ Jf
            JftJf_inv = jnp.linalg.inv(JftJf + jnp.eye(self.latent_dimension) * jnp.exp(self.epsilon))

            Sigma_z = jnp.linalg.inv(JftJf / jnp.exp(self.epsilon) + jnp.eye(self.latent_dimension))
            Entropy = -0.5 * (d * (jnp.log(2 * jnp.pi) + self.epsilon) - jnp.linalg.slogdet(Sigma_z)[1])
            # MC_samples = random.multivariate_normal(key, mu_z, Sigma_z, [self.nsamples])
            # logpdf = partial(multivariate_normal.logpdf, mean=jnp.zeros(self.latent_dimension),
                             # cov=jnp.eye(self.latent_dimension))
            # logpz = jax.vmap(logpdf)(MC_samples)
            # mu_x = self.model.apply_inverse(MC_samples)
            # datas = jnp.repeat(jnp.expand_dims(example, axis=0), MC_samples.shape[0], axis=0)
            # going to sub in the expectation of p(x\mid z) since it converges slowly
            # logpdf = partial(multivariate_normal.logpdf, cov=jnp.eye(self.output_dimension) * jnp.exp(self.epsilon))
            # logpxz = jax.vmap(logpdf)(datas, mu_x)
            # logpxz = z.shape[-1] / 2
            # return Entropy + (logpz + logpxz).mean()
            return Entropy - jnp.trace(Sigma_z) / 2 - jnp.sum(jnp.square(z_0)) / 2
        score = jax.vmap(score_example, [0])(batch)
        return score.mean()

    def get_batch(self, size, epsilon=None, return_latents=False):
        latent_key = self.get_key()
        latents = random.normal(latent_key, (size, self.latent_dimension))
        samples = self.model.apply_inverse(latents)
        noise_key = self.get_key()
        stdev = jnp.exp(self.epsilon / 2)
        noise_samples = random.normal(noise_key, (size, self.output_dimension)) * stdev
        noised_samples = samples + noise_samples
        if return_latents:
            return noised_samples, None
        return noised_samples

    def plot_batch(self, batch, fn):
        pass

    @property
    def shape(self):
        return (self.output_dimension,)

    def save(self, fn):
        data = flax.serialization.to_state_dict(self.model)
        with open(fn, "wb") as f:
            pkl.dump(data, f)

    def load(self, fn):
        with open(fn, "rb") as f:
            data = pkl.load(f)
        self.model = flax.serialization.from_state_dict(self.model, data)


class SparseCodingDataset(DistributionDataset):
    def __init__(self, seed, dimension, sparsity):
        super().__init__(seed)
        self.A = ortho_group(dimension)
        self.dimension = dimension
        self.sparsity = sparsity

    @property
    def shape(self):
        return (self.dimension,)

    def score_batch(self, batch):
        return None

    def get_batch(self, size, return_latents=False):
        single = jnp.concatenate((jnp.ones(self.sparsity), jnp.zeros(self.dimension - self.sparsity)))
        tiled = jnp.tile(single, (size, 1))
        shuffled = random.shuffle(self.get_key(), tiled, axis=1).T
        flipped = shuffled * random.rademacher(self.get_key(), shuffled.shape)
        output = self.A @ flipped
        if return_latents:
            return output, flipped
        return output

    def plot_batch(self, batch, fn):
        pass

    def save(self, fn):
        slug = {"A": self.A, "dimension": self.dimension, "sparsity": self.sparsity}
        with open(fn, "wb") as f:
            pkl.dump(slug, f)

    def load(self, fn):
        with open(fn, "rb") as f:
            slug = pkl.load(f)
        self.A = slug['A']
        self.dimension = slug['dimension']
        self.sparsity = slug['sparsity']


class CircleDataset(DistributionDataset):
    def __init__(self, seed):
        super().__init__(seed)
        self.R = 1

    @property
    def shape(self):
        return (3,)

    def score_batch(self, batch):
        return None

    def get_batch(self, size, return_latents=False):
        # theta then phi
        angles = jnp.random.uniform(self.get_key(), (size, 2), minval=-jnp.pi, maxval=jnp.pi)
        theta = angles[:, 0]
        phi = angles[:, 1]
        x = jnp.sin(theta) * jnp.cos(phi)
        y = jnp.sin(theta) * jnp.sin(phi)
        z = jnp.cos(theta)
        output = jnp.stack((x, y, z), axis=-1)
        return output

    def plot_batch(self, batch, fn):
        pass

    def save(self, fn):
        pass

    def load(self, fn):
        pass

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


class GraphDataset(DistributionDataset):
    def __init__(self, seed, dimension=3, padding_dimension=0):
        super().__init__(seed)
        self.dim = dimension
        self.padding_dim = padding_dimension
        self.ndim = dimension + padding_dimension

    @property
    def shape(self):
        return (self.ndim,)

    def score_batch(self, batch):
        domain = batch[:, :(self.dim - 1)]
        codomain = batch[:, self.dim - 1]
        codomain_hat = jnp.sum(jnp.square(domain), axis=1)
        manifold_error = jnp.mean(jnp.square((codomain_hat - codomain)))
        padding = batch[:, self.dim:]
        padding_score = jnp.mean(jnp.linalg.norm(padding, axis=1) ** 2)
        domain_norm = jnp.mean(jnp.linalg.norm(domain, axis=1) ** 2)
        return {"Manifold Error": manifold_error,
                "Padding Error": padding_score,
                "Domain Norm": domain_norm}

    def get_batch(self, size, return_latents=False):
        norm_samps = random.normal(self.get_key(), (size, self.dim - 1))
        ss = jnp.sum(jnp.square(norm_samps), axis=1, keepdims=True)
        padding = jnp.zeros((size, self.padding_dim))
        samps = jnp.concatenate([norm_samps, ss, padding], axis=1)

        if return_latents:
            return samps, None
        return samps

    def plot_batch(self, batch, fn):
        pass

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


class GaussianMixtureDataset(DistributionDataset):
    def __init__(self, seed, means, padding_dimension):
        super().__init__(seed)
        self.padding_dimension = padding_dimension
        self.data_dimension = padding_dimension + means.shape[1]
        self.n_clusters = means.shape[0]
        self.cov = jnp.eye(self.dimension)  # TODO: maybe fuck with this
        self.means = jnp.concatenate((means, jnp.zeros((self.n_clusters, padding_dimension))), axis=1)
        self.distribution = GaussianMixture(n_components=self.n_clusters,
                                            covariance_type='tied',
                                            weights_init=np.ones(self.n_clusters) / self.n_clusters,
                                            means_init=self.means,
                                            # hopefully a no-op if cov is identity
                                            precisions_init=jnp.linalg.inv(self.cov),
                                            # hopefully this does nothing
                                            max_iter=0,
                                            )
        # need to manually fill this in because we don't actually want to fit anything
        self.distribution.weights_ = np.ones(self.n_clusters) / self.n_clusters
        self.distribution.means_ = self.means
        self.distribution.precisions_cholesky_ = _compute_precision_cholesky(self.cov, 'tied')

    @property
    def shape(self):
        return (self.data_dimension,)

    def get_batch(self, size, return_latents=False):
        mix_key, self.key = random.split(self.key)
        # TODO: sometime maybe add support for non-uniform mixtures
        # TODO: maybe go to the gaussian mixture instead of manual sampling
        mixtures = random.randint(mix_key, shape=(size,), minval=0, maxval=self.n_clusters)
        means = self.means[mixtures, :]
        gaussian_key, self.key = random.split(self.key)
        batch = random.multivariate_normal(gaussian_key, means, self.cov)
        if return_latents:
            return batch, None
        else:
            return batch

    def score_batch(self, batch):
        return self.distribution.score(batch)

    def plot_batch(self, batch, fn=None):
        plt.hist2d(x=batch[:, 0], y=batch[:, 1], bins=50)
        plt.savefig(fn)
        plt.show(block=False)

    def save(self, fn):
        pass


class GaussianMixture4(GaussianMixtureDataset):

    def __init__(self, seed, padding_dimension):
        means = jnp.array([[-5, -5], [-5, 5], [5, -5], [5, 5]])
        super().__init__(seed, means=means, padding_dimension=padding_dimension)


class GaussianMixture25(GaussianMixtureDataset):

    def __init__(self, seed, padding_dimension):
        n_square = 5
        start = -8
        offset = 4
        mean_list = []
        for i in range(n_square):
            for j in range(n_square):
                mean_list.append([start + i * offset, start + j * offset])
        means = jnp.array(mean_list)
        super().__init__(seed, means=means, padding_dimension=padding_dimension)


class MNISTDataset(EpochDataset):
    def __init__(self, seed, batch_size):
        self.seed = seed
        # TODO: is there anything the seed is for?
        self.train_dataset = datasets.MNIST('../input_data', train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))]))
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True)
        self.test_dataset = datasets.MNIST('../input_data', train=False, download=True,
                                           transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))]))
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=True)

    @property
    def train_dataloader(self):
        return self.train_loader

    @property
    def test_dataloader(self):
        return self.test_loader

    def get_batch(self, batch_size, return_latents=False):
        loader = iter(self.test_dataloader)
        batch = jnp.array(next(loader)[0])
        batch = batch.reshape((batch.shape[0], -1))
        while batch.shape[0] < batch_size:
            new_batch = jnp.array(next(loader)[0])
            new_batch = new_batch.reshape((new_batch.shape[0], -1))
            batch = jnp.concatenate((batch, new_batch), axis=0)
        batch = batch[:batch_size, ...]
        if return_latents:
            return batch, None
        return batch

    @property
    def shape(self):
        return (28, 28, 1)

    def plot_batch(self, batch, fn=None):
        batch = batch[:100, ...]
        shape = [batch.shape[0]] + list(self.shape)
        batch = batch.reshape(shape)
        save = True
        if fn is None:
            fn = ''
            save = False
        img_tile(batch, fn, save)
        plt.show()

    def save(self, fn):
        pass


class LinearFunctionDataset(DistributionDataset):
    def __init__(self, seed, dimension):
        super().__init__(seed)
        # need to generate a random invertible matrix d x d
        # for now, generate a random matrix and make sure det > eps
        det_eps = 1e-4
        self.key, mat_key = random.split(self.key)
        self.dim = dimension
        mat = random.normal(mat_key, (dimension, dimension))
        det = jnp.linalg.det(mat)
        while det < det_eps:
            self.key, mat_key = random.split(self.key)
            mat = random.normal(mat_key, (dimension, dimension))
            det = jnp.linalg.det(mat)
        self.A = mat
        self.transformed_cov = self.A @ self.A.T

    def get_batch(self, size, return_latents=False):
        self.key, x_key = random.split(self.key)
        X = random.normal(x_key, (size, self.dim))
        Y = (self.A @ X.T).T
        if return_latents:
            return Y, X
        return Y

    @property
    def shape(self):
        return (self.dim,)

    def plot_batch(self, batch, fn=None):
        pass

    def score_batch(self, batch):
        scores = multivariate_normal.logpdf(batch, mean=jnp.zeros(batch.shape[1]), cov=self.transformed_cov)
        return scores.mean()

    def save(self, fn):
        data = {"A": self.A}
        with open(fn, "wb") as f:
            pkl.dump(data, f)

    def load(self, fn):
        with open(fn, "rb") as f:
            data = pkl.load(f)
        self.A = data['A']


class ToeplitzDataset(LinearFunctionDataset):
    def __init__(self, seed, dimension):
        super().__init__(seed, dimension)
        self.key, t_key = random.split(self.key)
        c = random.normal(t_key, (dimension,))
        self.A = jnp.asarray(toeplitz(c))
        self.transformed_cov = self.A @ self.A.T


class TanhDataset(DistributionDataset):
    def __init__(self, seed, dimension):
        super().__init__(seed)
        self.dim = dimension

    @property
    def shape(self):
        return (self.dim,)

    def get_batch(self, size, return_latents=False):
        self.key, z_key = random.split(self.key)
        z = random.normal(z_key, (size, self.dim))
        tanh_z = jnp.tanh(z)
        if return_latents:
            return tanh_z, z
        return tanh_z

    def plot_batch(self, batch, fn=None):
        pass

    def score_batch(self, batch):
        lognorm = norm.logpdf(batch)
        log_d_inv_tanh = jnp.log(1 / (1 - batch ** 2))
        return (lognorm + log_d_inv_tanh).sum(axis=1).mean(axis=0)

    def save(self, fn):
        pass

    def load(self, fn):
        pass


class ReLUDataset(DistributionDataset):
    def __init__(self, seed, dimension):
        super().__init__(seed)
        self.dim = dimension

    @property
    def shape(self):
        return (self.dim,)

    def get_batch(self, size, return_latents=False):
        self.key, z_key = random.split(self.key)
        z = random.normal(z_key, (size, self.dim))
        output = jnp.maximum(0, z)
        if return_latents:
            return output, z
        return output

    def plot_batch(self, batch, fn=None):
        pass

    def score_batch(self, batch):
        return 0

    def save(self, fn):
        pass

    def load(self, fn):
        pass


def test_flow_datasets():
    seed = 69
    dimension = 32
    n_passes = 4
    batch_size = 128
    layer_sizes = "128|128"
    flow_scores_flow = []
    flow_scores_noisy = []
    noisy_scores_flow = []
    noisy_scores_noisy = []
    x = np.arange(-16, 4, 0.5)
    for epsilon in x:
        flow_dataset = FlowDataset(seed, dimension, n_passes, layer_sizes)
        noisy_flow_dataset = NoisyFlowDataset(seed, dimension, n_passes, layer_sizes, epsilon)
        noisy_flow_batch = noisy_flow_dataset.get_batch(batch_size)
        flow_batch = flow_dataset.get_batch(batch_size)
        flow_scores_noisy += [flow_dataset.score_batch(noisy_flow_batch).item()]
        noisy_scores_noisy += [noisy_flow_dataset.score_batch(noisy_flow_batch).item()]
        flow_scores_flow += [flow_dataset.score_batch(flow_batch).item()]
        noisy_scores_flow += [noisy_flow_dataset.score_batch(flow_batch).item()]

    fig, [ax1, ax2] = plt.subplots(ncols=2)
    ax1.plot(x, flow_scores_flow, label="Naive Flow Scores")
    ax1.plot(x, noisy_scores_flow, label="Variational Scores")
    ax2.plot(x, flow_scores_noisy, label="Naive Flow Scores")
    ax2.plot(x, noisy_scores_noisy, label="Variational Scores")
    ax1.set_title("Data generated by flow without noise")
    ax2.set_title("Data generated by flow with noise")
    ax2.legend()
    ax1.set_ylabel("Log Likelihood")
    ax2.set_ylabel("Log Likelihood")
    ax1.set_xlabel("logvar added for noisy flow")
    ax2.set_xlabel("logvar added for noisy flow")
    plt.show()


if __name__ == '__main__':
    test_flow_datasets()
