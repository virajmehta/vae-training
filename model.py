import os
from abc import ABC, abstractmethod
from jax import numpy as jnp, random
from jax.scipy.stats import norm, logistic
import flax
from copy import deepcopy
import torch
import pickle as pkl
from torchvision import transforms, datasets
from tqdm import trange, tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import cross_entropy_loss, compute_accuracy


class Model(ABC):
    def __init__(self,
                 dirname,
                 batch_size,
                 learning_rate,
                 state_dict,
                 tqdm=False,
                 ):
        self.dirname = dirname
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.key = random.PRNGKey(0)
        self.state_dict = state_dict
        self.optimizer = None
        self.model = None
        self.state = None
        self.tqdm = tqdm
        self.stats = defaultdict(list)

    def load_model(self):
        if self.optimizer is None or self.state_dict is None:
            return
        with open(self.state_dict, 'rb') as f:
            self.state_dict = pkl.load(f)
        self.optimizer = flax.serialization.from_state_dict(self.optimizer, self.state_dict)
        self.model = self.optimizer.target

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def plot_epoch(self):
        pass

    @abstractmethod
    def train_one_batch(self, *args):
        pass

    def get_key(self):
        self.key, key = random.split(self.key)
        return key

    def compute_model_stats(self, real_batch, fake_batch):
        '''
        Return a dict of {stat_name: value}
        '''
        return {}

    @abstractmethod
    def compute_stats(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def model_save_data(self):
        '''
        return dict from stat name -> ndarray of model specific stuff
        '''
        return {}

    @abstractmethod
    def save(self):
        pass

    def save_model(self):
        state_dict = flax.serialization.to_state_dict(self.optimizer)
        model_fn = os.path.join(self.dirname, "model.pkl")
        with open(model_fn, "wb") as f:
            pkl.dump(state_dict, f)

    def load(self):
        if self.data_fn is not None:
            self.dataset.load(self.data_fn)
        self.load_model()


class GenerativeModel(Model):
    def __init__(self,
                 dirname,
                 num_batches,
                 num_epochs,
                 batch_size,
                 learning_rate,
                 latent_distribution,
                 state_dict,
                 dataset,
                 data_fn,
                 tqdm=False,
                 latent_dimension=None,
                 ):
        super().__init__(
                dirname,
                batch_size,
                learning_rate,
                state_dict,
                tqdm)
        self.num_batches = num_batches
        self.num_epochs = num_epochs
        self.latent_distribution = latent_distribution

        self.dataset = dataset

        self.n_plot = 50000
        self.n_print = 5000
        self.plot_batch_size = 1000
        self.print_batch_size = 1000
        self.average_log_likelihoods = []
        self.latent_dim = latent_dimension if latent_dimension else self.dataset.dimension
        self.data_fn = data_fn
        self.epoch_num = 0

    def plot_model_specific(self):
        pass

    def plot(self):
        self.plot_model_specific()
        plt.clf()

    def plot_epoch(self):
        key, self.key = random.split(self.key)
        batch = self.sample_batch(key, self.plot_batch_size)[0]
        if self.dataset.is_epochs:
            fn = os.path.join(self.dirname, f"output_{self.epoch_num}.png")
        else:
            fn = os.path.join(self.dirname, f"output_{self.batchnum}.png")
        self.dataset.plot_batch(batch, fn=fn)


    @abstractmethod
    def sample_batch(self, key, batch_size):
        pass

    def compute_stats(self):
        key, self.key = random.split(self.key)
        real_batch, latents = self.dataset.get_batch(self.print_batch_size, return_latents=True)
        if latents is None or latents.shape[-1] != self.latent_dim:
            latents = None
        fake_batch, latents = self.sample_batch(key, self.print_batch_size, latents=latents)
        # TODO: figure out this for epochs
        stats = self.compute_model_stats(real_batch, fake_batch, latents)
        if not self.dataset.is_epochs:
            score = self.dataset.score_batch(fake_batch)
            if type(score) is not dict:
                stats["Average Log Likelihood"] = score
                self.average_log_likelihoods.append(score)
            else:
                stats.update(score)
        return stats

    def train(self):
        if self.dataset.is_epochs:
            self.train_epochs()
        else:
            self.train_distribution()

    def train_epochs(self):
        self.batchnum = 0
        stats = self.compute_stats()
        self.write_stats(stats)
        epoch_iterator = trange(self.num_epochs) if self.tqdm else range(self.num_epochs)
        for self.epoch_num in epoch_iterator:
            self.current_train_accuracies, self.current_train_losses = [], []
            dataset_iterator = tqdm(self.dataset.train_dataloader) if self.tqdm else self.dataset.train_dataloader
            for batch, labels in dataset_iterator:
                batch = np.array(batch)
                labels = np.array(labels)
                self.train_one_batch(batch)
                self.batchnum += 1
            stats = self.compute_stats()
            print(f"Completed Epoch {self.epoch_num}")
            self.write_stats(stats)
            self.plot_epoch()
            self.save()

    def write_stats(self, stats):
        message = f"Epoch | {self.epoch_num}" if self.dataset.is_epochs else f"Batch | {self.batchnum}"
        for stat, val in stats.items():
            self.stats[stat].append(val)
            try:
                val = float(val)
            except Exception:
                self.stats[stat].append(val)
                continue;
            message = message + f" | {stat} | {val:.3f}"
        tqdm.write(message)

    def train_distribution(self):
        eval_batch_key, self.key = random.split(self.key)
        eval_batch = self.dataset.get_batch(self.print_batch_size)
        score = self.dataset.score_batch(eval_batch)
        print(f"Score for real data: {score}")
        batch_iterator = trange(self.num_batches) if self.tqdm else range(self.num_batches)
        for self.batchnum in batch_iterator:
            if self.batchnum % self.n_print == 0:
                # print statistics and maybe save them later
                stats = self.compute_stats()
                self.write_stats(stats)
            if self.batchnum % self.n_plot == 0 or self.batchnum == self.num_batches - 1:
                self.plot_epoch()
                self.save()
            batch = self.dataset.get_batch(self.batch_size)
            self.train_one_batch(batch)


    def sample_latent(self, key, batch_size):
        if self.latent_distribution == 'gaussian':
            output = random.normal(key, shape=(batch_size, self.latent_dim + self.dataset.dimension))
            return output
        elif self.latent_distribution == 'logistic':
            while True:
                key, tmp_key = random.split(key)
                sample = random.logistic(tmp_key, shape=(batch_size, self.latent_dim))
                if jnp.isfinite(sample).all():
                    return sample
        else:
            raise NotImplementedError(f"distribution {self.latent_distribution} is not implemented")

    def latent_likelihood(self, latent_batch):
        if self.latent_distribution == 'gaussian':
            return jnp.mean(jnp.sum(norm.logpdf(latent_batch), axis=1), axis=0)
        elif self.latent_distribution == 'logistic':
            return jnp.mean(jnp.sum(logistic.logpdf(latent_batch), axis=1), axis=0)
        else:
            raise NotImplementedError(f"distribution {self.latent_distribution} is not implemented")

    def save(self, final=False):
        data = self.model_save_data(final=final)
        data["Average Log Likelihood"] = np.array(self.average_log_likelihoods)
        fn = os.path.join(self.dirname, "losses")
        stats = deepcopy(self.stats)
        stats.update(data)
        np.savez(fn, **stats)
        self.save_model()
        data_fn = os.path.join(self.dirname, "dataset.pk")
        self.dataset.save(data_fn)


class ClassificationModel(Model):
    def __init__(self,
                 dirname,
                 batch_size,
                 learning_rate,
                 dataset_name,
                 num_epochs,
                 tqdm=False):
        # these 2 categories get filled in by the super call. Just putting them here so it's documented
        self.input_shape = None
        self.n_classes = None
        super().__init__(
                dirname,
                batch_size,
                learning_rate,
                tqdm)
        self.num_epochs = num_epochs
        self.epoch_num = None
        self.load_data(dataset_name)
        self.train_accuracies = []
        self.test_accuracies = []
        self.train_losses = []
        self.test_losses = []
        self.current_train_accuracies, self.current_train_losses = [], []

    @abstractmethod
    def predict(self, batch):
        pass

    def train(self):
        # initial evaluation with epoch num 0
        stats = self.compute_stats()
        self.write_stats(stats)
        for self.epoch_num in trange(self.num_epochs):
            self.current_train_accuracies, self.current_train_losses = [], []
            for batch, labels in tqdm(self.train_dataloader):
                batch = np.array(batch)
                labels = np.array(labels)
                train_accuracy, train_loss = self.train_one_batch(batch, labels)
                self.current_train_accuracies.append(train_accuracy)
                self.current_train_losses.append(train_loss)
            stats = self.compute_stats()
            self.write_stats(stats)

    def write_stats(self, stats):
        message = f"Epoch | {self.epoch_num}"
        for stat, val in stats.items():
            if val:
                message = message + f" | {stat} | {val:.3f}"
        tqdm.write(message)

    @abstractmethod
    def train_one_batch(self, batch, labels):
        '''
        needs to return train_accuracy, train_loss
        '''
        pass

    def plot(self):
        fig, axes = plt.subplots(ncols=2)
        axes[0].plot(self.train_accuracies, label="Train", color="blue")
        axes[0].plot(self.test_accuracies, label="Test", color="red")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Accuracy")
        axes[1].plot(self.train_losses, label="Train", color="blue")
        axes[1].plot(self.test_losses, label="Test", color="red")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Cross-Entropy Loss")
        # handles, labels = ax.get_legend_handles_labels()
        fig.legend()
        plt.show()

    def plot_epoch(self):
        pass

    def compute_stats(self):
        accs = []
        losses = []
        for batch, labels in tqdm(self.test_dataloader):
            batch = np.array(batch)
            labels = np.array(labels)
            accuracy, loss = self.evaluate(batch, labels)
            accs.append(accuracy)
            losses.append(loss)
        if len(self.current_train_accuracies) > 0:
            train_accuracy = jnp.mean(jnp.array(self.current_train_accuracies))
            train_loss = jnp.mean(jnp.array(self.current_train_losses))
            self.train_accuracies.append(train_accuracy)
            self.train_losses.append(train_loss)
        else:
            train_accuracy = None
            train_loss = None

        test_accuracy = jnp.mean(jnp.array(accs))
        test_loss = jnp.mean(jnp.array(losses))
        self.test_accuracies.append(test_accuracy)
        self.test_losses.append(test_loss)
        stats = {
                'Train Accuracy': train_accuracy,
                'Test Accuracy': test_accuracy,
                'Train Loss': train_loss,
                'Test Loss': test_loss,
                }
        return stats

    def evaluate(self, batch, labels):
        logits = self.predict(batch)
        loss = jnp.mean(cross_entropy_loss(logits, labels))
        accuracy = compute_accuracy(logits, labels)
        return accuracy, loss

    def save(self):
        data = self.model_save_data()
        data['Train Accuracy'] = self.train_accuracies
        data['Test Accuracy'] = self.test_accuracies
        data['Train Loss'] = self.train_losses
        data['Test Loss'] = self.test_losses
        fn = os.path.join(self.dirname, "losses")
        np.savez(fn, **data)
        data_fn = os.path.join(self.dirname, "dataset.pk")
        self.dataset.save(data_fn)

    def load_data(self, dataset_name):
        if dataset_name == 'mnist':
            train_dataset = datasets.MNIST('../input_data', train=True, download=True,
                                           transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))]))
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True)
            test_dataset = datasets.MNIST('../input_data', train=False, download=True,
                                          transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ]))
            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True)
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not supported")
        self.train_dataloader = train_loader
        self.test_dataloader = test_loader
        self.input_shape = (784,)
        self.n_classes = 10


class RegressionModel(Model):
    def __init__(self,
                 dirname,
                 batch_size,
                 learning_rate,
                 dataset,
                 num_batches,
                 state_dict,
                 data_fn,
                 num_epochs,
                 tqdm=False,):
        # these 2 categories get filled in by the super call. Just putting them here so it's documented
        self.input_shape = None
        self.output_shape = None
        super().__init__(
                dirname,
                batch_size,
                learning_rate,
                state_dict,
                tqdm)
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.epoch_num = None
        self.dataset = dataset
        self.train_accuracies = []
        self.test_accuracies = []
        self.train_losses = []
        self.test_losses = []
        self.losses = []
        self.n_print = 1000
        self.print_batch_size = 10000
        self.current_train_accuracies, self.current_train_losses = [], []
        self.data_fn = data_fn

    @abstractmethod
    def predict(self, batch):
        pass

    def train_epochs(self):
        # initial evaluation with epoch num 0
        stats = self.compute_stats()
        self.write_stats(stats)
        epoch_iterator = trange(self.num_epochs) if self.tqdm else range(self.num_epochs)
        for self.epoch_num in epoch_iterator:
            self.current_train_accuracies, self.current_train_losses = [], []
            dataset_iterator = tqdm(self.train_dataloader) if self.tqdm else self.train_dataloader
            for batch, labels in dataset_iterator:
                batch = np.array(batch)
                labels = np.array(labels)
                train_accuracy, train_loss = self.train_one_batch(batch, labels)
                self.current_train_accuracies.append(train_accuracy)
                self.current_train_losses.append(train_loss)
            stats = self.compute_stats()
            self.write_stats(stats)

    def train_distribution(self):
        batch_iterator = trange(self.num_batches) if self.tqdm else range(self.num_batches)
        for self.batchnum in batch_iterator:
            # TODO: fix for other distributional regression datasets besides linear and tanh
            # if a distribution: samples, latents
            X_batch, Y_batch = self.dataset.get_batch(self.batch_size, return_latents=True)
            self.train_one_batch(X_batch, Y_batch)
            if self.batchnum % self.n_print == 0:
                # print statistics and maybe save them later
                stats = self.compute_stats()
                self.write_stats(stats)

    def train(self):
        if self.dataset.is_epochs:
            self.train_epochs()
        else:
            self.train_distribution()

    def write_stats(self, stats):
        message = f"Epoch | {self.epoch_num}" if self.dataset.is_epochs else f"Batch | {self.batchnum}"
        for stat, val in stats.items():
            if val:
                message = message + f" | {stat} | {val:.3f}"
        tqdm.write(message)

    @abstractmethod
    def train_one_batch(self, X_batch, Y_batch):
        '''
        needs to return train_loss
        '''
        pass

    def plot(self):
        plt.plot(self.train_losses)
        plt.ylabel("L2 Loss")
        plt.yscale('log')
        plt.xlabel("Training Batches")
        plt.title("Train Loss")
        plt.show()

    def plot_epoch(self):
        pass

    def compute_stats(self):
        key, self.key = random.split(self.key)
        real_batch, latents = self.dataset.get_batch(self.print_batch_size, return_latents=True)
        fake_batch, latents = self.sample_batch(key, self.print_batch_size, latents=latents)

        stats = self.compute_model_stats(real_batch, fake_batch, latents)
        # stats = {}
        stats['Loss'] = np.mean(self.train_losses[-self.n_print:])
        return stats

    def evaluate(self, batch, labels):
        logits = self.predict(batch)
        loss = jnp.mean(cross_entropy_loss(logits, labels))
        accuracy = compute_accuracy(logits, labels)
        return accuracy, loss

    def save(self):
        data = self.model_save_data()
        data['Train Loss'] = self.train_losses
        fn = os.path.join(self.dirname, "losses")
        np.savez(fn, **data)
        self.save_model()
        data_fn = os.path.join(self.dirname, "dataset.pk")
        self.dataset.save(data_fn)
