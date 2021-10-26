import os
import jax
import jax.numpy as jnp
from jax import lax
import jax.nn.initializers as initializers
import flax
import numpy as np
import cv2
import json

DATA_DIR = 'data/'



class Constants:
    """
    Recommended hyperparameters (Feel free to add/remove/modify these values).
    """
    lambd = 10
    alpha = 0.1
    # TODO: figure out if this is worth computing
    epsilon_singular_value = 1e-7


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def relu(x):
    return jnp.maximum(x, 0)


def leaky_relu(x):
    return jnp.maximum(x, x * Constants.alpha)


def inv_leaky_relu(x):
    return jnp.minimum(x, x / Constants.alpha)


def inv_dense(x, weight, bias):
    inv_weight = jnp.linalg.inv(weight)
    return jnp.dot((x - bias), inv_weight)


def make_output_dir(name, overwrite, args):
    dirname = get_output_dir(name)
    if os.path.exists(dirname):
        if overwrite:
            for fn in os.listdir(dirname):
                os.remove(os.path.join(dirname, fn))
        else:
            raise ValueError(f"{dirname} already exists! Use a different name")
    else:
        os.mkdir(dirname)
    args_name = os.path.join(dirname, 'args.json')
    args = vars(args)
    with open(args_name, 'w') as f:
        json.dump(args, f)
    return dirname


def get_output_dir(name):
    dirname = os.path.join(DATA_DIR, name)
    return dirname


@jax.jit
@jax.vmap
def cross_entropy_loss(logits, label):
    return -logits[label]


@jax.jit
def compute_accuracy(logits, labels):
    return jnp.mean(jnp.argmax(logits, -1) == labels)


def img_tile(imgs, fn, save, aspect_ratio=1.0, border=1, border_color=0):
    """
    Visualize the WGAN result for each step
    :param imgs: Numpy array of the generated images
    :param path: Path to save visualized results for each epoch
    :param epoch: Epoch index
    :param save: Boolean value to determine whether you want to save the result or not
    """

    if imgs.ndim != 3 and imgs.ndim != 4:
        raise ValueError('imgs has wrong number of dimensions.')
    n_imgs = imgs.shape[0]

    tile_shape = None
    # Grid shape
    img_shape = np.array(imgs.shape[1:3])
    if tile_shape is None:
        img_aspect_ratio = img_shape[1] / float(img_shape[0])
        aspect_ratio *= img_aspect_ratio
        tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
        tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
        grid_shape = np.array((tile_height, tile_width))
    else:
        assert len(tile_shape) == 2
        grid_shape = np.array(tile_shape)

    # Tile image shape
    tile_img_shape = np.array(imgs.shape[1:])
    tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

    # Assemble tile image
    tile_img = np.empty(tile_img_shape)
    tile_img[:] = border_color
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            img_idx = j + i * grid_shape[1]
            if img_idx >= n_imgs:
                # No more images - stop filling out the grid.
                break

            # -1~1 to 0~1
            img = (imgs[img_idx] + 1) / 2.0  # * 255.0

            yoff = (img_shape[0] + border) * i
            xoff = (img_shape[1] + border) * j
            tile_img[yoff:yoff + img_shape[0], xoff:xoff + img_shape[1], ...] = img

    ##########################################
    # Change code below if you want to save results using PIL
    ##########################################
    tile_img = cv2.resize(tile_img, (256, 256))
    # cv2.imshow("Results", tile_img)
    # cv2.waitKey(1)
    if save:
        cv2.imwrite(fn, tile_img * 255)


def _absolute_dims(rank, dims):
    return tuple([rank + dim if dim < 0 else dim for dim in dims])


class InvertibleBatchNorm(flax.nn.Module):
    """Invertible BatchNorm Module."""

    def apply(self,
              x,
              batch_stats=None,
              use_running_average=False,
              axis=-1,
              momentum=0.99,
              epsilon=1e-5,
              dtype=jnp.float32,
              bias=True,
              scale=True,
              bias_init=initializers.zeros,
              scale_init=initializers.ones,
              axis_name=None,
              axis_index_groups=None):
        """Normalizes the input using batch statistics.

        Args:
          x: the input to be normalized.
          batch_stats: a `flax.nn.Collection` used to store an exponential moving
            average of the batch statistics (default: None).
          use_running_average: if true, the statistics stored in batch_stats
            will be used instead of computing the batch statistics on the input.
          axis: the feature or non-batch axis of the input.
          momentum: decay rate for the exponential moving average of
            the batch statistics.
          epsilon: a small float added to variance to avoid dividing by zero.
          dtype: the dtype of the computation (default: float32).
          bias:  if True, bias (beta) is added.
          scale: if True, multiply by scale (gamma).
            When the next layer is linear (also e.g. nn.relu), this can be disabled
            since the scaling will be done by the next layer.
          bias_init: initializer for bias, by default, zero.
          scale_init: initializer for scale, by default, one.
          axis_name: the axis name used to combine batch statistics from multiple
            devices. See `jax.pmap` for a description of axis names (default: None).
          axis_index_groups: groups of axis indices within that named axis
            representing subsets of devices to reduce over (default: None). For example,
            `[[0, 1], [2, 3]]` would independently batch-normalize over the examples
            on the first two and last two devices. See `jax.lax.psum` for more details.

        Returns:
          Normalized inputs (the same shape as inputs).
        """
        x = jnp.asarray(x, jnp.float32)
        axis = axis if isinstance(axis, tuple) else (axis,)
        axis = _absolute_dims(x.ndim, axis)
        feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
        reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
        reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)
        if self.is_stateful() or batch_stats:
            ra_mean = self.state('mean', reduced_feature_shape,
                                 initializers.zeros, collection=batch_stats)
            ra_var = self.state('var', reduced_feature_shape,
                                initializers.ones, collection=batch_stats)
            state_mul = self.state('recent_mul', reduced_feature_shape,
                                   initializers.ones, collection=batch_stats)
            state_mean = self.state('recent_mean', feature_shape,
                                    initializers.zeros, collection=batch_stats)
        else:
            ra_mean = None
            ra_var = None
            state_mul = None
            state_mean = None

        if use_running_average:
            if ra_mean is None:
                raise ValueError('when use_running_averages is True '
                                 'either use a stateful context or provide batch_stats')
            mean, var = ra_mean.value, ra_var.value
        else:
            mean = jnp.mean(x, axis=reduction_axis, keepdims=False)
            mean2 = jnp.mean(lax.square(x), axis=reduction_axis, keepdims=False)
            if axis_name is not None and not self.is_initializing():
                concatenated_mean = jnp.concatenate([mean, mean2])
                mean, mean2 = jnp.split(
                    lax.pmean(
                        concatenated_mean,
                        axis_name=axis_name,
                        axis_index_groups=axis_index_groups), 2)
            var = mean2 - lax.square(mean)

            if ra_mean and not self.is_initializing():
                ra_mean.value = momentum * ra_mean.value + (1 - momentum) * mean
                ra_var.value = momentum * ra_var.value + (1 - momentum) * var

        recent_mean = mean.reshape(feature_shape)
        y = x - recent_mean
        if state_mean and not self.is_initializing():
            state_mean.value = recent_mean
        mul = lax.rsqrt(var + epsilon)
        if state_mul and not self.is_initializing():
            state_mul.value = mul
        if scale:
            mul = mul * self.param(
                'scale', reduced_feature_shape, scale_init).reshape(feature_shape)
        y = y * mul
        if bias:
            y = y + self.param(
              'bias', reduced_feature_shape, bias_init).reshape(feature_shape)
        return jnp.asarray(y, dtype)


def inv_batch_norm(y,
                   params,
                   collection,
                   bias=True,
                   scale=True,
                   ):
    mul = collection['recent_mul']
    mean = collection['recent_mean']
    if bias:
        bias = params['bias']
        y = y - bias
    y = y / mul
    if scale:
        scale = params['scale']
        y = y / scale
    x = y + mean
    return x


def get_mask(shape, reverse, use_checkerboard=True):
    '''
    Assumes shape is (batch, height, width, channels) or (height, width, channels)
    '''
    height = shape[-3]
    width = shape[-2]
    channels = shape[-1]
    if use_checkerboard:
        checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)]
        mask = jnp.array(checkerboard).reshape(height, width, 1)
        if reverse:
            mask = 1 - mask
        if len(shape) == 4:
            return mask[jnp.newaxis, ...]
        else:
            return mask
    else:
        half = channels // 2
        zero_mask = jnp.zeros((height, width, half))
        one_mask = jnp.ones((height, width, half))
        if reverse:
            mask = jnp.concatenate((zero_mask, one_mask), axis=-1)
        else:
            mask = jnp.concatenate((one_mask, zero_mask), axis=-1)
        if len(shape) == 4:
            return mask[jnp.newaxis, ...]
        else:
            return mask


def squeeze_2x2(x, reverse=False):
    # block_size = 2
    assert x.ndim == 4
    b, h, w, c = x.shape
    if reverse:
        if c % 4 != 0:
            raise ValueError(f"Number of channels {c} is not divisible by 4")
        x = x.reshape((b, h, w, c // 4, 2, 2))
        x = jnp.transpose(x, (0, 1, 4, 2, 5, 3))
        x = x.reshape((b, 2 * h, 2 * w, c // 4))
    else:
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"Expected even spatial dims HxW got {h}x{w}")
        x = x.reshape((b, h // 2, 2, w // 2, 2, c))
        x = jnp.transpose(x, (0, 1, 3, 5, 2, 4))
        x = x.reshape((b, h // 2, w // 2, c * 4))
    return x


def split_layer_sizes(layer_sizes):
    return [int(size) for size in layer_sizes.split('|')]


def sin_theta_distance(A, B):
    '''
    Assumes A and B are orthogonal matrices
    '''
    U, _, _ = jnp.linalg.svd(A)
    Uprime, _, _ = jnp.linalg.svd(B)
    # eye = jnp.eye(B.shape[0])
    distmat = (U - Uprime)
    return 0.5 * jnp.linalg.norm(distmat, ord="fro")
