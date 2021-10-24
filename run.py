import argparse
import jax

from datasets import MNISTDataset, GaussianMixture4, GaussianMixture25, LinearFunctionDataset, TanhDataset, ReLUDataset
from datasets import SmallGANDataset, SwissRollDataset, CheckerboardDataset, TwoMoonsDataset, ToeplitzDataset
from datasets import GeneratorDataset, SparseCodingDataset, CircleDataset, FlowDataset, NoisyFlowDataset, SquareDataset
from datasets import SphereDataset, GaussianDataset, LinearGaussianDataset, GraphDataset, SigmoidDataset
from wgan import WGAN
from gin import GINModel
from mlp import MLPModel, MLPReg
from nice import NICEModel
from vae import VAEModel, ACVAEModel, VAEModel_2Stage, gammaVAEModel, MLP_ACVAEModel, SqVAEModel
from real_nvp import RealNVPModel, RealNVPRegressionModel, RealNVPWassersteinModel, ConvNVPModel
from real_nvp import ConvNVPRegressionModel, RealNVPImageModel
from utils import make_output_dir
from linear_partition import PartitionedLinearModel, PartitionedLinearRegressionModel


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help="The name of the experiment and output directory.")
    parser.add_argument('--model', dest='model', default='WGAN',
                        choices=["WGAN", "GIN", "MLP", "NICE", "RealNVP", "PLN", "PLNR", "RealNVPR", "RealNVPW",
                                 "MLPR", "ConvNVP", "ConvNVPR", "RealNVPImage", "VAE", "ACVAE", "MLP_ACVAE",
                                 "2Stage_VAE", "gammaVAE", "SqVAE"])
    parser.add_argument('--num_batches', dest='num_batches', type=int, default=15000,
                        help="Number of batches to train on.")
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=100)
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float, default=0.0001)
    parser.add_argument('--n_critic', dest='n_critic', type=int, default=5)
    parser.add_argument('--n_passes', dest='n_passes', type=int, default=1)
    parser.add_argument('--n_conv', type=int, default=8)
    # TODO: handle these appropriately in the code
    parser.add_argument('--generator_num_layers', type=int, dest='generator_num_layers', default=5)
    parser.add_argument('--padding_dim', type=int, dest='padding_dim', default=0)
    parser.add_argument('-ow', dest='overwrite', action='store_true')
    parser.add_argument('--dataset', dest='dataset', default='4gaussian',
                        choices=["mnist", "ReLU", "4gaussian", "25gaussian", "linear", "tanh", "small_gan",
                                 "swissroll", "2moons", "checkerboard", "toeplitz", "generator", "sparse_coding",
                                 "circle", "flow", "noisyflow", "square", "sphere", "gaussian", "linear_gaussian", "graph",
                                 "sigmoid"])
    parser.add_argument('--layer_sizes', dest='layer_sizes', default='512|512', help="Specify layer sizes for MLP (possibly others later) as integers separated by pipes. Example: 512|512|512")  # NOQA
    parser.add_argument('--dataset_layer_sizes', dest='dataset_layer_sizes', default='128|128', help="Specify layer sizes for MLP (possibly others later) as integers separated by pipes. Example: 512|512|512")  # NOQA
    parser.add_argument('--dataset_n_passes', type=int, default=4)
    parser.add_argument('--encoder_layer_sizes', dest='encoder_layer_sizes', default='512|512', help="Specify layer sizes for MLP (possibly others later) as integers separated by pipes. Example: 512|512|512")  # NOQA
    parser.add_argument('--latent_distribution', dest='latent_distribution', default='gaussian',
                        choices=['gaussian', 'logistic'])
    parser.add_argument('--latent_dim', dest='latent_dimension', type=int, default=100)
    # parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, default=0.01)
    parser.add_argument('-nojit', dest='nojit', action='store_true', help="Disables just-in-time compilation for debugging")  # NOQA
    parser.add_argument('--padding_type', dest='padding_type', default="none", choices=["zero", "gaussian", "none"])
    parser.add_argument('-ds', '--dataset_seed', dest='dataset_seed', type=int, default=69)
    parser.add_argument('--state_dict', dest='state_dict', default=None)
    parser.add_argument('--data_fn', dest="data_fn", default=None)
    parser.add_argument('--critic_layer_sizes', dest='critic_layer_sizes', default='256|256|256', help="Specify layer sizes for critic as integers separated by pipes. Example: 512|512|512")  # NOQA
    parser.add_argument('-cfd', '--copy_flow_dataset', dest='copy_flow_dataset', action='store_true')
    parser.add_argument('-ws', '--warm_start', action='store_true')

    parser.add_argument('-ii', '--initialize_inverse', action='store_true')
    parser.add_argument('-ufc', '--use_fred_covariance', action='store_true')
    parser.add_argument('-bn', '--batch_norm', dest='batch_norm', action='store_true')
    parser.add_argument('-mult', '--mult_layer', dest='mult_layer', action='store_true')
    parser.add_argument('-notqdm', dest="tqdm", action="store_false")
    parser.add_argument('-tbo', '--train_bias_only', dest='train_bias_only', action='store_true')
    parser.add_argument('-e', '--epsilon', type=float, default=0.)
    parser.add_argument('-ig', dest='if_grid', action='store_true')
    parser.add_argument('-tdv', dest='tunable_decoder_var', action='store_true')
    parser.add_argument('-dn', '--dataset_noise', type=float, default=0.)
    parser.add_argument('-dd', '--dataset_dimension', type=int, default=3)
    parser.add_argument('-wsl', '--warm_start_linear', action='store_true')
    parser.add_argument('-did', '--dataset_intrinsic_dimension', type=int, default=3)
    parser.add_argument('-off', '--latent_off_dimension', type=int, default=1)
    return parser.parse_args()


def get_dataset(name, seed, padding_dimension, batch_size, layer_sizes, args):
    if name == "mnist":
        return MNISTDataset(seed, batch_size)
    elif name == "4gaussian":
        return GaussianMixture4(seed, padding_dimension=padding_dimension)
    elif name == "25gaussian":
        return GaussianMixture25(seed, padding_dimension=padding_dimension)
    elif name == "linear":
        return LinearFunctionDataset(seed, dimension=padding_dimension)
    elif name == "swissroll":
        return SwissRollDataset(seed)
    elif name == "2moons":
        return TwoMoonsDataset(seed, padding_dimension, noise_level=args.dataset_noise)
    elif name == "checkerboard":
        return CheckerboardDataset(seed)
    elif name == 'tanh':
        return TanhDataset(seed, dimension=padding_dimension)
    elif name == 'ReLU':
        return ReLUDataset(seed, dimension=padding_dimension)
    elif name == 'small_gan':
        return SmallGANDataset(seed, dimension=padding_dimension)
    elif name == 'toeplitz':
        return ToeplitzDataset(seed, dimension=padding_dimension)
    elif name == 'generator':
        return GeneratorDataset(seed, dimension=padding_dimension, layer_sizes=layer_sizes)
    elif name == 'sparse_coding':
        return SparseCodingDataset(seed, latent_dimension=padding_dimension, sparsity=args.sparsity)
    elif name == 'circle':
        return CircleDataset(seed)
    elif name == 'flow':
        return FlowDataset(seed, dimension=padding_dimension, layer_sizes=layer_sizes, n_passes=args.dataset_n_passes)
    elif name == 'noisyflow':
        return NoisyFlowDataset(seed, dimension=padding_dimension, layer_sizes=layer_sizes,
                                n_passes=args.dataset_n_passes, epsilon=args.dataset_noise)
    elif name == "square":
        return SquareDataset(seed, dimension=padding_dimension, layer_sizes=layer_sizes, epsilon=args.dataset_noise)
    elif name == "sphere":
        return SphereDataset(seed, dimension=args.dataset_dimension, padding_dimension=args.padding_dim)
    elif name == "graph":
        return GraphDataset(seed, dimension=args.dataset_dimension, padding_dimension=args.padding_dim)
    elif name == "gaussian":
        return GaussianDataset(seed, dimension=args.dataset_dimension, padding_dimension=args.padding_dim, noise_level=args.dataset_noise)
    elif name == "linear_gaussian":
<<<<<<< HEAD
        return LinearGaussianDataset(seed, dimension=args.dataset_dimension, intrinsic_dimension=args.dataset_intrinsic_dimension,
                                     padding_dimension=args.padding_dim, var_added=args.dataset_noise)

=======
        return LinearGaussianDataset(seed, dimension=args.dataset_dimension, padding_dimension=args.padding_dim,
                                     var_added=args.dataset_noise)
    elif name == "sigmoid":
        return SigmoidDataset(seed, dimension=args.dataset_dimension, padding_dimension=args.padding_dim)
>>>>>>> fb9c07fe7914aa6056f4d0766fd66ba1f5c35124

def get_model(args, dataset, output_dir):
    if args.model == "WGAN":
        model = WGAN(
                dirname=output_dir,
                num_batches=args.num_batches,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                n_critic=args.n_critic,
                learning_rate=args.learning_rate,
                layer_sizes=args.layer_sizes,
                critic_layer_sizes=args.critic_layer_sizes,
                latent_distribution=args.latent_distribution,
                latent_dimension=args.latent_dimension,
                dataset=dataset,
                state_dict=args.state_dict,
                data_fn=args.data_fn,
                )
    elif args.model == "GIN":
        model = GINModel(
                dirname=output_dir,
                num_batches=args.num_batches,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                gin_num_layers=args.generator_num_layers,
                latent_distribution=args.latent_distribution,
                dataset=dataset,
                state_dict=args.state_dict,
                tqdm=args.tqdm,
                data_fn=args.data_fn,
                )
    elif args.model == "MLP":
        model = MLPModel(
                dirname=output_dir,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                dataset_name=args.dataset,
                layer_sizes=args.layer_sizes,
                state_dict=args.state_dict,
                )
    elif args.model == "MLPR":
        model = MLPReg(
                dirname=output_dir,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                dataset=dataset,
                num_batches=args.num_batches,
                num_epochs=args.num_epochs,
                layer_sizes=args.layer_sizes,
                state_dict=args.state_dict,
                data_fn=args.data_fn,
                tqdm=args.tqdm,)
    elif args.model == "NICE":
        model = NICEModel(
                dirname=output_dir,
                num_batches=args.num_batches,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                layer_sizes=args.layer_sizes,
                dataset=dataset,
                padding_type=args.padding_type,
                epsilon=args.epsilon,
                n_passes=args.n_passes,
                latent_distribution=args.latent_distribution,
                state_dict=args.state_dict,
                )
    elif args.model == "RealNVP":
        model = RealNVPModel(
                dirname=output_dir,
                num_batches=args.num_batches,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                batch_norm=args.batch_norm,
                mult_layer=args.mult_layer,
                learning_rate=args.learning_rate,
                layer_sizes=args.layer_sizes,
                dataset=dataset,
                padding_type=args.padding_type,
                padding_dimension=args.padding_dim,
                data_fn=args.data_fn,
                n_passes=args.n_passes,
                latent_distribution=args.latent_distribution,
                tqdm=args.tqdm,
                state_dict=args.state_dict,)
    elif args.model == "ConvNVP":
        model = ConvNVPModel(
                dirname=output_dir,
                num_batches=args.num_batches,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                batch_norm=args.batch_norm,
                mult_layer=args.mult_layer,
                learning_rate=args.learning_rate,
                channel_sizes=args.layer_sizes,
                dataset=dataset,
                padding_type=args.padding_type,
                data_fn=args.data_fn,
                n_passes=args.n_passes,
                latent_distribution=args.latent_distribution,
                tqdm=args.tqdm,
                state_dict=args.state_dict,)
    elif args.model == "RealNVPW":
        model = RealNVPWassersteinModel(
                dirname=output_dir,
                num_batches=args.num_batches,
                num_epochs=args.num_epochs,
                n_critic=args.n_critic,
                batch_size=args.batch_size,
                batch_norm=args.batch_norm,
                mult_layer=args.mult_layer,
                learning_rate=args.learning_rate,
                layer_sizes=args.layer_sizes,
                critic_layer_sizes=args.critic_layer_sizes,
                dataset=dataset,
                padding_type=args.padding_type,
                data_fn=args.data_fn,
                n_passes=args.n_passes,
                latent_distribution=args.latent_distribution,
                tqdm=args.tqdm,
                state_dict=args.state_dict,)
    elif args.model == "RealNVPR":
        model = RealNVPRegressionModel(
                dirname=output_dir,
                num_batches=args.num_batches,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                batch_norm=args.batch_norm,
                mult_layer=args.mult_layer,
                learning_rate=args.learning_rate,
                layer_sizes=args.layer_sizes,
                dataset=dataset,
                padding_type=args.padding_type,
                padding_dimension=args.padding_dim,
                n_passes=args.n_passes,
                data_fn=args.data_fn,
                tqdm=args.tqdm,
                state_dict=args.state_dict,)
    elif args.model == "RealNVPImage":
        model = RealNVPImageModel(
                dirname=output_dir,
                num_batches=args.num_batches,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                batch_norm=args.batch_norm,
                mult_layer=args.mult_layer,
                learning_rate=args.learning_rate,
                layer_sizes=args.layer_sizes,
                dataset=dataset,
                padding_type=args.padding_type,
                padding_dimension=args.padding_dim,
                data_fn=args.data_fn,
                n_passes=args.n_passes,
                n_conv=args.n_conv,
                latent_distribution=args.latent_distribution,
                tqdm=args.tqdm,
                state_dict=args.state_dict,)
    elif args.model == "ConvNVPR":
        model = ConvNVPRegressionModel(
                dirname=output_dir,
                num_batches=args.num_batches,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                batch_norm=args.batch_norm,
                mult_layer=args.mult_layer,
                learning_rate=args.learning_rate,
                channel_sizes=args.layer_sizes,
                dataset=dataset,
                padding_type=args.padding_type,
                n_passes=args.n_passes,
                data_fn=args.data_fn,
                tqdm=args.tqdm,
                state_dict=args.state_dict,)
    elif args.model == "PLN":
        model = PartitionedLinearModel(
                dirname=output_dir,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                n_passes=args.n_passes,
                dataset=dataset,
                num_batches=args.num_batches,
                data_fn=args.data_fn,
                state_dict=args.state_dict,)
    elif args.model == "PLNR":
        model = PartitionedLinearRegressionModel(
                dirname=output_dir,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                n_passes=args.n_passes,
                dataset=dataset,
                num_batches=args.num_batches,
                data_fn=args.data_fn,
                state_dict=args.state_dict)
    elif args.model == "VAE":
        model = VAEModel(
                dirname=output_dir,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                dataset=dataset,
                num_batches=args.num_batches,
                num_epochs=args.num_epochs,
                layer_sizes=args.layer_sizes,
                encoder_layer_sizes=args.encoder_layer_sizes,
                state_dict=args.state_dict,
                data_fn=args.data_fn,
                epsilon=args.epsilon,
                tqdm=args.tqdm,
                latent_dimension=args.latent_dimension,
                tunable_decoder_var=args.tunable_decoder_var,
                warm_start = args.warm_start,
                dataset_name = args.dataset,
                latent_off_dimension = args.latent_off_dimension)
    elif args.model == 'ACVAE':
        print(args.epsilon)
        model = ACVAEModel(
                dirname=output_dir,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                dataset=dataset,
                num_batches=args.num_batches,
                num_epochs=args.num_epochs,
                layer_sizes=args.layer_sizes,
                n_passes=args.n_passes,
                state_dict=args.state_dict,
                data_fn=args.data_fn,
                epsilon=args.epsilon,
                tqdm=args.tqdm,
                copy_flow_dataset=args.copy_flow_dataset,
                initialize_inverse=args.initialize_inverse,
                use_fred_covariance=args.use_fred_covariance)
    elif args.model == 'SqVAE':
        model = SqVAEModel(
                dirname=output_dir,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                dataset=dataset,
                num_batches=args.num_batches,
                num_epochs=args.num_epochs,
                layer_sizes=args.layer_sizes,
                encoder_layer_sizes=args.encoder_layer_sizes,
                state_dict=args.state_dict,
                data_fn=args.data_fn,
                epsilon=args.epsilon,
                warm_start=args.warm_start,
                tqdm=args.tqdm,
                train_bias_only=args.train_bias_only,
                latent_dimension = args.latent_dimension,
                if_grid = args.if_grid)
    elif args.model == 'MLP_ACVAE':
        model = MLP_ACVAEModel(
                dirname=output_dir,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                dataset=dataset,
                num_batches=args.num_batches,
                num_epochs=args.num_epochs,
                layer_sizes=args.layer_sizes,
                n_passes=args.n_passes,
                state_dict=args.state_dict,
                data_fn=args.data_fn,
                tqdm=args.tqdm)
    elif args.model == "gammaVAE":
        model = gammaVAEModel(
            dirname=output_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dataset=dataset,
            num_batches=args.num_batches,
            num_epochs=args.num_epochs,
            layer_sizes=args.layer_sizes,
            encoder_layer_sizes=args.encoder_layer_sizes,
            state_dict=args.state_dict,
            data_fn=args.data_fn,
            tqdm=args.tqdm,
            latent_dimension=args.latent_dimension)
    elif args.model == "2Stage_VAE":
        model = VAEModel_2Stage(
                dirname=output_dir,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                dataset=dataset,
                num_batches=args.num_batches,
                num_epochs=args.num_epochs,
                layer_sizes=args.layer_sizes,
                encoder_layer_sizes=args.encoder_layer_sizes,
                state_dict=args.state_dict,
                data_fn=args.data_fn,
                tqdm=args.tqdm,
                latent_dimension=args.latent_dimension,
                stop_point=args.num_batches/2)
    return model


def main(args):
    output_dir = make_output_dir(args.name, args.overwrite, args)
    if args.model != 'MLP':
        dataset = get_dataset(args.dataset, args.dataset_seed, args.padding_dim, args.batch_size,
                              args.dataset_layer_sizes, args)
    model = get_model(args, dataset, output_dir)

    model.train()
    model.plot()
    model.save(final=True)
    return 0


if __name__ == '__main__':
    args = parse_arguments()
    if args.nojit:
        with jax.disable_jit():
            main(args)
    else:
        main(args)
