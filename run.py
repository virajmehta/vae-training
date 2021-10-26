import argparse
import jax
from datasets import SphereDataset, LinearGaussianDataset, SigmoidDataset
from vae import VAEModel
from utils import make_output_dir


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help="The name of the experiment and output directory.")
    parser.add_argument('--num_batches', dest='num_batches', type=int, default=15000,
                        help="Number of batches to train on.")
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=100)
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float, default=0.0001)
    parser.add_argument('--padding_dim', type=int, dest='padding_dim', default=0)
    parser.add_argument('-ow', dest='overwrite', action='store_true')
    parser.add_argument('--dataset', dest='dataset', default='4gaussian',
                        choices=["sphere", "linear_gaussian", "sigmoid"])
    parser.add_argument('--layer_sizes', dest='layer_sizes', default='512|512', help="Specify layer sizes for MLP (possibly others later) as integers separated by pipes. Example: 512|512|512")  # NOQA
    parser.add_argument('--encoder_layer_sizes', dest='encoder_layer_sizes', default='512|512', help="Specify layer sizes for MLP (possibly others later) as integers separated by pipes. Example: 512|512|512")  # NOQA
    parser.add_argument('--latent_dim', dest='latent_dimension', type=int, default=100)
    # parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, default=0.01)
    parser.add_argument('-nojit', dest='nojit', action='store_true', help="Disables just-in-time compilation for debugging")  # NOQA
    parser.add_argument('--padding_type', dest='padding_type', default="none", choices=["zero", "gaussian", "none"])
    parser.add_argument('-ds', '--dataset_seed', dest='dataset_seed', type=int, default=69)
    parser.add_argument('--state_dict', dest='state_dict', default=None)
    parser.add_argument('--data_fn', dest="data_fn", default=None)
    parser.add_argument('-ws', '--warm_start', action='store_true')
    parser.add_argument('-ii', '--initialize_inverse', action='store_true')
    parser.add_argument('-ufc', '--use_fred_covariance', action='store_true')
    parser.add_argument('-e', '--epsilon', type=float, default=0.)
    parser.add_argument('-tdv', dest='tunable_decoder_var', action='store_true')
    parser.add_argument('-dn', '--dataset_noise', type=float, default=0.)
    parser.add_argument('-dd', '--dataset_dimension', type=int, default=3)
    parser.add_argument('-wsl', '--warm_start_linear', action='store_true')
    parser.add_argument('-did', '--dataset_intrinsic_dimension', type=int, default=3)
    parser.add_argument('-off', '--latent_off_dimension', type=int, default=1)
    args = parser.parse_args()
    args.model="VAE"
    args.latent_distribution = 'gaussian'
    args.tqdm = True
    return args


def get_dataset(name, seed, padding_dimension, batch_size, args):
    if name == "sphere":
        return SphereDataset(seed, dimension=args.dataset_dimension, padding_dimension=args.padding_dim)
    elif name == "linear_gaussian":
        return LinearGaussianDataset(seed, dimension=args.dataset_dimension, intrinsic_dimension=args.dataset_intrinsic_dimension,
                                     padding_dimension=args.padding_dim, var_added=args.dataset_noise)

    elif name == "sigmoid":
        return SigmoidDataset(seed, dimension=args.dataset_dimension, padding_dimension=args.padding_dim)

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
                              args)
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
