python run.py sphere_dd3_pd3_ld_6_eps-3 --dataset sphere --encoder_layer_sizes "200|200|200" --layer_sizes "200|200|200" -ow --latent_dim 6  --padding_dim 3 -dd 3 --num_batches 150000 --epsilon -3 -tdv
python run.py sphere_dd3_pd13_ld_8_eps-3 --dataset sphere --encoder_layer_sizes "200|200|200" --layer_sizes "200|200|200" -ow --latent_dim 8  --padding_dim 13 -dd 3 --num_batches 150000 --epsilon -3 -tdv
python run.py sphere_dd5_pd16_ld_16_eps-3 --dataset sphere --encoder_layer_sizes "200|200|200" --layer_sizes "200|200|200" -ow --latent_dim 16  --padding_dim 16 -dd 5 --num_batches 150000 --epsilon -3 -tdv
python run.py sphere_dd5_pd5_ld_10_eps-3 --dataset sphere --encoder_layer_sizes "200|200|200" --layer_sizes "200|200|200" -ow --latent_dim 10  --padding_dim 5 -dd 5 --num_batches 150000 --epsilon -3 -tdv
python run.py sphere_dd7_pd7_ld_13_eps-3 --dataset sphere --encoder_layer_sizes "200|200|200" --layer_sizes "200|200|200" -ow --latent_dim 13  --padding_dim 7 -dd 7 --num_batches 150000 --epsilon -3 -tdv

python run.py sphere_dd3_pd3_ld_6_eps-3_seed24 --dataset sphere --encoder_layer_sizes "200|200|200" --layer_sizes "200|200|200" -ow --latent_dim 6  --padding_dim 3 -dd 3 --num_batches 150000 --epsilon -3 -tdv --dataset_seed 24
python run.py sphere_dd3_pd13_ld_8_eps-3_seed24 --dataset sphere --encoder_layer_sizes "200|200|200" --layer_sizes "200|200|200" -ow --latent_dim 8  --padding_dim 13 -dd 3 --num_batches 150000 --epsilon -3 -tdv --dataset_seed 24
python run.py sphere_dd5_pd16_ld_16_eps-3_seed24 --dataset sphere --encoder_layer_sizes "200|200|200" --layer_sizes "200|200|200" -ow --latent_dim 16  --padding_dim 16 -dd 5 --num_batches 150000 --epsilon -3 -tdv --dataset_seed 24
python run.py sphere_dd5_pd5_ld_10_eps-3_seed24 --dataset sphere --encoder_layer_sizes "200|200|200" --layer_sizes "200|200|200" -ow --latent_dim 10  --padding_dim 5 -dd 5 --num_batches 150000 --epsilon -3 -tdv --dataset_seed 24
python run.py sphere_dd7_pd7_ld_13_eps-3_seed24 --dataset sphere --encoder_layer_sizes "200|200|200" --layer_sizes "200|200|200" -ow --latent_dim 13  --padding_dim 7 -dd 7 --num_batches 150000 --epsilon -3 -tdv --dataset_seed 24

python run.py sphere_dd3_pd3_ld_6_eps-3_seed48 --dataset sphere --encoder_layer_sizes "200|200|200" --layer_sizes "200|200|200" -ow --latent_dim 6  --padding_dim 3 -dd 3 --num_batches 150000 --epsilon -3 -tdv --dataset_seed 48
python run.py sphere_dd3_pd13_ld_8_eps-3_seed48 --dataset sphere --encoder_layer_sizes "200|200|200" --layer_sizes "200|200|200" -ow --latent_dim 8  --padding_dim 13 -dd 3 --num_batches 150000 --epsilon -3 -tdv --dataset_seed 48
python run.py sphere_dd5_pd16_ld_16_eps-3_seed48 --dataset sphere --encoder_layer_sizes "200|200|200" --layer_sizes "200|200|200" -ow --latent_dim 16  --padding_dim 16 -dd 5 --num_batches 150000 --epsilon -3 -tdv --dataset_seed 48
python run.py sphere_dd5_pd5_ld_10_eps-3_seed48 --dataset sphere --encoder_layer_sizes "200|200|200" --layer_sizes "200|200|200" -ow --latent_dim 10  --padding_dim 5 -dd 5 --num_batches 150000 --epsilon -3 -tdv --dataset_seed 48
python run.py sphere_dd7_pd7_ld_13_eps-3_seed48 --dataset sphere --encoder_layer_sizes "200|200|200" --layer_sizes "200|200|200" -ow --latent_dim 13  --padding_dim 7 -dd 7 --num_batches 150000 --epsilon -3 -tdv --dataset_seed 48

