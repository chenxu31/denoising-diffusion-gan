# -*- coding: utf-8 -*-

import argparse
import os
import logging
import sys
import pdb
import test_ddgan_pelvic
import torch


def main(args, device):
    netG = test_ddgan_pelvic.NCSNpp(args).to(device)
    ckpt = torch.load(os.path.join(args.checkpoint_dir, "netG_last.pth"), map_location=device)

    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()

    T = test_ddgan_pelvic.get_time_schedule(args, device)
    pos_coeff = test_ddgan_pelvic.Posterior_Coefficients(args, device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    x_t_1 = torch.randn(4, args.num_channels, args.image_size, args.image_size).to(device)
    fake_sample = test_ddgan_pelvic.sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, T, args)

    pdb.set_trace()

    print("xxx")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=int, default=0, help="gpu device id")
    parser.add_argument('--data_dir', type=str, default=r'/home/chenxu/datasets/pelvic/h5_data_nonrigid', help='path of the dataset')
    parser.add_argument('--num_channels', type=int, default=1, help="depth of patch")
    parser.add_argument('--image_size', type=int, default=256, help="depth of patch")
    parser.add_argument('--checkpoint_dir', type=str, default='', help="directory of pretrained checkpoint files")
    parser.add_argument('--output_dir', type=str, default='', help="the output directory")
    parser.add_argument('--num_channels_dae', type=int, default=64, help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3, help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 1, 2, 2, 4, 4], help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0., help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True, help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True, help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True, help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True, help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan', help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'], help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'], help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'], help='progressive combine method.')
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'], help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16., help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=2)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--centered', action='store_false', default=True, help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1, help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20., help='beta_max for diffusion')

    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")

    main(args, device)
