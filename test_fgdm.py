# -*- coding: utf-8 -*-

import argparse
import os
import logging
import sys
import platform
import pdb
import test_ddgan_pelvic
import train_ddgan_pelvic
import torch
import skimage.io
from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from skimage.metrics import structural_similarity as SSIM


if platform.system() == 'Windows':
    sys.path.append(r"E:\我的坚果云\sourcecode\python\util")
else:
    sys.path.append("/home/chenxu/我的坚果云/sourcecode/python/util")

import common_pelvic_pt as common_pelvic
import common_net_pt as common_net
import common_metrics


def produce(args, netG, netSobel, x, coeff, pos_coeff, T=4, eta=10):
    lpf = train_ddgan_pelvic.q_sample(coeff, x, T)
    with torch.no_grad():
        sobel_x, sobel_y = netSobel(test_img)
        hpf = torch.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
        hpf = torch.where(hpf < eta, 0, hpf)

    fake_sample = sample_from_model(pos_coeff, netG, netSobel, T, lpf, T, args, hpf)
    pdb.set_trace()
    return fake_sample


def main(args, device):
    netG = NCSNpp(args, double_channels=True).to(device)
    ckpt = torch.load(os.path.join(args.checkpoint_dir, "netG_last.pth"), map_location=device)

    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()

    netSobel = train_ddgan_pelvic.Sobel(args.num_channels).to(device)
    netSobel.eval()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    test_data_s, test_data_t, _, _ = common_pelvic.load_test_data(args.data_dir, valid=True)


    patch_shape = (args.num_channels, args.image_size, args.image_size)
    coeff = train_ddgan_pelvic.Diffusion_Coefficients(args, device)
    pos_coeff = train_ddgan_pelvic.Posterior_Coefficients(args, device)

    psnr_list = numpy.zeros((len(test_data_t.shape[0],)), numpy.float32)
    ssim_list = numpy.zeros((len(test_data_t.shape[0],)), numpy.float32)
    for i in range(len(test_data_t)):
        im_ts = common_net.produce_results(device, lambda x: produce(args, netG, netSobel, x, coeff, pos_coeff),
                                           [patch_shape, ], [test_data_t[i], ], data_shape=test_data_t[i].shape,
                                           patch_shape=patch_shape)
        psnr_list[i] = common_metrics.psnr(im_ts, test_data_s[i])
        ssim_list[i] = SSIM(im_ts, test_data_s[i])

        common_pelvic.save_nii(im_ts, "syn_ts_%d.nii.gz" % i)

    msg = "psnr_list:%s/%s  ssim_list:%s/%s" % (psnr_list.mean(), psnr_list.std(), ssim_list.mean(), ssim_list.std())
    print(msg)
    with open(os.path.join(args.output_dir, "result.txt"), "w") as f:
        f.write(msg)

    return
    ####----
    eta = 10
    test_img = torch.from_numpy(test_data_s[0][128:129, :, :]).unsqueeze(0).to(device)

    coeff = train_ddgan_pelvic.Diffusion_Coefficients(args, device)
    pos_coeff = train_ddgan_pelvic.Posterior_Coefficients(args, device)

    lpf = train_ddgan_pelvic.q_sample(coeff, test_img, torch.full((1,), args.num_timesteps, device=device, dtype=torch.long))

    with torch.no_grad():
        sobel_x, sobel_y = netSobel(test_img)
        hpf = torch.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
        hpf = torch.where(hpf < eta, 0, hpf)

    fake_sample = sample_from_model(pos_coeff, netG, netSobel, args.num_timesteps, lpf, None, args, hpf)

    fake_sample_np = fake_sample.detach().cpu().numpy()
    gen_images = common_pelvic.generate_display_image(fake_sample_np, is_seg=False)
    skimage.io.imsave(os.path.join(args.output_dir, "gen_images.jpg"), gen_images)

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
    parser.add_argument('--num_timesteps', type=int, default=4)
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
