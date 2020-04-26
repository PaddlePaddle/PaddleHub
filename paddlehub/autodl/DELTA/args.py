import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--prefix', default=None, type=str, help='prefix for model id')
parser.add_argument('--dataset', default='PetImages', type=str, help='dataset')
parser.add_argument(
    '--seed',
    default=None,
    type=int,
    help='random seed (default: None, i.e., not fix the randomness).')
parser.add_argument('--batch_size', default=20, type=int, help='batch_size.')
parser.add_argument('--delta_reg', default=0.1, type=float, help='delta_reg.')
parser.add_argument('--wd_rate', default=1e-4, type=float, help='wd_rate.')
parser.add_argument(
    '--use_cuda', default=0, type=int, help='use_cuda device. -1 cpu.')
parser.add_argument('--num_epoch', default=100, type=int, help='num_epoch.')
parser.add_argument('--outdir', default='outdir', type=str, help='outdir')
parser.add_argument(
    '--pretrained_model',
    default='./pretrained_models/ResNet101_pretrained',
    type=str,
    help='pretrained model pathname')

args = parser.parse_args()
