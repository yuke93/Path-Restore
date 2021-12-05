import argparse
from dynamic_model import DynamicModel
from dynamic_model_test import DynamicModelTest


def read_bool(x: str):
    return x == 'True'


def read_list(x: str):
    return [int(y) for y in x.split(',')]


parser = argparse.ArgumentParser()

# build graph
parser.add_argument('--is_train', type=read_bool, default=True,
                    help='whether to train')
parser.add_argument('--out_channel', type=int, default=3,
                    help='No. of output channels')
parser.add_argument('--filters', type=int, default=32,
                    help='No. of intermediate channels')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--num_path', type=int, default=4,
                    help='No. of dynamic paths')
parser.add_argument('--rb_num', type=int, default=3,
                    help='No. of dynamic routing blocks')
parser.add_argument('--rb_num_share', type=int, default=2,
                    help='No. of shared residual blocks in one dynamic block')
parser.add_argument('--is_share_down_up', type=read_bool, default=False,
                    help='whether to use down/up-sample in shared block')
parser.add_argument('--share_filter_size', type=int, default=3,
                    help='filter size for shared block (not support down-up)')
parser.add_argument('--share_layers', type=int, default=2,
                    help='No. of shared layers in each dynamic block')
parser.add_argument('--shared_mid_filters', type=int, default=-1,
                    help='filters of the middle shared layer')
parser.add_argument('--is_router_action', type=read_bool, default=True,
                    help='whether to use router (pathfinder) action')
parser.add_argument('--is_router_parallel', type=read_bool, default=True,
                    help='whether router parallel to shared block')
parser.add_argument('--router_conv_filters', type=read_list, default=[4, 24],
                    help='architecture of the router')
parser.add_argument('--up_scale', type=int, default=1,
                    help='scale of up-sampling')

# test
parser.add_argument('--data_size', type=int, default=63,
                    help='patch size')
parser.add_argument('--data_stride', type=int, default=53,
                    help='stride for test')
parser.add_argument('--test_set', type=str, default='mine',
                    help='test dataset')
parser.add_argument('--is_save', type=read_bool, default=True,
                    help='whether to save results')
parser.add_argument('--save_folder', type=str, default='../results/',
                    help='directory to save results')
parser.add_argument('--limit_batch', type=int, default=-1,
                    help='max number of patches per inference')

# load model
parser.add_argument('--load_dir_dyn', type=str, default='../model/',
                    help='directory to load dynamic model')
parser.add_argument('--load_which', type=str, default='restorer',
                    help='load `restorer`, `router` or `all`')
parser.add_argument('--load_dir_test', type=str, default='../model/',
                    help='directory to load model for testing')

args = parser.parse_args()

if __name__ == '__main__':
    if args.is_train:
        dynamic_model = DynamicModel(**vars(args))
        dynamic_model.train_dynamic()
    else:
        dynamic_model = DynamicModelTest(**vars(args))

        if args.test_set == 'mine':
            dynamic_model.test_mine()
        else:
            raise NotImplementedError(args.test_set)
