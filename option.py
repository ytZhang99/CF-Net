import argparse

parser = argparse.ArgumentParser(description='CF_Net')

parser.add_argument('--debug', action='store_true', help='Enables debug mode')

parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--validation', '-v', action='store_true',
                    help='set this option to validate after training')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')

# Data specifications
parser.add_argument('--dir_train', type=str, default='dataset/train_data/',
                    help='training dataset directory')
parser.add_argument('--dir_val', type=str, default='dataset/val_data/',
                    help='validation dataset directory')
parser.add_argument('--dir_test', type=str, default='dataset/test_data/',
                    help='test dataset directory')
parser.add_argument('--model_path', type=str, default='model/',
                    help='trained model directory')
parser.add_argument('--model', type=str, default='model_x4.pth',
                    help='model name')
parser.add_argument('--ext', type=str, default='.png',
                    help='extension of image files')
parser.add_argument('--scale', type=int, default=4,
                    help='super resolution scale')
parser.add_argument('--batch_size', type=int, default=4,
                    help='number of batches each time')
parser.add_argument('--patch_size', type=int, default=64,
                    help='input patch size')
parser.add_argument('--save_dir', type=str, default='test_results',
                    help='test results directory')

# Model specifications
parser.add_argument('--in_channels', type=int, default=3,
                    help='number of input channels')
parser.add_argument('--out_channels', type=int, default=3,
                    help='number of output channels')
parser.add_argument('--num_features', type=int, default=64,
                    help='number of features')
parser.add_argument('--num_groups', type=int, default=6,
                    help='number of projection groups in SRB and CFB')
parser.add_argument('--num_cfbs', type=int, default=3,
                    help='number of CFBs in CF_Net')
parser.add_argument('--num_steps', type=int, default=2)
parser.add_argument('--act_type', type=str, default='prelu',
                    help='type of activation function')

parser.add_argument('--eval', action='store_true',
                    help='evaluate the test results')

args = parser.parse_args()
