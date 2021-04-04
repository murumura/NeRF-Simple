import configargparse
from utils import boolean_string

def get_options():
    parser = configargparse.ArgumentParser()
    
    # output/logging/saving options
    parser.add_argument("--log_dir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')

    parser.add_argument('--out_dir', type=str, default='./output', 
                        help='Directory for training output.')

    parser.add_argument('--eval_dir', type=str, default='./evals', 
                        help='Directory for evaluation output.')

    parser.add_argument('--state_dir', type=str, default=None, 
                        help='where to load state from.')

    parser.add_argument('--validate_every', type=int, default=500, 
                        help='number of training iterations after which to validate.')
    
    parser.add_argument('--save_every', type=int, default=1, 
                        help='number of training epoches after which to validate.')

    parser.add_argument("--render_img_every", type=int, default=500, 
                        help='number of training iterations after which to render image logging')

    parser.add_argument("--render_video_every", type=int, default=50000, 
                        help='number of training iterations after which to render video logging')

    # training/evaluation misc options
    parser.add_argument('--cuda', type=boolean_string, default=True, 
                        help='enable CUDA.')

    parser.add_argument('--train', type=boolean_string, default=False, 
                        help='train mode.')

    parser.add_argument('--eval', type=boolean_string, default=False, 
                        help='evaluation mode.')
                        
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')

    # training main options
    parser.add_argument('--epochs', type=int, default=16, 
                        help='number of epochs')

    parser.add_argument('--batch_size', type=int, default=2*1024,
                        help='batch size')

    parser.add_argument('--chunk_size', type=int, default=1024,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type:sgd/adam')

    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate')

    # volume rendering option
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='factor to perturb the model prediction of sigma')

    # position encoding option
    parser.add_argument('--num_encoding_fn_xyz', type=int, default=10,
                        help='number of encoding functions to use in the positional encoding of the coordinates.')

    parser.add_argument('--num_encoding_fn_dir', type=int, default=4,
                        help='number of encoding functions to use in the positional encoding of the direction.')

    # sampling option
    parser.add_argument('--N_coarse_samples', type=int, default=64,
                        help='number of coarse samples')

    parser.add_argument('--N_importance_samples', type=int, default=128,
                        help='number of additional fine samples')

    parser.add_argument('--use_disp', type=boolean_string, default=False,
                        help='whether to sample in disparity space (inverse depth)')

    parser.add_argument('--perturb', type=boolean_string, default=True,
                        help='factor to perturb depth sampling points')
    
    parser.add_argument("--lindisp", type=boolean_string, default=False, 
                        help='sampling linearly in disparity rather than depth, using in LLFF dataset')

    parser.add_argument("--log_sampling_xyz", type=boolean_string, default=True, 
                        help='Whether or not to perform log sampling in the positional encoding of the coordinates.')

    parser.add_argument("--log_sampling_dir", type=boolean_string, default=True, 
                        help='Whether or not to perform log sampling in the positional encoding of the direction.')

    # dataset options
    parser.add_argument('--dataset_dir', type=str,
                        default='/mvwf/data/datasets/nerf_synthetic/lego',
                        help='root directory of dataset')

    parser.add_argument("--dataset_type", type=str, default='blender', 
                        help='options: blender / deepvoxels')

    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')

    parser.add_argument("--exp_name", type=str, 
                        help='experiment name')

    ## blender flags
    parser.add_argument("--white_bkgd", type=boolean_string, default=False, 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    return parser