import argparse

from utils.utils import *


def get_parser():
    r"""
    Create custom ArgumentParser
    
    :return: an ArgumentParser instance with all default + custom parameters passed at execution
    """
    parser = argparse.ArgumentParser()
    
    # Constants and seed
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed')
    parser.add_argument('--device', default="cuda:0", type=str,
                        help='training device')
    parser.add_argument('--epochs', default=200, type=int,
                        help='max number of epochs')
    parser.add_argument('--max_time', default=72000, type=int,
                        help='maximum time in seconds for training')

    # Tuned hyper-parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='size of each batch')
    parser.add_argument('--lr_rate', default=3e-4, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=0.00001, type=float,
                        help='weight decay for encoder and decoder optimizers')
    parser.add_argument('--dropout', default=.0, type=float,
                        help='dropout in internal lstm layers')
    parser.add_argument('--n_heads', default=8, type=int,
                        help='number of heads in MultiHeadAttention layers')
    parser.add_argument('--N', default=12, type=int,
                        help='number of GGT layers in DecoderGGT')
    parser.add_argument('--hidden_size', default=16, type=int,
                        help='size of the hidden layer output from GraphRNN node-level lstm')
    parser.add_argument('--lamb', default=0.5, type=float,
                        help='scale for the coordinate loss component wrt to adjacency component')
    
    # Fixed hyper-parameters
    parser.add_argument('--augment', default=False, type=bool,
                        help='use augmented data with flip and rotation')
    parser.add_argument('--features_dim', default=900, type=int,
                        help='dimension of the visual features')
    parser.add_argument('--pretrained_encoder', default=True, type=bool,
                        help='use visual features, otherwise use raw images')
    parser.add_argument('--d_model', default=256, type=int,
                        help='dimension of the hidden state in decoder')
    parser.add_argument('--n_hidden', default=1, type=int,
                        help='number of hidden layers in the lstm')
    parser.add_argument('--all_history', default=True, type=bool,
                        help='input all history in Context Attention, otherwise only the last generated node')
    parser.add_argument('--max_prev_node', default=4, type=int,
                        help='max number of nodes in the graph')
    parser.add_argument('--max_n_nodes', default=19 + 2, type=int,
                        help='max number of nodes in the graph = max number of nodes + 2 termination tokens')

    # plot self-attention and context attention
    parser.add_argument('--visualize_attention_sequence', default=False, type=bool,
                        help='Plot visualizations for attention over sequence')
    parser.add_argument('--visualize_attention_image', default=False, type=bool,
                        help='Plot visualizations for attention over sequence')
    
    # paths for outputs
    parser.add_argument('--dataset_path', default="./data/", type=str,
                        help='data path')
    parser.add_argument('--tensorboard_path', default="./output_graph/tensorboard/", type=str,
                        help='tensorboard path')
    parser.add_argument('--logs_path', default="./output_graph/logs/", type=str,
                        help='logs path')
    parser.add_argument('--plots_path', default="./output_graph/plots/", type=str,
                        help='plots path')
    parser.add_argument('--checkpoints_base', default="./output_graph/checkpoints/", type=str,
                        help='checkpoints path')
    parser.add_argument('--losses_path', default="./output_graph/losses/", type=str,
                        help='losses path')
    parser.add_argument('--statistics_path', default="./output_graph/statistics/", type=str,
                        help='statistics path')
    
    # experiment configuration and testing
    parser.add_argument('--experiment', default="GraphRNNAtt", type=str,
                        help='name of the experiment to load from the known experiments in config.py. Overrides decoder and encoder')
    parser.add_argument('--decoder', default="DecoderGGT", type=str,
                        help='name of the decoder model')
    parser.add_argument('--encoder', default="EncoderCNNAtt", type=str,
                        help='name of the encoder model')
    parser.add_argument('--notes', default="", type=str,
                        help='additional notes on the experiment, to put in the file name')
    parser.add_argument('--is_test', default=True, type=bool,
                        help='Test, otherwise train')

    return parser


def set_default_args(args):
    r"""
    Modify arguments applying constraints, generates file_name for outputs, ensure existence of output
    directories and files.
    
    :param args: ArgumentParser instance
    """
    # clamp lambda
    args.lamb = min(args.lamb, 1)
    args.lamb = max(args.lamb, 0)
    args.device = torch.device(args.device)
    
    # define experiment file name
    args.file_name = '{} {} {} lr{}_wd{}_bsz{}'.format(args.decoder, args.encoder, args.notes, args.lr_rate,
                                                       args.weight_decay, args.batch_size, args.seed)
    if args.decoder == "DecoderGGT":
        args.file_name += '_N{}_heads{}'.format(args.N, args.n_heads)
    if "DecoderGraphRNN" in args.decoder:
        args.file_name += '_hidden{}'.format(args.hidden_size)
    args.file_name += "_seed{}".format(args.seed)

    # set output files and directories
    args.plots_path = "./output_graph/plots/" + args.file_name + "/"
    args.file_tensorboard = args.tensorboard_path + args.file_name
    ensure_dir(args.checkpoints_base)
    args.checkpoints_path = args.checkpoints_base + args.file_name
    args.file_logs = args.logs_path + args.file_name + ".txt"
    args.train_split = "augment" if args.augment else "train"
    
    # ensure that all directories in the args exist, and clear the log files if already existing
    ensure_paths(args)
    if not args.is_test:
        clear_log(args.file_logs)
        clear_log(args.losses_path + args.file_name + ".txt")
    
    if args.is_test:
        args.epochs = 0
        args.batch_size = 1  # testing is implemented without batching
        args.plots_path += "test/"
        ensure_dir(args.plots_path)
    
    # for one-shot generation, model all the adjacency matrix
    if args.decoder == "DecoderMLP":
        args.max_prev_node = -1
    
    return args


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args = set_default_args(args)
    print("\n".join([str(x) for x in args.__dict__.items()]))
