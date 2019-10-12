import math
import os
import torch
import numpy as np
import torchvision
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import skimage

plt.switch_backend('agg')


#########################################################################################
# Training/Testing utils
#########################################################################################

def generate_input_sequence(x_coord, x_adj, img):
    r"""
    Generate input sequence for the decoder concatenating visual features and last generated node in the graph.
    
    :param x_coord: x_(t-1)
    :param x_adj: a_(t-1)
    :param img: conditionin vector emitted by the encoder to compress the input image
    :param bottleneck_decoder:
    :return: input for the decoder
    """
    # if the img features has been processed with attention, we have B x SEQ_LEN x VISUAL_SIZE
    # otherwise, only B X VISUAL_SIZE so we wan't to add the SEQ_LEN dimension
    if len(img.shape) == 2:
        img = img.unsqueeze(1).repeat(1, x_adj.shape[1], 1)
    input_sequence = torch.cat([x_coord, x_adj, img], dim=2)
    return input_sequence


def generate_mask_sequence(size, device="cuda:0"):
    """
    :param size: seq_len
    :param device: cuda or cpu
    :return: mask with future timesteps zero-valued. shape 1 x size x size
    """
    x = torch.ones((size, size), device=device)
    x = torch.tril(x, diagonal=-1)
    return x.unsqueeze(0)


def sample_sigmoid(args, y, sample=False):
    r"""
    Sample from scores between 0 and 1 as means of Bernouolli distribution, or threshold over 0.5
    
    :param args: parsed arguments
    :param y: values to threshold
    :param sample: if True, sample, otherwise, threshold
    :return: sampled/thresholed values, in {0., 1.}
    """
    thresh = 0.5
    if sample:
        y_thresh = torch.rand(y.size(0), y.size(1), y.size(2)).to(args.device)
        y_result = torch.gt(y, y_thresh).float()
    else:
        y_thresh = (torch.ones(y.size(0), y.size(1), y.size(2)) * thresh).to(args.device)
        y_result = torch.gt(y, y_thresh).float()
    return y_result


def decode_adj(adj_output):
    r"""
    Recover the adj matrix A from adj_output
    note: here adj_output has shape [N x max_prev_node], while A has shape [N x N]
    
    :param adj_output: outputs of the decoder
    :return: adjacency matrix A
    """
    '''
    
    '''
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i,::-1][output_start:output_end] # reverse order
    adj_full = np.zeros((adj_output.shape[0]+1, adj_output.shape[0]+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full[1:, 1:]


#########################################################################################
# Utils for logging, handling files and saving pickle data
#########################################################################################

def clear_log(file):
    r"""
    Clear content of the file for logging

    :param file: file location
    """
    with open(file, 'w') as dfile:
        dfile.write("")


def ensure_paths(args):
    r"""
    Ensure that all the directories are existing, create otherwise

    :param args: parsed arguments
    """
    ensure_dir("./output_graph/")
    ensure_dir(args.dataset_path)
    ensure_dir(args.tensorboard_path)
    ensure_dir(args.logs_path)
    ensure_dir(args.plots_path)
    ensure_dir(args.checkpoints_path)
    ensure_dir(args.losses_path)
    ensure_dir(args.statistics_path)


def ensure_dir(dir_path):
    r"""
    Ensure that all a directory exists, create otherwise

    :param args: path of a directory
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def update_writer(writer, measure, value_train, value_valid, epoch):
    r"""
    Updates tensorboard writer after one train/validation epoch
    
    :param writer: Tensorboard writer
    :param measure: measure name
    :param value_train: value for train epoch
    :param value_valid: value for validation epoch
    :param epoch: epoch number
    """
    writer.add_scalar(f'{measure} train', value_train, epoch)
    writer.add_scalar(f'{measure} valid', value_valid, epoch)


def print_and_log(s, file):
    r"""
    Print a string and log it to file. Used to store logs of the experiments.
    
    :param s: string
    :param file: file to log (append)
    """
    print(s)
    with open(file, 'a') as dfile:
        dfile.write(s + "\n")

    
def save_statistics(args, avg, std):
    r"""
    Save statistics for the current experiment (test) to a file.
    Appending is used to possibly save results from multiple experiments (e.g. to gather means and stds)
    
    :param args: parsed arguments
    :param avg: average score over the test set
    :param std: std over the test set
    """
    file_name = args.statistics_path + args.file_name + ".txt"
    avg = list(avg)
    std = list(std)
    row = avg + std
    row = [str(i) for i in row]
    row = ",".join(row) + "\n"
    with open(file_name, "a") as f:
        f.write(row)


def save_losses(args, loss_train, loss_train_adj, loss_train_coord, loss_valid, loss_valid_adj, loss_valid_coord):
    r"""
    Save values for train and validation losses and losses subcomponents.
    
    :param args: parsed arguments
    :param loss_train: total loss value for train epoch
    :param loss_train_adj: BCE on A for train epoch
    :param loss_train_coord: MSE on X for train epoch
    :param loss_valid: total loss value for valid epoch
    :param loss_valid_adj: BCE on A for valid epoch
    :param loss_valid_coord: MSE on X for valid epoch
    """
    file_name = args.losses_path + args.file_name + ".txt"
    row = loss_train, loss_train_adj, loss_train_coord, loss_valid, loss_valid_adj, loss_valid_coord
    row = [str(i) for i in row]
    row = ",".join(row) + "\n"
    with open(file_name, "a") as f:
        f.write(row)


#########################################################################################
# Utils for showing or saving plots
#########################################################################################


def full_frame(plt, width=0.64, height=0.64):
    r"""
    Generates a particular tight layout for Pyplot plots

    :param plt: pyplot
    :param width: width, default is 64 pixels
    :param height: height, default is 64 pixels
    :return:
    """
    import matplotlib as mpl
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)


def full_frame_high_res(plt, width=3.2, height=3.2):
    r"""
    Generates a particular tight layout for Pyplot plots, at higher resolution
    
    :param plt: pyplot
    :param width: width, default is 320 pixels
    :param height: height, default is 320 pixels
    :return:
    """
    import matplotlib as mpl
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)


def save_cnn_plots(epoch, i, y, output, plot_each=10, all_dataset=False, dir_plots=""):
    r"""
    Save plots generated during the cnn pre-training, output by the decoder
    
    :param epoch: current epoch
    :param i: index in the dataloader
    :param y: the real reconstructed image
    :param output: the reconstructed reconstructed image
    :param plot_each: plot each x epochs
    :param all_dataset: plot all data
    :param dir_plots: directory for plotting
    """
    if epoch % plot_each == 0:
        if i == 0 or all_dataset:
            img_recon = torchvision.utils.make_grid(output, nrow=int(math.sqrt(output.shape[0])))
            torchvision.utils.save_image(img_recon, dir_plots + "/{}_{}_recon.png".format(i, epoch))
            if epoch == 0:
                img_real = torchvision.utils.make_grid(y, nrow=int(math.sqrt(y.shape[0])))
                torchvision.utils.save_image(img_real, dir_plots + "/{}_{}_real.png".format(i, epoch))


def plot_output_graph(epoch, id, adj, coord, plots_path, no_edges=False, is_eval=True, no_points=True):
    r"""
    Plot graphs reconstructed by the model.
    
    :param epoch: current epoch
    :param id: id of the datapoint in the data
    :param adj: adjacency matrix A
    :param coord: node coordinate matrix X
    :param plots_path: path for saving plots
    :param no_edges: whether to not plot edges
    :param is_eval: if it is validation or train epoch
    :param no_points: whether to not plot points
    """
    full_frame(plt)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    seq_len = adj.shape[0]
    max_prev_nodes = 5
    greedy_adj = adj > 0.5
    points = set()
    
    for i in range(seq_len - 1):
        for n in range(1, min(i + 1, max_prev_nodes)):
            if greedy_adj[i, n - 1]:
                if not no_edges:
                    plt.plot((coord[i][0].item(), coord[i - n][0].item()), (coord[i][1].item(), coord[i - n][1].item()),
                             c='k', linewidth=2)
                points.add((coord[i - n][0].item(), coord[i - n][1].item()))
                points.add((coord[i][0].item(), coord[i][1].item()))
    
    if not no_points and len(points):
        points = list(zip(*list(points)))
        plt.plot(points[0], points[1], 'ro', alpha=0.5)
    
    plt.savefig(plots_path + f"{'' if is_eval else 'train'}{id}_{epoch}.png")
    plt.clf()
    plt.close('all')


def plot_point_cloud(adj, coord, points):
    r"""
    Plot point cloud generated while computing StreetMover distance, over the original graph
    
    :param adj: adjacency matrix A
    :param coord: node coordinate matrix X
    :param points: point cloud
    """
    full_frame_high_res(plt)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    seq_len = adj.shape[0]
    max_prev_nodes = 5
    greedy_adj = adj > 0.5
    
    for i in range(seq_len - 1):
        for n in range(1, min(i + 1, max_prev_nodes)):
            if greedy_adj[i, n - 1]:
                plt.plot((coord[i][0].item(), coord[i - n][0].item()), (coord[i][1].item(), coord[i - n][1].item()),
                         c='k', linewidth=4)
    
    if len(points):
        points = list(zip(*list(points)))
        plt.plot(points[0], points[1], 'ro', alpha=1, markersize=2)
    
    plt.show()
    # plt.savefig("point_cloud.png")
    plt.clf()
    plt.close('all')


def plot_histogram_streetmover(x, args):
    r"""
    Plot histogram of streetmover distance, to better analyze the performance of different models
    
    :param x: streetmover distances
    :param args: parsed arguments
    """
    
    sns.set(color_codes=True, style="white")
    
    sns_plot = sns.distplot(x, bins=int(np.max(x) / 0.002), kde=False)  # kde_kws=dict(bw=0.01))
    sns_plot.set_xticks(np.linspace(0, 0.08, 5))
    sns_plot.set_yticks(np.linspace(0, 10000, 6))
    sns_plot.set_xlim(0, 0.09)
    sns_plot.set_ylim(0, 10000)
    plt.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=1, alpha=.5)
    
    mean = np.mean(x)
    median = np.median(x)
    std = np.std(x)
    plt.axvline(mean, 0, 40, ls='--', c='r', label="mean")
    plt.axvline(median, 0, 40, ls='--', c='g', label="median")
    plt.text(0.088, 4000, f"Mean: {str(mean)[:6]}\nMedian: {str(median)[:6]}\nStd: {str(std)[:6]}", fontsize=20,
                       fontdict=dict(horizontalalignment='right'))
    sns_plot.set_ylabel("N datapoints", fontsize=20)
    sns_plot.set_xlabel("StreetMover distance", fontsize=20)
    sns.despine(ax=sns_plot, left=True, bottom=True)
    # sns_plot.set_title(f"Mean: {str(mean)[:6]}\nMedian: {str(median)[:6]}\nStd: {str(std)[:6]}", fontsize=20,
    #                   fontdict=dict(horizontalalignment='right'))
    sns_plot.legend(prop={'size': 20})
    sns_plot.figure.savefig(f"{args.statistics_path}/{args.file_name}.png", bbox_inches='tight')


def visualize_attention_image(original_img, alphas):
    r"""
    Visualize context attention weights over the image.
    
    :param original_img: original image
    :param alphas: attention weight matrices
    """
    if not isinstance(alphas, list) and alphas.shape[-1] == 900:
        alphas = [alphas[0, i, :].reshape(30, 30) for i in range(alphas.shape[1])]
    for t, current_alpha in enumerate(alphas):
        if t > 50:
            break
        plt.subplot(np.ceil(len(alphas) / 5.), 5, t + 1)
        
        plt.text(0, 1, '%s' % (t), color='black', backgroundcolor='white', fontsize=12)
        if t == 0:
            plt.imshow(original_img[0, 0].cpu(), cmap='gray')
        smooth = True
        if smooth:
            alpha = skimage.transform.pyramid_expand(np.clip(current_alpha.cpu().numpy(), 0.5, 1.5), upscale=24, sigma=16)
        else:
            alpha = skimage.transform.resize(current_alpha.cpu().numpy(), [14 * 24, 14 * 24])
        if t > 0:
            plt.imshow(alpha, alpha=0.5, cmap=cm.Reds)
            plt.colorbar()
        plt.axis('off')
    plt.show()


def visualize_attention_sequence(a, idx, plot_values=False):
    r"""
    Visualize self-attention weights over the sequence.
    
    :param a: attention weights
    :param idx: index of the current datapoint
    :param plot_values: if True, plot attention values
    """
    from matplotlib.pyplot import figure
    figure(num=None, figsize=(8, 8), dpi=80)
    fig, ax = plt.subplots()
    ax.imshow(a, cmap=plt.cm.Blues, interpolation='nearest') # plt.cm.binary
    plt.xticks([])
    plt.yticks([])
    if plot_values:
        for i in range(len(a)):
            for j in range(len(a[0])):
                text = ax.text(j, i, round(a[i, j], 2),
                               ha="center", va="center", color="black")
    fig.tight_layout()
    
    plt.savefig(f"attention/{idx}.png", bbox_inches='tight')