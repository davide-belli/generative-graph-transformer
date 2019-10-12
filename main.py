from torch.utils.data import DataLoader
from torch.nn import BCELoss, MSELoss
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import pickle
import time
from tensorboardX import SummaryWriter

from arguments import get_parser, set_default_args
from utils.configs import Configs
from utils.dataset import ToulouseRoadNetworkDataset, custom_collate_fn
from models.models_encoder import CNNEncoderSimple, CNNEncoderAtt
from models.models_decoder import DecoderGRU, DecoderMLP, DecoderGRUAtt, DecoderGGT, DecoderGraphRNN, DecoderGraphRNNAtt
from metrics.statistics import compute_statistics, compute_statistics_MLP


# ########################################################################################
# ########################################################################################
# ####################################  HELPERS  #########################################
# ########################################################################################
# ########################################################################################


def load_encoder(args):
    r"""
    Load the PyTorch model for the encoder specified in args.
    If pretrained_encoder, load the checkpoint of the encoder saved after trainin for reconstruction (pretrain_encoder.py).
    If test, load the state dictionary from checkpoint.
    Create optimizer for the encoder.
    
    :param args: parsed arguments
    :return: encoder network, optimizer for the encoder
    """
    if args.encoder == "EncoderCNNAtt":
        if args.all_history:
            encoder = CNNEncoderAtt(adj_size=args.max_prev_node * args.max_n_nodes, coord_size=2 * args.max_n_nodes)
        else:
            encoder = CNNEncoderAtt()
        if not args.is_test and args.pretrained_encoder:
            encoder.cnn.load_state_dict(torch.load(
                f'./output_cnn/CNN_autoencoder/checkpoints_for_context_attention/CNN_encoder.pth'))
    elif args.encoder == "EncoderCNN":
        encoder = CNNEncoderSimple()
        if not args.is_test and args.pretrained_encoder:
            encoder.load_state_dict(torch.load(
                f'./output_cnn/CNN_autoencoder/checkpoints/CNN_encoder.pth'))
    else:
        raise ValueError("Encoder type should be 'EncoderCNNAtt' or 'EncoderCNN'")
    
    encoder = encoder.to(args.device)
    if args.is_test:
        encoder.load_state_dict(torch.load(args.checkpoints_path + "/encoder.pth"))
        encoder.eval()
    
    optimizer_enc = torch.optim.Adam(list(encoder.parameters()), lr=args.lr_rate, weight_decay=args.weight_decay)
    
    return encoder, optimizer_enc


def load_decoder(args):
    r"""
    Load the PyTorch model for the decoder specified in args.
    If test, load the state dictionary from checkpoint.
    Create optimizer for the decoder.
    
    :param args: parsed arguments
    :return: decoder network, optimizer for the decoder
    """
    if args.decoder == "DecoderGRU":
        decoder = DecoderGRU(d_model=args.d_model, n_hidden=args.n_hidden, dropout=args.dropout,
                             adj_size=args.max_prev_node, coord_size=2, visual_size=args.features_dim)
    elif args.decoder == "DecoderGRUAtt":
        decoder = DecoderGRUAtt(d_model=args.d_model, n_hidden=args.n_hidden, dropout=args.dropout,
                                adj_size=args.max_prev_node, coord_size=2, visual_size=args.features_dim,
                                concat=False)
    elif args.decoder == "DecoderGGT":
        decoder = DecoderGGT(d_model=args.d_model, N=args.N, dropout=args.dropout, adj_size=args.max_prev_node,
                             coord_size=2, visual_size=args.features_dim, heads=args.n_heads,
                             max_n_nodes=args.max_n_nodes, use_pe=True)
    elif args.decoder == "DecoderMLP":
        decoder = DecoderMLP(max_n_nodes=args.max_n_nodes)
    elif args.decoder == "DecoderGraphRNN":
        decoder = DecoderGraphRNN(d_model=args.d_model, n_hidden=args.n_hidden, dropout=args.dropout,
                                  adj_size=args.max_prev_node, coord_size=2, visual_size=args.features_dim,
                                  hidden_size=args.hidden_size)
    elif args.decoder == "DecoderGraphRNNAtt":
        decoder = DecoderGraphRNNAtt(d_model=args.d_model, n_hidden=args.n_hidden, dropout=args.dropout,
                                     adj_size=args.max_prev_node, coord_size=2, visual_size=args.features_dim,
                                     hidden_size=args.hidden_size, concat=False)
    else:
        raise ValueError("Unknown decoder type!")
    
    decoder = decoder.to(args.device)
    if args.is_test:
        decoder.load_state_dict(torch.load(args.checkpoints_path + "/decoder.pth"))
        decoder.eval()
    
    optimizer = torch.optim.Adam(list(decoder.parameters()), lr=args.lr_rate, weight_decay=args.weight_decay)
    
    return decoder, optimizer


def get_epoch_fn(args):
    r"""
    Return the function to run one epoch of train/test using the model specified in arguments.
    
    :param args: parsed arguments
    :return: training/test function
    """
    if args.is_test:
        if args.decoder == "DecoderMLP":
            return epoch_test_MLP
        elif args.decoder == "DecoderGRU":
            return epoch_test
        elif args.decoder == "DecoderGRUAtt":
            return epoch_test
        elif args.decoder == "DecoderGraphRNN":
            return epoch_test_GraphRNN
        elif args.decoder == "DecoderGraphRNNAtt":
            return epoch_test_GraphRNN
        elif args.decoder == "DecoderGGT":
            return epoch_test
        else:
            raise ValueError("Unknown decoder type!")
    else:
        if args.decoder == "DecoderMLP":
            return epoch_train_MLP
        elif args.decoder == "DecoderGRU":
            return epoch_train
        elif args.decoder == "DecoderGRUAtt":
            return epoch_train
        elif args.decoder == "DecoderGraphRNN":
            return epoch_train
        elif args.decoder == "DecoderGraphRNNAtt":
            return epoch_train
        elif args.decoder == "DecoderGGT":
            return epoch_train
        else:
            raise ValueError("Unknown decoder type!")


# ########################################################################################
# ########################################################################################
# #################################  TRAIN FUNCTIONS #####################################
# ########################################################################################
# ########################################################################################


def epoch_train(args, epoch, dataloader, decoder, encoder, optimizer_decoder, optimizer_encoder, criterions,
                is_eval=False):
    r"""
    Execute one epoch of training or validation for the recurrent models.
    (GRU, GRUAtt, GraphRNN, GraphRNNAtt, GGT)
    
    :param args: parsed arguments
    :param epoch: epoch number
    :param dataloader: PyTorch dataloader for train or valid split
    :param decoder: decoder network
    :param encoder: encoder network
    :param optimizer_decoder: optimizer for decoder
    :param optimizer_encoder: optimizer for encoder
    :param criterions: dictionary of loss functions
    :param is_eval: True if is validation epoch
    :return: (average total loss, average BCE loss, average MSE loss)
    """
    losses = [], [], []
    mask_sequence = generate_mask_sequence(args.max_n_nodes)  # mask used to hide future steps in self-attention
    if is_eval:
        encoder.eval()
        decoder.eval()
    else:
        encoder.train()
        decoder.train()
    
    for i, data in enumerate(dataloader):
        decoder.reset_hidden()
        
        # ===================get batch===================
        x_adj, x_coord, y_adj, y_coord, img, seq_len, ids = data
        x_adj, x_coord, y_adj, y_coord, img, seq_len = x_adj.to(args.device), x_coord.to(args.device), y_adj.to(
            args.device), y_coord.to(args.device), img.to(args.device), seq_len.to(args.device)
        ids = list(ids)
        
        # ====================encode=====================
        if encoder is not None:
            if args.encoder == "EncoderCNNAtt":
                # CNN encoder with context attention
                if args.all_history:
                    # pass the whole history of generated nodes to the context attention encoder
                    history_adj = torch.zeros(x_adj.shape[0], args.max_n_nodes,
                                              args.max_n_nodes * args.max_prev_node).to(args.device)
                    history_coord = torch.zeros(x_adj.shape[0], args.max_n_nodes, args.max_n_nodes * 2).to(args.device)
                    for k in range(x_adj.shape[1]):
                        history_adj[:, k:k + 1, 0:(k + 1) * args.max_prev_node] = x_adj[:, 0:k + 1].view(x_adj.shape[0],
                                                                                                         -1).unsqueeze(
                            1)
                        history_coord[:, k:k + 1, 0:(k + 1) * 2] = x_coord[:, 0:k + 1].view(x_coord.shape[0],
                                                                                            -1).unsqueeze(1)
                    img = encoder(img, history_coord, history_adj)[:, :x_adj.shape[1]]
                else:
                    # pass only the last generated node to the context attention encoder
                    img = encoder(img, x_coord, x_adj)
            else:
                # simple CNN encoder
                img = encoder(img)
        
        # concatenate conditioning vector from the encoder and representation of lastly generated node
        input_sequence = generate_input_sequence(x_coord, x_adj, img)
        
        # max length in this batch
        current_max_seq_len = seq_len[0].item()
        
        # ====================decode=====================
        if "DecoderGraphRNN" in args.decoder:
            output_adj, output_coord = decoder(input_sequence, y_adj, input_len=seq_len)
        else:
            output_adj, output_coord = decoder(input_sequence, input_len=seq_len)
        
        # clean the padded part of the sequence and the part where prev_node goes before zero
        output_adj = pack_padded_sequence(output_adj, seq_len, batch_first=True)
        output_adj = pad_packed_sequence(output_adj, batch_first=True)[0]
        output_adj = output_adj * mask_sequence[:, :current_max_seq_len, :args.max_prev_node]
        output_coord = pack_padded_sequence(output_coord, seq_len, batch_first=True)
        output_coord = pad_packed_sequence(output_coord, batch_first=True)[0]
        y_adj = pack_padded_sequence(y_adj, seq_len, batch_first=True)
        y_adj = pad_packed_sequence(y_adj, batch_first=True)[0]
        y_adj = y_adj * mask_sequence[:, :current_max_seq_len, :args.max_prev_node]
        y_coord = pack_padded_sequence(y_coord, seq_len, batch_first=True)
        y_coord = pad_packed_sequence(y_coord, batch_first=True)[0]
        
        # ================compute losses=================
        loss_adj = criterions['bce'](output_adj, y_adj)
        loss_coord = criterions['mse'](output_coord, y_coord)
        loss = args.lamb * loss_adj + (1 - args.lamb) * loss_coord
        losses[0].append(loss.item())
        losses[1].append(loss_adj.item())
        losses[2].append(loss_coord.item())
        
        # ===================backward====================
        if not is_eval:
            optimizer_decoder.zero_grad()
            optimizer_encoder.zero_grad()
            loss.backward()
            optimizer_decoder.step()
            optimizer_encoder.step()
        
        # =====================plot reconstructions======================
        if i == 0:
            if epoch == 1:
                for b in range(y_adj.shape[0]):
                    plot_output_graph(args, "", ids[b], y_adj[b, :seq_len[b].item() - 1],
                                      y_coord[b, :seq_len[b].item() - 1], args.plots_path, is_eval=is_eval)
            if epoch % 1 == 0 and epoch > 0:
                for b in range(y_adj.shape[0]):
                    plot_output_graph(args, epoch, ids[b], output_adj[b, :seq_len[b].item() - 1],
                                      output_coord[b, :seq_len[b].item() - 1],
                                      args.plots_path, is_eval=is_eval, no_edges=False)
    
    res = sum(losses[0]) / len(losses[0])
    res_adj = sum(losses[1]) / len(losses[1])
    res_coord = sum(losses[2]) / len(losses[2])
    
    return res, res_adj, res_coord


def epoch_train_MLP(args, epoch, dataloader, decoder, encoder, optimizer_decoder, optimizer_encoder, criterions,
                    is_eval=False):
    r"""
    Execute one epoch of training or validation for the one-shot model (MLP decoder)
    
    :param args: parsed arguments
    :param epoch: epoch number
    :param dataloader: PyTorch dataloader for train or valid split
    :param decoder: decoder network
    :param encoder: encoder network
    :param optimizer_decoder: optimizer for decoder
    :param optimizer_encoder: optimizer for encoder
    :param criterions: dictionary of loss functions
    :param is_eval: True if is validation epoch
    :return: (average total loss, average BCE loss, average MSE loss)
    """
    losses = [], [], []
    if is_eval:
        encoder.eval()
        decoder.eval()
    else:
        encoder.train()
        decoder.train()
    
    for i, data in enumerate(dataloader):
        # ===================get batch===================
        x_adj, x_coord, y_adj, y_coord, img, seq_len, ids = data
        y_adj, y_coord, img, seq_len = y_adj.to(args.device), y_coord.to(args.device), img.to(args.device), seq_len.to(
            args.device)
        ids = list(ids)
        
        # In the MLP setting, we consider consider one output token at the end of the sequence
        # (modeled with adj_vector = 0) to be used as termination token at inference time. In this case, we don't need
        # to add the termination of connected components because we are modeling, for every node, also the future
        # connection in BFS, so we will never have an adjacency row in the matrix except for the last one
        # (termination token)
        current_max_seq_len = seq_len[0].item() - 1
        
        # =========get representation of X and A=========
        y_A = torch.zeros((y_adj.shape[0], current_max_seq_len, current_max_seq_len)).to(args.device)
        y_X = torch.zeros((y_adj.shape[0], current_max_seq_len, 2)).to(args.device)
        mask_A = torch.zeros((y_adj.shape[0], current_max_seq_len, current_max_seq_len)).to(args.device)
        mask_X = torch.zeros((y_adj.shape[0], current_max_seq_len, 2)).to(args.device)
        
        # get a fixed size representation of A and X from the sequential representation in the data
        # get masks for data cleaning and padding
        for i in range(y_adj.shape[0]):
            A = decode_adj(y_adj[i, :seq_len[i] - 1].cpu().numpy())
            A = torch.FloatTensor(A).to(args.device)
            y_A[i, :A.shape[0], :A.shape[1]] = A[:, :]
            y_X[i, :seq_len[i] - 1, :] = y_coord[i, :seq_len[i] - 1, :]
            mask = torch.ones_like(A)
            mask_A[i, :mask.shape[0], :mask.shape[1]] = mask[:, :]
            mask_X[i, :seq_len[i] - 1, :] = torch.ones((seq_len[i] - 1, 2)).to(args.device)
        
        # ====================encode=====================
        img = encoder(img)
        input_sequence = img
        
        # ====================decode=====================
        output_A, output_X = decoder(input_sequence)
        # clean the padded part and the part where prev_node goes before zero
        output_A = output_A[:, :current_max_seq_len, :current_max_seq_len] * mask_A
        output_X = output_X[:, :current_max_seq_len, :] * mask_X
        
        # ================compute losses=================
        loss_adj = criterions['bce'](output_A, y_A)
        loss_coord = criterions['mse'](output_X, y_X)
        loss = args.lamb * loss_adj + (1 - args.lamb) * loss_coord
        losses[0].append(loss.item())
        losses[1].append(loss_adj.item())
        losses[2].append(loss_coord.item())
        
        # ===================backward====================
        if not is_eval:
            optimizer_decoder.zero_grad()
            optimizer_encoder.zero_grad()
            loss.backward()
            optimizer_decoder.step()
            optimizer_encoder.step()
        
        # =====================plot reconstructions======================
        if i == 0:
            if epoch == 1:
                for b in range(y_adj.shape[0]):
                    plot_output_graph(args, "", ids[b], y_adj[b, :seq_len[b].item() - 1],
                                      y_coord[b, :seq_len[b].item() - 1], args.plots_path, is_eval=is_eval)
            if epoch % 1 == 0 and epoch > 0:
                for b in range(y_adj.shape[0]):
                    # if is_eval:
                    plot_output_graph(args, epoch, ids[b], output_A[b, :seq_len[b].item() - 1],
                                      output_X[b, :seq_len[b].item() - 1],
                                      args.plots_path, is_eval=is_eval)
    
    res = sum(losses[0]) / len(losses[0])
    res_adj = sum(losses[1]) / len(losses[1])
    res_coord = sum(losses[2]) / len(losses[2])
    
    return res, res_adj, res_coord


# ########################################################################################
# ########################################################################################
# ##################################  TEST FUNCTIONS #####################################
# ########################################################################################
# ########################################################################################


def epoch_test(args, dataloader, decoder, encoder):
    r"""
    Execute test for the recurrent models (GRU, GRUAtt, GGT)

    :param args: parsed arguments
    :param dataloader: PyTorch dataloader for test split
    :param decoder: decoder network
    :param encoder: encoder network
    :returns: np.array of means and np.array of std for metrics: (streetmover, loss, loss_adj, loss_coord, acc_A,
    delta_n_edges, delta_n_nodes, dist_degree, dist_diam, |delta_n_edges|, |delta_n_nodes|)
    """
    stats = []
    mask_sequence = generate_mask_sequence(args.max_n_nodes)
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            decoder.reset_hidden()

            # ===================get batch===================
            x_adj, x_coord, y_adj, y_coord, original_img, seq_len, ids = data
            x_adj, x_coord, y_adj, y_coord, original_img, seq_len = x_adj.to(args.device), x_coord.to(
                args.device), y_adj.to(args.device), y_coord.to(args.device), original_img.to(args.device), seq_len.to(
                args.device)
            x_coord_0, x_adj_0 = x_coord[:, 0].unsqueeze(1), x_adj[:, 0].unsqueeze(1)
            
            # =================encode at t=0=================
            if args.encoder == "EncoderCNNAtt":
                # CNN encoder with context attention
                if args.all_history:
                    # pass the whole history of generated nodes to the context attention encoder
                    history_adj = torch.zeros(x_adj.shape[0], args.max_n_nodes,
                                              args.max_n_nodes * args.max_prev_node).to(args.device)
                    history_coord = torch.zeros(x_adj.shape[0], args.max_n_nodes, args.max_n_nodes * 2).to(
                        args.device)
                    img = encoder(original_img, history_coord[:, 0:1], history_adj[:, 0:1], get_att_weights=False)
                    img = img[:, 0:1]
                else:
                    # pass only the last generated node to the context attention encoder
                    img = encoder(original_img, x_coord_0, x_adj_0, get_att_weights=False)
            else:
                # simple CNN encoder
                img = encoder(original_img)
            
            y_seq_len = seq_len[0].item()  # seq_len of target graph
            
            # initialize inputs for t=0
            output_adj = torch.zeros(args.batch_size, args.max_n_nodes, args.max_prev_node).to(args.device)
            output_coord = torch.zeros(args.batch_size, args.max_n_nodes, 2).to(args.device)
            output_seq_len = args.max_n_nodes  # seq_len of output graph is set to max if it does not terminate earlier
            input_sequence = generate_input_sequence(x_coord_0, x_adj_0, img)
            
            # ====================decode=====================
            for j in range(args.max_n_nodes):
                decoder.reset_hidden()  # because we are refeeding the whole sequence to the model
                x_adj, x_coord = decoder(input_sequence, input_len=[j + 1])
                sampled_x_adj = sample_sigmoid(args, x_adj[:, j:j + 1])  # sample or threshold
                output_adj[:, j] = sampled_x_adj  # store sampled adjacency vector in output A
                output_coord[:, j] = x_coord[:, j:j + 1]  # store emitted feature vector in output X
                # mask_sequence used to zero out connections that go earlier than first node
                output_adj = output_adj * mask_sequence[:, :args.max_n_nodes, :args.max_prev_node]
                
                # ==============check for termination============
                if j > 3:
                    a1 = torch.sum(output_adj[0, j] > 0.5)
                    a2 = torch.sum(output_adj[0, j - 1] > 0.5)
                    if a1 + a2 == 0:
                        # the generation completes where the previous connected component is closed (a1 == 0)
                        # and new connected component is empty, i.e. we do not want to generate anything else (a2 == 0)
                        output_seq_len = j + 1
                        break
                if j == args.max_n_nodes - 1:
                    # terminate generation when maximum number of nodes is reached
                    break
                
                # ================encode at t=j+1================
                # if we are using context attention, encode the image again, otherwise uses the initial encoded image
                if args.encoder == "EncoderCNNAtt":
                    # CNN encoder with context attention
                    if args.all_history:
                        # pass the whole history of generated nodes to the context attention encoder
                        history_adj[:, j + 1:j + 2, 0:(j + 2) * args.max_prev_node] = \
                            torch.cat([x_adj_0, output_adj[:, :j + 1]], dim=1).view(x_adj.shape[0], -1).unsqueeze(1)
                        history_coord[:, j + 1:j + 2, 0:(j + 2) * 2] = \
                            torch.cat([x_coord_0, output_coord[:, :j + 1]], dim=1).view(x_coord.shape[0], -1).unsqueeze(
                                1)
                        this_img, att_image = encoder(original_img, history_coord[:, j + 1:j + 2],
                                                      history_adj[:, j + 1:j + 2],
                                                      get_att_weights=True)
                    else:
                        # pass only the last generated node to the context attention encoder
                        this_img, att_image = encoder(original_img, x_coord[:, j:j + 1], sampled_x_adj,
                                                      get_att_weights=True)
                    img = torch.cat([img, this_img], dim=1)
                
                # new input sequence is zero vector as beginning plus the sequence generated so far
                input_sequence = generate_input_sequence(torch.cat([x_coord_0, output_coord[:, :j + 1]], dim=1),
                                                         torch.cat([x_adj_0, output_adj[:, :j + 1]], dim=1),
                                                         img)
            
            # =======================stats=====================
            this_stats = compute_statistics(output_adj, output_coord, output_seq_len, y_adj, y_coord, y_seq_len,
                                            lamb=args.lamb)
            streetmover, loss, loss_adj, loss_coord, acc_A, delta_n_edges, delta_n_nodes, dist_degree, dist_diam = this_stats
            stats.append(this_stats)
            
            # =====================plot reconstructions======================
            if i < 50:
                plot_output_graph(args, "real", ids[0], y_adj[0], y_coord[0], args.plots_path, is_eval=True)
                plot_output_graph(args, "recon", ids[0], output_adj[0, :output_seq_len],
                                  output_coord[0, :output_seq_len],
                                  args.plots_path, is_eval=True)

    # compute means and stds
    stats = np.array(stats)
    avg = np.mean(stats, axis=0)
    std = np.std(stats, axis=0)
    avg_pos = np.mean(np.absolute(stats[:, -4:-2]), axis=0)
    std_pos = np.std(np.absolute(stats[:, -4:-2]), axis=0)
    
    plot_histogram_streetmover(stats[:, 0], args)  # plot histogram of StreetMover distances
    pickle.dump(stats[:, :2], open(f"{args.statistics_path}/{args.file_name}.pickle", "wb"))  # store all stats
    
    return np.concatenate((avg, avg_pos)), np.concatenate((std, std_pos))


def epoch_test_GraphRNN(args, dataloader, decoder, encoder):
    r"""
    Execute test for the recurrent models based on GraphRNN (GraphRNN, GraphRNNAtt)

    :param args: parsed arguments
    :param dataloader: PyTorch dataloader for test split
    :param decoder: decoder network
    :param encoder: encoder network
    :returns: np.array of means and np.array of std for metrics: (streetmover, loss, loss_adj, loss_coord, acc_A,
    delta_n_edges, delta_n_nodes, dist_degree, dist_diam, |delta_n_edges|, |delta_n_nodes|)
    """
    stats = []
    mask_sequence = generate_mask_sequence(args.max_n_nodes)
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            decoder.reset_hidden()
            
            # ===================get batch===================
            x_adj, x_coord, y_adj, y_coord, img, seq_len, ids = data
            x_adj, x_coord, y_adj, y_coord, img, seq_len = x_adj.to(args.device), x_coord.to(args.device), \
                                                           y_adj.to(args.device), y_coord.to(args.device), \
                                                           img.to(args.device), seq_len.to(args.device)
            y_seq_len = seq_len[0].item()
            
            # =====================encode====================
            img = encoder(img)
            
            # initialize inputs for t=0
            x_coord_0, x_adj_0 = x_coord[:, 0].unsqueeze(1), x_adj[:, 0].unsqueeze(1)
            output_adj = torch.zeros(args.batch_size, args.max_n_nodes, args.max_prev_node).to(args.device)
            output_coord = torch.zeros(args.batch_size, args.max_n_nodes, 2).to(args.device)
            input_sequence = generate_input_sequence(x_coord_0, x_adj_0, img)
            output_seq_len = args.max_n_nodes  # if it does not terminate earlier
            
            # ====================decode=====================
            for j in range(seq_len[0]):
                x_adj, x_coord = decoder.generate(input_sequence, args=args)
                sampled_x_adj = sample_sigmoid(args, x_adj[:, :])  # sample or threshold
                output_adj[:, j] = sampled_x_adj  # store sampled adjacency vector in output A
                output_coord[:, j] = x_coord[:, :]  # store emitted feature vector in output X
                # mask_sequence used to zero out connections that go earlier than first node
                output_adj = output_adj * mask_sequence[:, :args.max_n_nodes, :args.max_prev_node]
                
                # ==============check for termination============
                if j > 3:
                    a1 = torch.sum(output_adj[0, j] > 0.5)
                    a2 = torch.sum(output_adj[0, j - 1] > 0.5)
                    if a1 + a2 == 0:
                        # the generation completes where the previous connected component is closed (a1 == 0)
                        # and new connected component is empty, i.e. we do not want to generate anything else (a2 == 0)
                        output_seq_len = j + 1
                        break
                
                # new input sequence is zero vector as beginning plus all the sampled sequence so far
                if j < args.max_n_nodes - 1:
                    input_sequence = generate_input_sequence(output_coord[:, j:j + 1], output_adj[:, j:j + 1], img)
            
            # =======================stats=====================
            this_stats = compute_statistics(output_adj, output_coord, output_seq_len, y_adj, y_coord, y_seq_len,
                                            lamb=args.lamb)
            streetmover, loss, loss_adj, loss_coord, acc_A, delta_n_edges, delta_n_nodes, dist_degree, dist_diam = this_stats
            stats.append(this_stats)
            
            # =====================plot reconstructions======================
            if i < 50:
                plot_output_graph(args, "real", ids[0], y_adj[0], y_coord[0], args.plots_path, is_eval=True)
                plot_output_graph(args, "recon", ids[0], output_adj[0, :output_seq_len],
                                  output_coord[0, :output_seq_len],
                                  args.plots_path, is_eval=True)
    
    # compute means and stds
    stats = np.array(stats)
    avg = np.mean(stats, axis=0)
    std = np.std(stats, axis=0)
    avg_pos = np.mean(np.absolute(stats[:, -4:-2]), axis=0)
    std_pos = np.std(np.absolute(stats[:, -4:-2]), axis=0)
    
    plot_histogram_streetmover(stats[:, 0], args)  # plot histogram of StreetMover distances
    pickle.dump(stats[:, :2], open(f"{args.statistics_path}/{args.file_name}.pickle", "wb"))  # store all stats
    
    return np.concatenate((avg, avg_pos)), np.concatenate((std, std_pos))


def epoch_test_MLP(args, dataloader, decoder, encoder):
    r"""
    Execute test for the one-shot model (MLP)

    :param args: parsed arguments
    :param dataloader: PyTorch dataloader for test split
    :param decoder: decoder network
    :param encoder: encoder network
    :returns: np.array of means and np.array of std for metrics: (streetmover, acc_A,
    delta_n_edges, delta_n_nodes, dist_degree, dist_diam, |delta_n_edges|, |delta_n_nodes|)
    """
    stats = []
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
    
            # ===================get batch===================
            x_adj, x_coord, y_adj, y_coord, img, seq_len, ids = data
            y_adj, y_coord, img, seq_len = y_adj.to(args.device), y_coord.to(args.device), \
                                           img.to(args.device), seq_len.to(args.device)
            y_seq_len = seq_len[0].item()
            y_X = x_coord[0, :-2, :]
            y_A = decode_adj(y_adj[0, :seq_len - 2].cpu().numpy())
            
            # =====================encode====================
            img = encoder(img)

            # =====================decode====================
            output_A, output_X = decoder(img)
            output_A = sample_sigmoid(args, output_A, sample=False)
            output_A, output_X = output_A[0], output_X[0]
            
            output_seq_len = args.max_n_nodes - 1
            
            # post-process output to find the length
            for j in range(2, output_A.shape[1], 1):
                a = torch.sum(output_A[j, :j])
                b = torch.sum(output_A[:j, j])
                if a + b == 0:
                    # the generation completes where the row and column of A for a particular node j have no edges
                    output_seq_len = j + 1
                    output_A = output_A[:output_seq_len, :output_seq_len]
                    output_X = output_X[:output_seq_len, :]
                    break
            output_A = output_A.cpu().numpy()

            # =======================stats=====================
            this_stats = compute_statistics_MLP(y_A, y_X, output_A, output_X, y_seq_len, output_seq_len)
            streetmover, acc_A, delta_n_edges, delta_n_nodes, dist_degree, dist_diam = this_stats
            stats.append(this_stats)
            
            # =====================plot reconstructions======================
            if i < 50:
                plot_output_graph(args, "real", ids[0], y_adj[0], y_coord[0], args.plots_path, is_eval=True)
                plot_output_graph(args, "recon", ids[0], output_A, output_X,
                                  args.plots_path, is_eval=True)

    # compute means and stds
    stats = np.array(stats)
    avg = np.mean(stats, axis=0)
    std = np.std(stats, axis=0)
    avg_pos = np.mean(np.absolute(stats[:, -4:-2]), axis=0)
    std_pos = np.std(np.absolute(stats[:, -4:-2]), axis=0)
    
    plot_histogram_streetmover(stats[:, 0], args)  # plot histogram of StreetMover distances
    pickle.dump(stats[:, :2], open(f"{args.statistics_path}/{args.file_name}.pickle", "wb"))  # store all stats
    
    return np.concatenate((avg, avg_pos)), np.concatenate((std, std_pos))


# ########################################################################################
# ########################################################################################
# ####################################  TRAIN/TEST  ######################################
# ########################################################################################
# ########################################################################################

def train(args):
    r"""
    Run training for the chosen model using the configurations in args
    
    :param args: parsed arguments
    """
    # =====================set seeds========================
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.cuda.manual_seed(args.seed)
    
    # =====================import data======================
    dataset_train = ToulouseRoadNetworkDataset(split=args.train_split, max_prev_node=args.max_prev_node)
    dataset_valid = ToulouseRoadNetworkDataset(split="valid", max_prev_node=args.max_prev_node)
    
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=custom_collate_fn)
    dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=custom_collate_fn)
    
    print("Dataset splits -> Train: {} | Valid: {}\n".format(len(dataset_train), len(dataset_valid)))
    
    # =====================init models======================
    encoder, optimizer_enc = load_encoder(args)
    decoder, optimizer_dec = load_decoder(args)
    run_epoch = get_epoch_fn(args)
    
    # use this to continue training using existing checkpoints
    # encoder.load_state_dict(torch.load(args.checkpoints_path + "/encoder.pth"))
    # decoder.load_state_dict(torch.load(args.checkpoints_path + "/decoder.pth"))

    # =====================init losses=======================
    criterion_mse = MSELoss(reduction='mean')
    criterion_bce = BCELoss(reduction='mean')
    criterions = {"bce": criterion_bce, "mse": criterion_mse}
    
    # ===================random guessing====================
    loss_valid, loss_valid_adj, loss_valid_coord = run_epoch(args, 0, dataloader_valid, decoder, encoder, optimizer_dec,
                                                             optimizer_enc, criterions, is_eval=True)
    print_and_log(
        'Epoch {}/{} || Train loss: {:.4f} Adj: {:.4f} Coord: {:.4f} ||'
        ' Valid loss: {:.4f} Adj: {:.4f} Coord: {:.4f}'
            .format(0, args.epochs, 0, 0, 0, loss_valid, loss_valid_adj, loss_valid_coord), args.file_logs)
    
    # ========================train=========================
    start_time = time.time()
    writer = SummaryWriter(args.file_tensorboard)
    min_loss_valid = (10000000, 0)
    
    for epoch in range(args.epochs):
        if time.time() - start_time > args.max_time:
            break
        
        loss_train, loss_train_adj, loss_train_coord = run_epoch(args, epoch + 1, dataloader_train, decoder, encoder,
                                                                 optimizer_dec, optimizer_enc, criterions,
                                                                 is_eval=False)
        loss_valid, loss_valid_adj, loss_valid_coord = run_epoch(args, epoch + 1, dataloader_valid, decoder, encoder,
                                                                 optimizer_dec, optimizer_enc, criterions, is_eval=True)
        
        # ========================log and plot=========================
        if epoch % 1 == 0:
            print_and_log(
                'Epoch {}/{} || Train loss: {:.4f} Adj: {:.4f} Coord: {:.4f} ||'
                ' Valid loss: {:.4f} Adj: {:.4f} Coord: {:.4f}'
                    .format(epoch + 1, args.epochs, loss_train, loss_train_adj, loss_train_coord,
                            loss_valid, loss_valid_adj, loss_valid_coord), args.file_logs)
        
        # ========================update curves========================
        save_losses(args, loss_train, loss_train_adj, loss_train_coord, loss_valid, loss_valid_adj, loss_valid_coord)
        update_writer(writer, "Loss", loss_train, loss_valid, epoch)
        update_writer(writer, "Loss Adj", loss_train_adj, loss_valid_adj, epoch)
        update_writer(writer, "Loss Coord", loss_train_coord, loss_valid_coord, epoch)
        
        # =========================save models=========================
        if min_loss_valid[0] > loss_valid:
            min_loss_valid = (loss_valid, epoch + 1)
            torch.save(decoder.state_dict(), args.checkpoints_path + "/decoder.pth")
            if encoder is not None:
                torch.save(encoder.state_dict(), args.checkpoints_path + "/encoder.pth")
    
    print("\nTraining Completed!")
    print_and_log("Minimum loss on validation set: {} at epoch {}".format(min_loss_valid[0], min_loss_valid[1]),
                  args.file_logs)


def test(args):
    r"""
    Run test on test set for the chosen model using the configurations in args.
    
    :param args: parsed arguments
    """
    # =====================set seeds========================
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.cuda.manual_seed(args.seed)
    
    # =====================import data======================
    dataset_test = ToulouseRoadNetworkDataset(split="test", max_prev_node=args.max_prev_node)
    
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False,
                                 collate_fn=custom_collate_fn)
    
    print("Dataset splits -> Test: {}\n".format(len(dataset_test)))
    
    # =====================init models======================
    encoder, _ = load_encoder(args)
    decoder, _ = load_decoder(args)
    run_epoch = get_epoch_fn(args)
    
    # =========================test=========================
    avg, std, = run_epoch(args, dataloader_test, decoder, encoder)
    print(f'Mean: {avg}')
    print(f'Std: {std}')
    save_statistics(args, avg, std)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    # Load the experiment if specified in args.experiment
    configs = Configs()
    args = configs.load_experiment(args)
    
    # set default arguments
    args = set_default_args(args)
    print("\n".join([str(x) for x in args.__dict__.items()]))
    print()
    
    # run train or test functions
    if args.is_test:
        test(args)
    else:
        train(args)
