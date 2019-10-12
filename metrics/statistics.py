import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance
import torch
from torch.nn import BCELoss, MSELoss

from utils.utils import decode_adj
from metrics.streetmover_distance import StreetMoverDistance

criterion_mse = MSELoss(reduction='mean')
criterion_bce = BCELoss(reduction='mean')

streetmover_distance = StreetMoverDistance(eps=0.00001, max_iter=10)


def compute_statistics(output_adj, output_coord, output_seq_len, y_adj, y_coord, y_seq_len, lamb=0.5):
    r"""
    Compute statistics for the current data point.
    
    :param output_adj: predicted A
    :param output_coord: predicted X
    :param output_seq_len: predicted |V|
    :param y_adj: target A
    :param y_coord: target X
    :param y_seq_len: target |V|
    :param lamb: lambda parameter for the loss in this experiment
    :return: streetmover, loss, loss_adj, loss_coord, acc_A, delta_n_edges, delta_n_nodes, dist_degree, dist_diam
    """
    output_A = decode_adj(output_adj[0, :output_seq_len - 2].cpu().numpy())  # not include the last 1)
    y_A = decode_adj(y_adj[0, :y_seq_len - 2].cpu().numpy())
    output_nodes = output_coord[0, :output_seq_len - 2]
    y_nodes = y_coord[0, :y_seq_len - 2]
    output_graph = nx.from_numpy_matrix(output_A)
    y_graph = nx.from_numpy_matrix(y_A)
    
    assert output_A.shape[0] == output_nodes.shape[0] == output_seq_len - 2
    assert y_A.shape[0] == y_nodes.shape[0] == y_seq_len - 2
    
    output_n_edges = output_adj.reshape(-1).sum()
    y_n_edges = y_adj.reshape(-1).sum()
    
    output_degree = get_degree_hist(output_graph)
    y_degree = get_degree_hist(y_graph)
    dist_degree = wasserstein_distance(output_degree, y_degree)
    
    output_diam = get_diameters(output_graph)
    y_diam = get_diameters(y_graph)
    dist_diam = wasserstein_distance(output_diam, y_diam) if len(output_diam) > 0 else 1
    
    delta_n_nodes = int(output_seq_len - y_seq_len)
    delta_n_edges = (output_n_edges - y_n_edges).item()
    
    acc_A = get_accuracy_A(output_A, y_A)
    
    loss_adj = get_BCE_adj(output_adj[0], y_adj[0])
    loss_coord = get_MSE_coord(output_nodes, y_nodes)
    loss = lamb * loss_adj + (1 - lamb) * loss_coord
    
    (y_pc, output_pc), (streetmover, P, C) = streetmover_distance(y_A, y_nodes, output_A, output_nodes, n_points=100)
    # print("Streetmover distance: {:.3f}".format(streetmover.item()))
    
    # possibly, plot assignments and/or point clouds
    # show_assignments(y_pc, output_pc, P, title=str(streetmover.item())[:8])
    # plot_point_cloud(y_adj[0], y_coord[0], y_pc)
    # plot_point_cloud(output_adj[0], output_coord[0], output_pc)
    
    return streetmover.item(), loss, loss_adj, loss_coord, acc_A, delta_n_edges, delta_n_nodes, dist_degree, dist_diam


def compute_statistics_MLP(y_A, y_nodes, output_A, output_nodes, y_seq_len, output_seq_len):
    r"""
    Compute statistics for the current data point, based on the one-shot output from the MLP decoder.

    :param output_adj: predicted A
    :param output_coord: predicted X
    :param output_seq_len: predicted |V|
    :param y_adj: target A
    :param y_coord: target X
    :param y_seq_len: target |V|
    :param lamb: lambda parameter for the loss in this experiment
    :return: streetmover, acc_A, delta_n_edges, delta_n_nodes, dist_degree, dist_diam
    """
    output_graph = nx.from_numpy_matrix(output_A)
    y_graph = nx.from_numpy_matrix(y_A)
    
    output_degree = get_degree_hist(output_graph)
    y_degree = get_degree_hist(y_graph)
    dist_degree = wasserstein_distance(output_degree, y_degree)
    
    output_diam = get_diameters(output_graph)
    y_diam = get_diameters(y_graph)
    dist_diam = wasserstein_distance(output_diam, y_diam) if len(output_diam) > 0 else 1
    
    delta_n_nodes = int(output_seq_len - y_seq_len)
    delta_n_edges = output_A.sum() - y_A.sum()
    
    acc_A = get_accuracy_A(output_A, y_A)
    
    (y_pc, output_pc), (streetmover, P, C) = streetmover_distance(y_A, y_nodes, output_A, output_nodes, n_points=100)
    # print("Streetmover distance: {:.3f}".format(streetmover.item()))
    
    return streetmover.item(), acc_A, delta_n_edges, delta_n_nodes, dist_degree, dist_diam


def get_clustering_hist(graph):
    r"""
    Compute histogram of clustering coefficient for a graph.
    Not used in our experiments.
    
    :param graph: graph representation in networkx format: nx.from_numpy_matrix(A)
    :return: histogram of clustering coefficients
    """
    clustering_coeffs_list = list(nx.clustering(graph).values())
    hist_c, _ = np.histogram(clustering_coeffs_list, bins=100, range=(0.0, 1.0), density=False)
    return hist_c


def get_degree_hist(graph):
    r"""
    Compute histogram of node degrees for a graph.
    
    :param graph: graph representation in networkx format: nx.from_numpy_matrix(A)
    :return: histogram of degrees
    """
    hist_d = nx.degree_histogram(graph)
    return np.array(hist_d)


def get_diameters(graph):
    r"""
    Compute histogram of connected components diameters for a graph.
    
    :param graph: graph representation in networkx format: nx.from_numpy_matrix(A)
    :return: list of connected components diameters
    """
    diams = []
    for g in nx.connected_component_subgraphs(graph):
        diams.append(nx.diameter(g))
    diams = list(filter(lambda a: a != 0, diams))
    return np.array(diams)


def get_BCE_adj(out_np, y_np):
    r"""
    Compute the BCE between predicted and target adjacency matrices.
    
    :param out_np: predicted A
    :param y_np: target A
    :return: BCE score
    """
    max_size = max(out_np.shape[0], y_np.shape[0])
    max_prev_node = y_np.shape[1]
    out = torch.zeros(1, max_size, max_prev_node).to(device="cuda:0")
    y = torch.zeros(1, max_size, max_prev_node).to(device="cuda:0")
    out[0:, :out_np.shape[0], :out_np.shape[1]] = out_np
    y[0:, :y_np.shape[0], :y_np.shape[1]] = y_np
    assert out.shape == y.shape
    bce = criterion_bce(out.view(1, -1), y.view(1, -1))
    return bce.item()


def get_accuracy_A(out_np, y_np):
    r"""
    Compute the accuracy ratio between predicted and target adjacency matrices, as the number of correct values in the
    binary adjacency matrices divided by the size of the adjacency matrix.
    
    :param out_np: predicted A
    :param y_np: target A
    :return: Accuracy
    """
    max_size = max(out_np.shape[0], y_np.shape[0])
    out = torch.zeros(1, max_size, max_size)
    y = torch.zeros(1, max_size, max_size)
    out[0:, :out_np.shape[0], :out_np.shape[0]] = torch.FloatTensor(out_np)
    y[0:, :y_np.shape[0], :y_np.shape[0]] = torch.FloatTensor(y_np)
    acc = float(torch.sum(out.view(-1) == y.view(-1))) / (max_size ** 2)
    return acc


def get_MSE_coord(out_np, y_np):
    r"""
    Compute the MSE between predicted and target node coordinate matrices.
    
    :param out_np: predicted X
    :param y_np: target X
    :return: MSE score
    """
    max_size = max(out_np.shape[0], y_np.shape[0])
    out = torch.zeros(1, max_size, 2).to(device="cuda:0")
    y = torch.zeros(1, max_size, 2).to(device="cuda:0")
    out[0:, :out_np.shape[0], :] = out_np
    y[0:, :y_np.shape[0], :] = y_np
    mse = criterion_mse(out.view(1, -1), y.view(1, -1))
    return mse.item()

