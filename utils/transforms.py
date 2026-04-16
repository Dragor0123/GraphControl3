from typing import List, Union
from torch_geometric.utils import to_undirected, remove_self_loops, to_dense_adj
import torch.nn.functional as F
import torch
import scipy

from .normalize import similarity, get_laplacian_matrix

def obtain_attributes(data, use_adj=False, threshold=0.1, num_dim=32):
    save_node_border = 30000

    if use_adj:
        # to undirected and remove self-loop
        edges = to_undirected(data.edge_index)
        edges, _ = remove_self_loops(edges)
        tmp = to_dense_adj(edges)[0]
    else:
        tmp = similarity(data.x, data.x)

        # discretize the similarity matrix by threshold
        tmp = torch.where(tmp>threshold, 1.0, 0.0)

    tmp = get_laplacian_matrix(tmp)
    if tmp.shape[0] > save_node_border:
        L, V = scipy.linalg.eigh(tmp)
        L = torch.from_numpy(L)
        V = torch.from_numpy(V)
    else:
        L, V = torch.linalg.eigh(tmp) # much faster than torch.linalg.eig

    x = V[:, :num_dim].float()
    import sklearn.preprocessing as preprocessing
    x = preprocessing.normalize(x.cpu(), norm="l2")
    x = torch.tensor(x, dtype=torch.float32)

    return x


def obtain_attributes_dissimilarity(data, num_dim=32, dissim_threshold=0.5):
    """
    Produce dissimilarity-based condition PE.
    A'_dissim = edges that exist in original A AND have low cosine similarity (K < dissim_threshold).
    This selects "real heterophilic edges" as the condition topology.

    Args:
        data: PyG data object with data.x (original node features) and data.edge_index
        num_dim: number of eigenvectors to extract (default 32)
        dissim_threshold: cosine similarity below which an edge is considered dissimilar.

    Returns:
        x: (N, num_dim) float tensor of Laplacian PE from A'_dissim
        stats: dict with topology statistics
    """
    import sklearn.preprocessing as preprocessing

    X = data.x
    N = X.size(0)
    device = X.device

    # Step 1: Cosine similarity matrix
    K = similarity(X, X)  # (N, N), values in [-1, 1]

    # Step 2: Original adjacency as dense matrix (undirected, no self-loops)
    edges = to_undirected(data.edge_index)
    edges, _ = remove_self_loops(edges)
    A_orig = to_dense_adj(edges, max_num_nodes=N)[0].to(device)  # (N, N)

    # Step 3: A_dissim = low similarity AND exists in original A
    A_dissim = ((K < dissim_threshold).float()) * A_orig

    # Step 4: Remove self-loops, ensure symmetry
    A_dissim.fill_diagonal_(0)
    A_dissim = (A_dissim + A_dissim.T).clamp(max=1.0)

    # Compute topology statistics
    num_edges_dissim = int(A_dissim.sum().item()) // 2
    num_edges_orig = int(A_orig.sum().item()) // 2
    overlap_ratio = num_edges_dissim / max(num_edges_orig, 1)
    stats = {
        'num_edges': num_edges_dissim,
        'num_edges_orig': num_edges_orig,
        'overlap_ratio': overlap_ratio,
        'dissim_threshold': dissim_threshold,
    }

    # Step 5: Handle empty A_dissim — fall back to original obtain_attributes
    if A_dissim.sum() == 0:
        print(f"WARNING: A_dissim is empty for dissim_threshold={dissim_threshold}. Falling back to similarity A'.")
        x = obtain_attributes(data, use_adj=False, threshold=0.17, num_dim=num_dim)
        stats['fallback'] = True
        return x, stats

    stats['fallback'] = False

    # Step 6: Normalized Laplacian of A_dissim
    L_norm = get_laplacian_matrix(A_dissim)

    save_node_border = 30000
    if N > save_node_border:
        L, V = scipy.linalg.eigh(L_norm.cpu().numpy())
        L = torch.from_numpy(L)
        V = torch.from_numpy(V)
    else:
        L, V = torch.linalg.eigh(L_norm)

    # Step 7: Smallest eigenvectors → PE
    x = V[:, :num_dim].float()
    x = preprocessing.normalize(x.cpu(), norm="l2")
    x = torch.tensor(x, dtype=torch.float32)

    return x, stats


def compute_condition_homophily(data, x_sim_dense, threshold=None):
    """
    Compute edge homophily ratio of the condition graph A'.
    Homophily ratio = fraction of edges in A' connecting nodes of the same class.

    Args:
        data: full-graph PyG data with data.y (node labels) and data.x
        x_sim_dense: (N, N) adjacency matrix of A' (already discretized)

    Returns:
        homophily_ratio: float
    """
    if not hasattr(data, 'y'):
        return None
    y = data.y.cpu()
    N = y.size(0)
    edges = x_sim_dense.nonzero(as_tuple=False)  # (E, 2)
    if edges.size(0) == 0:
        return 0.0
    src, dst = edges[:, 0], edges[:, 1]
    same_class = (y[src] == y[dst]).float().mean().item()
    return same_class
    

def process_attributes(data, use_adj=False, threshold=0.1, num_dim=32, soft=False, kernel=False):
    '''
    Replace the node attributes with positional encoding. Warning: this function will replace the node attributes!
    
    Args:
      data: a single graph contains x (if use_adj=False) and edge_index.
      use_adj: use the eigen-vectors of adjacent matrix or similarity matrix as node attributes.
      threshold: only work when use_adj=False, used for discretize the similarity matrix. 1 if Adj(i,j)>0.1 else 0
      soft: only work when use_adj=False, if soft=True, we will use soft similarity matrix.
      
    Returns:
      modified data.
    '''
    
    if use_adj:
        # to undirected and remove self-loop
        edges = to_undirected(data.edge_index)
        if edges.size(1) > 1:
            edges, _ = remove_self_loops(edges)
        else:
            edges = torch.tensor([[0],[0]]) # for isolated nodes
        Adj = to_dense_adj(edges)[0]
    else:
        
        if kernel:      
            # memory efficient
            XY = (data.x@data.x.T) # 2xy
            deg = torch.diag(XY)
            Y_norm = deg.repeat(XY.shape[0],1)
            X_norm = Y_norm.T
            Adj = X_norm - 2*XY + Y_norm # |X-Y|^2
            Adj = torch.exp(-0.05*Adj) # rbf kernel
        else:
            Adj = similarity(data.x, data.x) # equal to linear kernel
        if soft:
            L, V = torch.linalg.eigh(Adj)
            x = V[:, :num_dim].float()
            x = F.normalize(x, dim=1)
            data.x = x
            return data
        else:
            # discretize the similarity matrix by threshold
            Adj = torch.where(Adj>threshold, 1.0, 0.0)
    Lap = get_laplacian_matrix(Adj)
    
    L, V = torch.linalg.eigh(Lap) # much faster than torch.linalg.eig, if this line triggers bugs please refer to https://github.com/pytorch/pytorch/issues/70122#issuecomment-1232766638
    L_sort, _ = torch.sort(L, descending=False)
    hist = torch.histc(L, bins=32, min=0, max=2)
    hist = hist.unsqueeze(0)

    # Padding
    import sklearn.preprocessing as preprocessing
    if V.shape[0] < num_dim:
        V = preprocessing.normalize(V, norm="l2")
        V = torch.tensor(V, dtype=torch.float32)
        x = torch.nn.functional.pad(V, (0, num_dim-V.shape[0]))
        data.x = x.float()
        data.eigen_val = torch.nn.functional.pad(L_sort, (0, num_dim-L_sort.shape[0])).unsqueeze(0)
    else:
        x = V[:, 0:num_dim].float()
        x = preprocessing.normalize(x, norm="l2")
        x = torch.tensor(x, dtype=torch.float32)
        data.x = x.float()
        data.eigen_val = L_sort[:num_dim].unsqueeze(0)

    return data