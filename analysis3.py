"""
Idea 3 Pre-experiment: Monophily-Based 2-Hop Feature Similarity Condition Analysis (Claude.md)

Compares 1-hop feature similarity matrix A' vs 2-hop feature similarity matrix A'':
  - A'  : cosine_sim(x_i, x_j) > delta for ALL node pairs
  - A'' : cosine_sim(x_i, x_j) > delta only for node pairs EXACTLY 2 hops apart in original graph

Measures: total edges, homo/hetero edges, H_ratio for both matrices.

Output: ./results/idea3_analysis/analysis_{dataset}.json

Usage:
  python idea3_analysis.py --datasets Cora_ML DBLP Photo Chameleon Squirrel Actor
"""

import argparse
import json
import os

import torch
from torch_geometric.utils import to_dense_adj

from datasets import NodeDataset
from utils.normalize import similarity


# ── constants ─────────────────────────────────────────────────────────────────

DELTA = 0.1

DATASET_TYPES = {
    'Cora_ML':   'Homophilic',
    'DBLP':      'Homophilic',
    'Photo':     'Homophilic',
    'Chameleon': 'Heterophilic',
    'Squirrel':  'Heterophilic',
    'Actor':     'Heterophilic',
}

DEFAULT_DATASETS = ['Cora_ML', 'DBLP', 'Photo', 'Chameleon', 'Squirrel', 'Actor']


# ── helpers ───────────────────────────────────────────────────────────────────

def compute_stats(adj_matrix: torch.Tensor, labels: torch.Tensor):
    """
    adj_matrix: [N, N] symmetric binary matrix (no self-loops)
    labels: [N] node label tensor
    Returns: (total_edges, homo_edges, hetero_edges, h_ratio)
    """
    triu = torch.triu(adj_matrix, diagonal=1)
    edges = triu.nonzero(as_tuple=False)  # [E, 2]
    total = edges.shape[0]

    if total == 0:
        return 0, 0, 0, 0.0

    src_labels = labels[edges[:, 0]]
    dst_labels = labels[edges[:, 1]]
    homo = int((src_labels == dst_labels).sum().item())
    hetero = total - homo
    h_ratio = homo / total

    return total, homo, hetero, h_ratio


def build_A_prime(data, delta: float) -> torch.Tensor:
    """1-hop feature similarity matrix."""
    X = data.x.cpu().float()
    sim = similarity(X, X)
    A_prime = (sim > delta).float()
    A_prime.fill_diagonal_(0)
    return A_prime


def build_two_hop_mask(data) -> torch.Tensor:
    """Boolean mask of node pairs exactly 2 hops apart in the original graph."""
    N = data.x.shape[0]
    A_orig = to_dense_adj(data.edge_index, max_num_nodes=N)[0].cpu()
    A_orig = (A_orig > 0).float()
    A_orig.fill_diagonal_(0)

    # 2-hop reachability (A^2 > 0)
    A_orig_sq = (A_orig @ A_orig > 0).float()

    # Exactly 2-hop: reachable in 2 steps but NOT directly connected, no self-loops
    two_hop_mask = A_orig_sq.bool() & ~A_orig.bool()
    two_hop_mask.fill_diagonal_(False)
    return two_hop_mask


def build_A_double_prime(data, delta: float, two_hop_mask: torch.Tensor) -> torch.Tensor:
    """2-hop feature similarity matrix restricted to 2-hop neighbor pairs."""
    X = data.x.cpu().float()
    sim = similarity(X, X)
    A_double_prime = ((sim > delta) & two_hop_mask).float()
    return A_double_prime


# ── per-dataset analysis ──────────────────────────────────────────────────────

def analyze_dataset(dataset_name: str, delta: float) -> dict:
    dtype = DATASET_TYPES.get(dataset_name, 'Unknown')
    print(f'\n{"="*65}')
    print(f'  Dataset: {dataset_name} ({dtype})  |  δ = {delta}')
    print(f'{"="*65}')

    dataset_obj = NodeDataset(dataset_name)
    data = dataset_obj.data
    N = dataset_obj.num_nodes

    labels = data.y.cpu()
    if labels.ndim > 1:
        labels = labels.argmax(dim=1)

    # ── Build matrices ─────────────────────────────────────────────────────────
    print('  Building A\' (1-hop feature similarity)...')
    A_prime = build_A_prime(data, delta)

    print('  Building 2-hop mask from original graph...')
    two_hop_mask = build_two_hop_mask(data)
    n_two_hop_candidates = int(torch.triu(two_hop_mask.float(), diagonal=1).sum().item())

    print('  Building A\'\' (2-hop feature similarity)...')
    A_double_prime = build_A_double_prime(data, delta, two_hop_mask)

    # ── Compute statistics ─────────────────────────────────────────────────────
    total_p, homo_p, hetero_p, hratio_p = compute_stats(A_prime, labels)
    total_pp, homo_pp, hetero_pp, hratio_pp = compute_stats(A_double_prime, labels)

    # ── Print per-dataset output ───────────────────────────────────────────────
    sep = '─' * 54
    print(f'\n  Threshold δ = {delta}')
    print(f'  {sep}')
    col_a  = "A' (1-hop sim)"
    col_b  = "A'' (2-hop sim)"
    row_dh = "dH_ratio (A'' - A')"
    print(f'  {"":30s}  {col_a:>14}  {col_b:>14}')
    print(f'  {sep}')
    print(f'  {"Total edges":<30}  {total_p:>14,}  {total_pp:>14,}')
    print(f'  {"2-hop candidates":<30}  {"  -":>14}  {n_two_hop_candidates:>14,}')
    print(f'  {"Homo edges":<30}  {homo_p:>14,}  {homo_pp:>14,}')
    print(f'  {"Hetero edges":<30}  {hetero_p:>14,}  {hetero_pp:>14,}')
    print(f'  {"H_ratio":<30}  {hratio_p:>14.4f}  {hratio_pp:>14.4f}')
    delta_h = hratio_pp - hratio_p
    sign = '+' if delta_h >= 0 else ''
    print(f'  {row_dh:<30}  {sign}{delta_h:.4f}')
    print(f'  {sep}')

    return {
        'dataset':            dataset_name,
        'type':               dtype,
        'num_nodes':          N,
        'delta':              delta,
        'two_hop_candidates': n_two_hop_candidates,
        'A_prime': {
            'total_edges':  total_p,
            'homo_edges':   homo_p,
            'hetero_edges': hetero_p,
            'h_ratio':      hratio_p,
        },
        'A_double_prime': {
            'total_edges':  total_pp,
            'homo_edges':   homo_pp,
            'hetero_edges': hetero_pp,
            'h_ratio':      hratio_pp,
        },
        'delta_h_ratio': delta_h,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Idea 3 Pre-experiment: 2-Hop Feature Similarity Analysis')
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    parser.add_argument('--delta', type=float, default=DELTA)
    args = parser.parse_args()

    os.makedirs('./results/idea3_analysis', exist_ok=True)

    all_results = []
    for dataset_name in args.datasets:
        result = analyze_dataset(dataset_name, args.delta)
        all_results.append(result)

        out_path = f'./results/idea3_analysis/analysis_{dataset_name}.json'
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\n  Saved → {out_path}')

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f'\n\n{"="*89}')
    print('  SUMMARY TABLE')
    print(f'{"="*89}')
    h1 = "|E(A')|"
    h2 = "H_ratio(A')"
    h3 = "|E(A'')|"
    h4 = "H_ratio(A'')"
    h5 = "dH_ratio"
    header = f"  {'Dataset':<12} | {'Type':<10} | {h1:>9} | {h2:>11} | {h3:>9} | {h4:>12} | {h5:>9}"
    print(header)
    print(f'  {"─"*85}')
    for r in all_results:
        ap = r['A_prime']
        app = r['A_double_prime']
        dh = r['delta_h_ratio']
        sign = '+' if dh >= 0 else ''
        print(
            f"  {r['dataset']:<12} | {r['type']:<10} | {ap['total_edges']:>9,} | "
            f"{ap['h_ratio']:>11.4f} | {app['total_edges']:>9,} | "
            f"{app['h_ratio']:>12.4f} | {sign}{dh:.4f}"
        )
    print(f'{"="*89}')


if __name__ == '__main__':
    main()