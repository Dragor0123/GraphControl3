"""
analyze.py — Diagnostic Experiments for GraphControl Branch Complementarity Analysis

Usage (development):
    python analyze.py --dataset Cora_ML --lr 0.5 --optimizer adamw --use_adj --seeds 0 --model GCC_GraphControl

Usage (full):
    python analyze.py --dataset Cora_ML --lr 0.5 --optimizer adamw --threshold 0.17 --use_adj --seeds 0 1 2 3 4 --model GCC_GraphControl
    python analyze.py --dataset Squirrel --lr 0.5 --optimizer adamw --threshold 0.15 --use_adj --seeds 0 1 2 3 4 --model GCC_GraphControl
"""

import os
import copy
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.utils import homophily as pyg_homophily
from tqdm import tqdm

from utils.random import reset_random_seed
from utils.args import Arguments
from utils.sampling import collect_subgraphs
from utils.transforms import process_attributes, obtain_attributes
from models import load_model
from datasets import NodeDataset
from optimizers import create_optimizer


# ─────────────────────────────────────────────────────────────────────────────
# Output directories
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR = 'analysis_results'   # overridden in main() to analysis_results/{dataset}


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two (N, D) representation matrices."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic_xy = torch.norm(Y.T @ X, 'fro') ** 2
    hsic_xx = torch.norm(X.T @ X, 'fro') ** 2
    hsic_yy = torch.norm(Y.T @ Y, 'fro') ** 2
    denom = torch.sqrt(hsic_xx) * torch.sqrt(hsic_yy)
    if denom < 1e-10:
        return 0.0
    return (hsic_xy / denom).item()


def effective_rank(X: torch.Tensor) -> float:
    """Effective rank (entropy-based) of (N, D) matrix X."""
    s = torch.linalg.svdvals(X - X.mean(0))
    s = s[s > 1e-10]
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    p = p[p > 1e-10]
    return torch.exp(-torch.sum(p * torch.log(p))).item()


def compute_local_homophily(edge_index: torch.Tensor,
                             y: torch.Tensor,
                             num_nodes: int) -> torch.Tensor:
    """Returns tensor (num_nodes,) with local homophily per node."""
    h   = torch.zeros(num_nodes)
    deg = torch.zeros(num_nodes)
    row, col = edge_index.cpu()
    y_cpu = y.cpu()
    if y_cpu.dim() > 1:
        y_cpu = y_cpu.argmax(1)
    same_label = (y_cpu[row] == y_cpu[col]).float()
    deg.scatter_add_(0, row, torch.ones(row.shape[0]))
    h.scatter_add_(0, row, same_label)
    return h / deg.clamp(min=1)


def compute_normalized_laplacian(edge_index: torch.Tensor,
                                  num_nodes: int,
                                  device='cpu') -> torch.Tensor:
    """Dense normalized Laplacian from sparse edge_index."""
    if num_nodes == 0:
        return torch.zeros(0, 0, device=device)
    if edge_index.shape[1] == 0:
        return torch.eye(num_nodes, device=device)
    adj = torch.zeros(num_nodes, num_nodes, device=device)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj = (adj + adj.T).clamp(0.0, 1.0)
    adj.fill_diagonal_(0.0)
    deg = adj.sum(dim=1).clamp(min=1.0)
    D_rsqrt = torch.diag(deg.rsqrt())
    I = torch.eye(num_nodes, device=device)
    return I - D_rsqrt @ adj @ D_rsqrt


def compute_avg_rayleigh_quotient(H_nodes: torch.Tensor,
                                   edge_index: torch.Tensor,
                                   num_nodes: int) -> float:
    """Average Rayleigh Quotient across feature dims for node representations."""
    device = H_nodes.device
    L = compute_normalized_laplacian(edge_index.to(device), num_nodes, device=device)
    rqs = []
    for d in range(H_nodes.shape[1]):
        h = H_nodes[:, d]
        denom = (h @ h).item()
        if denom > 1e-10:
            numer = (h @ L @ h).item()
            rqs.append(numer / denom)
    return float(np.mean(rqs)) if rqs else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper for extracting intermediate representations
# ─────────────────────────────────────────────────────────────────────────────

class GraphControlAnalyzer:
    """
    Wraps a fine-tuned GCC_GraphControl model for diagnostic analysis.
    All forward passes run in eval/no_grad mode.
    Does NOT modify the underlying model.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def extract_graph_representations(self, x, x_sim, edge_index, batch, root_n_id):
        """
        Returns (h_frozen, h_tc_raw, h_control, h_combined) — each (B, D).
        Uses the same forward logic as GCC_GraphControl.forward_subgraph(frozen=True)
        but exposes all intermediate tensors.
        """
        with torch.no_grad():
            m = self.model
            m.encoder.eval()

            h_frozen = m.encoder.forward_subgraph(x, edge_index, batch, root_n_id)

            x_down   = m.zero_conv1(x_sim) + x
            h_tc_raw = m.trainable_copy.forward_subgraph(x_down, edge_index, batch, root_n_id)
            h_control  = m.zero_conv2(h_tc_raw)
            h_combined = h_frozen + h_control

        return h_frozen, h_tc_raw, h_control, h_combined

    def extract_node_level_representations(self, x, x_sim, edge_index, batch, root_n_id):
        """
        Returns (h_frozen_nodes, h_control_nodes) — node-level (N_total, D),
        extracted from the last GIN hidden layer before pooling.

        Uses forward hooks on the last BatchNorm of each branch's UnsupervisedGIN,
        then applies ReLU (matching UnsupervisedGIN.forward internals).
        """
        captured = {}

        def make_hook(key):
            def fn(module, inp, out):
                captured[key] = out
            return fn

        h1 = self.model.encoder.gnn.batch_norms[-1].register_forward_hook(
            make_hook('frozen'))
        h2 = self.model.trainable_copy.gnn.batch_norms[-1].register_forward_hook(
            make_hook('control'))

        with torch.no_grad():
            self.model.encoder.eval()
            self.model.encoder.forward_subgraph(x, edge_index, batch, root_n_id)

            x_down = self.model.zero_conv1(x_sim) + x
            self.model.trainable_copy.forward_subgraph(x_down, edge_index, batch, root_n_id)

        h1.remove()
        h2.remove()

        # Apply ReLU to match the final activation in UnsupervisedGIN.forward
        h_frozen_nodes  = F.relu(captured['frozen'])
        h_control_nodes = F.relu(captured['control'])
        return h_frozen_nodes, h_control_nodes


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing & fine-tuning  (mirrors graphcontrol.py)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(config, dataset_obj, device):
    kwargs_train = dict(batch_size=config.batch_size, num_workers=4,
                        persistent_workers=True, pin_memory=True, shuffle=True)
    kwargs_test  = dict(batch_size=config.batch_size, num_workers=4,
                        persistent_workers=True, pin_memory=True, shuffle=False)
    kwargs_anal  = dict(batch_size=1, num_workers=2, persistent_workers=True,
                        shuffle=False)

    print('Generating subgraphs...')
    train_idx = dataset_obj.data.train_mask.nonzero().squeeze()
    test_idx  = dataset_obj.data.test_mask.nonzero().squeeze()

    train_graphs = collect_subgraphs(train_idx, dataset_obj.data,
                                     walk_steps=config.walk_steps,
                                     restart_ratio=config.restart)
    test_graphs  = collect_subgraphs(test_idx,  dataset_obj.data,
                                     walk_steps=config.walk_steps,
                                     restart_ratio=config.restart)

    [process_attributes(g, use_adj=config.use_adj,
                        threshold=config.threshold,
                        num_dim=config.num_dim) for g in train_graphs]
    [process_attributes(g, use_adj=config.use_adj,
                        threshold=config.threshold,
                        num_dim=config.num_dim) for g in test_graphs]

    train_loader    = DataLoader(train_graphs, **kwargs_train)
    test_loader     = DataLoader(test_graphs,  **kwargs_test)
    analysis_loader = DataLoader(test_graphs,  **kwargs_anal)
    return train_loader, test_loader, analysis_loader


def eval_subgraph(model, test_loader, device, full_x_sim):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            if not hasattr(batch, 'root_n_id'):
                batch.root_n_id = batch.root_n_index
            x_sim = full_x_sim[batch.original_idx]
            preds = model.forward_subgraph(
                batch.x, x_sim, batch.edge_index, batch.batch,
                batch.root_n_id, frozen=True).argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total   += batch.y.shape[0]
    return correct / total


def finetune(config, model, train_loader, device, full_x_sim, test_loader):
    for k, v in model.named_parameters():
        if 'encoder' in k:
            v.requires_grad = False

    model.reset_classifier()
    eval_steps = 3
    patience   = 15
    count      = 0
    best_acc   = 0.0
    best_state = None

    params    = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = create_optimizer(name=config.optimizer, parameters=params,
                                 lr=config.lr, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    pbar      = tqdm(range(config.epochs), desc='Fine-tuning')

    for epoch in pbar:
        for data in train_loader:
            optimizer.zero_grad()
            model.train()
            data = data.to(device)
            if not hasattr(data, 'root_n_id'):
                data.root_n_id = data.root_n_index

            sign_flip = torch.rand(data.x.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip <  0.5] = -1.0
            x = data.x * sign_flip.unsqueeze(0)

            x_sim = full_x_sim[data.original_idx]
            preds = model.forward_subgraph(x, x_sim, data.edge_index, data.batch,
                                           data.root_n_id, frozen=True)
            loss = criterion(preds, data.y)
            loss.backward()
            optimizer.step()

        if epoch % eval_steps == 0:
            acc = eval_subgraph(model, test_loader, device, full_x_sim)
            pbar.set_postfix({'epoch': epoch, 'acc': f'{acc:.4f}'})
            if acc > best_acc:
                best_acc   = acc
                best_state = copy.deepcopy(model.state_dict())
                count      = 0
            else:
                count += 1

        if count == patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc


# ─────────────────────────────────────────────────────────────────────────────
# Single-pass data collection  (batch_size=1 for clean per-subgraph tracking)
# ─────────────────────────────────────────────────────────────────────────────

def collect_all_test_data(analyzer, analysis_loader, full_x_sim, device,
                           local_homophily):
    """
    Iterates the test set (batch_size=1) and collects everything needed for
    all four experiments in a single forward pass per subgraph.
    Returns a list of per-subgraph record dicts.
    """
    records = []

    for data in tqdm(analysis_loader, desc='Collecting representations', leave=False):
        data = data.to(device)
        if not hasattr(data, 'root_n_id'):
            data.root_n_id = data.root_n_index

        x     = data.x
        x_sim = full_x_sim[data.original_idx]
        ei    = data.edge_index
        b     = data.batch
        rnid  = data.root_n_id

        # Graph-level representations
        h_frozen, h_tc_raw, h_control, h_combined = \
            analyzer.extract_graph_representations(x, x_sim, ei, b, rnid)

        # Node-level representations (for Exp 4)
        try:
            h_fn, h_cn = analyzer.extract_node_level_representations(
                x, x_sim, ei, b, rnid)
        except Exception as e:
            h_fn = h_cn = None

        # Classification logits using the shared linear_classifier
        with torch.no_grad():
            clf = analyzer.model.linear_classifier
            logits_frozen   = clf(h_frozen)
            logits_control  = clf(h_control)
            logits_combined = clf(h_combined)

        y_true        = data.y.item()
        pred_frozen   = logits_frozen.argmax(dim=1).item()
        pred_control  = logits_control.argmax(dim=1).item()
        pred_combined = logits_combined.argmax(dim=1).item()

        # Center node's full-graph index → look up local homophily
        rnid_val = rnid.item() if torch.is_tensor(rnid) else int(rnid)
        center_global = data.original_idx[rnid_val].item()
        lh = local_homophily[center_global].item()

        num_nodes = x.shape[0]

        records.append({
            'y_true':          y_true,
            'pred_frozen':     pred_frozen,
            'pred_control':    pred_control,
            'pred_combined':   pred_combined,
            'h_frozen':        h_frozen.squeeze(0).cpu(),   # (D,)
            'h_control':       h_control.squeeze(0).cpu(),  # (D,)
            'h_combined':      h_combined.squeeze(0).cpu(), # (D,)
            'local_homophily': lh,
            'center_global':   center_global,
            # Exp 4
            'h_frozen_nodes':  h_fn.cpu() if h_fn is not None else None,
            'h_control_nodes': h_cn.cpu() if h_cn is not None else None,
            'edge_index':      ei.cpu(),
            'num_nodes':       num_nodes,
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Per-Branch Classification Performance
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_1(records, dataset_name, homophily_ratio, acc_list):
    n = len(records)
    if n == 0:
        return
    acc_frozen   = sum(r['pred_frozen']   == r['y_true'] for r in records) / n
    acc_control  = sum(r['pred_control']  == r['y_true'] for r in records) / n
    acc_combined = sum(r['pred_combined'] == r['y_true'] for r in records) / n

    print(f'\n[Exp 1] {dataset_name} | homophily={homophily_ratio:.4f}')
    print(f'  Frozen-only  : {acc_frozen:.4f}')
    print(f'  Control-only : {acc_control:.4f}')
    print(f'  Combined     : {acc_combined:.4f}')

    acc_list.append({
        'Dataset':        dataset_name,
        'Homophily_Ratio': round(homophily_ratio, 4),
        'Frozen_Acc':     round(acc_frozen,   4),
        'Control_Acc':    round(acc_control,  4),
        'Combined_Acc':   round(acc_combined, 4),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Representation Redundancy Metrics
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_2(records, dataset_name, redundancy_list):
    if not records:
        return

    H_frozen  = torch.stack([r['h_frozen']  for r in records])  # (N, D)
    H_control = torch.stack([r['h_control'] for r in records])  # (N, D)

    cos_sims = F.cosine_similarity(H_frozen, H_control, dim=1)
    cos_mean = cos_sims.mean().item()
    cos_std  = cos_sims.std().item()

    cka        = linear_CKA(H_frozen, H_control)
    er_frozen  = effective_rank(H_frozen)
    er_control = effective_rank(H_control)
    er_concat  = effective_rank(torch.cat([H_frozen, H_control], dim=1))
    er_ratio   = er_concat / max(er_frozen, er_control)

    print(f'\n[Exp 2] {dataset_name}')
    print(f'  Cosine sim : {cos_mean:.4f} ± {cos_std:.4f}')
    print(f'  CKA        : {cka:.4f}')
    print(f'  ERank  frozen={er_frozen:.2f}  control={er_control:.2f}  '
          f'concat={er_concat:.2f}  ratio={er_ratio:.3f}')

    redundancy_list.append({
        'Dataset':       dataset_name,
        'Cosine_Mean':   round(cos_mean,   4),
        'Cosine_Std':    round(cos_std,    4),
        'CKA':           round(cka,        4),
        'ERank_Frozen':  round(er_frozen,  3),
        'ERank_Control': round(er_control, 3),
        'ERank_Concat':  round(er_concat,  3),
        'ERank_Ratio':   round(er_ratio,   4),
    })

    # Histogram plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cos_sims.numpy(), bins=50, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(cos_mean, color='red', linestyle='--', linewidth=1.5,
               label=f'mean = {cos_mean:.3f}')
    ax.set_xlabel('Cosine Similarity (frozen vs. control)')
    ax.set_ylabel('Count')
    ax.set_title(f'{dataset_name} — Branch Pairwise Cosine Similarity\n'
                 f'mean={cos_mean:.3f}, std={cos_std:.3f}, CKA={cka:.3f}')
    ax.legend()
    plt.tight_layout()
    fig.savefig(f'{OUT_DIR}/exp2_cosine_histograms/{dataset_name}.png', dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Local Homophily vs. Branch Contribution
# ─────────────────────────────────────────────────────────────────────────────

def _assign_quartiles_value(lh_vals: np.ndarray) -> np.ndarray:
    """
    값 기반 분류: q25/q50/q75 경계값으로 분류.
    동일 homophily 값이 많은 편향 분포에서 특정 분위가 비어버릴 수 있음.
    """
    q25, q50, q75 = np.quantile(lh_vals, [0.25, 0.50, 0.75])
    labels = np.where(lh_vals >= q75, 'Q1',
             np.where(lh_vals >= q50, 'Q2',
             np.where(lh_vals >= q25, 'Q3', 'Q4')))
    return labels


def _assign_quartiles_rank(lh_vals: np.ndarray) -> np.ndarray:
    """
    순위 기반 분류: 정렬 순위를 기준으로 분류.
    항상 4개 분위가 균등하게 채워지지만 경계 근처 동일 값이 다른 분위로 분리될 수 있음.
    """
    n = len(lh_vals)
    ranks = np.argsort(np.argsort(lh_vals, kind='stable'), kind='stable')
    labels = np.where(ranks >= int(0.75 * n), 'Q1',
             np.where(ranks >= int(0.50 * n), 'Q2',
             np.where(ranks >= int(0.25 * n), 'Q3', 'Q4')))
    return labels


def _compute_quartile_stats(records, quartile_labels):
    """분위 레이블 배열로 통계 계산 → table_rows 반환."""
    buckets = {q: [] for q in ['Q1', 'Q2', 'Q3', 'Q4']}
    for r, q in zip(records, quartile_labels):
        hf = r['h_frozen']
        hc = r['h_control']
        hb = r['h_combined']
        norm_f = hf.norm().item()
        norm_c = hc.norm().item()
        norm_ratio = norm_c / (norm_f + norm_c + 1e-10)
        cos_f = F.cosine_similarity(hf.unsqueeze(0), hb.unsqueeze(0)).item()
        cos_c = F.cosine_similarity(hc.unsqueeze(0), hb.unsqueeze(0)).item()
        buckets[str(q)].append({
            'norm_ratio':           norm_ratio,
            'cos_frozen_combined':  cos_f,
            'cos_control_combined': cos_c,
            'correct_frozen':   int(r['pred_frozen']   == r['y_true']),
            'correct_control':  int(r['pred_control']  == r['y_true']),
            'correct_combined': int(r['pred_combined'] == r['y_true']),
        })

    table_rows = []
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        bd = buckets[q]
        if not bd:
            continue
        table_rows.append({
            'Quartile':        q,
            'N':               len(bd),
            'NormRatio_Mean':  round(np.mean([d['norm_ratio']           for d in bd]), 4),
            'CosFrozen_Mean':  round(np.mean([d['cos_frozen_combined']  for d in bd]), 4),
            'CosControl_Mean': round(np.mean([d['cos_control_combined'] for d in bd]), 4),
            'Acc_Frozen':      round(np.mean([d['correct_frozen']       for d in bd]), 4),
            'Acc_Control':     round(np.mean([d['correct_control']      for d in bd]), 4),
            'Acc_Combined':    round(np.mean([d['correct_combined']     for d in bd]), 4),
        })
    return table_rows


def _save_quartile_results(table_rows, tag_name, suffix, mode_label):
    """통계 테이블을 CSV + 플롯으로 저장."""
    df = pd.DataFrame(table_rows)
    print(f'\n[Exp 3 | {mode_label}] {tag_name}')
    print(df.to_string(index=False))

    out_base = f'{OUT_DIR}/exp3_homophily_contribution/{tag_name}_{suffix}'
    df.to_csv(f'{out_base}_table.csv', index=False)

    qs    = [r['Quartile'] for r in table_rows]
    x_pos = np.arange(len(qs))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{tag_name} [{mode_label}] — Branch Contribution by Local Homophily Quartile\n'
                 f'(Q1 = highest homophily, Q4 = lowest)')

    axes[0].bar(x_pos, [r['NormRatio_Mean'] for r in table_rows],
                color='coral', edgecolor='white')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(qs)
    axes[0].set_ylabel('Mean Norm Ratio  ‖control‖ / (‖frozen‖ + ‖control‖)')
    axes[0].set_title('ControlNet Branch Norm Ratio')
    axes[0].set_ylim(0, 1)

    acc_f = [r['Acc_Frozen']   for r in table_rows]
    acc_c = [r['Acc_Control']  for r in table_rows]
    acc_b = [r['Acc_Combined'] for r in table_rows]
    axes[1].bar(x_pos - width, acc_f, width, label='Frozen',   color='steelblue',     edgecolor='white')
    axes[1].bar(x_pos,         acc_c, width, label='Control',  color='coral',          edgecolor='white')
    axes[1].bar(x_pos + width, acc_b, width, label='Combined', color='mediumseagreen', edgecolor='white')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(qs)
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Per-Variant Accuracy by Quartile')
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(f'{out_base}_plot.png', dpi=150)
    plt.close(fig)


def run_experiment_3(records, tag_name, mode='both'):
    """
    mode 옵션:
      'value' — 값 기반 분류 (q25/q50/q75 경계값)
      'rank'  — 순위 기반 분류 (항상 균등 분위)
      'both'  — 두 방식 모두 수행하여 각각 저장 (default)
    tag_name: 파일명 prefix (e.g. 'Cora_ML' or 'Cora_ML_seed0').
    """
    if not records:
        return

    lh_vals = np.array([r['local_homophily'] for r in records])

    if mode in ('value', 'both'):
        labels = _assign_quartiles_value(lh_vals)
        rows   = _compute_quartile_stats(records, labels)
        _save_quartile_results(rows, tag_name, suffix='value', mode_label='Value-based')

    if mode in ('rank', 'both'):
        labels = _assign_quartiles_rank(lh_vals)
        rows   = _compute_quartile_stats(records, labels)
        _save_quartile_results(rows, tag_name, suffix='rank', mode_label='Rank-based')


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4: Spectral Energy (Rayleigh Quotient) Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_4(records, tag_name, rq_agg_list):
    """
    rq_agg_list: accumulator list for CSV rows; pass [] if you only want plots.
    """
    rq_frozen_list  = []
    rq_control_list = []
    lh_list         = []

    for r in tqdm(records, desc=f'  RQ [{tag_name}]', leave=False):
        h_fn = r['h_frozen_nodes']
        h_cn = r['h_control_nodes']
        if h_fn is None or h_cn is None:
            continue
        ei = r['edge_index']
        nn = r['num_nodes']
        if nn < 2 or h_fn.shape[0] != nn:
            continue

        rq_f = compute_avg_rayleigh_quotient(h_fn, ei, nn)
        rq_c = compute_avg_rayleigh_quotient(h_cn, ei, nn)
        rq_frozen_list.append(rq_f)
        rq_control_list.append(rq_c)
        lh_list.append(r['local_homophily'])

    if not rq_frozen_list:
        print(f'[Exp 4] {tag_name}: no valid subgraphs, skipping.')
        return

    rq_f = np.array(rq_frozen_list)
    rq_c = np.array(rq_control_list)
    lh   = np.array(lh_list)
    diff = rq_f - rq_c

    print(f'\n[Exp 4] {tag_name}')
    print(f'  RQ Frozen  : {rq_f.mean():.4f} ± {rq_f.std():.4f}')
    print(f'  RQ Control : {rq_c.mean():.4f} ± {rq_c.std():.4f}')
    print(f'  RQ Diff (F-C): {diff.mean():.4f}')

    rq_agg_list.append({
        'Dataset':         tag_name,
        'RQ_Frozen_Mean':  round(float(rq_f.mean()), 4),
        'RQ_Frozen_Std':   round(float(rq_f.std()),  4),
        'RQ_Control_Mean': round(float(rq_c.mean()), 4),
        'RQ_Control_Std':  round(float(rq_c.std()),  4),
        'RQ_Diff':         round(float(diff.mean()), 4),
    })

    out_base = f'{OUT_DIR}/exp4_spectral_analysis/{tag_name}'

    # RQ distribution histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rq_f, bins=40, alpha=0.6, color='steelblue',
            label=f'Frozen  (μ={rq_f.mean():.3f})')
    ax.hist(rq_c, bins=40, alpha=0.6, color='coral',
            label=f'Control (μ={rq_c.mean():.3f})')
    ax.set_xlabel('Average Rayleigh Quotient')
    ax.set_ylabel('Count')
    ax.set_title(f'{tag_name} — RQ Distribution (node-level, before pooling)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(f'{out_base}_rq_histogram.png', dpi=150)
    plt.close(fig)

    # Scatter: local homophily vs RQ
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{tag_name} — Local Homophily vs Rayleigh Quotient')
    for ax, rq_vals, label, color in [
        (axes[0], rq_f, 'Frozen',  'steelblue'),
        (axes[1], rq_c, 'Control', 'coral'),
    ]:
        ax.scatter(lh, rq_vals, alpha=0.3, s=8, color=color)
        ax.set_xlabel('Local Homophily')
        ax.set_ylabel('Rayleigh Quotient')
        ax.set_title(f'{label} Branch')
    plt.tight_layout()
    fig.savefig(f'{out_base}_rq_scatter.png', dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Summary writer
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(dataset_name, df1, df2):
    lines = [
        f'GraphControl Branch Complementarity Analysis — {dataset_name}',
        '=' * 60,
        '',
        'Experiment 1: Per-Branch Classification Performance',
        '(Aggregated over all seeds)',
        df1.to_string(index=False),
        '',
        'Experiment 2: Representation Redundancy Metrics',
        '(Aggregated over all seeds)',
        df2.to_string(index=False),
        '',
        'Interpretation guide:',
        '  CKA → 1.0       : branches are representationally similar (redundant)',
        '  CKA → 0.0       : branches are representationally diverse (complementary)',
        '  ERank_Ratio > 1.2: concatenated space is richer (complementary signal)',
        '  Cosine_Mean > 0.9: largely aligned representations',
        '  RQ_Diff > 0      : frozen branch captures higher-freq (heterophilic) signals',
        '  RQ_Diff < 0      : control branch captures higher-freq signals',
        '',
        'See exp3_homophily_contribution/ for quartile analysis plots.',
        'See exp4_spectral_analysis/ for Rayleigh Quotient histograms & scatter plots.',
    ]
    with open(f'{OUT_DIR}/summary.txt', 'w') as f:
        f.write('\n'.join(lines))
    print(f'\nSummary saved → {OUT_DIR}/summary.txt')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config):
    global OUT_DIR
    OUT_DIR = f'analysis_results/{config.dataset}'
    for d in [
        OUT_DIR,
        f'{OUT_DIR}/exp2_cosine_histograms',
        f'{OUT_DIR}/exp3_homophily_contribution',
        f'{OUT_DIR}/exp4_spectral_analysis',
    ]:
        os.makedirs(d, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Dataset: {config.dataset}  Seeds: {config.seeds}')
    print(f'Output dir: {OUT_DIR}')

    # ── Dataset loading ───────────────────────────────────────────────────────
    dataset_obj = NodeDataset(config.dataset, n_seeds=config.seeds)
    dataset_obj.print_statistics()

    homophily_ratio = float(pyg_homophily(
        dataset_obj.data.edge_index, dataset_obj.data.y))
    print(f'Global homophily ratio: {homophily_ratio:.4f}')

    if dataset_obj.num_nodes < 30000:
        dataset_obj.to(device)
    x_sim = obtain_attributes(dataset_obj.data, use_adj=False,
                               threshold=config.threshold).to(device)
    dataset_obj.to('cpu')

    # ── Pre-compute local homophily (full graph) ───────────────────────────────
    y_full = dataset_obj.data.y
    if y_full.dim() > 1:
        y_full = y_full.argmax(1)
    local_hom = compute_local_homophily(
        dataset_obj.data.edge_index, y_full, dataset_obj.num_nodes)
    print(f'Local homophily — mean={local_hom.mean():.4f}  std={local_hom.std():.4f}')

    train_masks = dataset_obj.data.train_mask
    test_masks  = dataset_obj.data.test_mask

    # ── Accumulators ──────────────────────────────────────────────────────────
    exp1_rows         = []   # one row per seed
    exp2_rows         = []   # one row per seed
    exp4_rows         = []   # one row per seed
    all_records       = []   # all records from all seeds (for combined Exp 3/4)

    # ── Per-seed loop ─────────────────────────────────────────────────────────
    for seed in config.seeds:
        print(f'\n{"="*60}')
        print(f'Seed {seed}')
        print(f'{"="*60}')
        reset_random_seed(seed)

        if dataset_obj.random_split:
            dataset_obj.data.train_mask = train_masks[:, seed]
            dataset_obj.data.test_mask  = test_masks[:, seed]
        elif dataset_obj.data.train_mask.dim() > 1:
            dataset_obj.data.train_mask = train_masks[:, seed]
            dataset_obj.data.test_mask  = test_masks[:, seed]

        train_loader, test_loader, analysis_loader = preprocess(
            config, dataset_obj, device)

        model = load_model(config.num_dim, dataset_obj.num_classes, config)
        model = model.to(device)

        # Fine-tune
        best_acc = finetune(config, model, train_loader, device, x_sim, test_loader)
        print(f'Seed {seed}  best combined acc = {best_acc:.4f}')

        # Analyse
        model.eval()
        analyzer = GraphControlAnalyzer(model, device)

        records = collect_all_test_data(
            analyzer, analysis_loader, x_sim, device, local_hom)
        all_records.extend(records)

        run_experiment_1(records, config.dataset, homophily_ratio, exp1_rows)
        run_experiment_2(records, config.dataset, exp2_rows)
        run_experiment_3(records, f'{config.dataset}_seed{seed}', mode=config.quartile_mode)
        run_experiment_4(records, f'{config.dataset}_seed{seed}', exp4_rows)

    # ── Aggregate across seeds ─────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print('AGGREGATE RESULTS (all seeds)')
    print(f'{"="*60}')

    # Exp 1 — average per dataset
    df1 = pd.DataFrame(exp1_rows)
    num_cols = ['Frozen_Acc', 'Control_Acc', 'Combined_Acc']
    agg1 = df1.groupby('Dataset').agg(
        Homophily_Ratio=('Homophily_Ratio', 'first'),
        **{c: (c, 'mean')     for c in num_cols},
        **{f'{c}_Std': (c, 'std') for c in num_cols},
    ).reset_index().round(4)
    print(f'\n[Exp 1 AGGREGATE]\n{agg1.to_string(index=False)}')
    agg1.to_csv(f'{OUT_DIR}/exp1_branch_accuracy.csv', index=False)

    # Exp 2 — average per dataset
    df2 = pd.DataFrame(exp2_rows)
    agg2 = df2.groupby('Dataset').mean(numeric_only=True).reset_index().round(4)
    print(f'\n[Exp 2 AGGREGATE]\n{agg2.to_string(index=False)}')
    agg2.to_csv(f'{OUT_DIR}/exp2_redundancy_metrics.csv', index=False)

    # Exp 3 — combined over all seeds
    print('\n[Exp 3 COMBINED (all seeds)]')
    run_experiment_3(all_records, config.dataset, mode=config.quartile_mode)

    # Exp 4 — combined over all seeds
    print('\n[Exp 4 COMBINED (all seeds)]')
    combined_rq = []
    run_experiment_4(all_records, config.dataset, combined_rq)
    # Save per-seed rows + combined row
    all_rq = exp4_rows + combined_rq
    if all_rq:
        pd.DataFrame(all_rq).to_csv(
            f'{OUT_DIR}/exp4_spectral_analysis/rayleigh_quotient_table.csv',
            index=False)

    # Summary text
    write_summary(config.dataset, agg1, agg2)
    print(f'\nAll results saved to {OUT_DIR}/')


class AnalysisArguments(Arguments):
    """Arguments 를 상속하여 analyze.py 전용 인자를 추가."""
    def __init__(self):
        super().__init__()
        self.parser.add_argument(
            '--quartile_mode', type=str, default='both',
            choices=['value', 'rank', 'both'],
            help=(
                'Exp 3 quartile assignment strategy. '
                '"value": q25/q50/q75 경계값 기반 (편향 분포에서 빈 분위 가능), '
                '"rank": 순위 기반 (항상 균등 분위), '
                '"both": 두 방식 모두 수행 (default).'
            )
        )


if __name__ == '__main__':
    config = AnalysisArguments().parse_args()
    main(config)
