"""
analyze_margin.py — Sample-wise margin diagnostics for GraphControl

Usage (development):
    python analyze_margin.py --dataset Cora_ML --lr 0.5 --optimizer adamw --use_adj --seeds 0 --model GCC_GraphControl

Usage (full):
    python analyze_margin.py --dataset Cora_ML --lr 0.5 --optimizer adamw --threshold 0.17 --use_adj --seeds 0 1 2 3 4 --model GCC_GraphControl
    python analyze_margin.py --dataset Squirrel --lr 0.5 --optimizer adamw --threshold 0.15 --use_adj --seeds 0 1 2 3 4 --model GCC_GraphControl
"""

import os
import copy
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from utils.random import reset_random_seed
from utils.args import Arguments
from utils.metrics import compute_margin_gain, compute_true_class_margin
from utils.sampling import collect_subgraphs
from utils.transforms import process_attributes, obtain_attributes
from models import load_model
from datasets import NodeDataset
from optimizers import create_optimizer


OUT_DIR = 'analysis_results'


class GraphControlMarginAnalyzer:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def forward_diagnostics(self, x, x_sim, edge_index, batch, root_n_id):
        with torch.no_grad():
            return self.model.forward_with_diagnostics(
                x, x_sim, edge_index, batch, root_n_id, frozen=True)


def preprocess(config, dataset_obj, device):
    kwargs_train = dict(batch_size=config.batch_size, num_workers=4,
                        persistent_workers=True, pin_memory=True, shuffle=True)
    kwargs_test = dict(batch_size=config.batch_size, num_workers=4,
                       persistent_workers=True, pin_memory=True, shuffle=False)
    kwargs_margin = dict(batch_size=1, num_workers=2, persistent_workers=True,
                         shuffle=False)

    print('Generating subgraphs...')
    train_idx = dataset_obj.data.train_mask.nonzero().squeeze()
    test_idx = dataset_obj.data.test_mask.nonzero().squeeze()

    train_graphs = collect_subgraphs(
        train_idx, dataset_obj.data,
        walk_steps=config.walk_steps, restart_ratio=config.restart)
    test_graphs = collect_subgraphs(
        test_idx, dataset_obj.data,
        walk_steps=config.walk_steps, restart_ratio=config.restart)

    [process_attributes(g, use_adj=config.use_adj,
                        threshold=config.threshold,
                        num_dim=config.num_dim) for g in train_graphs]
    [process_attributes(g, use_adj=config.use_adj,
                        threshold=config.threshold,
                        num_dim=config.num_dim) for g in test_graphs]

    train_loader = DataLoader(train_graphs, **kwargs_train)
    test_loader = DataLoader(test_graphs, **kwargs_test)
    margin_loader = DataLoader(test_graphs, **kwargs_margin)
    return train_loader, test_loader, margin_loader


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
            total += batch.y.shape[0]
    return correct / total


def finetune(config, model, train_loader, device, full_x_sim, test_loader):
    for k, v in model.named_parameters():
        if 'encoder' in k:
            v.requires_grad = False

    model.reset_classifier()
    eval_steps = 3
    patience = 15
    count = 0
    best_acc = 0.0
    best_state = None

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = create_optimizer(name=config.optimizer, parameters=params,
                                 lr=config.lr, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    pbar = tqdm(range(config.epochs), desc='Fine-tuning')

    for epoch in pbar:
        for data in train_loader:
            optimizer.zero_grad()
            model.train()
            data = data.to(device)
            if not hasattr(data, 'root_n_id'):
                data.root_n_id = data.root_n_index

            sign_flip = torch.rand(data.x.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            x = data.x * sign_flip.unsqueeze(0)

            x_sim = full_x_sim[data.original_idx]
            diagnostics = model.forward_with_diagnostics(
                x, x_sim, data.edge_index, data.batch, data.root_n_id, frozen=True)
            ce_loss = criterion(diagnostics['z_fc'], data.y)
            loss = ce_loss
            if config.use_l_gain:
                loss = loss + config.lambda_gain * torch.zeros(
                    (), device=ce_loss.device, dtype=ce_loss.dtype)
            loss.backward()
            optimizer.step()

        if epoch % eval_steps == 0:
            acc = eval_subgraph(model, test_loader, device, full_x_sim)
            pbar.set_postfix({'epoch': epoch, 'acc': f'{acc:.4f}'})
            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc


def collect_margin_records(analyzer, margin_loader, full_x_sim, device):
    records = []

    for data in tqdm(margin_loader, desc='Collecting margins', leave=False):
        data = data.to(device)
        if not hasattr(data, 'root_n_id'):
            data.root_n_id = data.root_n_index

        diagnostics = analyzer.forward_diagnostics(
            data.x,
            full_x_sim[data.original_idx],
            data.edge_index,
            data.batch,
            data.root_n_id,
        )

        logits_frozen = diagnostics['z_f']
        logits_combined = diagnostics['z_fc']
        labels = data.y.view(-1)

        margin_f = compute_true_class_margin(logits_frozen, labels).item()
        margin_fc = compute_true_class_margin(logits_combined, labels).item()
        margin_gain = compute_margin_gain(
            logits_frozen, logits_combined, labels).item()

        rnid = data.root_n_id
        rnid_val = rnid.item() if torch.is_tensor(rnid) else int(rnid)
        center_global = data.original_idx[rnid_val].item()

        records.append({
            'sample_index': center_global,
            'node_id': center_global,
            'label': data.y.item(),
            'pred_frozen': logits_frozen.argmax(dim=1).item(),
            'pred_combined': logits_combined.argmax(dim=1).item(),
            'margin_f': margin_f,
            'margin_fc': margin_fc,
            'margin_gain': margin_gain,
        })

    return records


def export_margin_metrics(records, dataset_name, seed, summary_rows):
    if not records:
        return

    raw_dir = f'{OUT_DIR}/margin_raw'
    os.makedirs(raw_dir, exist_ok=True)

    raw_rows = []
    for r in records:
        raw_rows.append({
            'dataset': dataset_name,
            'seed': seed,
            'split': 'test',
            'sample_index': r['sample_index'],
            'node_id': r['node_id'],
            'label': r['label'],
            'pred_frozen': r['pred_frozen'],
            'pred_combined': r['pred_combined'],
            'margin_f': r['margin_f'],
            'margin_fc': r['margin_fc'],
            'margin_gain': r['margin_gain'],
        })

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(
        f'{raw_dir}/margin_metrics_{dataset_name}_seed{seed}.csv',
        index=False,
    )

    summary_rows.append({
        'dataset': dataset_name,
        'seed': seed,
        'split': 'test',
        'num_samples': len(raw_df),
        'mean_margin_f': raw_df['margin_f'].mean(),
        'mean_margin_fc': raw_df['margin_fc'].mean(),
        'mean_margin_gain': raw_df['margin_gain'].mean(),
        'std_margin_gain': raw_df['margin_gain'].std(ddof=0),
        'positive_margin_gain_frac': (raw_df['margin_gain'] > 0).mean(),
        'negative_margin_gain_frac': (raw_df['margin_gain'] < 0).mean(),
    })


def write_summary(dataset_name, df_margin):
    lines = [
        f'GraphControl Margin Analysis — {dataset_name}',
        '=' * 60,
        '',
        'Margin diagnostics',
        '(Aggregated over all seeds)',
        df_margin.to_string(index=False),
    ]
    with open(f'{OUT_DIR}/margin_summary/summary_{dataset_name}.txt', 'w') as f:
        f.write('\n'.join(lines))
    print(f'\nSummary saved → {OUT_DIR}/margin_summary/summary_{dataset_name}.txt')


def main(config):
    global OUT_DIR
    OUT_DIR = f'analysis_results/{config.dataset}'
    for d in [
        OUT_DIR,
        f'{OUT_DIR}/margin_raw',
        f'{OUT_DIR}/margin_summary',
    ]:
        os.makedirs(d, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Dataset: {config.dataset}  Seeds: {config.seeds}')
    print(f'Output dir: {OUT_DIR}')

    dataset_obj = NodeDataset(config.dataset, n_seeds=config.seeds)
    dataset_obj.print_statistics()

    if dataset_obj.num_nodes < 30000:
        dataset_obj.to(device)
    x_sim = obtain_attributes(
        dataset_obj.data, use_adj=False, threshold=config.threshold).to(device)
    dataset_obj.to('cpu')

    train_masks = dataset_obj.data.train_mask
    test_masks = dataset_obj.data.test_mask

    margin_rows = []

    for seed in config.seeds:
        print(f'\n{"="*60}')
        print(f'Seed {seed}')
        print(f'{"="*60}')
        reset_random_seed(seed)

        if dataset_obj.random_split:
            dataset_obj.data.train_mask = train_masks[:, seed]
            dataset_obj.data.test_mask = test_masks[:, seed]
        elif dataset_obj.data.train_mask.dim() > 1:
            dataset_obj.data.train_mask = train_masks[:, seed]
            dataset_obj.data.test_mask = test_masks[:, seed]

        train_loader, test_loader, margin_loader = preprocess(
            config, dataset_obj, device)

        model = load_model(config.num_dim, dataset_obj.num_classes, config)
        model = model.to(device)

        best_acc = finetune(config, model, train_loader, device, x_sim, test_loader)
        print(f'Seed {seed}  best combined acc = {best_acc:.4f}')

        model.eval()
        analyzer = GraphControlMarginAnalyzer(model)
        records = collect_margin_records(analyzer, margin_loader, x_sim, device)
        export_margin_metrics(records, config.dataset, seed, margin_rows)

    df_margin = pd.DataFrame(margin_rows)
    margin_num_cols = [
        'num_samples',
        'mean_margin_f',
        'mean_margin_fc',
        'mean_margin_gain',
        'std_margin_gain',
        'positive_margin_gain_frac',
        'negative_margin_gain_frac',
    ]
    agg_margin = df_margin.groupby('dataset').agg(
        **{col: (col, 'mean') for col in margin_num_cols},
    ).reset_index().round(4)

    print(f'\n[Margin AGGREGATE]\n{agg_margin.to_string(index=False)}')
    agg_margin.to_csv(
        f'{OUT_DIR}/margin_summary/margin_summary_{config.dataset}.csv',
        index=False,
    )
    write_summary(config.dataset, agg_margin)
    print(f'\nMargin analysis results saved to {OUT_DIR}/')


if __name__ == '__main__':
    config = Arguments().parse_args()
    main(config)
