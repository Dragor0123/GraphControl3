import copy
import json
import os
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np


from utils.random import reset_random_seed
from utils.args import Arguments
from utils.metrics import compute_margin_gain, compute_margin_gain_loss
from utils.sampling import collect_subgraphs
from utils.transforms import (
    process_attributes,
    obtain_attributes,
    obtain_attributes_dissimilarity,
    compute_condition_homophily,
)
from models import load_model
from datasets import NodeDataset
from optimizers import create_optimizer


def preprocess(config, dataset_obj, device):
    kwargs = {'batch_size': config.batch_size, 'num_workers': 4, 'persistent_workers': True, 'pin_memory': True}
    
    print('generating subgraphs....')
    
    train_idx = dataset_obj.data.train_mask.nonzero().squeeze()
    test_idx = dataset_obj.data.test_mask.nonzero().squeeze()
    
    train_graphs = collect_subgraphs(train_idx, dataset_obj.data, walk_steps=config.walk_steps, restart_ratio=config.restart)
    test_graphs = collect_subgraphs(test_idx, dataset_obj.data, walk_steps=config.walk_steps, restart_ratio=config.restart)

    [process_attributes(g, use_adj=config.use_adj, threshold=config.threshold, num_dim=config.num_dim) for g in train_graphs]
    [process_attributes(g, use_adj=config.use_adj, threshold=config.threshold, num_dim=config.num_dim) for g in test_graphs]
    
        
    train_loader = DataLoader(train_graphs, shuffle=True, **kwargs)
    test_loader = DataLoader(test_graphs, **kwargs)

    return train_loader, test_loader


def finetune(config, model, train_loader, device, full_x_sim, test_loader):
    # freeze the pre-trained encoder (left branch)
    if hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = False
            
    model.reset_classifier()
    eval_steps = 3
    patience = 15
    count = 0
    best_acc = 0
    best_state = None
    best_train_ce = None
    best_train_l_gain = None

    params  = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = create_optimizer(name=config.optimizer, parameters=params, lr=config.lr, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    process_bar = tqdm(range(config.epochs))

    for epoch in process_bar:
        ce_loss_sum = 0.0
        l_gain_sum = 0.0
        num_batches = 0
        num_l_gain_batches = 0
        for data in train_loader:
            optimizer.zero_grad()
            model.train()

            data = data.to(device)
            
            if not hasattr(data, 'root_n_id'):
                data.root_n_id = data.root_n_index

            sign_flip = torch.rand(data.x.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            x = data.x * sign_flip.unsqueeze(0)
            
            x_sim = full_x_sim[data.original_idx]
            if config.use_l_gain:
                if not hasattr(model, 'forward_with_diagnostics'):
                    raise ValueError('L_gain requires a model with forward diagnostics support')
                diagnostics = model.forward_with_diagnostics(
                    x,
                    x_sim,
                    data.edge_index,
                    data.batch,
                    data.root_n_id,
                    frozen=True,
                )
                ce_loss = criterion(diagnostics['z_fc'], data.y)
                l_gain = compute_margin_gain_loss(
                    diagnostics['z_f'],
                    diagnostics['z_fc'],
                    data.y.view(-1),
                )
                loss = ce_loss + config.lambda_gain * l_gain
                ce_loss_sum += ce_loss.item()
                l_gain_sum += l_gain.item()
                num_l_gain_batches += 1
            else:
                preds = model.forward_subgraph(
                    x, x_sim, data.edge_index, data.batch, data.root_n_id, frozen=True)
                ce_loss = criterion(preds, data.y)
                loss = ce_loss
                ce_loss_sum += ce_loss.item()
            loss.backward()
            optimizer.step()
            num_batches += 1
    
        if epoch % eval_steps == 0:
            acc = eval_subgraph(config, model, test_loader, device, full_x_sim)
            mean_ce_loss = ce_loss_sum / max(num_batches, 1)
            postfix = {"Epoch": epoch, "Accuracy": f"{acc:.4f}", "CE": f"{mean_ce_loss:.4f}"}
            mean_l_gain = None
            if num_l_gain_batches > 0:
                mean_l_gain = l_gain_sum / num_l_gain_batches
                postfix["L_gain"] = f"{mean_l_gain:.4f}"
            process_bar.set_postfix(postfix)
            if best_acc < acc:
                best_acc = acc
                best_state = copy.deepcopy(model.state_dict())
                best_train_ce = mean_ce_loss
                best_train_l_gain = mean_l_gain
                count = 0
            else:
                count += 1

        if count == patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        'best_acc': best_acc,
        'best_train_ce': best_train_ce,
        'best_train_l_gain': best_train_l_gain,
    }


def _get_config_name(config):
    """Derive experiment subdirectory name from condition_mode and operator_mode."""
    c = getattr(config, 'condition_mode', 'similarity')
    o = getattr(config, 'operator_mode', 'standard')
    if c == 'similarity' and o == 'standard':
        return 'baseline'
    elif c == 'dissimilarity' and o == 'standard':
        thr = getattr(config, 'dissim_threshold', 0.5)
        return f'C1_dissim/threshold_{thr}'
    elif c == 'similarity' and o == 'anti_smoothing':
        return 'O1_antismooth'
    else:
        thr = getattr(config, 'dissim_threshold', 0.5)
        return f'C1_O1_combined/threshold_{thr}'


def _save_results(config, seed_metrics, condition_stats):
    config_name = _get_config_name(config)
    out_dir = os.path.join('experiment_results', 'dual_track', config_name)
    os.makedirs(out_dir, exist_ok=True)

    result = {
        'config': {
            'dataset': config.dataset,
            'seeds': config.seeds,
            'condition_mode': getattr(config, 'condition_mode', 'similarity'),
            'dissim_threshold': getattr(config, 'dissim_threshold', 0.5),
            'operator_mode': getattr(config, 'operator_mode', 'standard'),
            'epochs': config.epochs,
            'lr': config.lr,
            'threshold': config.threshold,
        },
        'condition_stats': condition_stats,
        'seed_metrics': seed_metrics,
        'summary': {
            'combined_acc_mean': float(np.mean([m['combined_acc'] for m in seed_metrics])),
            'combined_acc_std': float(np.std([m['combined_acc'] for m in seed_metrics])),
            'frozen_acc_mean': float(np.mean([m['frozen_acc'] for m in seed_metrics])),
            'control_acc_mean': float(np.mean([m['control_acc'] for m in seed_metrics])),
        },
    }

    fname = os.path.join(out_dir, f'{config.dataset}.json')
    with open(fname, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {fname}")


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_obj = NodeDataset(config.dataset, n_seeds=config.seeds)
    dataset_obj.print_statistics()

    condition_mode = getattr(config, 'condition_mode', 'similarity')
    dissim_threshold = getattr(config, 'dissim_threshold', 0.5)

    # For large graph, use cpu to preprocess rather than gpu (OOM prevention).
    if dataset_obj.num_nodes < 30000:
        dataset_obj.to(device)

    condition_stats = {}
    if condition_mode == 'dissimilarity':
        x_sim, cstats = obtain_attributes_dissimilarity(
            dataset_obj.data, num_dim=config.num_dim, dissim_threshold=dissim_threshold
        )
        x_sim = x_sim.to(device)
        # Compute homophily ratio of A'_dissim if labels are available
        if hasattr(dataset_obj.data, 'y') and not cstats.get('fallback', False):
            from torch_geometric.utils import to_undirected, remove_self_loops, to_dense_adj
            import torch as _torch
            from utils.normalize import similarity as _sim
            X = dataset_obj.data.x
            N = X.size(0)
            K = _sim(X, X)
            edges = to_undirected(dataset_obj.data.edge_index)
            edges, _ = remove_self_loops(edges)
            A_orig = to_dense_adj(edges, max_num_nodes=N)[0]
            A_dissim = ((_torch.lt(K, dissim_threshold)).float()) * A_orig
            A_dissim.fill_diagonal_(0)
            A_dissim = (A_dissim + A_dissim.T).clamp(max=1.0)
            hom = compute_condition_homophily(dataset_obj.data, A_dissim)
            cstats['homophily_ratio'] = hom
        condition_stats = cstats
        print(
            f"[Condition] dissimilarity A': edges={cstats['num_edges']}, "
            f"overlap_ratio={cstats['overlap_ratio']:.4f}, "
            f"homophily_ratio={cstats.get('homophily_ratio', 'N/A')}"
        )
    else:
        x_sim = obtain_attributes(dataset_obj.data, use_adj=False, threshold=config.threshold).to(device)
        # Compute stats for similarity A' as well
        from torch_geometric.utils import to_undirected, remove_self_loops, to_dense_adj
        from utils.normalize import similarity as _sim
        X = dataset_obj.data.x
        N = X.size(0)
        K = _sim(X, X)
        edges_orig = to_undirected(dataset_obj.data.edge_index)
        edges_orig, _ = remove_self_loops(edges_orig)
        A_orig = to_dense_adj(edges_orig, max_num_nodes=N)[0]
        A_sim = torch.where(K > config.threshold, 1.0, 0.0)
        A_sim.fill_diagonal_(0)
        num_edges_sim = int(A_sim.sum().item()) // 2
        num_edges_orig = int(A_orig.sum().item()) // 2
        hom = compute_condition_homophily(dataset_obj.data, A_sim)
        condition_stats = {
            'num_edges': num_edges_sim,
            'num_edges_orig': num_edges_orig,
            'overlap_ratio': (A_sim * A_orig).sum().item() / max(A_sim.sum().item(), 1),
            'homophily_ratio': hom,
            'threshold': config.threshold,
        }
        print(
            f"[Condition] similarity A': edges={num_edges_sim}, "
            f"overlap_ratio={condition_stats['overlap_ratio']:.4f}, "
            f"homophily_ratio={hom}"
        )

    dataset_obj.to('cpu')  # Otherwise the deepcopy will raise an error
    num_node_features = config.num_dim

    train_masks = dataset_obj.data.train_mask
    test_masks = dataset_obj.data.test_mask

    seed_metrics = []

    for i, seed in enumerate(config.seeds):
        reset_random_seed(seed)
        if dataset_obj.random_split:
            dataset_obj.data.train_mask = train_masks[:, seed]
            dataset_obj.data.test_mask = test_masks[:, seed]
        elif dataset_obj.data.train_mask.dim() > 1:
            dataset_obj.data.train_mask = train_masks[:, seed]
            dataset_obj.data.test_mask = test_masks[:, seed]

        train_loader, test_loader = preprocess(config, dataset_obj, device)

        model = load_model(num_node_features, dataset_obj.num_classes, config)
        model = model.to(device)

        # finetuning model
        train_summary = finetune(config, model, train_loader, device, x_sim, test_loader)
        eval_metrics = eval_subgraph_diagnostics(config, model, test_loader, device, x_sim)

        seed_metrics.append({
            'seed': seed,
            'best_acc': train_summary['best_acc'],
            'best_train_ce': train_summary['best_train_ce'],
            'best_train_l_gain': train_summary['best_train_l_gain'],
            **eval_metrics,
        })
        print(
            f"Seed: {seed}, "
            f"Combined: {eval_metrics['combined_acc']:.4f}, "
            f"Frozen: {eval_metrics['frozen_acc']:.4f}, "
            f"Control: {eval_metrics['control_acc']:.4f}, "
            f"MeanMarginGain: {eval_metrics['mean_margin_gain']:.4f}, "
            f"TestLGain: {eval_metrics['mean_test_l_gain']:.4f}"
        )

    final_acc = np.mean([m['combined_acc'] for m in seed_metrics])
    final_acc_std = np.std([m['combined_acc'] for m in seed_metrics])
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(
        f"# final_frozen_acc: "
        f"{np.mean([m['frozen_acc'] for m in seed_metrics]):.4f}"
        f"±{np.std([m['frozen_acc'] for m in seed_metrics]):.4f}"
    )
    print(
        f"# final_control_acc: "
        f"{np.mean([m['control_acc'] for m in seed_metrics]):.4f}"
        f"±{np.std([m['control_acc'] for m in seed_metrics]):.4f}"
    )
    print(
        f"# final_combined_acc: "
        f"{np.mean([m['combined_acc'] for m in seed_metrics]):.4f}"
        f"±{np.std([m['combined_acc'] for m in seed_metrics]):.4f}"
    )
    print(
        f"# final_mean_margin_gain: "
        f"{np.mean([m['mean_margin_gain'] for m in seed_metrics]):.4f}"
        f"±{np.std([m['mean_margin_gain'] for m in seed_metrics]):.4f}"
    )
    print(
        f"# final_test_l_gain: "
        f"{np.mean([m['mean_test_l_gain'] for m in seed_metrics]):.4f}"
        f"±{np.std([m['mean_test_l_gain'] for m in seed_metrics]):.4f}"
    )
    train_l_gain_values = [
        m['best_train_l_gain'] for m in seed_metrics if m['best_train_l_gain'] is not None
    ]
    if train_l_gain_values:
        print(
            f"# final_train_l_gain: "
            f"{np.mean(train_l_gain_values):.4f}"
            f"±{np.std(train_l_gain_values):.4f}"
        )

    _save_results(config, seed_metrics, condition_stats)


def eval_subgraph(config, model, test_loader, device, full_x_sim):
    model.eval()
    
    correct = 0
    total_num = 0
    for batch in test_loader:
        batch = batch.to(device)
        if not hasattr(batch, 'root_n_id'):
            batch.root_n_id = batch.root_n_index
        x_sim = full_x_sim[batch.original_idx]
        preds = model.forward_subgraph(batch.x, x_sim, batch.edge_index, batch.batch, batch.root_n_id, frozen=True).argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total_num += batch.y.shape[0]
    acc = correct / total_num
    return acc


def eval_subgraph_diagnostics(config, model, test_loader, device, full_x_sim):
    model.eval()

    frozen_correct = 0
    control_correct = 0
    combined_correct = 0
    total_num = 0
    margin_gain_sum = 0.0
    test_l_gain_sum = 0.0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            if not hasattr(batch, 'root_n_id'):
                batch.root_n_id = batch.root_n_index

            x_sim = full_x_sim[batch.original_idx]
            diagnostics = model.forward_with_diagnostics(
                batch.x,
                x_sim,
                batch.edge_index,
                batch.batch,
                batch.root_n_id,
                frozen=True,
            )

            labels = batch.y.view(-1)
            pred_frozen = diagnostics['z_f'].argmax(dim=1)
            pred_control = diagnostics['z_c'].argmax(dim=1)
            pred_combined = diagnostics['z_fc'].argmax(dim=1)
            margin_gain = compute_margin_gain(
                diagnostics['z_f'],
                diagnostics['z_fc'],
                labels,
            )

            batch_size = labels.shape[0]
            frozen_correct += (pred_frozen == labels).sum().item()
            control_correct += (pred_control == labels).sum().item()
            combined_correct += (pred_combined == labels).sum().item()
            margin_gain_sum += margin_gain.sum().item()
            test_l_gain_sum += torch.relu(-margin_gain).sum().item()
            total_num += batch_size

    return {
        'frozen_acc': frozen_correct / total_num,
        'control_acc': control_correct / total_num,
        'combined_acc': combined_correct / total_num,
        'mean_margin_gain': margin_gain_sum / total_num,
        'mean_test_l_gain': test_l_gain_sum / total_num,
    }

if __name__ == '__main__':
    config = Arguments().parse_args()
    
    main(config)
