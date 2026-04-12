import copy
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops, to_undirected
from tqdm import tqdm

from datasets import NodeDataset
from models import load_model
from optimizers import create_optimizer
from utils.args import Arguments
from utils.normalize import similarity
from utils.random import reset_random_seed
from utils.sampling import add_remaining_selfloop_for_isolated_nodes, collect_subgraphs
from utils.transforms import obtain_attributes, process_attributes


class OperatorSeparationGraphControl(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model.encoder
        self.trainable_copy = base_model.trainable_copy
        self.zero_conv1 = base_model.zero_conv1
        self.zero_conv2 = base_model.zero_conv2
        self.linear_classifier = base_model.linear_classifier

    def reset_classifier(self):
        self.linear_classifier.reset_parameters()

    def forward_subgraph(
        self,
        x,
        x_sim,
        edge_index,
        control_edge_index,
        batch,
        root_n_id,
        frozen=True,
        return_components=False,
    ):
        if not frozen:
            raise NotImplementedError("This experiment keeps the frozen branch fixed.")

        with torch.no_grad():
            self.encoder.eval()
            h_frozen = self.encoder.forward_subgraph(x, edge_index, batch, root_n_id)

        x_down = self.zero_conv1(x_sim) + x
        h_control = self.trainable_copy.forward_subgraph(x_down, control_edge_index, batch, root_n_id)
        h_control = self.zero_conv2(h_control)

        combined = h_frozen + h_control
        combined_logits = self.linear_classifier(combined)
        if not return_components:
            return combined_logits

        return {
            "combined_logits": combined_logits,
            "frozen_logits": self.linear_classifier(h_frozen),
            "control_logits": self.linear_classifier(h_control),
        }


def build_parser():
    parser = Arguments().parser
    parser.description = "Run GraphControl operator separation experiments."
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "condition_operator"],
        choices=["baseline", "condition_operator"],
    )
    parser.add_argument("--output_root", type=str, default="operator_separation_results")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser


def get_mask_column(mask_tensor, seed_idx):
    if mask_tensor.dim() == 1:
        return mask_tensor.clone()
    return mask_tensor[:, seed_idx].clone()


def split_masks(data, seed, seed_idx, val_ratio):
    train_mask = get_mask_column(data.train_mask, seed_idx)
    test_mask = get_mask_column(data.test_mask, seed_idx)

    if hasattr(data, "val_mask"):
        val_mask = get_mask_column(data.val_mask, seed_idx)
        return train_mask, val_mask, test_mask

    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    if train_idx.numel() <= 1:
        return train_mask, train_mask.clone(), test_mask

    reset_random_seed(seed)
    perm = train_idx[torch.randperm(train_idx.numel())]
    val_size = max(1, int(round(train_idx.numel() * val_ratio)))
    if val_size >= train_idx.numel():
        val_size = train_idx.numel() - 1

    val_idx = perm[:val_size]
    new_train_idx = perm[val_size:]

    new_train_mask = torch.zeros_like(train_mask)
    val_mask = torch.zeros_like(train_mask)
    new_train_mask[new_train_idx] = True
    val_mask[val_idx] = True
    return new_train_mask, val_mask, test_mask


def make_loader(graphs, batch_size, shuffle, num_workers):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(graphs, **kwargs)


def build_condition_edge_index(x_sim_local, threshold):
    adj = similarity(x_sim_local, x_sim_local)
    adj = torch.where(adj > threshold, 1.0, 0.0)
    adj.fill_diagonal_(0.0)

    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=x_sim_local.device)
    else:
        edge_index = to_undirected(edge_index)
        edge_index, _ = remove_self_loops(edge_index)
    return add_remaining_selfloop_for_isolated_nodes(edge_index, x_sim_local.size(0)).long()


def attach_control_topology(graphs, full_x_sim, threshold):
    for graph in graphs:
        local_x_sim = full_x_sim[graph.original_idx].cpu()
        graph.control_edge_index = build_condition_edge_index(local_x_sim, threshold)


def build_subgraph_loaders(config, dataset_obj, train_mask, val_mask, test_mask, full_x_sim):
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    val_idx = val_mask.nonzero(as_tuple=False).view(-1)
    test_idx = test_mask.nonzero(as_tuple=False).view(-1)

    train_graphs = collect_subgraphs(
        train_idx, dataset_obj.data, walk_steps=config.walk_steps, restart_ratio=config.restart
    )
    val_graphs = collect_subgraphs(
        val_idx, dataset_obj.data, walk_steps=config.walk_steps, restart_ratio=config.restart
    )
    test_graphs = collect_subgraphs(
        test_idx, dataset_obj.data, walk_steps=config.walk_steps, restart_ratio=config.restart
    )

    for graph in train_graphs + val_graphs + test_graphs:
        process_attributes(graph, use_adj=config.use_adj, threshold=config.threshold, num_dim=config.num_dim)

    attach_control_topology(train_graphs + val_graphs + test_graphs, full_x_sim.cpu(), config.threshold)

    train_loader = make_loader(train_graphs, config.batch_size, True, config.num_workers)
    val_loader = make_loader(val_graphs, config.batch_size, False, config.num_workers)
    test_loader = make_loader(test_graphs, config.batch_size, False, config.num_workers)
    return train_loader, val_loader, test_loader


def train_variant(config, model, variant, train_loader, val_loader, device, full_x_sim):
    for name, parameter in model.named_parameters():
        if "encoder" in name:
            parameter.requires_grad = False

    model.reset_classifier()
    optimizer = create_optimizer(
        name=config.optimizer,
        parameters=filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    eval_steps = 3
    patience = 15
    patience_count = 0
    best_val_acc = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    progress = tqdm(range(config.epochs))
    for epoch in progress:
        for data in train_loader:
            optimizer.zero_grad()
            model.train()

            data = data.to(device)
            if not hasattr(data, "root_n_id"):
                data.root_n_id = data.root_n_index

            sign_flip = torch.rand(data.x.size(1), device=device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            x = data.x * sign_flip.unsqueeze(0)
            x_sim = full_x_sim[data.original_idx]
            control_edge_index = data.edge_index if variant == "baseline" else data.control_edge_index

            logits = model.forward_subgraph(
                x, x_sim, data.edge_index, control_edge_index, data.batch, data.root_n_id, frozen=True
            )
            loss = criterion(logits, data.y)
            loss.backward()
            optimizer.step()

        if epoch % eval_steps == 0:
            val_metrics = evaluate_variant(model, variant, val_loader, device, full_x_sim)
            progress.set_postfix({"Epoch": epoch, "ValAcc": f"{val_metrics['ValAcc']:.4f}"})
            if val_metrics["ValAcc"] > best_val_acc:
                best_val_acc = val_metrics["ValAcc"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_count = 0
            else:
                patience_count += 1

        if patience_count >= patience:
            break

    model.load_state_dict(best_state)
    return best_epoch, best_val_acc


@torch.no_grad()
def evaluate_variant(model, variant, loader, device, full_x_sim):
    model.eval()
    correct = {"combined": 0, "frozen": 0, "control": 0}
    total = 0

    for batch in loader:
        batch = batch.to(device)
        if not hasattr(batch, "root_n_id"):
            batch.root_n_id = batch.root_n_index
        x_sim = full_x_sim[batch.original_idx]
        control_edge_index = batch.edge_index if variant == "baseline" else batch.control_edge_index

        outputs = model.forward_subgraph(
            batch.x,
            x_sim,
            batch.edge_index,
            control_edge_index,
            batch.batch,
            batch.root_n_id,
            frozen=True,
            return_components=True,
        )
        correct["combined"] += (outputs["combined_logits"].argmax(dim=1) == batch.y).sum().item()
        correct["frozen"] += (outputs["frozen_logits"].argmax(dim=1) == batch.y).sum().item()
        correct["control"] += (outputs["control_logits"].argmax(dim=1) == batch.y).sum().item()
        total += batch.y.size(0)

    denom = max(total, 1)
    return {
        "ValAcc": correct["combined"] / denom,
        "FrozenOnlyAcc": correct["frozen"] / denom,
        "ControlOnlyAcc": correct["control"] / denom,
        "CombinedAcc": correct["combined"] / denom,
    }


def format_metric(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.6f}"


def write_variant_report(output_dir, variant, per_seed_rows, aggregate_row):
    report_path = output_dir / f"{variant}.txt"
    lines = [f"Dataset: {aggregate_row['Dataset']}", f"Variant: {variant}", ""]
    lines.append("Per-seed results:")
    for row in per_seed_rows:
        lines.append(
            "  Seed={Seed} ValAcc={ValAcc:.6f} TestAcc={TestAcc:.6f} BestEpoch={BestEpoch} FrozenOnlyAcc={FrozenOnlyAcc:.6f} ControlOnlyAcc={ControlOnlyAcc:.6f} CombinedAcc={CombinedAcc:.6f}".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Aggregate:",
            f"  ValAcc={aggregate_row['ValAcc']:.6f}",
            f"  TestAcc={aggregate_row['TestAcc']:.6f}",
            f"  BestEpoch={aggregate_row['BestEpoch']:.2f}",
            f"  FrozenOnlyAcc={aggregate_row['FrozenOnlyAcc']:.6f}",
            f"  ControlOnlyAcc={aggregate_row['ControlOnlyAcc']:.6f}",
            f"  CombinedAcc={aggregate_row['CombinedAcc']:.6f}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


def aggregate_rows(dataset_name, variant, rows):
    return {
        "Dataset": dataset_name,
        "Variant": variant,
        "ValAcc": float(np.mean([row["ValAcc"] for row in rows])),
        "TestAcc": float(np.mean([row["TestAcc"] for row in rows])),
        "BestEpoch": float(np.mean([row["BestEpoch"] for row in rows])),
        "FrozenOnlyAcc": float(np.mean([row["FrozenOnlyAcc"] for row in rows])),
        "ControlOnlyAcc": float(np.mean([row["ControlOnlyAcc"] for row in rows])),
        "CombinedAcc": float(np.mean([row["CombinedAcc"] for row in rows])),
    }


def write_summary_csv(output_dir, rows):
    summary_path = output_dir / "summary.csv"
    fieldnames = [
        "Dataset",
        "Variant",
        "ValAcc",
        "TestAcc",
        "BestEpoch",
        "FrozenOnlyAcc",
        "ControlOnlyAcc",
        "CombinedAcc",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_metric(value) if key not in {"Dataset", "Variant"} else value for key, value in row.items()})


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_obj = NodeDataset(config.dataset, n_seeds=config.seeds)
    dataset_obj.print_statistics()

    if dataset_obj.num_nodes < 30000:
        dataset_obj.to(device)
    full_x_sim = obtain_attributes(dataset_obj.data, use_adj=False, threshold=config.threshold).to(device)

    dataset_obj.to("cpu")
    num_node_features = config.num_dim

    train_masks = dataset_obj.data.train_mask
    test_masks = dataset_obj.data.test_mask
    val_masks = getattr(dataset_obj.data, "val_mask", None)

    output_dir = Path(config.output_root) / config.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for variant in config.variants:
        per_seed_rows = []
        print(f"Running operator variant: {variant}")
        for seed_idx, seed in enumerate(config.seeds):
            reset_random_seed(seed)

            dataset_obj.data.train_mask = get_mask_column(train_masks, seed_idx)
            dataset_obj.data.test_mask = get_mask_column(test_masks, seed_idx)
            if val_masks is not None:
                dataset_obj.data.val_mask = get_mask_column(val_masks, seed_idx)

            train_mask, val_mask, test_mask = split_masks(dataset_obj.data, seed, seed_idx, config.val_ratio)
            train_loader, val_loader, test_loader = build_subgraph_loaders(
                config, dataset_obj, train_mask, val_mask, test_mask, full_x_sim
            )

            base_model = load_model(num_node_features, dataset_obj.num_classes, config).to(device)
            model = OperatorSeparationGraphControl(base_model).to(device)

            best_epoch, best_val_acc = train_variant(
                config, model, variant, train_loader, val_loader, device, full_x_sim
            )
            test_metrics = evaluate_variant(model, variant, test_loader, device, full_x_sim)

            row = {
                "Seed": seed,
                "ValAcc": best_val_acc,
                "TestAcc": test_metrics["CombinedAcc"],
                "BestEpoch": best_epoch,
                "FrozenOnlyAcc": test_metrics["FrozenOnlyAcc"],
                "ControlOnlyAcc": test_metrics["ControlOnlyAcc"],
                "CombinedAcc": test_metrics["CombinedAcc"],
            }
            per_seed_rows.append(row)
            print(
                f"Seed: {seed}, Variant: {variant}, ValAcc: {best_val_acc:.4f}, "
                f"TestAcc: {test_metrics['CombinedAcc']:.4f}, BestEpoch: {best_epoch}"
            )

        aggregate_row = aggregate_rows(config.dataset, variant, per_seed_rows)
        write_variant_report(output_dir, variant, per_seed_rows, aggregate_row)
        summary_rows.append(aggregate_row)

    write_summary_csv(output_dir, summary_rows)
    print(f"Saved operator separation results to {output_dir}")


if __name__ == "__main__":
    main(build_parser().parse_args())
