import torch


def compute_true_class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError('logits must have shape [N, C]')
    if labels.ndim != 1:
        raise ValueError('labels must have shape [N]')
    if logits.size(0) != labels.size(0):
        raise ValueError('logits and labels must agree on batch dimension')

    true_class_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    masked_logits = logits.clone()
    masked_logits.scatter_(1, labels.unsqueeze(1), float('-inf'))
    largest_non_true_logits = masked_logits.max(dim=1).values
    return true_class_logits - largest_non_true_logits


def compute_margin_gain(
    frozen_logits: torch.Tensor,
    combined_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    margin_f = compute_true_class_margin(frozen_logits, labels)
    margin_fc = compute_true_class_margin(combined_logits, labels)
    return margin_fc - margin_f


def compute_margin_gain_loss(
    frozen_logits: torch.Tensor,
    combined_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    margin_gain = compute_margin_gain(frozen_logits, combined_logits, labels)
    return torch.relu(-margin_gain).mean()
