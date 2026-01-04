"""Collate functions for MIL batching"""
import torch


def collate_mil(batch):
    """
    Collate variable-length bags with padding.
    Returns: (features, labels)
    """
    bags, coords, labels = zip(*batch)

    # Pad to max length in batch
    max_len = max(b.size(0) for b in bags)
    dim = bags[0].size(1)
    batch_size = len(bags)

    padded = torch.zeros(batch_size, max_len, dim)
    for i, bag in enumerate(bags):
        padded[i, :bag.size(0)] = bag

    labels = torch.tensor(labels, dtype=torch.long)
    return padded, labels


def collate_mil_subsample(batch, max_instances=1000):
    """Collate with random subsampling for memory efficiency."""
    bags, coords, labels = zip(*batch)

    sampled = []
    for bag in bags:
        if bag.size(0) > max_instances:
            idx = torch.randperm(bag.size(0))[:max_instances]
            sampled.append(bag[idx])
        else:
            sampled.append(bag)

    return collate_mil([(s, None, l) for s, l in zip(sampled, labels)])
