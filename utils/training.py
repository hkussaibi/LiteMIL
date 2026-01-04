"""Training utilities for nested cross-validation"""
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast
import numpy as np


def train_epoch(model, loader, optimizer, criterion, device, use_amp=True,
                clip_grad=1.0):
    """Single training epoch."""
    model.train()
    scaler = GradScaler(enabled=use_amp)

    total_loss = 0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=use_amp):
            outputs, _ = model(features)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        'loss': total_loss / len(loader),
        'accuracy': correct / total
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True):
    """Evaluation epoch."""
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        with autocast('cuda', enabled=use_amp):
            outputs, _ = model(features)
            loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return {
        'loss': total_loss / len(loader),
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }


def get_optimizer_scheduler(model, lr=1e-4, weight_decay=0.01, epochs=100):
    """Create optimizer and scheduler."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - 5), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[5])

    return optimizer, scheduler


def get_class_weights(labels, num_classes, device):
    """Compute balanced class weights."""
    counts = np.bincount(labels, minlength=num_classes)
    weights = np.where(counts > 0, 1.0 / counts, 0.0)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)
