"""Evaluation metrics"""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report)
from collections import Counter, defaultdict


def compute_metrics(predictions, labels, probabilities, class_names):
    """Compute classification metrics."""
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

    # AUC
    try:
        if len(class_names) == 2:
            auc = roc_auc_score(labels, probabilities[:, 1])
        else:
            auc = roc_auc_score(labels, probabilities, multi_class='ovr',
                                average='macro')
    except:
        auc = None

    report = classification_report(labels, predictions,
                                   target_names=class_names,
                                   output_dict=True, zero_division=0)

    return {
        'accuracy': acc,
        'f1_weighted': f1,
        'auc': auc,
        'report': report
    }


def aggregate_slide_predictions(predictions, labels, wsi_ids, probabilities):
    """Aggregate chunk predictions to slide-level via majority voting."""
    slide_preds = defaultdict(list)
    slide_probs = defaultdict(list)
    slide_labels = {}

    for wsi, pred, prob, label in zip(wsi_ids, predictions, probabilities, labels):
        slide_preds[wsi].append(pred)
        slide_probs[wsi].append(prob)
        slide_labels[wsi] = label

    # Majority vote
    final_preds = []
    final_labels = []
    final_probs = []

    for wsi in slide_preds:
        vote = Counter(slide_preds[wsi]).most_common(1)[0][0]
        final_preds.append(vote)
        final_labels.append(slide_labels[wsi])
        final_probs.append(np.mean(slide_probs[wsi], axis=0))

    return np.array(final_preds), np.array(final_labels), np.array(final_probs)


def aggregate_slide_predictions_weighted(predictions, labels, wsi_ids, probabilities, attention_scores):
    """
    Aggregate chunk predictions to slide-level using attention-weighted aggregation.
    """
    slide_probs = defaultdict(list)
    slide_attn = defaultdict(list)
    slide_labels = {}

    for wsi, prob, attn, label in zip(wsi_ids, probabilities, attention_scores, labels):
        slide_probs[wsi].append(prob)
        slide_attn[wsi].append(attn)
        slide_labels[wsi] = label

    final_preds = []
    final_labels = []
    final_probs = []

    for wsi in slide_probs:
        probs = np.stack(slide_probs[wsi])  # (num_chunks, num_classes)
        attn = np.array(slide_attn[wsi])    # (num_chunks,)

        # Normalize attention scores to sum to 1
        attn_sum = attn.sum()
        if attn_sum == 0:
            attn = np.ones_like(attn) / len(attn)
        else:
            attn = attn / attn_sum

        weighted_probs = np.sum(probs * attn[:, np.newaxis], axis=0)
        pred = np.argmax(weighted_probs)
        final_preds.append(pred)
        final_labels.append(slide_labels[wsi])
        final_probs.append(weighted_probs)

    return np.array(final_preds), np.array(final_labels), np.array(final_probs)
