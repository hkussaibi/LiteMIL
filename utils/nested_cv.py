"""Nested Cross-Validation Implementation"""
import os
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset
from .training import train_epoch, evaluate, get_optimizer_scheduler, get_class_weights
from .metrics import compute_metrics, aggregate_slide_predictions_weighted
from .collate import collate_mil
import json
import warnings


class NestedCrossValidation:
    """Nested CV for unbiased model evaluation."""

    def __init__(self, model_factory, dataset, class_names, output_dir,
                 n_outer=5, n_inner=4, batch_size=128, epochs=100,
                 patience=10, device='cuda', use_amp=True):
        self.model_factory = model_factory
        self.dataset = dataset
        self.class_names = class_names
        self.output_dir = output_dir
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.use_amp = use_amp

        os.makedirs(output_dir, exist_ok=True)

    def run(self):
        """Execute nested cross-validation."""
        # Get labels and groups
        labels = np.array([self.dataset[i][2] for i in range(len(self.dataset))])
        case_ids = [self.dataset.get_case_id(i) for i in range(len(self.dataset))]

        # Outer CV splitter (patient-level)
        outer_cv = StratifiedGroupKFold(n_splits=self.n_outer, shuffle=True,
                                        random_state=42)

        all_results = []
        best_overall_acc = 0
        best_overall_model = None

        for fold, (train_idx, test_idx) in enumerate(
                outer_cv.split(np.zeros(len(labels)), labels, case_ids), 1
        ):
            print(f"\n{'=' * 70}\nOUTER FOLD {fold}/{self.n_outer}\n{'=' * 70}")

            # Inner CV for model selection
            best_model, best_val_acc = self._inner_cv(train_idx, labels, case_ids, fold)

            # Evaluate on test set
            fold_results = self._evaluate_fold(best_model, test_idx, fold)
            all_results.append(fold_results)

            # Track best model across all folds
            fold_test_acc = fold_results['slide']['accuracy']
            if fold_test_acc > best_overall_acc:
                best_overall_acc = fold_test_acc
                best_overall_model = best_model.state_dict()
                print(f"  → New best model! Slide accuracy: {fold_test_acc:.3f}")

        # Save best model across all folds
        if best_overall_model is not None:
            torch.save({
                'model_state_dict': best_overall_model,
                'slide_accuracy': best_overall_acc,
                'model_config': {
                    'class_names': self.class_names,
                    'num_classes': len(self.class_names)
                }
            }, f"{self.output_dir}/all_folds_best.pth")
            print(f"\n✓ Best model saved: {self.output_dir}/all_folds_best.pth")
            print(f"  Best slide accuracy: {best_overall_acc:.3f}")

        # Aggregate outputs
        final_results = self._aggregate_results(all_results)
        final_results['best_overall_accuracy'] = best_overall_acc
        self._save_results(final_results)

        return final_results

    def _inner_cv(self, train_idx, labels, case_ids, outer_fold):
        """Inner CV for hyperparameter selection."""
        inner_labels = labels[train_idx]
        inner_cases = [case_ids[i] for i in train_idx]

        inner_cv = StratifiedGroupKFold(n_splits=self.n_inner, shuffle=True,
                                        random_state=42)

        best_val_acc = 0
        best_model_state = None

        for inner_fold, (tr_idx, val_idx) in enumerate(
                inner_cv.split(np.zeros(len(inner_labels)), inner_labels, inner_cases), 1
        ):
            print(f"  Inner fold {inner_fold}/{self.n_inner}")

            # Map to original indices
            train_indices = train_idx[tr_idx]
            val_indices = train_idx[val_idx]

            # Create dataloaders
            train_loader = DataLoader(Subset(self.dataset, train_indices),
                                      batch_size=self.batch_size, shuffle=True,
                                      collate_fn=collate_mil, num_workers=4,
                                      pin_memory=True)
            val_loader = DataLoader(Subset(self.dataset, val_indices),
                                    batch_size=self.batch_size, shuffle=False,
                                    collate_fn=collate_mil, num_workers=4,
                                    pin_memory=True)

            # Train model
            model = self.model_factory().to(self.device)
            optimizer, scheduler = get_optimizer_scheduler(model, epochs=self.epochs)

            # Class weights
            train_labels = labels[train_indices]
            weights = get_class_weights(train_labels, len(self.class_names),
                                        self.device)
            criterion = torch.nn.CrossEntropyLoss(weight=weights)

            # Training loop
            best_acc = 0
            no_improve = 0

            for epoch in range(self.epochs):
                train_metrics = train_epoch(model, train_loader, optimizer,
                                            criterion, self.device, self.use_amp)
                val_metrics = evaluate(model, val_loader, criterion,
                                       self.device, self.use_amp)
                scheduler.step()

                if val_metrics['loss'] < float('inf'):
                    val_acc = accuracy_score(val_metrics['labels'],
                                             val_metrics['predictions'])

                    if val_acc > best_acc:
                        best_acc = val_acc
                        no_improve = 0
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_model_state = model.state_dict()
                    else:
                        no_improve += 1
                        if no_improve >= self.patience:
                            break

        # Load best model
        if best_model_state is not None:
            best_model = self.model_factory()
            best_model.load_state_dict(best_model_state)
            torch.save(best_model_state,
                       f"{self.output_dir}/fold_{outer_fold}_best.pth")
        else:
            best_model = self.model_factory()

        return best_model, best_val_acc

    def _evaluate_fold(self, model, test_idx, fold):
        """Evaluate on outer test fold."""
        model = model.to(self.device).eval()

        test_loader = DataLoader(Subset(self.dataset, test_idx),
                                 batch_size=self.batch_size, shuffle=False,
                                 collate_fn=collate_mil, num_workers=4,
                                 pin_memory=True)

        # Get predictions
        all_preds = []
        all_labels = []
        all_probs = []
        all_wsi_ids = []
        all_attn_scores = []

        with torch.no_grad():
            for i, (features, labels) in enumerate(test_loader):
                features = features.to(self.device)
                outputs, aux = model(features)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

                # Extract attention scores for chunk importance
                attn = aux.get('attn')
                if attn is not None:
                    try:
                        # attn from MultiheadAttention: (batch, num_queries, num_instances)
                        # Strategy: max attention across instances, then mean across queries
                        attn_max_per_query = attn.max(dim=-1)[0]  # (batch, num_queries)
                        attn_scores = attn_max_per_query.mean(dim=-1).cpu().numpy()  # (batch,)

                        # Validate shape
                        if len(attn_scores) != len(labels):
                            warnings.warn(f"Attention score shape mismatch: {len(attn_scores)} vs {len(labels)}")
                            attn_scores = np.ones(len(labels))
                    except Exception as e:
                        warnings.warn(f"Error extracting attention scores: {e}. Using uniform weights.")
                        attn_scores = np.ones(len(labels))
                else:
                    # No attention - uniform weights
                    attn_scores = np.ones(len(labels))

                all_attn_scores.extend(attn_scores)

                # Get WSI IDs for this batch
                batch_start = i * self.batch_size
                batch_wsi_ids = [self.dataset.get_wsi_id(test_idx[j])
                                 for j in range(batch_start,
                                                min(batch_start + len(labels),
                                                    len(test_idx)))]
                all_wsi_ids.extend(batch_wsi_ids)

        # Chunk-level metrics
        chunk_metrics = compute_metrics(np.array(all_preds), np.array(all_labels),
                                        np.array(all_probs), self.class_names)

        # Slide-level metrics with attention-weighted aggregation
        slide_preds, slide_labels, slide_probs = aggregate_slide_predictions_weighted(
            all_preds, all_labels, all_wsi_ids, all_probs, all_attn_scores
        )
        slide_metrics = compute_metrics(slide_preds, slide_labels, slide_probs,
                                        self.class_names)

        return {
            'fold': fold,
            'chunk': chunk_metrics,
            'slide': slide_metrics
        }

    def _aggregate_results(self, fold_results):
        """Aggregate all fold outputs."""
        chunk_accs = [r['chunk']['accuracy'] for r in fold_results]
        slide_accs = [r['slide']['accuracy'] for r in fold_results]

        return {
            'n_folds': self.n_outer,
            'chunk_accuracy_mean': np.mean(chunk_accs),
            'chunk_accuracy_std': np.std(chunk_accs),
            'slide_accuracy_mean': np.mean(slide_accs),
            'slide_accuracy_std': np.std(slide_accs),
            'fold_results': fold_results
        }

    def _save_results(self, results):
        """Save outputs to JSON."""
        with open(f"{self.output_dir}/outputs.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 70}")
        print("FINAL RESULTS")
        print(f"{'=' * 70}")
        print(f"Chunk Acc: {results['chunk_accuracy_mean']:.3f} ± "
              f"{results['chunk_accuracy_std']:.3f}")
        print(f"Slide Acc: {results['slide_accuracy_mean']:.3f} ± "
              f"{results['slide_accuracy_std']:.3f}")