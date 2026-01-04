import torch
import torch.nn as nn
import torch.nn.functional as F


# Your original meanPool for context (no changes needed here)
class meanPool(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super(meanPool, self).__init__()
        self.instance_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, bag):
        # bag shape: (batch_size, num_instances, input_dim)
        x = self.instance_embed(bag)  # (batch_size, num_instances, hidden_dim)
        bag_representation = x.mean(dim=1)  # (batch_size, hidden_dim)
        output = self.classifier(bag_representation)
        return output, bag_representation


# Your original maxPool for context (no changes needed here)
class maxPool(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super(maxPool, self).__init__()
        self.instance_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, bag):
        # bag shape: (batch_size, num_instances, input_dim)
        x = self.instance_embed(bag)  # (batch_size, num_instances, hidden_dim)
        bag_representation, _ = x.max(dim=1)  # (batch_size, hidden_dim)
        output = self.classifier(bag_representation)
        return output, bag_representation


# --- New Top-K Pooling Classifier ---
class topKPool(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, k=1,
                 dropout=0.0):  # k is the number of top instances to select
        super(topKPool, self).__init__()
        self.k = k  # Number of top instances to select

        self.instance_embed = nn.Linear(input_dim, hidden_dim)

        # New: A layer to learn instance importance scores
        # This can be a simple linear layer mapping instance embedding to a scalar score
        self.instance_score_layer = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)  # Using dropout for regularization

    def forward(self, bag):
        # bag shape: (batch_size, num_instances, input_dim)

        # 1. Embed instances
        instance_embeddings = self.instance_embed(bag)  # (batch_size, num_instances, hidden_dim)
        instance_embeddings = torch.relu(instance_embeddings)
        instance_embeddings = self.dropout(instance_embeddings)  # Apply dropout after ReLU

        # 2. Learn instance importance scores
        # scores shape: (batch_size, num_instances, 1)
        instance_scores = self.instance_score_layer(instance_embeddings)
        # Squeeze the last dimension to get (batch_size, num_instances)
        instance_scores = instance_scores.squeeze(-1)

        # 3. Select Top-K instances based on scores
        # Check if k is valid (not more than num_instances)
        num_instances = instance_embeddings.size(1)
        k_val = min(self.k, num_instances)  # Ensure k doesn't exceed available instances

        # topk returns values (scores) and indices
        # We need the indices to select the corresponding embeddings
        # scores_topk shape: (batch_size, k_val)
        # indices_topk shape: (batch_size, k_val)
        _, indices_topk = torch.topk(instance_scores, k_val, dim=1)

        # Reshape indices for gathering
        # indices_topk needs to be (batch_size, k_val, hidden_dim) to gather correctly
        # We expand indices_topk to match hidden_dim for broadcasting
        indices_topk_expanded = indices_topk.unsqueeze(-1).expand(-1, -1, instance_embeddings.size(2))

        # Gather the top-k instance embeddings
        # topk_embeddings shape: (batch_size, k_val, hidden_dim)
        topk_embeddings = torch.gather(instance_embeddings, 1, indices_topk_expanded)

        # 4. Aggregate the Top-K embeddings (e.g., using mean or max)
        # Here, I'll use mean pooling over the selected top-k instances.
        # You could also use max pooling here if desired: topk_embeddings.max(dim=1)[0]
        bag_representation = topk_embeddings.mean(dim=1)  # (batch_size, hidden_dim)

        # 5. Classify the bag
        output = self.classifier(bag_representation)

        return output, bag_representation  # Return output and bag_representation for potential downstream use