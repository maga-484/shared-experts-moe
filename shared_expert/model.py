import torch
import torch.nn as nn

class SharedExpert(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return self.net(x)


class SpecializedHead(nn.Module):
    def __init__(self, in_features=128, out_features=32):
        super().__init__()
        self.head = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.head(x)


class SharedExpertMoE(nn.Module):
    def __init__(self, num_specialized=10, input_dim=32, hidden_dim=128, output_dim=32):
        super().__init__()
        self.shared = SharedExpert(input_dim, hidden_dim, output_dim)
        self.heads = nn.ModuleList([SpecializedHead(hidden_dim, output_dim) for _ in range(num_specialized)])
        self.num_specialized = num_specialized

    def forward(self, x, silo_id):
        features = self.shared(x)
        return self.heads[silo_id](features)

    def freeze_shared(self):
        for param in self.shared.parameters():
            param.requires_grad = False

    def unfreeze_shared(self):
        for param in self.shared.parameters():
            param.requires_grad = True
