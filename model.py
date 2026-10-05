import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, h1=10, num_layers=3, in_features=2, out_features=10, freeze_layers=[]):
        super().__init__()
        self.num_layers = num_layers
        self.in_features = in_features
        self.out_features = out_features
        self.layers = nn.ModuleList()  # List to hold hidden layers

        # First hidden layer (maps input features to h1 neurons)
        self.layers.append(nn.Linear(in_features, h1))

        # Additional hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(h1, h1))

        # Output layer (maps h1 neurons to out_features)
        self.out = nn.Linear(h1, out_features)

        # Freeze selected layers for transfer learning
        if freeze_layers:
            for layer_idx in freeze_layers:
                if 0 <= layer_idx < len(self.layers):
                    for param in self.layers[layer_idx].parameters():
                        param.requires_grad = False
                    print(f"Layer {layer_idx} frozen.")

    def forward(self, x):
        for i in range(self.num_layers):
            x = F.elu(self.layers[i](x))  # Apply ELU activation after each hidden layer
        x = self.out(x)  # Output layer (linear activation for regression)
        return x

