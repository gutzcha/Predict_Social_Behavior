import torch
import torch.nn as nn


class JointModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[], batch_norm=True, weight_decay=0.0, dropout_prob=0.0, joint_name=None):
        super(JointModel, self).__init__()
        self.joint_name = joint_name
        layers = []
        prev_layer_size = input_dim

        # Add Batch Normalization if specified
        if batch_norm:
            layers.append(nn.BatchNorm1d(input_dim))

        for layer_size in hidden_layers:
            layers.append(nn.Linear(prev_layer_size, layer_size))
            layers.append(nn.ReLU())

            # Add Batch Normalization if specified
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_size))

            # Add Dropout if specified
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))

            prev_layer_size = layer_size
        layers.append(nn.Linear(prev_layer_size, output_dim))

        self.layers = nn.Sequential(*layers)

        # Apply He initialization to the layers
        self.apply(self.init_weights)

        # Add L2 regularization (weight decay) to all linear layers if weight_decay is specified
        if weight_decay > 0:
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    layer.weight_decay = weight_decay

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        return self.layers(x)
