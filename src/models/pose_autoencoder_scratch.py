import numpy as np
import torch
import torch.nn as nn


class DownPose1dBlock(nn.Module):
    def __init__(self, input_dim,output_dim ,kernel_size=3):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.conv1d(x)

class UpPose1dBlock(nn.Module):
    def __init__(self, input_dim,output_dim ,kernel_size=3):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.ConvTranspose1d(input_dim, output_dim, kernel_size=kernel_size, padding=1),
            nn.ReLU(),)
            # nn.Upsample(scale_factor=2))

    def forward(self, x):
        return self.conv1d(x)


class PoseConv1D(nn.Module):
    def __init__(self, input_dim, hidden_dim, cnn_out_dim):
        super().__init__()
        # Define your 1D convolutional layers
        num_layers = int(np.log2(input_dim))-1
        layers = []
        first_in_dim = 1
        first_hidden = hidden_dim
        in_dim = first_hidden
        layers.append(DownPose1dBlock(input_dim=first_in_dim, output_dim=first_hidden))

        for layer in range(num_layers):
            layers.append(DownPose1dBlock(input_dim=in_dim, output_dim=in_dim*2))
            in_dim *= 2
        self.layers = nn.Sequential(*layers)

        self.shrink_net = nn.Conv1d(in_dim, cnn_out_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # Reshape the data to [batch_size, num_joints * num_dims, num_frames]
        x = x.permute(0, 2, 1, 3).reshape(x.size(0), 1, x.size(2) * x.size(1) * x.size(3))
        # Pass the data through the convolutional layers
        x = self.layers(x)
        x = self.shrink_net(x)
        return x


class PoseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, lstm_out_dim=8):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers)

        self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=lstm_out_dim, num_layers=num_layers)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x

class PoseDecoder(nn.Module):
    def __init__(self, pose_shape, inputs_dim, num_layers):
        super().__init__()
        self.pose_shape = pose_shape
        layers = []

        for layer in range(num_layers):
            layers.append(UpPose1dBlock(input_dim=inputs_dim, output_dim=inputs_dim*2))
            inputs_dim = inputs_dim*2
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        x = self.layers(x)
        # x = x.permute(0, 1, 2, 3).reshape(x.shape[0], *self.pose_shape)
        return x
class PoseAutoencoder(nn.Module):
    def __init__(self, input_dim=32,
                 hidden_dim=64,cnn_out_dim=32,
                 n_hidden_lstm=64, n_lstm=3,
                 lstm_out_dim=8, n_up_layers=3):
        super().__init__()
        self.pose_shape = (8,2,2)
        self.pose_encoder = PoseConv1D(input_dim, hidden_dim, cnn_out_dim)
        self.lstm = PoseLSTM(cnn_out_dim, n_hidden_lstm, n_lstm, lstm_out_dim)
        self.decoder = PoseDecoder(pose_shape=None, inputs_dim=lstm_out_dim, num_layers=n_up_layers)

    def forward(self, x):
        x = self.pose_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = self.decoder(x)
        x = x.reshape(x.shape[0], *self.pose_shape)
        return x

def get_model():
    # Example dimensions

    num_joints = 8
    num_dims = 2
    num_subjects = 2

    input_dim = num_joints * num_dims * num_subjects
    hidden_dim = 64

    # Generate random synthetic pose data
    # num_frames = 100
    # pose_data = torch.rand((num_frames, num_joints, num_dims, num_subjects))

    encoder = PoseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim,
                              cnn_out_dim=8, n_hidden_lstm=64, n_lstm=3, lstm_out_dim=8, n_up_layers=2)
    return encoder

if __name__ == "__main__":
    # Example dimensions
    num_frames = 100
    num_joints = 8
    num_dims = 2
    num_subjects = 2

    input_dim = num_joints*num_dims*num_subjects
    hidden_dim = 64

    # Generate random synthetic pose data
    pose_data = torch.rand((num_frames, num_joints, num_dims, num_subjects))

    encoder = PoseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim,
                              cnn_out_dim=8, n_hidden_lstm=64, n_lstm=3, lstm_out_dim=8, n_up_layers=2)
    encoded_pose = encoder(pose_data)
    print(encoded_pose.shape)