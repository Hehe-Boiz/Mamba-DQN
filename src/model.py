import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Hàm trợ giúp để tự động tính kích thước vector sau khi đi qua các lớp tích chập
        dummy_input = torch.zeros(1, *input_shape)
        conv_output_size = self._get_conv_output_size(dummy_input)

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(in_features=conv_output_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_actions)
        )

    def _get_conv_output_size(self, x):
        """Tính toán kích thước output của các lớp convolutional."""
        x = self.convolutional_layers(x)
        return x.flatten().shape[0]

    def forward(self, x):
        """Forward pass qua mạng."""
        # Chuẩn hóa giá trị pixel về khoảng [0, 1]
        x = x / 255.0
        conv_out = self.convolutional_layers(x)
        # Flatten tensor để đưa vào lớp fully connected
        flattened = conv_out.view(x.size(0), -1)
        return self.fully_connected_layers(flattened)