import torch.nn as nn

class LungCancerNN(nn.Module):
    def __init__(self):
        super(LungCancerNN, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=246016, out_features=16),
            nn.Linear(in_features=16, out_features=3)
        )

    def forward(self, x):
        return self.model(x)