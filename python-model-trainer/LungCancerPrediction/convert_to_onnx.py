
import torch
import torch.nn as nn

class LungCancerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d   = nn.Conv2d(1, 64, 3)
        self.conv2d2  = nn.Conv2d(64, 64, 3)
        self.pool     = nn.MaxPool2d(2, 2)
        self.dense1   = nn.Linear(246016, 16)
        self.dense2   = nn.Linear(16, 3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv2d(x)))
        x = self.pool(torch.relu(self.conv2d2(x)))
        x = x.flatten(1)
        x = torch.relu(self.dense1(x))
        x = self.dense2(x)
        return x

from pathlib import Path
repo_root = Path(__file__).resolve().parents[2]

model = LungCancerCNN()
model.load_state_dict(torch.load(
    repo_root / "models" / "lung-cancer-prediction" / "csharp" / "lung-cancer-model.weights",
    weights_only=False
))
model.eval()

dummy = torch.zeros(1, 1, 256, 256)
model_dir = repo_root / "models" / "lung-cancer-prediction" / "csharp"

torch.onnx.export(
    model, dummy,
    str(model_dir / "lung-cancer-model.onnx"),
    input_names=["input"],
    output_names=["output"],
    opset_version=17
)
print("ONNX model saved")