import torch.nn as nn

class CaptchaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 100x40 → 50x20
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 50x20 → 25x10
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 25 * 10, 512),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.out = nn.ModuleList([nn.Linear(512, 10) for _ in range(4)])  # 每位数字输出一个10类softmax

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return [layer(x) for layer in self.out]
