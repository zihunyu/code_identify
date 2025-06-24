import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CaptchaModel
from dataset import CaptchaDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CaptchaModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_dataset = CaptchaDataset('data/train')
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

for epoch in range(100): # 假如得到的结果错误率高，可以增加训练轮次
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)  # List of 4 tensors
        loss = sum([criterion(out, labels[:, i]) for i, out in enumerate(outputs)])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), 'captcha_model.pth')
