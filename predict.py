import torch
from model import CaptchaModel
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CaptchaModel().to(device)
model.load_state_dict(torch.load('captcha_model.pth', map_location=device))
model.eval()

def predict(img_path):
    img = Image.open(img_path).convert('L').resize((100, 40))
    img = np.array(img, dtype=np.float32) / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
    with torch.no_grad():
        outputs = model(img)
        preds = [str(torch.argmax(o).item()) for o in outputs]
    return ''.join(preds)

# 测试
print(predict('data/test/1063.png'))
