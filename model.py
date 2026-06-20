import sys
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vit_b_16

device = torch.device("mps" if torch.mps.is_available() else "cpu")

dataset = ImageFolder("./data/train")
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

model = vit_b_16(weights=None)
model.heads.head = nn.Linear(model.heads.head.in_features, 6)

state_dict = torch.load(
    "best_model.pth",
    map_location=device,
)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print(f"[Model loaded] device={device}, classes={list(idx_to_class.values())}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0][pred_idx].item()

    print(f"\n{'='*50}")
    print(f"  Prediction: {idx_to_class[pred_idx]}")
    print(f"{'='*50}")
    for i in range(6):
        bar = "█" * int(probs[0, i].item() * 40)
        print(f"  {idx_to_class[i]:<12s} {probs[0, i].item():.4f}  {bar}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        # Interactive mode: keep asking forever
        print("\nEnter image path (or 'q' to quit):")
        while True:
            try:
                path = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if path.lower() in ("q", "quit", "exit"):
                print("Bye!")
                break
            if not path:
                continue

            try:
                predict(path)
            except FileNotFoundError:
                print(f"  [Error] File not found: {path}")
            except Exception as e:
                print(f"  [Error] {e}")
