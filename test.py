from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, vit_b_16
from tqdm import tqdm


CHECKPOINTS = {
    "vit1_b_16": "outputs/vit1_b_16_runs/vit1_epoch_3.pth",
    "vit2_b_16": "outputs/vit2_b_16_runs/model_epoch_5.pth",
    "vit3_b_16": "outputs/vit3_b_16_runs/model_epoch_5.pth",
    "resnet50": "outputs/resnet50_runs/model_epoch_6.0.pth",
}


def choose_device(device_name):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_test_model(model_name, num_classes):
    if model_name in {"vit1_b_16", "vit2_b_16", "vit3_b_16"}:
        model = vit_b_16(weights=None)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == "resnet50":
        model = resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown test_model: {model_name}")
    return model


def build_test_loader(batch_size=32, num_workers=0):
    input_size = (224, 224)
    transform = transforms.Compose([
        transforms.Resize(int(input_size[0] / 0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    testset = datasets.ImageFolder("./data/test/", transform=transform)
    return DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if checkpoint_path.stat().st_size == 0:
        raise RuntimeError(
            f"Checkpoint is empty: {checkpoint_path}. "
            "Please place a valid ResNet50 .pth file here or set test_checkpoint in config.yaml."
        )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)

    return correct / total, total


def main():
    with open("config.yaml", "r") as f:
        cfg = SimpleNamespace(**yaml.safe_load(f))

    model_name = cfg.test_model
    device = choose_device(getattr(cfg, "device", "cpu"))
    checkpoint_path = getattr(cfg, "test_checkpoint", CHECKPOINTS[model_name])

    model = build_test_model(model_name, cfg.num_classes)
    load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)

    testloader = build_test_loader(batch_size=32, num_workers=0)
    acc, total = evaluate(model, testloader, device)

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Testing ({total} samples), accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
