import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from torchvision.transforms import autoaugment
from torchvision.models import (
    resnet18, resnet50,vit_b_16,
    ResNet18_Weights, ResNet50_Weights, ViT_B_16_Weights
)
from tqdm import tqdm
from Utils.Config import Config

# 选择模型
def build_model(cfg):
    if cfg.model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if cfg.pretrained else None)
        model.fc = nn.Linear(512, cfg.num_classes)
    elif cfg.model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT if cfg.pretrained else None)
        model.fc = nn.Linear(2048, cfg.num_classes)
    elif cfg.model_name == "vit_b_16":
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if cfg.pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features,cfg.num_classes)
    else:
        raise ValueError("Unknown model")
    return model
# 选择优化器
def build_optimizer(model, cfg):
    if cfg.optimizer_name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    elif cfg.optimizer_name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    elif cfg.optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    else:
        raise ValueError("Unknown optimizer")
# 选择数据增强
def build_transform(cfg, input_size=(224, 224), is_train=True):
    """
    Args:
        cfg: 配置对象，需要包含 cfg.augmentation 字符串
        input_size: 目标图像尺寸 (H, W)
        is_train: 是否为训练集（验证/测试集通常只做 resize + center crop + normalize）
    """
    # 训练集的增强流水线（以下分支都是针对训练）
    if cfg.augmentation == "none":
        transform = T.Compose([
            T.Resize(input_size),
            T.ToTensor()
        ])

    elif cfg.augmentation == "flip":
        transform = T.Compose([
            T.Resize(input_size),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])

    elif cfg.augmentation == "crop":
        transform = T.Compose([
            T.RandomResizedCrop(input_size),
            T.ToTensor()
        ])

    elif cfg.augmentation == "all":
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(input_size),
            T.ToTensor()
        ])

    elif cfg.augmentation == "randaugment":
        # 🔥 推荐：RandAugment 是 ViT 官方训练使用的强增强方法
        # num_ops=2, magnitude=9 是常见的起点
        transform = T.Compose([
            T.RandomResizedCrop(input_size),
            T.RandomHorizontalFlip(),
            autoaugment.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor()
        ])
    elif cfg.augmentation == "all":
        # 默认使用 all
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(input_size),
            T.ToTensor()
        ])
    else:
        raise ValueError("Unknown augmentation")

    # ⚠️ 重要：无论哪种增强，最后必须加上归一化（针对 ImageNet 预训练模型）
    transform = T.Compose([
        transform,
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return transform



# 训练一个 epoch
def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc=f"Epoch {epoch}",leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


# =========================
# Main
# =========================
# def main():
#     device = torch.device(cfg.device)
#
#     model = build_model(cfg).to(device)
#     optimizer = build_optimizer(model, cfg)
#
#     transform = build_transform(cfg)
#
#     dataset = datasets.ImageFolder("./data/train/", transform=transform)
#     loader = DataLoader(dataset, batch_size=cfg.batch_size,
#                         shuffle=True, num_workers=cfg.num_workers)
#
#     os.makedirs(cfg.save_dir, exist_ok=True)
#
#     losses, accs = [], []
#
#     for epoch in range(1, cfg.epochs + 1):
#         loss, acc = train_one_epoch(model, loader, optimizer, device, epoch)
#
#         losses.append(loss)
#         accs.append(acc)
#         print(f"[Epoch {epoch}] loss={loss:.4f}, acc={acc:.4f}")
#         print("\n")
#
#         if epoch % 10 == 0:
#             path = f"{cfg.save_dir}/model_epoch_{epoch}.pth"
#             torch.save(model.state_dict(), path)
#             print("saved:", path)
#
#
# if __name__ == "__main__":
#     main()