import os
import csv
import torch
import yaml
from types import SimpleNamespace
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from Utils.models import build_model, build_optimizer, build_transform, train_one_epoch


def main():
    # 加载 YAML 配置

    with open("config.yaml", "r") as f:
        cfg = SimpleNamespace(**yaml.safe_load(f))
    device = torch.device(cfg.device)
    model = build_model(cfg).to(device)
    transform = build_transform(cfg)
    os.makedirs(cfg.log_dir, exist_ok=True)
    csv_file = os.path.join(cfg.log_dir, "train_log.csv")

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "acc"])
    dataset = datasets.ImageFolder("/root/Project1/data/train/", transform=transform)
    loader = DataLoader(dataset, batch_size=cfg.batch_size,
                        shuffle=True, num_workers=cfg.num_workers)
    os.makedirs(cfg.save_dir, exist_ok=True)
    losses, accs = [], []

    # for epoch in range(1, cfg.epochs + 1):
    #     loss, acc = train_one_epoch(model, loader, optimizer, device, epoch)  # 假设 train_one_epoch 内部也使用 tqdm
    #
    #     losses.append(loss)
    #     accs.append(acc)
    #     with open(csv_file, "a", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow([
    #             epoch,
    #             round(loss, 6),
    #             round(acc, 6)
    #         ])
    #     # 使用 tqdm.write 代替 print，避免与进度条冲突
    #     tqdm.write(f"[Epoch {epoch}] loss={loss:.4f}, acc={acc:.4f}")
    #     if epoch % 10 == 0:
    #         path = f"{cfg.save_dir}/model_epoch_{int(epoch/10)}.pth"
    #         torch.save(model.state_dict(), path)
    #         tqdm.write(f"saved: {path}")

    ### 第一阶段
    for param in model.parameters():
        param.requires_grad = False
    for param in model.heads.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(
        model.heads.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    for epoch in range(1, 5+1):
        loss, acc = train_one_epoch(model, loader, optimizer, device, epoch)  # 假设 train_one_epoch 内部也使用 tqdm

        losses.append(loss)
        accs.append(acc)
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(loss, 6),
                round(acc, 6)
            ])
        # 使用 tqdm.write 代替 print，避免与进度条冲突
        tqdm.write(f"[Epoch {epoch}] loss={loss:.4f}, acc={acc:.4f}")
        if epoch % 10 == 0:
            path = f"{cfg.save_dir}/model_epoch_{int(epoch/10)}.pth"
            torch.save(model.state_dict(), path)
            tqdm.write(f"saved: {path}")



    ### 第二阶段

    for layer in model.encoder.layers[-4:]:
        for param in layer.parameters():
            param.requires_grad = True
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.heads.parameters(),
                "lr": 1e-4
            },
            {
                "params":
                    model.encoder.layers[-4:].parameters(),
                "lr": 1e-5
            }
        ],
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=15
    )
    for epoch in range(11, 21):
        loss, acc = train_one_epoch(model, loader, optimizer, device, epoch)  # 假设 train_one_epoch 内部也使用 tqdm
        scheduler.step()
        losses.append(loss)
        accs.append(acc)
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(loss, 6),
                round(acc, 6)
            ])
        # 使用 tqdm.write 代替 print，避免与进度条冲突
        tqdm.write(f"[Epoch {epoch}] loss={loss:.4f}, acc={acc:.4f}")
        if epoch % 10 == 0:
            path = f"{cfg.save_dir}/model_epoch_{int(epoch/10)}.pth"
            torch.save(model.state_dict(), path)
            tqdm.write(f"saved: {path}")


    ### 第三阶段
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.heads.parameters(),
                "lr": 1e-4
            },
            {
                "params": model.encoder.parameters(),
                "lr": 5e-6
            }
        ],
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=20
    )

    for epoch in range(21, 50+1):
        loss, acc = train_one_epoch(model, loader, optimizer, device, epoch)  # 假设 train_one_epoch 内部也使用 tqdm
        scheduler.step()
        losses.append(loss)
        accs.append(acc)
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(loss, 6),
                round(acc, 6)
            ])
        # 使用 tqdm.write 代替 print，避免与进度条冲突
        tqdm.write(f"[Epoch {epoch}] loss={loss:.4f}, acc={acc:.4f}")
        if epoch % 10 == 0:
            path = f"{cfg.save_dir}/model_epoch_{int(epoch/10)}.pth"
            torch.save(model.state_dict(), path)
            tqdm.write(f"saved: {path}")

if __name__ == "__main__":
    main()
