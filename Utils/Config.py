class Config:
    device = "mps"

    model_name = "resnet50"
    pretrained = True
    num_classes = 6

    optimizer_name = "AdamW"
    lr = 2e-4 # 学习率
    weight_decay = 1e-4 # 权重衰减率

    augmentation = "randaugment"
    batch_size = 128
    epochs = 60
    num_workers = 8

    save_dir = "./outputs/demo"
    log_dir = "./logs/demo"

    test_model = "resnet50"
