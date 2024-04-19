import os
import json
from dataclasses import dataclass
from dotenv import load_dotenv
import time
import math

import wandb
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

config_path = "./config.json"
dataset_path = "./data"

torch.manual_seed(42)

@dataclass
class Config:
    num_classes: int
    lr: float
    batch_size: int
    num_epochs: int
    optimizer: str
    loss_function: str
    weight_decay: float
    dataset: str
    train_split: float
    shuffle: bool
    num_train_workers: int
    num_test_workers: int
    pin_memory: bool
    normalization: dict
    augmentation: dict
    checkpoint_frequency: int
    save_path: str

def check_tf32():
    # Check if TF32 is allowed for CUDA matrix multiplication
    tf32_enabled_for_matmul = torch.backends.cuda.matmul.allow_tf32
    print(f"TF32 enabled for matrix multiplication: {tf32_enabled_for_matmul}")
    # Check if TF32 is allowed for cuDNN operations
    tf32_enabled_for_cudnn = torch.backends.cudnn.allow_tf32
    print(f"TF32 enabled for cuDNN: {tf32_enabled_for_cudnn}")

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    config = Config(**config)
    return config

def load_data(config: Config):
    try:
        dataset_type = getattr(datasets, config.dataset)
    except AttributeError:
        raise ValueError(f"Invalid dataset name: {config.dataset}")

    transform = [transforms.ToTensor()]
    if config.normalization:
        transform.append(transforms.Normalize(config.normalization["mean"], config.normalization["std"]))
    if config.augmentation:
        if "random_crop" in config.augmentation:
            transform.append(transforms.RandomCrop(config.augmentation["random_crop"]))
        if "random_horizontal_flip" in config.augmentation:
            transform.append(transforms.RandomHorizontalFlip(config.augmentation["random_horizontal_flip"]))
        if "random_rotation" in config.augmentation:
            transform.append(transforms.RandomRotation(config.augmentation["random_rotation"]))
    
    dataset = dataset_type(root=(os.path.join(dataset_path, config.dataset)), transform=transforms.Compose(transform), download=True)
    train_size = int(len(dataset) * config.train_split)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, shuffle=config.shuffle, batch_size=config.batch_size, num_workers=config.num_train_workers, pin_memory=config.pin_memory)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size, num_workers=config.num_test_workers, pin_memory=config.pin_memory)
    return train_dataloader, test_dataloader

def wandb_table(model, test_loader, device):
    table = wandb.Table(columns=["image", "prediction_label", "true_label"])
    model.eval()
    with torch.inference_mode():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            preds = model(data)
            preds = torch.argmax(preds, dim=-1)
            for img, pred, true in zip(data, preds, target):
                img = wandb.Image(img.cpu())
                table.add_data(img, pred.item(), true.item())
    wandb.log({"prediction_table": table})

def train_full_precision(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: Config,
    device: str,
    allow_tf32: bool,
):
    torch.backends.cudnn.allow_tf32 = allow_tf32 # Disable tf32 and do float32
    run_name = "tf32" if allow_tf32 else "float32"
    run = wandb.init(project="precision-testing", name=run_name, config=config.__dict__)

    # Watch the model to log gradients and model parameters
    wandb.watch(model, criterion, log="all", log_freq=20)

    for epoch in range(config.num_epochs):
        start_time = time.time()

        model.train()
        train_loss = 0
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss

            # Log batch level metric
            wandb.log({"batch_train_loss": loss.item()})
        train_loss /= len(train_loader)
        epoch_duration = time.time()-start_time
        
        model.eval()
        correct, total = 0, 0
        with torch.inference_mode():
            test_loss = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data)
                loss = criterion(pred, target)
                test_loss += loss
                pred = torch.argmax(pred, -1)
                correct += (pred == target).sum().item()
                total += target.size(0)
            test_loss /= len(test_loader)

        wandb.log({
            "epoch": epoch + 1,
            "accuracy": correct / total,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "epoch_duration_seconds": epoch_duration
        })

        if (epoch + 1) % config.checkpoint_frequency == 0:
            save_path = os.path.join(config.save_path, f"model_epoch_{epoch+1}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            wandb.save(save_path)

        print(f"Epoch: {epoch+1} | Training Loss: {train_loss}, Test Loss: {test_loss}, Accuracy: {correct/total}")

    wandb_table(model, test_loader, device)
    run.finish()

def train_half_precision(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: Config,
    device: str,
    allow_bfloat: bool,
):
    run_name = "bfloat16" if allow_bfloat else "float16"
    run = wandb.init(project="precision-testing", name=run_name, config=config.__dict__)

    # Watch the model to log gradients and model parameters
    wandb.watch(model, criterion, log="all", log_freq=20)
    for epoch in range(config.num_epochs):
        start_time = time.time()
        model = model.to(dtype=torch.bfloat16) if allow_bfloat else model.half()

        model.train()
        train_loss = 0
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.to(dtype=torch.bfloat16) if allow_bfloat else data.half()
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss

            # Log batch level metric
            wandb.log({"batch_train_loss": loss.item()})
        train_loss /= len(train_loader)
        epoch_duration = time.time()-start_time
        
        model = model.to(dtype=torch.float32)
        model.eval()
        correct, total = 0, 0
        with torch.inference_mode():
            test_loss = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data)
                loss = criterion(pred, target)
                test_loss += loss
                pred = torch.argmax(pred, -1)
                correct += (pred == target).sum().item()
                total += target.size(0)
            test_loss /= len(test_loader)

        wandb.log({
            "epoch": epoch + 1,
            "accuracy": correct / total,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "epoch_duration_seconds": epoch_duration
        })

        if (epoch + 1) % config.checkpoint_frequency == 0:
            save_path = os.path.join(config.save_path, f"model_epoch_{epoch+1}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            wandb.save(save_path)

        print(f"Epoch: {epoch+1} | Training Loss: {train_loss}, Test Loss: {test_loss}, Accuracy: {correct/total}")

    run.finish()

def train_amp(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: Config,
    device: str,
):
    run_name = "amp"
    run = wandb.init(project="precision-testing", name=run_name, config=config.__dict__)

    # Watch the model to log gradients and model parameters
    wandb.watch(model, criterion, log="all", log_freq=20)
    scaler = GradScaler()

    for epoch in range(config.num_epochs):
        start_time = time.time()

        model.train()
        train_loss = 0
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            with autocast():
                pred = model(data)
                loss = criterion(pred, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss

            # Log batch level metric
            wandb.log({"batch_train_loss": loss.item()})
        train_loss /= len(train_loader)
        epoch_duration = time.time()-start_time
        
        model.eval()
        correct, total = 0, 0
        with torch.inference_mode():
            test_loss = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data)
                loss = criterion(pred, target)
                test_loss += loss
                pred = torch.argmax(pred, -1)
                correct += (pred == target).sum().item()
                total += target.size(0)
            test_loss /= len(test_loader)

        wandb.log({
            "epoch": epoch + 1,
            "accuracy": correct / total,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "epoch_duration_seconds": epoch_duration
        })

        if (epoch + 1) % config.checkpoint_frequency == 0:
            save_path = os.path.join(config.save_path, f"model_epoch_{epoch+1}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            wandb.save(save_path)

        print(f"Epoch: {epoch+1} | Training Loss: {train_loss}, Test Loss: {test_loss}, Accuracy: {correct/total}")

    run.finish()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18(weights=False)
    config = load_config(config_path)
    train_loader, test_loader = load_data(config)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=config.num_classes)
    try:
        optimizer = getattr(torch.optim, config.optimizer)(
            params=model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay
        )
    except AttributeError:
        raise ValueError(f"Invalid optimizer name: {config.optimizer}")

    try:
        criterion = getattr(nn, config.loss_function)()
    except AttributeError:
        raise ValueError(f"Invalid loss function name: {config.loss_function}")
    
    total_steps = len(train_loader) * config.num_epochs
    # Lambda Function for the learning rate
    def warmup_scheduler(step, total_steps, warmup_prop=0.15):
        if (step / total_steps) < warmup_prop:
            return step / (total_steps * warmup_prop)
        else:
            progress = (step - total_steps * warmup_prop) / (total_steps * (1 - warmup_prop))
            return 0.5 * (1 + math.cos(math.pi * progress))
        
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: warmup_scheduler(step, total_steps))

    model.to(device)
    # train_full_precision(model, train_loader, test_loader, optimizer, criterion, scheduler, config, device, allow_tf32=False) #float32 not tf32
    train_amp(model, train_loader, test_loader, optimizer, criterion, scheduler, config, device)

if __name__ == "__main__":
    main()