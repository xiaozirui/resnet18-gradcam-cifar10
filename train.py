import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.model import ResNet18
from src.dataset import get_dataloaders
from src.utils import calculate_accuracy, plot_metrics

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # 切换为训练模式（开启 Batch Normalization 的均值方差追踪）
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        
        _, preds = torch.max(outputs, dim=1)
        running_corrects += (preds == labels).sum().item()
        
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for images,labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            _, preds = torch.max(outputs, dim=1)
            running_corrects += (preds == labels).sum().item()
            
            total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


def main():
    BATCH_SIZE = 128
    EPOCHS = 20  
    LEARNING_RATE = 0.1
    CHECKPOINT_DIR = './checkpoints'
    OUTPUT_DIR = './outputs'

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device('mps')   # Apple Silicon GPU (Mac)
    else:
        device = torch.device('cpu')   # CPU
    print(device)
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE, num_workers=2)

    model = ResNet18(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    print("🔥 开始正式训练！")
    start_total_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, val_acc = validate(model, val_loader, criterion, device)


        scheduler.step()


        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        epoch_time = time.time() - epoch_start_time

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({epoch_time:.1f}s) | "
              f"Lr: {current_lr:.4f} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"  🏆 创下新高！最高验证准确率更新为: {best_val_acc*100:.2f}%，权重已保存至 {best_model_path}")

    total_time = time.time() - start_total_time
    print(f"\n🎉 训练全部结束！总耗时: {total_time / 60:.2f} 分钟，最高验证集准确率: {best_val_acc*100:.2f}%")


    plot_metrics(train_losses, val_losses, train_accs, val_accs, save_path=os.path.join(OUTPUT_DIR, 'loss_acc.png'))

if __name__ == "__main__":
    main()