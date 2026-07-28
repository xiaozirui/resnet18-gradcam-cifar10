import os
import torch
import matplotlib.pyplot as plt

def calculate_accuracy(outputs,targets):
    _,preds=torch.max(outputs,dim=1)
    correct_num = (preds==targets).sum().item()
    total_num = targets.size(0)
    return correct_num/total_num

def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_path='./outputs/loss_acc.png'):
    os.makedirs(os.path.dirname(save_path),exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', linestyle='-')
    plt.plot(epochs, val_losses, label='Val Loss', color='red', linestyle='--')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

 
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy', color='blue', linestyle='-')
    plt.plot(epochs, val_accs, label='Val Accuracy', color='red', linestyle='--')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)


    plt.tight_layout()


    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 训练曲线已成功保存至: {save_path}")


if __name__ == "__main__":
    print("=== 开始测试 src/utils.py 工具模块 ===")
    
    # 1. 单元测试: calculate_accuracy
    dummy_outputs = torch.tensor([
        [2.1, 0.5, 0.1],  # 预测索引 0
        [0.1, 3.4, 0.2],  # 预测索引 1
        [0.5, 0.2, 1.8]   # 预测索引 2
    ])
    dummy_targets = torch.tensor([0, 1, 0])  # 正确数 2/3
    acc = calculate_accuracy(dummy_outputs, dummy_targets)
    print(f"准确率计算结果: {acc:.4f}")
    assert abs(acc - (2 / 3)) < 1e-5, "准确率计算逻辑不匹配！"
    print("✅ calculate_accuracy 单元测试通过！")

    # 2. 单元测试: plot_metrics
    dummy_train_loss = [0.8, 0.6, 0.4, 0.2]
    dummy_val_loss = [0.9, 0.7, 0.5, 0.3]
    dummy_train_acc = [0.6, 0.7, 0.8, 0.9]
    dummy_val_acc = [0.55, 0.65, 0.75, 0.85]
    
    plot_metrics(dummy_train_loss, dummy_val_loss, dummy_train_acc, dummy_val_acc, save_path='./outputs/test_loss_acc.png')
    assert os.path.exists('./outputs/test_loss_acc.png'), "测试图片生成失败！"
    print("✅ plot_metrics 单元测试通过！")