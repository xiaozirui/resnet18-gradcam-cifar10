import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=128, num_workers=2, data_dir='./data'):
    """
    构建 CIFAR-10 的 DataLoader
    """
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN,CIFAR10_STD)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN,CIFAR10_STD)
    ])

   
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    val_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=64)
    images, labels = next(iter(train_loader))
    print("✅ DataLoader success")
    print("Images batch shape:", images.shape)
    print("Labels batch shape:", labels.shape) 