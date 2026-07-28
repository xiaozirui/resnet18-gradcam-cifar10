import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        x=self.conv1(x)
        x=self.bn1(x)
        x=F.relu(x)

        x=self.conv2(x)
        x=self.bn2(x)
        out = x +self.shortcut(identity)

        out=F.relu(out)

        return out

class ResNet18(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.in_channels=64

        self.layer1=self.__make_layer(BasicBlock,out_channels=64,num_blocks=2,stride=1)
        self.layer2=self.__make_layer(BasicBlock,out_channels=128,num_blocks=2,stride=2)
        self.layer3=self.__make_layer(BasicBlock,out_channels=256,num_blocks=2,stride=2)
        self.layer4=self.__make_layer(BasicBlock,out_channels=512,num_blocks=2,stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)

        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
    def __make_layer(self,block,out_channels,num_blocks,stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers=[]
        for s in strides:
            block_1=block(self.in_channels,out_channels,s)
            layers.append(block_1)
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)


        
if __name__ == "__main__":
    model = ResNet18(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print("ResNet-18 output shape:", out.shape)