#import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.flatten = nn.Flatten()
        # 全连接层
        self.fc1 = nn.Linear(100352, 512)
        self.fc2 = nn.Linear(512, 9)

        # 批量归一
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # dropout层防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
