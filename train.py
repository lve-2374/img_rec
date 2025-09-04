import torch
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter # 数据看板
from torch import nn



# 数据根目录 (子文件夹名为类别)
data_dir = "./data/train_and_val"  # 例如: data/cat/*.jpg, images/dog/*.jpg

# 定义 transform（独立管理）
transformx = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop((224,224), padding=25, fill=125, padding_mode="edge"),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()
    ])

# 创建数据集
from src import dataset
dataset = dataset.MyDataset(data_dir, transform=transformx)

# 打印类别映射
print("Class to index mapping:", dataset.class_list)

# 分割成训练集和验证集，比例为8：2，并加载成loader
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # 16张图一批，爆显存就改小点，下同
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 导入模型
from src import model
classifier = model.Classifier()
if torch.cuda.is_available():  # 如果cuda可用就用显卡跑
    print("Using GPU")
    NIIIC_model = classifier.cuda()
# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# 优化器
#learning_rate = 1e-2
#optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0001)
# 学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 200

# 添加tensorboard，记录数据
writer = SummaryWriter("log") # log就是储存的文件夹

# 开始训练及验证，训练一次验证一次
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i + 1))

    # 训练步骤开始
    classifier.train() # 训练模式
    for data in train_loader:
        imgs, targets = data
        if torch.cuda.is_available():# 图片和标签都放入显卡
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = classifier(imgs)  # 看当前输出
        loss = loss_fn(outputs, targets) # 对比网络输出和标签，得到损失（就是差了多少）

        # 优化器优化模型
        optimizer.zero_grad() # 先取消优化器的梯度
        loss.backward() # 反向传播，优化模型
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step) # 写到看板里

    # 测试步骤开始
    classifier.eval() # 测试模式
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad(): # 无梯度模式
        for data in val_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = classifier(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
                # argmax 用以指出概率最大项
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / val_size))
    writer.add_scalar("loss", total_test_loss, total_test_step)
    writer.add_scalar("train_accuracy", total_accuracy / val_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(classifier.state_dict(), "./pth/classifier_{}.pth".format(i))
    print("模型已保存")

writer.close()  # 关闭看板
