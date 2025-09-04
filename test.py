import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# 添加tensorboard，记录数据
writer = SummaryWriter("log")

# 测试集根目录
data_dir = "./data/test"

# 定义 transform（独立管理）
transformx = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop((224,224), padding=25, fill=125, padding_mode="edge"),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()
    ])

def main():
    # 创建数据集
    from src import dataset
    dataset = dataset.MyDataset(data_dir, transform=transformx)

    # 将数据集加载为loader
    test_size = len(dataset)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # 载入模型 注意训练和测试用的模型必须相同，否则储存的权值文件用不了
    from src import model
    classifier = model.Classifier()
    if torch.cuda.is_available():  # 如果cuda可用就用显卡跑
        print("Using GPU")
        classifier.cuda()
    classifier.load_state_dict(torch.load("./pth/classifier_198.pth", map_location=torch.device('cuda')))# 一张图用cpu跑也行，也可以像训练一样塞入gpu
    classifier.eval()
    # 注意，pth权值文件有两种不同保存形式，保存和载入的形式需要一致，这里都是state_dict

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    # 开始测试
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():  # 无梯度模式
        for data in test_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = classifier(imgs)
            loss = loss_fn(outputs, targets)
            test_loss = test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            test_accuracy = test_accuracy + accuracy

    # 输出测试集loss及准确率并添加到看板
    print("测试集loss: {}".format(test_loss))
    print("测试集准确率: {}".format(test_accuracy / test_size))
    writer.add_scalar("test_loss", test_loss, 1)
    writer.add_scalar("test_accuracy", test_accuracy / test_size, 1)
    writer.close()

if __name__ == "__main__":
    main()