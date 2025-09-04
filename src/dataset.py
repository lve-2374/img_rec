#import os
#import cv2
from PIL import Image   # 如果想用 PIL，可以取消注释
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os


# 准备一个列表，储存各所有的类别看，要和data文件夹里各类别名称完全对应
class_list = ["bass","headphones","phone"]

def get_img_paths_and_labels(data_dir):
    """
    从根目录获取所有图像路径和对应标签
    假设目录结构：
    data_dir/class1/*.jpg
    data_dir/class2/*.jpg
    """
    classes = sorted(os.listdir(data_dir))
    img_paths = []
    labels = []
    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for f in os.listdir(cls_dir):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                img_paths.append(os.path.join(cls_dir, f))
                labels.append(class_list.index(cls_name))

    print(class_list)
    return img_paths, labels, class_list


class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_paths, self.labels, self.class_list = get_img_paths_and_labels(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        label = self.labels[idx]

        # ---- OpenCV 读取 (BGR → RGB) ----
        #img = cv2.imread(path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ---- PIL 读取 (可选) ----
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, label




if __name__ == "__main__":
    # 数据根目录 (子文件夹名为类别)
    data_dir = os.path.abspath("C:/Users/lve/Desktop/img_rec/data/train_and_val")  # 例如: data/cat/*.jpg, images/dog/*.jpg

    # 定义 transform（独立管理）
    transformx = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    # 创建数据集
    dataset = MyDataset(data_dir, transform=transformx)

    # 打印类别映射
    print("Class to index mapping:", dataset.class_list)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for imgs, labels in dataloader:
        print("Batch images shape:", imgs.shape)   # [B, 3, 224, 224]
        print("Batch labels:", labels)            # [B]
        break
