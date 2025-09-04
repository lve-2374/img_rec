import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as f
from PIL import Image
from src import model
from src import dataset



# 定义 transform（独立管理）
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
    ])

def main():
    # 创建一个VideoCapture对象以访问摄像头
    cap = cv2.VideoCapture(0)

    flag = {'bass':False,'headphones':False,'phone':False} # 各类已训练物品是否已保存识别结果

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        # 读取一帧
        ret, frame = cap.read()

        if not ret:
            print("无法接收帧（流结束？）")
            break

        # 转成符合网络输入的尺寸，并转为tensor
        img  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = transform(img)
        img = img.unsqueeze(0)

        # 载入模型 注意训练和测试用的模型必须相同，否则储存的权值文件用不了
        classifier = model.Classifier()
        classifier.load_state_dict(
            torch.load("./pth/classifier_198.pth", map_location=torch.device('cpu')))  # 一张图用cpu跑也行，也可以像训练一样塞入gpu
        classifier.eval()
        # 注意，pth权值文件有两种不同保存形式，保存和载入的形式需要一致，这里都是state_dict

        # 零梯度下将图片输入模型
        with torch.no_grad():
            output = classifier(img)
        #print(output)  # 这个输出的是，图片是那一堆类别里每一个类别的可能性
        prediction = output.argmax(1)
        #print(prediction)  # 输出概率最大值对应的那个类别序号

        probabilities = f.softmax(output, dim=1)  # 用softmax将tensor转换为概率
        probability = probabilities[0,prediction].item()
        print('probability:',probability)

        # 若出现未训练的物品显示“未发现”
        if probability < 0.6:
            pre_name = 'Not found'
        else:
            pre_name = dataset.class_list[prediction] # 把类别序号转回文字
        print(pre_name) # 输出识别结果

        # 将 Tensor 转换回 numpy 数组
        image = img.squeeze(0).permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = np.ascontiguousarray(image)
        image = cv2.resize(image, (800, 600))

        # 显示结果帧
        cv2.putText(image, pre_name, (0, 180), 0, 2, (0, 0, 255), 3, 0)
        cv2.putText(image, str(probability), (0, 120), 0, 2, (0, 0, 255), 3, 0)
        cv2.imshow('Camera', image)

        # 保存结果画面
        if probability > 0.99 and not flag[pre_name]: # 补充：由于python语法特性，这里先判断概率就不会出现pre_name为Not found时flag中不存在的情况
            cv2.imwrite('./result/'+pre_name+'.jpg', image)
            flag[pre_name] = True

        # 按下q键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 完成所有操作后释放捕获器
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()