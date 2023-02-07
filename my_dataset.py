import os

import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class My_dataset(Dataset):
    def __init__(self,path_image,path_lable_1,path_lable_2,path_lable_3,path_lable_4):
        self.path_image=path_image
        self.path_lable_1=path_lable_1
        self.path_lable_2 = path_lable_2
        self.path_lable_3 = path_lable_3
        self.path_lable_4 = path_lable_4
        self.images=os.listdir(self.path_image)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_index=self.images[index]
        image_path=os.path.join(self.path_image,image_index)    #得到图像路径
        img=Image.open(image_path).convert('1')               #读取图像
        #1，2，3，4分别代表上右下左
        #1
        label_path=os.path.join(self.path_lable_1,image_index[:-4]+self.path_lable_1[-2:]+'.jpg')
        label_1=Image.open(label_path).convert('1')
        # 2
        label_path = os.path.join(self.path_lable_2, image_index[:-4] + self.path_lable_2[-2:] + '.jpg')
        label_2 = Image.open(label_path).convert('1')
        # 3
        label_path = os.path.join(self.path_lable_3, image_index[:-4] + self.path_lable_3[-2:] + '.jpg')
        label_3 = Image.open(label_path).convert('1')
        # 4
        label_path = os.path.join(self.path_lable_4, image_index[:-4] + self.path_lable_4[-2:] + '.jpg')
        label_4 = Image.open(label_path).convert('1')

        #把图像转成tensor格式
        to_tensor = transforms.ToTensor()  # 必须进行实例化
        img = to_tensor(img)
        img = img.view(128, 128)

        label_1=to_tensor(label_1)
        label_1=label_1.view(32,128)

        label_2 = to_tensor(label_2)
        label_2 = label_2.view(128, 32)

        label_3 = to_tensor(label_3)
        label_3 = label_3.view(32, 128)

        label_4= to_tensor(label_4)
        label_4 = label_4.view(128, 32)

        #生成上方的，从下往上生成
        img_1=torch.flip(img,[0])
        label_1=torch.flip(label_1,[0])

        # 生成右方的，从左往右生成

        img_2 = torch.t(img)
        label_2 = torch.t(label_2)

        # 生成下方的，从上往下生成
        img_3 =img
        label_3 = label_3

        # 生成右方的，从右往左生成
        img_4 = torch.t(img)
        label_4 = torch.t(label_4)
        img_4 = torch.flip(img_4, [0])
        label_4 = torch.flip(label_4, [0])

        return img_1,label_1,img_2,label_2,img_3,label_3,img_4,label_4



