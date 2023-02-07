
import numpy as np
import torch

from train_by_transformer import my_transformer
from PIL import Image
from torch import nn
from torchvision import transforms
device='cuda'
d_input = 256
heads = 8
dropout = 0.1

model =my_transformer(lenth=32)

image_path='E:/1.SCSF\Self_supervision\data/val\_0/30.jpg'
img=Image.open(image_path).convert('1')
label=Image.open('E:/1.SCSF\Self_supervision\data/val\_3/30_3.jpg').convert('1')

model.load_state_dict(torch.load( "save_weights/best_my_model_TF_self.pth"))

model.to(device)

to_tensor = transforms.ToTensor()  # 必须进行实例化

img = to_tensor(img)
img = img.view(128, 128)
img=img.unsqueeze(0).to(device)

label=to_tensor(label).to(device)
label=label.view(32,128)
label=label.unsqueeze(0).to(device)

#创建掩码矩阵，模拟预测过程
matrix = np.ones((32, 32))
T_matrix = torch.from_numpy(matrix)
mask1 = torch.triu(T_matrix, 0).to(device)
mask1=torch.t(mask1)

predict=model(img,label,tgt_mask=mask1)


#将概率值转成像素值：
for i in range(32):
    for j in range(128):

        if predict[0][i][j]>0.1:
            predict[0][i][j]=1
        else:
            predict[0][i][j]=0
print(predict)

#将标签图和预测图显示，观察结果差距
unloader = transforms.ToPILImage()#tensor转PIL
image = unloader(predict)
tensor_image=to_tensor(image).to(device)
image.show()
ima2=unloader(label)
ima2.show()

#输出loss值
loss1=nn.L1Loss()
loss=loss1(predict,label)
print(loss)
























