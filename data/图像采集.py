from PIL import Image
import os
from torchvision import transforms
Image.MAX_IMAGE_PIXELS = 230000000000


#image_large=Image.open ("E:/1.SCSF\红参组织切片大图/1.黑白图/根 27_Wholeslide_Default_Extended.jpg")
#image_large=image_large.convert('1')

#image_0 = image_large.crop((000,000,600,600))
#image_0.save("E:/1.SCSF\Self_supervision\data/0.jpg")
#print(image_large)


path='E:/1.SCSF\红参组织切片大图/1.20000X20000'#总文件路径
path_0="E:/1.SCSF\Self_supervision\data\_0/"
path_1="E:/1.SCSF\Self_supervision\data\_1/"
path_2="E:/1.SCSF\Self_supervision\data\_2/"
path_3="E:/1.SCSF\Self_supervision\data\_3/"
path_4="E:/1.SCSF\Self_supervision\data\_4/"
num=0
images=os.listdir(path)

num = 0
for index in range(20):

    image_index = images[index]
    image_path = os.path.join(path, image_index)  # 得到图像路径
    image_large = Image.open(image_path).convert('1')
    
    length = 20000

    
    for i in range(300, length - 300, 64):
        for j in range(300, length - 300, 64):

            point = image_large.getpixel((i, j))  # 得到中心点的像素值
            if point ==0 :  # 白色就不采集
                num += 1
                print(index,i, j,num)
                image_0 = image_large.crop((i - 64, j - 64, i + 64, j + 64))  # 保存中心图像
                image_0.save(path_0 + '{}'.format(num) + '.jpg')

                image_1 = image_large.crop((i - 64, j - 96, i + 64, j - 64))  # 保存上方图像
                image_1.save(path_1 + '{}'.format(num) + '_1.jpg')

                image_2 = image_large.crop((i + 64, j - 64, i + 96, j + 64))  # 保存右方图像
                image_2.save(path_2 + '{}'.format(num) + '_2.jpg')

                image_3 = image_large.crop((i - 64, j + 64, i + 64, j + 96))  # 保存下方图像
                image_3.save(path_3 + '{}'.format(num) + '_3.jpg')

                image_4 = image_large.crop((i - 96, j - 64, i - 64, j + 64))  # 保存左方图像
                image_4.save(path_4 + '{}'.format(num) + '_4.jpg')


