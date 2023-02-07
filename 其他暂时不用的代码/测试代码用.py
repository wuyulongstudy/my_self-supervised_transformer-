from Models import  IMAGE_TO_SEQ
from Models import  Encoder
from Models import  Dncoder
from Models import  self_model
from PIL import Image

from torch import nn
from torchvision import transforms
import numpy as np
import torch
import torchsummary
from einops import rearrange, repeat
start=torch.randn(1, 1,128)
start=repeat(start, '1 n d -> b n d', b = 256)
l1=torch.zeros(256,32,128)


l2=torch.cat((start,l1),dim=1)
print(l2)
print(l2.shape)