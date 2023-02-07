import torch
from torch import nn
import copy
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from Sublayers import FeedForward, MultiHeadAttention, Norm
# helpers
import math
import torch.nn.functional as F
import  numpy as np


#输入尺寸转换部分：
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class IMAGE_TO_SEQ(nn.Module):
    def __init__(self, *, image_size, patch_size, dim,  pool = 'mean', channels = 1, dim_head = 64, dropout = 0.1, emb_dropout = 0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.dropout = nn.Dropout(emb_dropout)


        self.pool = pool
        self.to_latent = nn.Identity()


    def forward(self, img):

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        return x


#Multi Head Attention部分：


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_input,d_output, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_input,d_output)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))

        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class Encoder_Decoder_Layer(nn.Module):
    def __init__(self, d_model, heads,d_output, dropout=0.1):#d_model是输入该层的维度，d_output是输出该层的维度
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_1_linear = Norm(d_model)
        self.norm_2_linear=Norm(int(d_model * 4))
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)

        self.ff = FeedForward(d_model,d_output, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model, int(d_model*4))
        self.linear_2 = nn.Linear(int(d_model * 4), d_model)
    def forward(self, x,mask=None):

        for i in range(1):
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn(x2, x2, x2,mask))

            x=self.norm_1_linear(x)
            x=self.linear_1(x)

            x = self.norm_2_linear(x)
            x = self.linear_2(x)

        x2 = self.norm_2(x)
        x = self.dropout_2(self.ff(x2))
        return x

#编码器端

class Encoder(nn.Module):
    def __init__(self, d_model, heads, d_output,dropout):
        super().__init__()

        self.encoderlayer1 =Encoder_Decoder_Layer(d_model,heads,int(d_model/2),dropout)

        self.encoderlayer2 = Encoder_Decoder_Layer(int(d_model/2),heads,int(d_model/4),dropout)

        self.encoderlayer3 = Encoder_Decoder_Layer(int(d_model/4), heads, d_output, dropout)

    def forward(self, x,mask=None):

        x = self.encoderlayer1(x,mask)

        x = self.encoderlayer2(x,mask)

        x = self.encoderlayer3(x,mask)

        return x



#解码器端：

class Dncoder(nn.Module):
    def __init__(self, d_model, heads, d_output,dropout):
        super().__init__()
        self.cls_token_up = nn.Parameter(torch.randn(1, 1, d_model))
        self.cls_token_left = nn.Parameter(torch.randn(1, 1, d_model))
        self.cls_token_right = nn.Parameter(torch.randn(1, 1, d_model))
        self.cls_token_down = nn.Parameter(torch.randn(1, 1, d_model))

        self.dncoderlayer1 =Encoder_Decoder_Layer(d_model,heads,int(d_model*2),dropout)

        self.dncoderlayer2 = Encoder_Decoder_Layer(int(d_model*2),heads,int(d_model*4),dropout)

        self.dncoderlayer3 = Encoder_Decoder_Layer(int(d_model*4), heads, d_output, dropout)



    def forward(self, x,derection):
        b, n, _ = x.shape           #b=batchsize,n=seqlength

        if derection == 'up':
            cls_tokens = repeat(self.cls_token_up, '1 n d -> b n d', b=b)

        if derection == 'left':
            cls_tokens = repeat(self.cls_token_left, '1 n d -> b n d', b=b)

        if derection == 'right':
            cls_tokens = repeat(self.cls_token_right, '1 n d -> b n d', b=b)

        if derection == 'down':
            cls_tokens = repeat(self.cls_token_down, '1 n d -> b n d', b=b)

        x = torch.cat((cls_tokens, x), dim=1)
        batch_size=len(x)
        device='cuda'
        matrix = np.ones((batch_size,65, 65))
        T_matrix = torch.from_numpy(matrix)
        mask1 = torch.triu(T_matrix, 1).to(device)

        matrix = np.ones((batch_size,33, 33))
        T_matrix = torch.from_numpy(matrix)
        mask2 = torch.triu(T_matrix, 1).to(device)

        matrix = np.ones((batch_size,17, 17))
        T_matrix = torch.from_numpy(matrix)
        mask3 = torch.triu(T_matrix, 1).to(device)

        x = self.dncoderlayer1(x, mask1)


        x=x[:,0:int(len(x[0])/2)+1,:]


        x = self.dncoderlayer2(x, mask2)
        x = x[:, 0:int(len(x[0]) / 2)+1, :]


        x = self.dncoderlayer3(x, mask3)
        x=x[:,1:,:]


        return x

class self_model(nn.Module):
    def __init__(self,d_input,heads,dropout):
        super().__init__()
        self.image_to_seq=IMAGE_TO_SEQ( image_size=128, patch_size=16, dim=d_input)
        self.encoder=Encoder(d_model=d_input,heads=heads,d_output=int(d_input/8),dropout=dropout)

        self.decoder=Dncoder(d_model=int(d_input/8),heads=heads,d_output=int(d_input),dropout=dropout)

    def forward(self, img,derection,mask=None):

        x=self.image_to_seq(img)

        encoder_outputs = self.encoder(x)

        decoder_outputs = self.decoder(encoder_outputs,derection)

        decoder_outputs=decoder_outputs.view(int(len(decoder_outputs)),1,int(len(decoder_outputs[0])*2),int(len(decoder_outputs[0][0])/2))

        return decoder_outputs