import os
import time
import datetime
import numpy as np
import torch

from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import My_dataset
from torch import nn
#参数列表：

class my_transformer(nn.Module):
    def __init__(self, *,d_model= 128, nhead = 8, num_encoder_layers = 6,num_decoder_layers = 6, dim_feedforward= 1024, dropout = 0.1,batch_first= True, norm_first = True,lenth=32):
        super().__init__()

        self.pos_embedding_src= nn.Parameter(torch.randn(1, 128, d_model))
        self.pos_embedding_tgt = nn.Parameter(torch.randn(1, lenth , d_model))
        self.transformer = torch.nn.Transformer(d_model= d_model, nhead = nhead, num_encoder_layers = num_encoder_layers,num_decoder_layers = num_decoder_layers, dim_feedforward= dim_feedforward, dropout =dropout,batch_first= batch_first, norm_first = norm_first,)

    def forward(self, src,tgt,tgt_mask=None):
        b, n1, _ = src.shape
        _,n2,_=tgt.shape
        #输入输出序列加入位置编码
        #tgt += self.pos_embedding_tgt[:, :(n2 + 1)]
        #src += self.pos_embedding_src[:, :(n1 + 1)]
        #实验发现，不加位置编码的效果还要更好一些
        outputs = self.transformer(src , tgt ,tgt_mask=tgt_mask)
        return outputs #第一行开始位，不进行输出

def main():

    # 一些训练参数
    lenth = 32
    batch_size = 64
    device = 'cuda'
    lr = 0.1
    weight_decay = 1e-4
    epochs = 100
    momentum = 0.9
    save_best = 0.5
    resume = ''


    #定义路径
    train_path_image = 'E:/1.SCSF\Self_supervision\data\_0'
    train_path_lable_1 = 'E:/1.SCSF\Self_supervision\data\_1'
    train_path_lable_2 = 'E:/1.SCSF\Self_supervision\data\_2'
    train_path_lable_3 = 'E:/1.SCSF\Self_supervision\data\_3'
    train_path_lable_4 = 'E:/1.SCSF\Self_supervision\data\_4'
    val_path_image = 'E:/1.SCSF\Self_supervision\data\_0'
    val_path_lable = 'E:/1.SCSF\Self_supervision\data\_1'

    #模型定义
    model =my_transformer(lenth=lenth)
    #model.load_state_dict(torch.load("save_weights/best_my_model_TF_self.pth"))

    #创建掩码矩阵，模拟预测过程
    matrix = np.ones((lenth, lenth))
    T_matrix = torch.from_numpy(matrix)
    mask1 = torch.triu(T_matrix, 0).to(device)
    mask1=torch.t(mask1)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    train_dataset = My_dataset(train_path_image,train_path_lable_1,train_path_lable_2,train_path_lable_3,train_path_lable_4)

    #val_dataset = My_dataset(val_path_image,val_path_lable)
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               )

    '''val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             )
'''
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs, warmup=True)

    if resume:
        checkpoint = torch.load(resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    start_time = time.time()


    for epoch in range(epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,lr_scheduler=lr_scheduler,mask=mask1,)


        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \

            f.write(train_info + "\n\n")


        if save_best > mean_loss:
            save_best = mean_loss
        else:
            continue



        if save_best <0.5:
            torch.save(model.state_dict(), "save_weights/best_my_model_TF_self.pth")
        else:
            pass
            #torch.save(save_file, "save_weights/model_{}.pth".format(epoch))
        print(save_best)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))





if __name__ == '__main__':

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main()




