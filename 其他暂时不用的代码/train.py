import os
import time
import datetime
import numpy as np
import torch
from torch.autograd import Variable
from Models import self_model
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import My_dataset
#参数列表：


def main():
    d_input = 256
    heads = 8
    dropout = 0.1
    model = self_model(d_input=d_input, heads=heads, dropout=dropout, )
    derection = 'up'
    device = 'cuda'
    batch_size=128
    lr=0.1
    weight_decay=1e-4
    epochs=100
    momentum=0.9
    save_best=0.5
    resume=''
    train_path_image='E:/1.SCSF\Self_supervision\data\_0'
    train_path_lable='E:/1.SCSF\Self_supervision\data\_1'
    val_path_image = 'E:/1.SCSF\Self_supervision\data\_0'
    val_path_lable = 'E:/1.SCSF\Self_supervision\data\_1'

    device = torch.device(device if torch.cuda.is_available() else "cpu")


    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))



    train_dataset = My_dataset(train_path_image,train_path_lable)

    val_dataset = My_dataset(val_path_image,val_path_lable)

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             )

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
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,lr_scheduler=lr_scheduler,)


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
            torch.save(model.state_dict(), "save_weights/best_my_model_0.4.pth")
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




