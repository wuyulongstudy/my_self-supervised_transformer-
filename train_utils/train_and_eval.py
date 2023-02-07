import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target
import numpy as np
from einops import rearrange, repeat


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value.item()


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    lr_scheduler, print_freq=10, scaler=None,mask=None,):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    loss1=nn.L1Loss()

    for img_1,label_1,img_2,label_2,img_3,label_3,img_4,label_4 in metric_logger.log_every(data_loader, print_freq, header):
        img_1,label_1,img_2,label_2,img_3,label_3,img_4,label_4= img_1.to(device),label_1.to(device),img_2.to(device),label_2.to(device),img_3.to(device),label_3.to(device),img_4.to(device),label_4.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):

            output_1 = model(img_1,label_1,tgt_mask=mask)
            output_2 = model(img_2, label_2, tgt_mask=mask)
            output_3 = model(img_3, label_3, tgt_mask=mask)
            output_4 = model(img_4, label_4, tgt_mask=mask)

            output_1=output_1.float()
            label_1=label_1.float()

            output_2 = output_2.float()
            label_2 = label_2.float()

            output_3 = output_3.float()
            label_3 = label_3.float()

            output_4 = output_4.float()
            label_4 = label_4.float()

            loss = (loss1(output_1,label_1)+loss1(output_2,label_2)+loss1(output_3,label_3)+loss1(output_4,label_4))/4

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
