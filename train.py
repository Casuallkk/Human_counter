"""训练， 对代码进行了优化"""
import datetime
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.hub import load_state_dict_from_url

from nets.yolo import YoloBody
from nets.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    Cuda = True
    sync_bn = False
    fp16 = False
    classes_path = 'model_data/my_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    model_path = 'model_data/yolov7_weights.pth'
    input_shape = [640, 640]
    pretrained = True

    # mosaic增强
    mosaic = True
    mosaic_prob = 0.5

    # mixup增强
    mixup = True
    mixup_prob = 0.5

    special_aug_ratio = 0.7

    # 标签平滑
    label_smoothing = 0

    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 8
    UnFreeze_Epoch = 300
    Unfreeze_batch_size = 4
    Freeze_Train = True

    max_lr = 1e-2
    min_lr = max_lr * 0.01
    optimizer_type = "sgd"
    momentum = 0.937
    weight_decay = 5e-4
    lr_decay_type = "cos"
    save_period = 10
    save_dir = 'logs'
    eval_flag = True
    eval_period = 10

    num_workers = 2

    train_annotation_path = 'model_data/2007_train.txt'
    val_annotation_path = 'model_data/2007_val.txt'

    ngpus_per_node = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rank = 0
    class_names, num_classes = YoloDataset.get_classes(classes_path)
    anchors, num_anchors = YoloDataset.get_anchors(anchors_path)

    if pretrained:
        model_dir = "./model_data"
        url = 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_backbone_weights.pth'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        load_state_dict_from_url(url, model_dir)

    model = YoloBody(anchors_mask, num_classes, pretrained=pretrained, index=4)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, anchors_mask, label_smoothing)
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    else:
        model_train = model.train()

    #   权值平滑
    ema = ModelEMA(model_train)

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * max_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
            'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        # 获得学习率下降的公式
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        if ema:
            ema.updates = epoch_step * Init_Epoch

        # 构建数据集加载器。
        train_dataset = YoloDataset(train_lines, input_shape, num_classes, anchors, anchors_mask,
                                    epoch_length=UnFreeze_Epoch, mosaic=mosaic, mixup=mixup,
                                    mosaic_prob=mosaic_prob, mixup_prob=mixup_prob,
                                    train=True, special_aug_ratio=special_aug_ratio)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, anchors, anchors_mask,
                                  epoch_length=UnFreeze_Epoch,
                                  mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0,
                                  train=False, special_aug_ratio=0)

        train_sampler = None
        val_sampler = None
        shuffle = True

        train_data = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                pin_memory=True,
                                drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
        val_data = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)

        # 记录eval的map曲线
        eval_callback = EvalCallback(model, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines,
                                     log_dir, Cuda, eval_flag=eval_flag, period=eval_period)

        # 开始训练
        for epoch in range(Init_Epoch, UnFreeze_Epoch):

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * max_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if ema:
                    ema.updates = epoch_step * epoch

                train_data = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                        pin_memory=True,
                                        drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
                val_data = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                      pin_memory=True,
                                      drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            train_data.dataset.epoch_now = epoch
            val_data.dataset.epoch_now = epoch

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                          epoch_step_val, train_data, val_data, UnFreeze_Epoch, Cuda, fp16, scaler, save_period,
                          save_dir)

        loss_history.writer.close()
