# -*- coding: utf-8 -*-

import argparse
import os
parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--cfg_path', '-c', default='config_qata', metavar='CFG_PATH',
                    type=str,
                    help='Path to the config file')
parser.add_argument('--gpu', '-g', default='0', metavar='cuda',
                    type=str,
                    help='device id')
parser.add_argument('--profile', action='store_true',
                    help='Print Params/FLOPs and continue training')
parser.add_argument('--profile_only', action='store_true',
                    help='Print Params/FLOPs and exit')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.cfg_path == "config_mosmed":
    import config_mosmed as config
else:
    import config_qata as config

import torch.optim
import torch.nn as nn
import math
from tensorboardX import SummaryWriter
import numpy as np
import random
from torch.backends import cudnn
from nets.tirnet import TIRNet
from torch.utils.data import DataLoader
import logging
from Train_one_epoch import train_one_epoch
from torchvision import transforms
from utils import DiceBCELoss, read_text, profile_params_flops

def set_global_seed(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


class SingleCycleCosineDecay:
    def __init__(self, optimizer, lr_max: float, eta_min: float, t_max: float):
        self.optimizer = optimizer
        self.lr_max = float(lr_max)
        self.eta_min = float(eta_min)
        self.t_max = float(max(1.0, t_max))
        self.last_t = None

    def step(self, t):
        t = float(t)
        if t < 0.0:
            t = 0.0
        if t > self.t_max:
            t = self.t_max
        lr = self.eta_min + (self.lr_max - self.eta_min) * 0.5 * (1.0 + math.cos(math.pi * (t / self.t_max)))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self.last_t = t

    def state_dict(self):
        return {"lr_max": self.lr_max, "eta_min": self.eta_min, "t_max": self.t_max, "last_t": self.last_t}

    def load_state_dict(self, state_dict):
        self.lr_max = float(state_dict.get("lr_max", self.lr_max))
        self.eta_min = float(state_dict.get("eta_min", self.eta_min))
        self.t_max = float(state_dict.get("t_max", self.t_max))
        self.last_t = state_dict.get("last_t", self.last_t)

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '' + \
                   'latest_model.pth.tar'
    logger.info('\t Saving to {}'.format(filename))
    torch.save(state, filename)


def worker_init_fn(worker_id):
    seed = int(config.seed) + int(worker_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


##################################################################################
# =================================================================================
#          Main Loop: load model,
# =================================================================================
##################################################################################
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    lr = config.learning_rate
    logger.info(model_type)

    config_vit = config.get_ViT_config()
    model = TIRNet(config, config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    if args.profile or args.profile_only:
        params_m, flops_g = profile_params_flops(
            model,
            img_size=int(getattr(config, "img_size", 224)),
            token_len=int(getattr(config, "token_len", 18)),
            n_channels=int(getattr(config, "n_channels", 3)),
        )
        logger.info(f"Params(M): {params_m:.3f}  FLOPs(G) inference: {flops_g:.3f}  FLOPs(G) train~: {flops_g * 3.0:.3f}")
        if args.profile_only:
            return model

    from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D

    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    train_text = read_text(os.path.join(config.train_dataset, 'Train_text.xlsx'))
    val_text = read_text(os.path.join(config.val_dataset, 'Val_text.xlsx'))
    train_dataset = ImageToImage2D(
        config.train_dataset, config.task_name, train_text, train_tf,
        image_size=config.img_size, data_name=config.task_name, token_len=config.token_len,
        config=config, mode="train")
    val_dataset = ImageToImage2D(
        config.val_dataset, config.task_name, val_text, val_tf,
        image_size=config.img_size, data_name=config.task_name, token_len=config.token_len,
        config=config, mode="val")

    data_generator = torch.Generator()
    data_generator.manual_seed(int(config.seed))

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              generator=data_generator,
                              num_workers=8,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            generator=data_generator,
                            num_workers=8,
                            pin_memory=True)
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # Choose optimize
    t_max = float(config.epochs)
    lr_max = float(getattr(config, "learning_rate", lr))
    eta_min = float(getattr(config, "cosine_eta_min", 1e-4))
    lr_scheduler = SingleCycleCosineDecay(optimizer, lr_max=lr_max, eta_min=eta_min, t_max=t_max)

    print(config.lr)
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    epoch = 0

    if config.resume:
        checkpoint = torch.load(config.resume_path, map_location='cpu')
        # print(type(checkpoint), type(checkpoint['model']), checkpoint.keys())
        model.load_state_dict(checkpoint['state_dict'], strict=True)

    model = model.cuda()

    if config.resume:
        checkpoint = torch.load(config.resume_path, map_location='cpu')
        logger.info('resume path: {}'.format(config.resume_path))
        print(model.load_state_dict(checkpoint['state_dict']))
        
    if torch.cuda.device_count() > 1:
        logger.info("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    if config.resume:
        print(optimizer.load_state_dict(checkpoint['optimizer']))
        if lr_scheduler is not None and checkpoint.get('lr_scheduler') is not None:
            print(lr_scheduler.load_state_dict(checkpoint['lr_scheduler']))
        epoch = checkpoint['epoch']
        print("resume optimizer and lr scheduler successfuly")
    else:
        epoch = -999

    max_dice = 0.0
    for epoch in range(max(0, epoch+1), config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_one_epoch(config, train_loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger)

        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice = train_one_epoch(config, val_loader, model, criterion,
                                                 optimizer, writer, epoch, None, model_type, logger)
        # =============================================================
        #       Save best model
        # =============================================================
        if val_dice > max_dice:
            if epoch + 1 > 0:
                logger.info(
                    '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path)
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice, max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))

        save_checkpoint({'epoch': epoch,
                        'best_model': False,
                        'model': model_type,
                        'state_dict': model.state_dict(),
                        'val_loss': val_loss,
                        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                        'optimizer': optimizer.state_dict()}, config.model_path)

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':

    deterministic = False
    set_global_seed(int(config.seed), deterministic=deterministic)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)

    with open(args.cfg_path+'.py', 'r') as file:  
        lines = file.readlines()  
    for line in lines:  
        logger.info(line[:-1])

    model = main_loop(model_type=config.model_name, tensorboard=True)
