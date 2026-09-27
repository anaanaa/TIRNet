# -*- coding: utf-8 -*-
import os
import time

import ml_collections
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'datasets')
MOSMED_ROOT = os.path.join(DATA_ROOT, 'MosMedData+Dataset')

save_model = True
tensorboard = True
use_cuda = torch.cuda.is_available()
seed = 3407
os.environ['PYTHONHASHSEED'] = str(seed)

lr = 'cosineLR'

n_channels = 3
n_labels = 1
epochs = 200

img_size = 224
print_frequency = 10
vis_frequency = 5000
early_stopping_patience = 100

task_name = 'MosMed'
learning_rate = 3e-4
batch_size = 32
token_len = 18
optimizer = "AdamW"
weight_decay = 1e-5

model_name = 'TIRNet'
loss_weight = {
    "loss_seg": 1.0,
    "loss_rgc": 1.0,
    "loss_bs": 0.5,
}
resume = False
cosine_eta_min = 1e-6

train_dataset = os.path.join(MOSMED_ROOT, 'Train_Folder')
val_dataset = os.path.join(MOSMED_ROOT, 'Val_Folder')
test_dataset = os.path.join(MOSMED_ROOT, 'Test_Folder')

session_name = 'session' + '_' + time.strftime('%m.%d_%Hh%M') + '_MosMed'
base_run_path = os.path.join(PROJECT_ROOT, 'outputs')
base_run_dir = os.path.join(base_run_path, task_name, model_name)
save_path = os.path.join(base_run_dir, session_name) + '/'
model_path = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path = save_path + session_name + ".log"
visualize_path = save_path + 'visualize_val/'

clip_pretrained_path = os.path.join(PROJECT_ROOT, 'pretrained')


def get_ViT_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.base_channel = 64
    config.clip_backbone = "ViT-B/32"
    config.frozen_clip = True
    return config
