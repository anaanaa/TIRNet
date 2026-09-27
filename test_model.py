
import os
import warnings

warnings.filterwarnings("ignore")
from nets.tirnet import TIRNet
from utils import *
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('--cfg_path', '-c', default='config_qata', metavar='CFG_PATH',
                    type=str,
                    help='Path to the config file')
parser.add_argument('--gpu', '-g', default='0', metavar='cuda',
                    type=str,
                    help='device id')
parser.add_argument('--test_session', '-t', default=None,
                    type=str,
                    help='session name')
parser.add_argument('--test_vis', '-v', default=False, type=bool, help='visilization')
parser.add_argument('--profile', action='store_true',
                    help='Print Params/FLOPs and continue testing')
parser.add_argument('--profile_only', action='store_true',
                    help='Print Params/FLOPs and exit')
parser.add_argument('--model_path', default='', type=str,
                    help='Explicit checkpoint path (*.pth or *.pth.tar)')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.cfg_path == "config_mosmed":
    import config_mosmed as config
else:
    import config_qata as config

def set_global_seed(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    from torch.backends import cudnn
    cudnn.benchmark = True
    cudnn.deterministic = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


red_color = (255, 0, 0)     # red
blue_color = (0, 0, 255)  # blue
green_color = (0, 255, 0)   # green
size = (224, 224)


def dice_iou_from_counts(tp, fp, fn):
    if tp + fp + fn == 0:
        return float('nan'), float('nan')
    dice = 2 * tp / (2 * tp + fp + fn)
    iou = tp / (tp + fp + fn)
    return float(dice), float(iou)


def foreground_counts(pred_bin, gt_bin):
    pred_fg = pred_bin.astype(bool)
    gt_fg = gt_bin.astype(bool)
    tp = int(np.logical_and(pred_fg, gt_fg).sum())
    fp = int(np.logical_and(pred_fg, ~gt_fg).sum())
    fn = int(np.logical_and(~pred_fg, gt_fg).sum())
    return tp, fp, fn


def compute_foreground_metrics(pred, gt):
    pred_bin = (pred >= 0.5).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    tp, fp, fn = foreground_counts(pred_bin, gt_bin)
    return dice_iou_from_counts(tp, fp, fn)


def pred_mix(ground_truth, prediction_mask, original_image):

    TP = np.sum(np.logical_and(ground_truth == 1, prediction_mask == 1))
    FP = np.sum(np.logical_and(ground_truth == 0, prediction_mask == 1))
    FN = np.sum(np.logical_and(ground_truth == 1, prediction_mask == 0))
    overlay = original_image.copy()

    # FN: red
    overlay[ground_truth == 1] = red_color
    # FP: blue
    overlay[np.logical_and(ground_truth == 0, prediction_mask == 1)] = blue_color
    # TP: green
    overlay[np.logical_and(ground_truth == 1, prediction_mask == 1)] = green_color
    
    return overlay

def draw_sub_plot(img, fig, nums, idx, mode="gray"):
    import cv2
    img = cv2.resize(img, size)
    ax = fig.add_subplot(1, nums, idx)
    if mode == "gray":
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)
    ax.axis('off')
    

def vis_and_save_heatmap(model, input_img, masks, text_token, text_mask, img_RGB, labs, vis_save_path, text=None, config=config):
    import cv2
    import matplotlib.pyplot as plt
    model.eval()

    output, img_weight, text_weight = model(input_img, masks, text_token, text_mask, mode="test")
    pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    labs = np.squeeze(labs)
    dice_pred_tmp, iou_tmp = compute_foreground_metrics(predict_save, labs)


    original_image = torch.squeeze(input_img, 0).cpu().numpy() * 255
    original_image = original_image.transpose(1, 2, 0).astype(np.uint8)

    if args.test_vis:
        nums = 1
        fig = plt.figure(figsize=(nums,1), dpi=size[0])
        fig.subplots_adjust(wspace=0.01, left=0, right=1, bottom=0,top=1)
        # draw img
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        draw_sub_plot(original_image, fig, nums, 1, mode="rgb")
        original_image_ = original_image.copy()

        ## draw GT
        ground_truth = labs.squeeze()
        original_image_[ground_truth == 1] = green_color
        draw_sub_plot(original_image_, fig, nums, 2)

        # draw pred
        pred = pred_mix(ground_truth, predict_save, original_image)
        draw_sub_plot(pred, fig, nums, 3)
        
        fig.subplots_adjust(wspace=0.01, left=0, right=1, bottom=0, top=1)

        f = plt.gcf() 
        f.savefig(vis_save_path+"_dice"+str(round(dice_pred_tmp,2))+".png")
        f.clear()  

    return dice_pred_tmp, iou_tmp, predict_save


if __name__ == '__main__':
    set_global_seed(int(getattr(config, "seed", 3407)), deterministic=False)
    test_session = args.test_session

    model_type = config.model_name
    if args.model_path:
        model_path = args.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型文件: {model_path}")
    elif test_session is None:
        raise ValueError("请通过 --test_session 或 --model_path 指定模型路径")
    else:
        candidate_model_paths = []
        if hasattr(config, "base_run_dir"):
            candidate_model_paths.append(
                os.path.join(config.base_run_dir, test_session, "models", f"best_model-{model_type}.pth.tar")
            )
            candidate_model_paths.append(
                os.path.join(config.base_run_dir, test_session, "models", f"best_model-{model_type}.pth")
            )
        candidate_model_paths.append(
            os.path.join(".", config.task_name, model_type, test_session, "models", f"best_model-{model_type}.pth.tar")
        )
        candidate_model_paths.append(
            os.path.join(".", config.task_name, model_type, test_session, "models", f"best_model-{model_type}.pth")
        )
        model_path = None
        for path in candidate_model_paths:
            if os.path.exists(path):
                model_path = path
                break
        if model_path is None:
            print("未找到模型文件，尝试的路径如下：")
            for path in candidate_model_paths:
                print(" -", path)
            raise FileNotFoundError("未找到模型文件")
        
    
    if model_type == 'TIRNet':
        config_vit = config.get_ViT_config()
        model = TIRNet(config, config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
    else:
        raise TypeError('Please enter a valid name for the model type')

    if args.profile or args.profile_only:
        params_m, flops_g = profile_params_flops(
            model,
            img_size=int(getattr(config, "img_size", 224)),
            token_len=int(getattr(config, "token_len", 18)),
            n_channels=int(getattr(config, "n_channels", 3)),
        )
        print(f"Params(M): {params_m:.3f}  FLOPs(G) inference: {flops_g:.3f}  FLOPs(G) train~: {flops_g * 3.0:.3f}")
        if args.profile_only:
            raise SystemExit(0)

    from Load_Dataset import ValGenerator, ImageToImage2D
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    if hasattr(config, "base_run_dir"):
        save_path = os.path.join(config.base_run_dir, test_session) + '/'
    else:
        save_path = config.task_name + '/' + model_type + '/' + test_session + '/'
    vis_path = save_path + config.task_name + '_visualize_test/'
    print("vis path is", vis_path)
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
       print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
       model = nn.DataParallel(model)
    load_res = model.load_state_dict(checkpoint['state_dict'], strict=False)
    print('missing keys---> ', load_res.missing_keys)
    print('*'* 100)
    print('unexpected keys---> ', load_res.unexpected_keys)
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_text = read_text(os.path.join(config.test_dataset, 'Test_text.xlsx'))
    test_dataset = ImageToImage2D(config.test_dataset, config.task_name, 
                                    test_text, tf_test, image_size=config.img_size, 
                                    data_name=config.task_name, token_len=config.token_len, 
                                    config=config, mode="val")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    per_image_dice = []
    per_image_iou = []
    global_tp = global_fp = global_fn = 0
    test_num = len(test_loader)
    with tqdm(total=test_num, desc='Test', unit='img', ncols=70, leave=True, dynamic_ncols=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label, text_token, text_mask, text = sampled_batch["image"], sampled_batch["label"], sampled_batch["text_token"], sampled_batch["text_mask"], sampled_batch["text"]
            lab = np.squeeze(test_label.data.numpy())

            test_data, test_label, text_token, text_mask = test_data.cuda(), test_label.cuda(), text_token.cuda(), text_mask.cuda()
            dice_pred_t, iou_pred_t, pred_np = vis_and_save_heatmap(
                model, test_data, test_label, text_token, text_mask, None, lab,
                vis_path + str(names[0]), text=text, config=config)
            per_image_dice.append(dice_pred_t)
            per_image_iou.append(iou_pred_t)

            tp, fp, fn = foreground_counts((pred_np >= 0.5).astype(np.uint8), (lab > 0).astype(np.uint8))
            global_tp += tp
            global_fp += fp
            global_fn += fn

            torch.cuda.empty_cache()
            mean_dice_so_far = float(np.nanmean(per_image_dice))
            mean_iou_so_far = float(np.nanmean(per_image_iou))
            pbar.set_postfix({"mDice": round(mean_dice_so_far, 4), "mIoU": round(mean_iou_so_far, 4)})
            pbar.update()

    mean_dice = float(np.nanmean(per_image_dice))
    mean_iou = float(np.nanmean(per_image_iou))
    global_dice, global_iou = dice_iou_from_counts(global_tp, global_fp, global_fn)

    print("=" * 60)
    print("Foreground metrics (background ignored)")
    print("  Mean per-image Dice:  {:.6f}".format(mean_dice))
    print("  Mean per-image IoU:   {:.6f}".format(mean_iou))
    print("  Global pixel Dice:    {:.6f}".format(global_dice))
    print("  Global pixel IoU:     {:.6f}".format(global_iou))
    print("=" * 60)
