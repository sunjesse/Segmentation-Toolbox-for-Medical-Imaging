# System libs
import os
import time
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import nibabel as nib
# Our libs
from data.augmentations import ComposeTest, PaddingCenterCropTest
from data.test_loader import AC17Test as AC17
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
import lib.utils.data as torchdata
import cv2
from tqdm import tqdm
from loss import Loss
import random
from PIL import Image, ImageOps
from skimage import transform

def round_num(x):
    return int(x) + 1 if (x-int(x)) >= 0.5 else int(x)

def undo_crop(img, pred): #img is original image:w
    pred = Image.fromarray(pred.astype(np.uint8), mode="L")
    img = Image.fromarray(img, mode="I")
    w, h = img.size
    tw, th = pred.size
    if w >= tw and h >= th:  # crop a center patch
        x1 = int(round_num((w - tw) / 2.)) 
        y1 = int(round_num((h - th) / 2.)) 
        rem_x = (w - tw) % 2
        rem_y = (h - th) % 2
        border = [x1, y1, x1-rem_x, y1-rem_y]
        return np.array(ImageOps.expand(pred, tuple(border), fill=0))
         
    else:  # pad zeros and do center crop
        pad_h = max(th - h, 0)
        pad_w = max(tw - w, 0)
        b = [pad_w//2, pad_h//2, pad_w//2 + w, pad_h//2+h]
        
        if pad_w == 0:
            b[2] = tw
        if pad_h == 0:
            b[3] = th

        pred = pred.crop(b)
        x1 = max(int(round_num((w - tw) / 2.)), 0)
        y1 = max(int(round_num((h - th) / 2.)), 0)
        rem_w = (w - tw) % 2 if (w-tw) >= 0 else 0
        rem_h = (h - th) % 2 if (h-th) >= 0 else 0
        border = [x1, y1, x1-rem_w, y1-rem_h]
        return np.array(ImageOps.expand(pred, tuple(border), fill=0))
    
def resample_to_orig(data, pred):
    #uncrop
    p_stack = np.zeros_like(data["post_scale"])
    for i in range(data["orig"].shape[-1]):
        p = undo_crop(data["post_scale"][:,:,i], pred[:,:,i])
        p_stack[:,:,i] = p
    #rescale
    p_stack = transform.resize(p_stack,
                          data['orig'].shape,
                          order=0,
                          preserve_range=True,
                          mode='constant')

    assert data["orig"].shape == p_stack.shape, "Error while resampling"
    return p_stack

def visualize_result(data, pred, args):
    (img, info) = data
    
    #normalize image to [0, 1] first.
    img = (img - img.min())/(img.max()-img.min())
    img = (img * 255).astype(np.uint8) #Then scale it up to [0, 255] to get the final image.
    pred_img = (pred * 85).astype(np.uint8)

    #heat = get_heatmap(LRP)
    im_vis = np.concatenate((img, pred_img), axis=1).astype(np.uint8)
    img_name = info.split('/')[-1] + '.png'
    cv2.imwrite(os.path.join(args.result,
                img_name), im_vis)


def save_as_nifti(pred, path, name):
    img = nib.Nifti1Image(pred, np.eye(4))
    img.to_filename(os.path.join(path, str(name)+'.nii.gz'))
    print("Saved " + str(name) + "!")

def evaluate(sm1, sm2, sm3, sm4, sm5, loader_val, args):
    time_meter = AverageMeter()

    sm1.eval()
    sm2.eval()
    sm3.eval()
    sm4.eval()
    sm5.eval()

    pbar = tqdm(total=len(loader_val))
    for batch_data in loader_val:
        batch_data = batch_data[0]
        batch_data["image"] = batch_data["image"].unsqueeze(0).cuda()
        torch.cuda.synchronize()
        pred_volume = np.zeros_like(batch_data["image"][0][0].cpu())
        for z in range(batch_data["image"].shape[-1]):
            slice_data = {"image":batch_data["image"][:,:,:,:,z]}
            tic = time.perf_counter()
            with torch.no_grad():
                feed_dict = batch_data.copy()

                # forward pass
                p1 = sm1(slice_data, epoch=0, segSize=True)
                p2 = sm2(slice_data, epoch=0, segSize=True)
                p3 = sm3(slice_data, epoch=0, segSize=True)
                p4 = sm4(slice_data, epoch=0, segSize=True)
                p5 = sm5(slice_data, epoch=0, segSize=True)
                scores = (p1+p2+p3+p4+p5)/5 

                _, pred = torch.max(scores, dim=1)
                pred = as_numpy(pred.squeeze(0).cpu())
                pred_volume[:,:,z] = pred
                
            time_meter.update(time.perf_counter() - tic)
        pv_resized = resample_to_orig(batch_data, pred_volume)
        save_as_nifti(pv_resized, args.save_test_path, batch_data["name"])
        if args.visualize:
            for z in range(batch_data['orig'].shape[-1]): 
                visualize_result(
                        (batch_data['orig'][:,:,z], batch_data["name"]+str(z)),
                        pv_resized[:,:, z], args)

        torch.cuda.synchronize()

        pbar.update(1)

def main(args):
    torch.cuda.set_device(args.gpu)

    # Network Builders
    builder = ModelBuilder()

    unet1 = builder.build_unet(num_class=args.num_class,
        arch=args.arch_unet,
        weights=args.weights_unet1)
    unet2 = builder.build_unet(num_class=args.num_class,
        arch=args.arch_unet,
        weights=args.weights_unet2)

    unet3 = builder.build_unet(num_class=args.num_class,
        arch=args.arch_unet,
        weights=args.weights_unet3)

    unet4 = builder.build_unet(num_class=args.num_class,
        arch=args.arch_unet,
        weights=args.weights_unet4)

    unet5 = builder.build_unet(num_class=args.num_class,
        arch=args.arch_unet,
        weights=args.weights_unet5)
    
    crit = Loss()
    sm1 = SegmentationModule(unet1, crit)
    sm2 = SegmentationModule(unet2, crit)
    sm3 = SegmentationModule(unet3, crit)
    sm4 = SegmentationModule(unet4, crit)
    sm5 = SegmentationModule(unet5, crit)

    test_augs = ComposeTest([PaddingCenterCropTest(384)])

    ac17 = AC17(
            root=args.data_root,
            augmentations=test_augs,
            img_norm=args.img_norm)
    
    loader_val = data.DataLoader(
        ac17,
        batch_size=1,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    sm1.cuda()
    sm2.cuda()
    sm3.cuda()
    sm4.cuda()
    sm5.cuda()

    # Main loop
    evaluate(sm1, sm2, sm3, sm4, sm5, loader_val, args)

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    DATA_ROOT = os.getenv('DATA_ROOT', '')

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', required=True,
                        help="a name for identifying the model to load")
    parser.add_argument('--arch_unet', default='unet',
                        help='UNet architecture?')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=4, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize. current only supports 1')
    
    parser.add_argument('--checkpoint1', type=str, required=True, help="checkpoint path")
    parser.add_argument('--checkpoint2', type=str, required=True, help="checkpoint path")
    parser.add_argument('--checkpoint3', type=str, required=True, help="checkpoint path")
    parser.add_argument('--checkpoint4', type=str, required=True, help="checkpoint path")
    parser.add_argument('--checkpoint5', type=str, required=True, help="checkpoint path")

    parser.add_argument('--test-split', type=str, default='val')
    parser.add_argument('--data-root', type=str, default=DATA_ROOT)
    parser.add_argument('--img-norm', default=True, action='store_true', help="normalize img value to [0, 1]")
    parser.add_argument('--k_split', default=1)
    
    # Misc argument
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--visualize', default=True, action='store_true',
                        help='output visualization?')
    parser.add_argument('--result', default='./result',
                        help='folder to output visualization results')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu id for evaluation')
    parser.add_argument('--show_SRmap', default=True, type=bool,
                        help='Show the saliency relevance mapping')
    parser.add_argument('--save_test_path', default='./test_files')

    args = parser.parse_args()
    args.arch_encoder = args.arch_encoder.lower()
    args.arch_decoder = args.arch_decoder.lower()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # absolute paths of model weights
    args.weights_unet1 = args.checkpoint1
    args.weights_unet2 = args.checkpoint2
    args.weights_unet3 = args.checkpoint3 
    args.weights_unet4 = args.checkpoint4 
    args.weights_unet5 = args.checkpoint5

    assert os.path.exists(args.weights_unet1), 'checkpoint1 does not exist!'
    assert os.path.exists(args.weights_unet2), 'checkpoint2 does not exist!'
    assert os.path.exists(args.weights_unet3), 'checkpoint3 does not exist!'
    assert os.path.exists(args.weights_unet4), 'checkpoint4 does not exist!'
    assert os.path.exists(args.weights_unet5), 'checkpoint5 does not exist!'

    args.result = os.path.join(args.result, args.id)
    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    main(args)
