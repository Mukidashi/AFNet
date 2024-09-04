#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
import torch
import torch.nn as nn
from options import MVS2DOptions, InferOptions
from datasets.road_video_data import RoadVidData
import networks
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import *
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':

    # options = MVS2DOptions()
    options = InferOptions()
    opts = options.parse()
    # opts.cfg = "/configs/DDAD.conf"


    model = networks.MVS2D(opt=opts).cuda()
    pretrained_dict = torch.load(opts.pretrain_model)

    model.load_state_dict(pretrained_dict)
    model.eval()

    dataset = RoadVidData(opts)

    os.makedirs(opts.out_dir,exist_ok=True)

    with torch.no_grad():
        for idx in range(dataset.__len__()):
            datas = dataset.get_data(idx)
            if datas is None:
                continue

            bname = os.path.basename(dataset.imgPathList[idx+1])
            bname,ext = os.path.splitext(bname)

            print("## Processing {0}".format(bname))

            ## Infer
            outputs = model(datas[0],datas[1],datas[2],datas[3],datas[4])

            depth_pred1 = outputs[("depth_pred", 0)]
            depth_pred2 = outputs[("depth_pred_2", 0)]

            depth_pred1 = depth_pred1.cpu().detach().numpy().squeeze()
            depth_pred2 = depth_pred2.cpu().detach().numpy().squeeze() 
            print(depth_pred1.shape,np.min(depth_pred1),np.max(depth_pred1))
            print(depth_pred2.shape,np.min(depth_pred2),np.max(depth_pred2))


            ## Correct Depthmap (shift & resize)
            depth1 = depth_pred1*1000.0
            depImg1 = depth1.astype(np.uint16)
            cv2.imwrite(os.path.join(opts.out_dir,"{0}_dep1.png".format(bname)),depImg1)

            depth2 = depth_pred2*1000.0
            depImg2 = depth2.astype(np.uint16)
            cv2.imwrite(os.path.join(opts.out_dir,"{0}_dep2.png".format(bname)),depImg2)


            


