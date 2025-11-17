import copy
import torch
import torch.nn as nn
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import pandas as pd

from core.trainmodel.model import CNN

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

def run(args):

    time_list = []
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        start = time.time()

        if model_str == "CNN":
            if "MNIST" in args.dataset:
                args.model = CNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
        
        else:
            raise NotImplementedError


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('-dev', "--device", type=str, default="cuda",choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-e', "--epochs", type=int, default=100)
    parser.add_argument('-pv', "--prev", type=int, default=0, help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1, help="Running times")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    
    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)

    run(args)
