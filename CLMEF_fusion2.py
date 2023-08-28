import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
from CLMEF_net_gray import CLMEFNet
import os
from PIL import Image, ImageFile, ImageFilter
from tqdm import tqdm
import cv2


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def load_img(img_path):
    # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = Image.open(img_path, mode='L')
    return _tensor(img).unsqueeze(0)


_tensor = transforms.ToTensor()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description='model save and load')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--ue_path', type=str, default='./MEFB_dataset_gray/under-exposed/')
parser.add_argument('--oe_path', type=str, default='./MEFB_dataset_gray/over-exposed/')
parser.add_argument('--model_path', type=str,
                    default='./train_result.pth')
parser.add_argument('--save_path', type=str, default='./clif_/')

args = parser.parse_args()

args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")

model = CLMEFNet().to(device)

state_dict = torch.load(args.model_path, map_location="cuda:0")['model']

if len(args.gpus) > 1:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(state_dict)


def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


mkdir(args.save_path)

for name in tqdm(os.listdir(args.ue_path)):
    img_path_ue = args.ue_path + name
    img_path_oe = args.oe_path + name

    model.eval()

    img_ue = np.array(Image.open(img_path_ue))
    img_oe = np.array(Image.open(img_path_oe))

    img_ue = _tensor(img_ue).unsqueeze(0).to(device)
    img_oe = _tensor(img_oe).unsqueeze(0).to(device)

    img_ue_feats = model.encoder(img_ue.float())
    img_oe_feats = model.encoder(img_oe.float())

    fusion_feats = (img_ue_feats + img_oe_feats) / 2

    fusion_img = model.decoder(fusion_feats).squeeze(0).detach().cpu().numpy()

    fusion_img = normalization(fusion_img)

    fusion_img = fusion_img.squeeze()

    # cv2.imwrite(args.save_path+name,fusion_img*255)

    fusion_img_array = Image.fromarray((fusion_img * 255).astype(np.uint8))
    fusion_img_array.save(args.save_path + name)

    # plt.figure()
    # plt.imshow(fusion_img)
    # plt.show()

    # print("")
