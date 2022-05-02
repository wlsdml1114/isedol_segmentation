
import os
import numpy as np
import torch
import math
import cv2
import sys
import time
import argparse
from PIL import Image
import torchvision
from torch import nn, Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from typing import List, Tuple, Dict, Optional
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from collections import Counter
from tqdm import tqdm
import wandb


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(nn.Module):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.to_tensor(image)
        return image, target

class CustomDataset(object):
    def __init__(self, root, transforms, num=0):
        self.root = root
        self.transforms = transforms
        self.name = name
        self.files = os.listdir(os.path.join(root,'Masks'))

    def __getitem__(self, idx):
        
        thrshold = 150
        img = cv2.imread(os.path.join(self.root,'Images',self.files[idx][:-3]+'jpg'))
        img = img.astype(np.float32)
        img = img/255

        mask = cv2.imread(os.path.join(self.root,'Masks',self.files[idx]))
        mask = mask[:,:,0]
        mask[mask <thrshold] = 0
        mask[mask >=thrshold] = 1

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmin == xmax:
                xmax = xmax+1
            if ymin == ymax :
                ymax = ymax+1
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = torch.tensor(100)
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.files)

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    return Compose(transforms)


def train_one_epoch(model, optimizer, data_loader, device, epoch, wandb, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        target = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if target[0]['boxes'].size()[0]==0 :
            continue
        
        loss_dict = model(images, target)

        wandb.log(loss_dict)
        
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


parser = argparse.ArgumentParser(description='mask rcnn')    
parser.add_argument('--name', required = True, help='folder name')
args = parser.parse_args()

#set root
name = args.name
data_path = os.path.join('/home/jini1114/git/data/augmentation',name)
model_path = '/home/jini1114/git/data/model/'

wandb.init(project="segmentation", entity="engui")
wandb.run.name = name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
num_classes = 2
dataset = CustomDataset(data_path, get_transform(train=True))


batch_size = 16
lr = 0.005
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=2,
    collate_fn=utils.collate_fn)

model = get_model_instance_segmentation(num_classes)

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

num_epochs = 2

wandb.config = {
  "learning_rate": lr,
  "epochs": num_epochs,
  "batch_size": batch_size
}

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, wandb, print_freq=1)
    lr_scheduler.step()

torch.save(model,os.path.join(model_path,'%s.pt'%(name)))
