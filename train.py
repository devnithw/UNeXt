import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90,Resize
import model
from loss import BCEDiceLoss
from dataset import BUSIDataset
from metrics import iou_score
from utils import AverageMeter, str2bool
from model import UNext

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
DATA_DIR = 'busi'
BATCH_SIZE = 16


# Hyper parameters
NUM_CLASSES = 1
EPOCHS = 100
MODEL_NAME = "unext_dev"
INPUT_CHANNELS = 3


# Pre-processing
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Optimizers
OPTIMIZER = 'Adam'
SCHEDULER = 'CosineAnnealingLR'
LEARNING_RATE = 1e-3
W_DECAY = 1e-4
MOMENTUM = 0.9
NESTEROV = False

# args = parser.parse_args()
def train(train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # Set model to train mode
    model.train()

    # Get batch
    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        # Send data to device
        input = input.to(device)
        target = target.to(device)

        # Compute output
        output = model(input)
        loss = criterion(output, target)
        iou, dice = iou_score(output, target)

        # Compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        # get batch
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)

            # Forward pass
            output = model(input)
            loss = criterion(output, target)
            iou,dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def main():
    # config = vars(parse_args())

    # if config['name'] is None:
    #     if config['deep_supervision']:
    #         config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
    #     else:
    #         config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    
    # os.makedirs('models/%s' % config['name'], exist_ok=True)

    # print('-' * 20)
    # for key in config:
    #     print('%s: %s' % (key, config[key]))
    # print('-' * 20)

    # with open('models/%s/config.yml' % config['name'], 'w') as f:
    #     yaml.dump(config, f)

    # define loss function (criterion)
    criterion = BCEDiceLoss()

    # create model
    model = UNext()
    model = model.to(device)

    # Get model parameters
    params = filter(lambda p: p.requires_grad, model.parameters())

    # Set optimizer and scheduler
    if OPTIMIZER == 'Adam':
        optimizer = optim.Adam(
            params, lr=LEARNING_RATE, weight_decay=W_DECAY)
    elif OPTIMIZER == 'SGD':
        optimizer = optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM,
                              nesterov=NESTEROV, weight_decay=W_DECAY)
    else:
        raise NotImplementedError

    # Set scheduler
    if SCHEDULER == 'CosineAnnealingLR': # Default
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    else:
        raise NotImplementedError

    # Load data
    img_ids = glob(os.path.join(DATA_DIR, 'images', '*' + 'png'))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # Split indices
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    # Apply transforms
    train_transform = Compose([
        RandomRotate90(),
        transforms.Flip(),
        Resize(IMG_HEIGHT, IMG_WIDTH),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(IMG_HEIGHT, IMG_WIDTH),
        transforms.Normalize(),
    ])

    # Create dataset objects
    train_dataset = BUSIDataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(DATA_DIR, 'images'),
        mask_dir=os.path.join(DATA_DIR, 'masks'),
        img_ext='.png',
        mask_ext='.png',
        num_classes=NUM_CLASSES,
        transform=train_transform)
    val_dataset = BUSIDataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(DATA_DIR, 'images'),
        mask_dir=os.path.join(DATA_DIR, 'masks'),
        img_ext='.png',
        mask_ext='.png',
        num_classes=NUM_CLASSES,
        transform=val_transform)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    # Store results
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

    best_iou = 0
    trigger = 0

    # Training Loop
    for epoch in range(EPOCHS):
        print('Epoch [%d/%d]' % (epoch, EPOCHS))

        # Train for one epoch
        train_log = train(train_loader, model, criterion, optimizer)

        # Evaluate on validation set
        val_log = validate(val_loader, model, criterion)

        if SCHEDULER == 'CosineAnnealingLR':
            scheduler.step()
        # elif SCHEDULER == 'ReduceLROnPlateau':
        #     scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        # Append train step details
        log['epoch'].append(epoch)
        log['lr'].append(LEARNING_RATE)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        # Save details to pandas dataframe and csv
        pd.DataFrame(log).to_csv('models/%s/log.csv' % MODEL_NAME, index=False)
        
        trigger += 1

        # Save checkpoint
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' % MODEL_NAME)
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()