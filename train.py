import os
from collections import OrderedDict
import glob
import torch
from torchvision import transforms
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import models
import yaml
import losses
from dataset import BUSIDataset
from metrics import iou_score
from utils import AverageMeter, str2bool
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from albumentations import RandomRotate90, Resize, HorizontalFlip

# Load configuration from YAML
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Assign loaded configuration to variables
MODEL_NAME = config['model_name']
LOSS_FUNCTION = losses.BCEDiceLoss()
MODEL = models.UNext
BATCH_SIZE = config['batch_size']
NUM_WORKERS = config['num_workers']
NUM_CLASSES = config['num_classes']
DEEP_SUPERVISION = config['deep_supervision']
INPUT_CHANNELS = config['input_channels']
OPTIMIZER = config['optimizer']
LEARNING_RATE = config['learning_rate']
WEIGHT_DECAY = config['weight_decay']
MOMENTUM = config['momentum']
SCHEDULER = config['scheduler']
EPOCHS = config['epochs']
MIN_LEARNING_RATE = float(config['min_learning_rate'])
SCHEDULER_FACTOR = config['scheduler_factor']
SCHEDULER_PATIENCE = config['scheduler_patience']
SCHEDULER_MILESTONES = config['scheduler_milestones']
SCHEDULER_GAMMA = config['scheduler_gamma']
DATA_PATH = config['data_path']
EARLY_STOPPING = config['early_stopping']
INPUT_W = config['input_w']
INPUT_H = config['input_h']
TRANSFORM_FLIP_PROBABILITY = config['transform_flip_probability']

MODEL_SAVE_PATH = f'models/{MODEL_NAME}/model2.pth'


# check the model name and losses
'''
ARCH_NAMES = models.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# args = parser.parse_args()
def train(train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.to(device)
        target = target.to(device)

        # compute output
        if DEEP_SUPERVISION:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou,dice = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou,dice = iou_score(output, target)

        # compute gradient and do optimizing step
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

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)

            # compute output
            if DEEP_SUPERVISION:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou,dice = iou_score(outputs[-1], target)
            else:
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

    
    os.makedirs('models/%s' % MODEL_NAME, exist_ok=True)

    #print config
    '''
    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)
    '''


    criterion = LOSS_FUNCTION.to(device)

    cudnn.benchmark = True

    # create model    
    model = MODEL(num_classes=NUM_CLASSES, deep_supervision=DEEP_SUPERVISION, input_channels=INPUT_CHANNELS)
    model = model.to(device)

    params = filter(lambda p: p.requires_grad, model.parameters())
    
    if OPTIMIZER == 'Adam':
        optimizer = optim.Adam(params=params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == 'SGD':
        optimizer = optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    else:
        raise NotImplementedError
    

    if SCHEDULER == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LEARNING_RATE)
    elif SCHEDULER == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE,
                                                   verbose=1, min_lr=MIN_LEARNING_RATE)
    elif SCHEDULER == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in SCHEDULER_MILESTONES.split(',')], gamma=SCHEDULER_GAMMA)
    elif SCHEDULER == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    img_ids = glob.glob(os.path.join(DATA_PATH, 'images', '*' + '.png'))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    train_transform = Compose([
        RandomRotate90(),
        HorizontalFlip(p=TRANSFORM_FLIP_PROBABILITY),
        Resize(INPUT_H, INPUT_W),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(INPUT_H, INPUT_W),
        transforms.Normalize(),
    ])


    train_dataset = BUSIDataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(DATA_PATH, 'images'),
        mask_dir=os.path.join(DATA_PATH, 'masks'),
        img_ext='.png',
        mask_ext='.png',
        num_classes=NUM_CLASSES,
        transform=train_transform)

    val_dataset = BUSIDataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(DATA_PATH, 'images'),
        mask_dir=os.path.join(DATA_PATH, 'masks'),
        img_ext='.png',
        mask_ext='.png',
        num_classes=NUM_CLASSES,
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False)

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
    for epoch in range(1,EPOCHS+1):
        print('Epoch [%d/%d]' % (epoch, EPOCHS))

        # train for one epoch
        train_log = train(train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(val_loader, model, criterion)

        if SCHEDULER == 'CosineAnnealingLR':
            scheduler.step()
        elif SCHEDULER == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(LEARNING_RATE)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv('models/%s/log_with_flip.csv' %
                                 MODEL_NAME, index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if EARLY_STOPPING >= 0 and trigger >= EARLY_STOPPING:
            print("=> early stopping")
            break


if __name__ == '__main__':
    main()
