import argparse
import os
from glob import glob
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import models
from dataset import BUSIDataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize, HorizontalFlip
import time

# Set device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Load configuration from YAML
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Assign loaded configuration to variables
MODEL_NAME = config['model_name']
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
OUT_PATH = config['output_path']

# Argument parsing
parser = argparse.ArgumentParser(description='Validation script for BUSI Dataset')
parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment')
parser.add_argument('--load_model', type=str, help='Number of epochs to train')

args = parser.parse_args()

MODEL_LOAD_PATH = f'models/saved_models/{args.load_model}.pth'
VALIDATION_NAME = f'{args.experiment_name}'


def main():

    #print config information
    for key, value in config.items():
        print(f'{key}: {value}')
    print(f"device: {device}")

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % MODEL_NAME)
    model = MODEL(num_classes=NUM_CLASSES, deep_supervision=DEEP_SUPERVISION, input_channels=INPUT_CHANNELS)
    model = model.to(device)

    # Data loading code
    img_ids = glob(os.path.join(DATA_PATH, 'images', '*' + '.png'))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load(MODEL_LOAD_PATH))
    model.eval()

    val_transform = Compose([
        Resize(INPUT_H, INPUT_W),
        transforms.Normalize(),
    ])

    val_dataset = BUSIDataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(DATA_PATH, 'images'),
        mask_dir=os.path.join(DATA_PATH, 'masks'),
        img_ext='.png',
        mask_ext='.png',
        num_classes=NUM_CLASSES,
        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    count = 0
    for c in range(NUM_CLASSES):
        os.makedirs(os.path.join(OUT_PATH, VALIDATION_NAME, str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)
            model = model.to(device)
            # compute output

            if count<=5:
                start = time.time()
                if DEEP_SUPERVISION:
                    output = model(input)[-1]
                else:
                    output = model(input)
                stop = time.time()

                gput.update(stop-start, input.size(0))

                start = time.time()
                model = model.cpu()
                input = input.cpu()
                output = model(input)
                stop = time.time()

                cput.update(stop-start, input.size(0))
                count=count+1

            iou,dice = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(NUM_CLASSES):
                    cv2.imwrite(os.path.join(OUT_PATH, VALIDATION_NAME, str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    print('CPU: %.4f' %cput.avg)
    print('GPU: %.4f' %gput.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
