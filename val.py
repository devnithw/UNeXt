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
import model
import argparse
from dataset import BUSIDataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize, HorizontalFlip
import time
from loss import BCEDiceLoss
from model import UNext

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

# Argument parsing
parser = argparse.ArgumentParser(description='Validation script for BUSI Dataset')
parser.add_argument('--name', type=str, required=True, help='Name of the experiment')
parser.add_argument('--load_model', required=True, type=str, help='Name of the model')
parser.add_argument('--device', type=str, default=device, help='Model name to use for training')

args = parser.parse_args()

# Assign loaded configuration to variables
MODEL = model.UNext
device = args.device
BATCH_SIZE = config['batch_size']
NUM_WORKERS = config['num_workers']
NUM_CLASSES = config['num_classes']
DEEP_SUPERVISION = config['deep_supervision']
INPUT_CHANNELS = config['input_channels']
DATA_PATH = config['data_path']
INPUT_W = config['input_w']
INPUT_H = config['input_h']
OUT_PATH = config['output_path']

MODEL_LOAD_PATH = f'models/saved_models/model_{args.load_model}.pth'
VALIDATION_NAME = f'{args.name}'

def main():

    cudnn.benchmark = True

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
        os.makedirs(os.path.join(OUT_PATH, VALIDATION_NAME,str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)
            model = model.to(device)
            # compute output
            output = model(input)


            iou,dice = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0

            for i in range(len(output)):
                for c in range(NUM_CLASSES):
                    cv2.imwrite(os.path.join(OUT_PATH, VALIDATION_NAME, str(c), meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
