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
import model
from dataset import BUSIDataset
from metrics import iou_score, f1_score
from utils import AverageMeter
from albumentations import Resize
import time
from thop import profile  # Ensure 'thop' library is installed

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configuration from YAML
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Argument parsing
parser = argparse.ArgumentParser(description='Validation script for BUSI Dataset')
parser.add_argument('--name', type=str, required=True, help='Name of the experiment')
parser.add_argument('--load_model', type=str, required=True, help='Path to load the model')
parser.add_argument('--device', type=str, default=device, help='Device to use')
args = parser.parse_args()

# Assign loaded configuration to variables
MODEL_NAME = config['model_name']
MODEL = model.UNext
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

    # create model
    print("=> creating model %s" % MODEL_NAME)
    model = MODEL(num_classes=NUM_CLASSES, deep_supervision=DEEP_SUPERVISION, input_channels=INPUT_CHANNELS)
    model = model.to(device)

    # Load model weights
    model.load_state_dict(torch.load(MODEL_LOAD_PATH))
    model.eval()


    # Calculate number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate GFLOPs
    dummy_input = torch.randn(1, INPUT_CHANNELS, INPUT_H, INPUT_W).to(device)
    flops, _ = profile(model, inputs=(dummy_input,))

    # Load dataset
    img_ids = glob(os.path.join(DATA_PATH, 'images', '*.png'))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)


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
    f1_avg_meter = AverageMeter()  # To store F1 score for pixels
    time_meter = AverageMeter()  # To measure inference time per image

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)

            # Time inference
            start_time = time.time()
            output = model(input)[-1] if DEEP_SUPERVISION else model(input)
            end_time = time.time()

            time_meter.update((end_time - start_time) / input.size(0))  # Time per image

            iou, dice = iou_score(output, target)
            f1 = f1_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            f1_avg_meter.update(f1, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            for i in range(len(output)):
                for c in range(NUM_CLASSES):
                    cv2.imwrite(os.path.join(OUT_PATH, VALIDATION_NAME, str(c), meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))
    
    print(f"Number of model parameters: {num_params/1e6:.4f} M")
    print(f"GFLOPs: {flops / 1e9:.3f}")
    print(f'IoU: {iou_avg_meter.avg:.4f}')
    print(f'Dice: {dice_avg_meter.avg:.4f}')
    print(f'Average F1 Score per Pixel: {f1_avg_meter.avg:.4f}')
    print(f'Average Inference Time per Image: {time_meter.avg*1e3:.4f} ms')

    # output a csv file with the results
    with open(os.path.join(OUT_PATH, VALIDATION_NAME, 'results.csv'), 'w') as f:
        f.write(f'Parameter Count,{num_params/1e6:.4f} M\n')
        f.write(f'GFLOPs,{flops/1e9:.3f}\n')
        f.write(f'IoU,{iou_avg_meter.avg*100:.4f}\n')
        f.write(f'Dice,{dice_avg_meter.avg*100:.4f}\n')
        f.write(f'Average F1 Score per Pixel,{f1_avg_meter.avg*100:.4f}\n')
        f.write(f'Average Inference Time per Image,{time_meter.avg*1e3:.4f} ms\n')

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
