import matplotlib.pyplot as plt
import pandas as pd

# Read CSV file
df = pd.read_csv('models/UNext/log.csv')

# Plotting
plt.figure(figsize=(12, 8))

# Loss
plt.subplot(2, 2, 1)
plt.plot(df['epoch'], df['loss'], label='Training Loss')
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# IoU
plt.subplot(2, 2, 2)
plt.plot(df['epoch'], df['iou'], label='Training IoU')
plt.plot(df['epoch'], df['val_iou'], label='Validation IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.title('Training and Validation IoU')
plt.legend()

# Dice coefficient
plt.subplot(2, 2, 3)
plt.plot(df['epoch'], df['val_dice'], label='Validation Dice')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.title('Validation Dice Coefficient')
plt.legend()

plt.tight_layout()
plt.show()