"""
Author: Rahul Tole
Training loop
"""

import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations import ToTensorV2
from torch.utils.data import DataLoader
from dataset import MarsDataset
from model import UNET
from utils import check_accuracy, model_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# hyperparams
LEARNING_RATE = 3e-5
IN_CHANNELS = 1
OUT_CHANNELS = 4
FEATURES = [64, 128, 256, 512]
WEIGHTS = [0.6713,  0.5020,  2.0772, 27.1316]
BATCH_SIZE = 3
NUM_EPOCHS = 50
NUM_WORKERS = 4
ACCUMULATION_STEPS = 8

# dataset info
IMAGE_DIR = "dataset/msl/images/edr"
TR_LABEL_DIR = "dataset/msl/labels/masked_train"
VAL_LABEL_DIR = "dataset/msl/labels/test/masked-gold-min2-100agree"
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024

def train_model(save_file, checkpoint=None):
    print(f"Using: {DEVICE}")

    # Transformations applied
    train_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.2291],
            std=[0.1124]
        ),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Normalize(
            mean=[0.2291],
            std=[0.1124]
        ),
        ToTensorV2()
    ])

    # train set
    train_dataset = MarsDataset(IMAGE_DIR, TR_LABEL_DIR, train_transform, "train")
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # validation set
    val_dataset = MarsDataset(IMAGE_DIR, VAL_LABEL_DIR, val_transform, "validate")
    val_loader = DataLoader(val_dataset, BATCH_SIZE, num_workers=NUM_WORKERS)

    model = UNET(IN_CHANNELS, OUT_CHANNELS, FEATURES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(WEIGHTS).to(DEVICE))
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)

    optimizer.zero_grad()

    # loads model & checks accuracy before training
    model_checkpoint("load", model, optimizer, checkpoint=checkpoint)
    check_accuracy(val_loader, model, WEIGHTS, DEVICE)

    for epoch in range(NUM_EPOCHS):
        print(f"Training Epoch: {epoch+1}/{NUM_EPOCHS}")

        model.train()
        running_loss = 0.0

        for batch_idx, (image, label) in enumerate(train_loader):
            image, label = image.to(DEVICE), label.to(DEVICE)

            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                out = model(image)
                loss = loss_fn(out, label)
                loss = loss / ACCUMULATION_STEPS # accounts for accumulation steps
            
            loss.backward()
            running_loss += loss.item() * ACCUMULATION_STEPS

            # Adjusts weights after accum steps or if dataset finished
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            if batch_idx % 100 == 99:
                print(f"Batch: {batch_idx + 1}\n Loss: {running_loss/100:.4f}")
                running_loss = 0.0

        accuracy = check_accuracy(val_loader, model, WEIGHTS, DEVICE)
    
        model_checkpoint("save", model, optimizer, save_file, epoch) # saves model
    accuracy = check_accuracy(val_loader, model, WEIGHTS, DEVICE)

    print("Training complete")

if __name__ == '__main__':
    train_model("checkpoints/v10/save", "checkpoints/v9/save_epoch1.pth")