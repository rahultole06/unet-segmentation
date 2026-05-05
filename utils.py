import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

SMOOTH = 1e-6
CLASSES = 4

def check_accuracy(loader, model, weights, device):
    model.eval()
    
    total_intersection = torch.zeros(CLASSES, device=device)
    total_union_sum = torch.zeros(CLASSES, device=device)

    with torch.no_grad():
        for image, label in loader:
            image = image.to(device)
            label = label.to(device)

            preds = model(image)
            preds = torch.argmax(preds, dim=1)

            pixel_mask = get_pixel_mask(label)
            
            for c in range(CLASSES):
                preds_valid = (preds == c) & pixel_mask
                label_valid = (label == c) & pixel_mask

                total_intersection[c] += (preds_valid & label_valid).sum()
                total_union_sum[c] += preds_valid.sum() + label_valid.sum()
            
    dice_scores = (2.0 * total_intersection + SMOOTH) / (total_union_sum + SMOOTH)
    dice_scores = dice_scores.cpu().numpy()
    
    accuracy = round(np.average(dice_scores, weights=weights), 4)
    
    print(f"Class-wise Dice: {dice_scores}")
    print(f"Avg Dice Accuracy: {accuracy}")

    return accuracy

def get_pixel_mask(label):
    valid_pixel_mask = label != -100
    return valid_pixel_mask

def model_checkpoint(mode, model, optimizer, save_file=None, epoch=0, checkpoint=None):
        if mode == "load":
            if checkpoint:
                print(f"Resuming from checkpoint {checkpoint}...")
                checkpoint = torch.load(checkpoint, weights_only=False)

                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                print("Training fresh model...")
        elif mode == "save":
            print("Saving checkpoint")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            cur_save_file = f"{save_file}_epoch{epoch}.pth"
            torch.save(checkpoint, cur_save_file)

def get_class_weights(dataloader):
    class_count = torch.zeros(CLASSES, dtype=torch.float32)

    for _, label in dataloader:
        valid_labels = label[label != -100]
        counts = torch.bincount(valid_labels, minlength=4)
        class_count += counts
    
    total_pixels = class_count.sum()
    avg_ppc = total_pixels / CLASSES
    weights = avg_ppc / (class_count + SMOOTH)
    return weights.float()

def plot_img(image, preds, label):
    color_mask = np.array([
        [139, 69, 19],
        [0, 255, 0],
        [238, 232, 170],
        [255, 0, 0]
    ], dtype=np.uint8)

    preds = color_mask[preds]
    valid_pixel_mask = get_pixel_mask(label)
    alpha_channel = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
    alpha_channel[valid_pixel_mask] = 100
    rgba_mask = np.concatenate([preds, alpha_channel], axis=-1)

    plt.figure(figsize=(16, 9))

    plt.imshow(image, cmap="gray")
    plt.imshow(rgba_mask)

    plt.title("Prediction")
    plt.axis("off")

    labels = {0: "Soil", 1: "Bedrock", 2: "Sand", 3: "Big Rock"}
    patches = [mpatches.Patch(color=color_mask[i]/255.0, label=labels[i]) for i in range(4)]

    plt.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

    plt.tight_layout()
    plt.show()