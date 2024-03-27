import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import random
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


plots_path = "./plots/"


def iou(pred, target, n_classes = 21):
    """
    Calculate the Intersection over Union (IoU) for predictions.

    Args:
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.
        n_classes (int, optional): Number of classes. Default is 21.

    Returns:
        float: Mean IoU across all classes.
    """
    ious = [] 

    for cls in range(n_classes):
        prediction_idxs = pred == cls
        target_idxs = target == cls

        intersection = (prediction_idxs[target_idxs]).sum()
        union = prediction_idxs.sum() + target_idxs.sum() - intersection

        ious.append(float(intersection) / max(union, 1))
    return ious

def pixel_acc(pred, target):
    """
    Calculate pixel-wise accuracy between predictions and targets.

    Args:
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.

    Returns:
        float: Pixel-wise accuracy.
    """
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total


class CommonTransforms:
    """
    A class that defines common image transformations.

    Args:
        size (tuple): The desired size of the transformed image. Default is (224, 224).

    """

    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, img, mask):
        """
        Applies common image transformations to the input image and mask.

        Args:
            img (PIL.Image.Image): The input image.
            mask (PIL.Image.Image): The input mask.

        Returns:
            tuple: A tuple containing the transformed image and mask.

        """
        # Random horizontal flip with the same decision for both img and mask
        if random.random() > 0.5:
            img = F.hflip(img)
            mask = F.hflip(mask)

        angle = random.randint(-60, 60)
        if random.random() > 0.5 and angle != 0:
            img = F.rotate(img, angle)
            mask = F.rotate(mask, angle)

        img = F.to_tensor(img)
        mask = F.to_tensor(mask)
        mask = mask.to(dtype=torch.int32).long().squeeze()

        
        # Additional transformations can be added here with the same parameters for img and mask

        return img, mask

def plots(trainEpochLoss, valEpochLoss, earlyStop):

    """
    Helper function for creating the plots
    """
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
        
    # Creates plot for training/val loss
    fig1, ax1 = plt.subplots(figsize=((24, 12)))
    epochs = np.arange(1,len(trainEpochLoss)+1,1)
    ax1.plot(epochs, trainEpochLoss, 'r', label="Training Loss")
    ax1.plot(epochs, valEpochLoss, 'g', label="Validation Loss")
    plt.scatter(epochs[earlyStop],valEpochLoss[earlyStop],marker='x', c='g',s=400,label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35 )
    plt.yticks(fontsize=35)
    ax1.set_title('Loss Plots', fontsize=35.0)
    ax1.set_xlabel('Epochs', fontsize=35.0)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=35.0)
    ax1.legend(loc="upper right", fontsize=35.0)
    plt.savefig(plots_path+"loss.png")
    plt.show()
    
    # Creates plot for training/val IoU
    # fig2, ax2 = plt.subplots(figsize=((24, 12)))
    # ax2.plot(epochs, trainEpochIOU, 'r', label="Training IoU")
    # ax2.plot(epochs, valEpochIOU, 'g', label="Validation IoU")
    # plt.scatter(epochs[earlyStop], valEpochIOU[earlyStop], marker='x', c='g', s=400, label='Early Stop Epoch')
    # plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35)
    # plt.yticks(fontsize=35)
    # ax2.set_title('IoU Plots', fontsize=35.0)
    # ax2.set_xlabel('Epochs', fontsize=35.0)
    # ax2.set_ylabel('IoU', fontsize=35.0)
    # ax2.legend(loc="lower right", fontsize=35.0)
    # plt.savefig(plots_path+"iou.png")
    # plt.show()
    
    # Creates plot for training/val accuracy
    # fig3, ax3 = plt.subplots(figsize=((24, 12)))
    # ax3.plot(epochs, trainPixelAcc, 'r', label="Training Pixel Acc")
    # ax3.plot(epochs, valPixelAcc, 'g', label="Validation Pixel Acc")
    # plt.scatter(epochs[earlyStop], valPixelAcc[earlyStop], marker='x', c='g', s=400, label='Early Stop Epoch')
    # plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35)
    # plt.yticks(fontsize=35)
    # ax3.set_title('Pixel Acc Plots', fontsize=35.0)
    # ax3.set_xlabel('Epochs', fontsize=35.0)
    # ax3.set_ylabel('Pixel Acc', fontsize=35.0)
    # ax3.legend(loc="lower right", fontsize=35.0)
    # plt.savefig(plots_path+"pixelacc.png")
    # plt.show()
    
