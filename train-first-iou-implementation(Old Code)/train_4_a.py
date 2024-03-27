from basic_fcn import *
import time
from torch.utils.data import DataLoader
import torch
import gc
import voc
import torchvision.transforms as standard_transforms
import util
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()
    
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

        # Random resized crop with the same parameters for both img and mask
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.08, 1.0), ratio=(3./4., 4./3.))
        img = F.resized_crop(img, i, j, h, w, self.size, InterpolationMode.BILINEAR)
        mask = F.resized_crop(mask, i, j, h, w, self.size, InterpolationMode.NEAREST)
        img = F.to_tensor(img)
        mask = F.to_tensor(mask)
        mask = mask.to(dtype=torch.int32).long().squeeze()
        
        # Additional transformations can be added here with the same parameters for img and mask

        return img, mask


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases



def getClassWeights(dataset):
    """
    Calculate the class weights for a given dataset to handle class imbalance.

    Parameters:
    dataset (torch.utils.data.Dataset): The dataset containing the samples and labels.

    Returns:
    torch.Tensor: The class weights for each class in the dataset, inversely proportional to class frequencies.
    """
    class_counts = torch.zeros(21, dtype=torch.long)
    for _, label in dataset:
        label = label.long()  # Ensure label is of type torch.long for bincount
        class_counts += torch.bincount(label.view(-1), minlength=21)
    
    # Avoid division by zero for classes not present in the dataset
    class_counts[class_counts == 0] = 1
    
    total_samples = class_counts.sum().float()
    class_weights = total_samples / class_counts
    
    # Normalize weights to sum to 1, if desired (optional, depending on use case)
    class_weights /= class_weights.sum()
    
    return class_weights

    

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

target_transform = MaskToTensor()

# Load the dataset
train_dataset =voc.VOC('train', transform=input_transform, target_transform=target_transform)
val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform)
test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform)

# Make the dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size= 16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= 16, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size= 16, shuffle=False)

epochs = 40 # Epochs to train for

n_class = 21 # Pascal VOC has 21 classes
class_weights = getClassWeights(train_dataset) # Get the class weights for the dataset
fcn_model = FCN(n_class=n_class) # Create the model
fcn_model.apply(init_weights) # Initialize the weights of the model using Xavier initialization

device = torch.device("cuda")

# Changed learning rate from 1e-4 to 1e-2 in baseline model
optimizer = torch.optim.Adam(fcn_model.parameters(), lr=1e-2, weight_decay=1e-5)

# Cosine annealing learning rate scheduler
T_max = 10 
eta_min = 0.001  
scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

# Use cross entropy loss with the class weights
criterion =  torch.nn.CrossEntropyLoss()

# Transfer the model to the device (GPU if available)
print(torch.cuda.is_available())
fcn_model =  fcn_model.to(device)
criterion = criterion.to(device)

# Save the best model
save_path = 'models/4a_best_model.pth'


def train():
    """
    Train a deep learning model using mini-batches.

    - Perform forward propagation in each epoch.
    - Compute loss and conduct backpropagation.
    - Update model weights.
    - Evaluate model on validation set for mIoU score.
    - Save model state if mIoU score improves.
    - Implement early stopping if necessary.

    Returns:
        None.
    """
    # Keep track of the best IoU score and early stopping
    best_iou_score, best_pixel_acc = 0.0, 0.0
    early_stop, early_stop_epoch = 0, 10

    # Keep track of the loss, IoU, and pixel accuracy for each epoch for plotting
    trainEpochLoss, trainEpochIOU, trainEpochPixAcc = [], [], []
    valEpochLoss, valEpochIOU, valEpochPixAcc = [], [], []

    # Early Stop counter    
    early_stop = 0

    for epoch in range(epochs):
        ts = time.time() # Start time for the epoch
        
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad() # Initialize the gradients to zero

            # Put the inputs and labels on the device
            inputs =  inputs.to(device)
            labels =   labels.to(device)

            # Forward pass, forwards automatically put model in device
            outputs =  fcn_model.forward(inputs) 
            # Calculate the loss
            loss = criterion(outputs,labels) 
            
            # backpropagate
            loss.backward()
            
            # update the weights
            optimizer.step()
            
            # Update the learning rate
            scheduler.step()

            # Print the loss every 10 iterations (For debugging purposes/checking if the model is learning)
            if i % 10 == 0:
                print("epoch {}, iter {}, loss: {}".format(epoch + 1, i + 1, loss.item()))
        
        
        
        print("Finish epoch {}, time elapsed {}".format(epoch + 1, time.time() - ts))
        print("_"*50)


        # PLOTS
        # # Validate the model on the validation set and test set
        # val_metrics = val(epoch)
        # test_metrics = modelTest()
        
        # # Append the loss, IoU, and pixel accuracy for each epoch for plotting
        # trainEpochLoss.append(test_metrics[1])
        # valEpochLoss.append(val_metrics[1])
        
        # trainEpochIOU.append(test_metrics[0])
        # valEpochIOU.append(val_metrics[0])
        
        # trainEpochPixAcc.append(test_metrics[2])
        # valEpochPixAcc.append(test_metrics[2])
        
        
        
        # Getting the Validation IoU score
        current_miou_score, _, current_pixel_acc = val(epoch)

        # Saving the best model based on the validation IoU score
        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            # save the best model
            torch.save(fcn_model.state_dict(), save_path)
            # Print statement to show that the model has been saved
            print("*"*50)
            print(f"Saved new best model with mIoU: {current_miou_score} at epoch {epoch + 1}")
            print("*"*50)
        if current_pixel_acc > best_pixel_acc:
            # keep track of the best pixel accuracy
            best_pixel_acc = current_pixel_acc
        
        # Early stopping
        if current_miou_score < best_iou_score:
            early_stop += 1  
        else:
            early_stop = 0
        if early_stop > early_stop_epoch:
            print("Early stopping at epoch: ", epoch + 1)
            break
    
    # Print statement for training tracking
    print("%"*50)
    print("Training finished")
    print(f"Best mIoU: {best_iou_score} at epoch {epoch + 1}")
    print(f"Best pixel acc: {best_pixel_acc} at epoch {epoch + 1}")   
    print("%"*50)
    
    # Plot the training and validation loss, IoU, and pixel accuracy
    # util.plots(trainEpochLoss, trainEpochIOU, valEpochLoss, valEpochIOU, trainEpochPixAcc, valEpochPixAcc, earlyStop)
            
    
def val(epoch):
    """
    Validate the deep learning model on a validation dataset.

    - Set model to evaluation mode.
    - Disable gradient calculations.
    - Iterate over validation data loader:
        - Perform forward pass to get outputs.
        - Compute loss and accumulate it.
        - Calculate and accumulate mean Intersection over Union (IoU) scores and pixel accuracy.
    - Print average loss, IoU, and pixel accuracy for the epoch.
    - Switch model back to training mode.

    Args:
        epoch (int): The current epoch number.

    Returns:
        tuple: Mean IoU score and mean loss for this validation epoch.
    """
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for _, (inputs, label) in enumerate(val_loader):
            inputs = inputs.to(device)
            label = label.to(device)
            
            cur_output = fcn_model.forward(inputs)
            cur_loss = criterion(cur_output, label).item()
            cur_IoU = util.iou(cur_output, label)
            cur_pix_acc = util.pixel_acc(cur_output,label)

            losses.append(cur_loss)
            mean_iou_scores.append(cur_IoU)
            accuracy.append(cur_pix_acc)
    
    avg_loss = np.mean(losses)
    avg_IoU = np.mean(mean_iou_scores)
    avg_acc = np.mean(accuracy)
    
    print("Validation results")
    print(f"Loss at epoch: {epoch} is {avg_loss}")
    print(f"IoU at epoch: {epoch} is {avg_IoU}")
    print(f"Pixel acc at epoch: {epoch} is {avg_acc}")

    fcn_model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return (avg_IoU, avg_loss, avg_acc)



def modelTest():
    """
    Test the deep learning model using a test dataset.

    - Load the model with the best weights.
    - Set the model to evaluation mode.
    - Iterate over the test data loader:
        - Perform forward pass and compute loss.
        - Accumulate loss, IoU scores, and pixel accuracy.
    - Print average loss, IoU, and pixel accuracy for the test data.
    - Switch model back to training mode.

    Returns:
        tuple: Mean IoU score and mean loss for the test dataset. (For Plotting)
    """

    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for _, (inputs, label) in enumerate(test_loader):
            inputs = inputs.to(device)
            label = label.to(device)
            
            cur_output = fcn_model.forward(inputs)
            cur_loss = criterion(cur_output, label).item()
            cur_IoU = util.iou(cur_output, label)
            cur_pix_acc = util.pixel_acc(cur_output,label)

            losses.append(cur_loss)
            mean_iou_scores.append(cur_IoU)
            accuracy.append(cur_pix_acc)
    
    # Calculate the average loss, IoU, and pixel accuracy for the test data
    avg_loss = np.mean(losses)
    avg_IoU = np.mean(mean_iou_scores)
    avg_acc = np.mean(accuracy)

    # Print the average loss, IoU, and pixel accuracy for the test data
    print("%"*50)
    print("Test results")
    print(f"Loss is {avg_loss}")
    print(f"IoU  is {avg_IoU}")
    print(f"Pixel acc is {avg_acc}")
    
    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!
    return (avg_IoU, avg_loss, avg_acc)

def exportModel(inputs):    
    """
    Export the output of the model for given inputs.

    - Set the model to evaluation mode.
    - Load the model with the best saved weights.
    - Perform a forward pass with the model to get output.
    - Switch model back to training mode.

    Args:
        inputs: Input data to the model.

    Returns:
        Output from the model for the given inputs.
    """

    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !

    
    saved_model_path = save_path
    # TODO Then Load your best model using saved_model_path
    model_weights = torch.load(saved_model_path)
    fcn_model.load_state_dict(model_weights)



    inputs = inputs.to(device)
    
    output_image = fcn_model(inputs)
    
    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!
    
    return output_image

if __name__ == "__main__":

    val(0)  # show the accuracy before training
    train()
    modelTest()
    
    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
