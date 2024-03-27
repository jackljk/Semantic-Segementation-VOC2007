import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FCN(nn.Module):
    def __init__(self, n_class):
        super(FCN, self).__init__()
        self.n_class = n_class
        
        # Load the pretrained ResNet50 model
        resnet50 = models.resnet50(pretrained=True)
        
        # Remove the last fully connected layer
        self.backbone = nn.Sequential(*list(resnet50.children())[:-2])
        
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Decoder layers
        self.conv1 = nn.Conv2d(2048, 1024, 1)
        self.conv2 = nn.Conv2d(1024, 512, 1)
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.classifier = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Pass the input through the backbone to get the feature map
        x = self.backbone(x)

        # Decoder
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.bn1(x)
        x = self.relu(self.deconv4(x))
        x = self.bn2(x)
        score = self.classifier(x)
        score = F.interpolate(score, size=(224, 224), mode='bilinear', align_corners=False)

        # During inference, apply argmax to get the class index for each pixel
        # score = torch.argmax(score, dim=1)  # Uncomment during inference if needed

        return score
