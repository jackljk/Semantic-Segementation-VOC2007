import torch.nn as nn
import torch
from torch.nn.functional import relu


class UNET(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        input_shape = (224, 224, 3)
        filter_size = [32, 64, 128, 256, 512]
        kernel_size = 3
        padding = 1
        stride = 2
        self.n_class = n_class
        self.crop = True
        
        # Encoder (Layer 1)
        self.conv11 = nn.Conv2d(input_shape[2], filter_size[0], kernel_size=kernel_size, padding=padding)
        self.bn11 = nn.BatchNorm2d(filter_size[0])
        self.conv1 = nn.Conv2d(filter_size[0], filter_size[0], kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(filter_size[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder (Layer 2)
        self.conv21 = nn.Conv2d(filter_size[0], filter_size[1], kernel_size=kernel_size, padding=padding)
        self.bn21 = nn.BatchNorm2d(filter_size[1])
        self.conv2 = nn.Conv2d(filter_size[1], filter_size[1], kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(filter_size[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder (Layer 3)
        self.conv31 = nn.Conv2d(filter_size[1], filter_size[2], kernel_size=kernel_size, padding=padding)
        self.bn31 = nn.BatchNorm2d(filter_size[2])
        self.conv3 = nn.Conv2d(filter_size[2], filter_size[2], kernel_size=kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(filter_size[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder (Layer 4)
        self.conv41 = nn.Conv2d(filter_size[2], filter_size[3], kernel_size=kernel_size, padding=padding)
        self.bn41 = nn.BatchNorm2d(filter_size[3])
        self.conv4 = nn.Conv2d(filter_size[3], filter_size[3], kernel_size=kernel_size, padding=padding)
        self.bn4 = nn.BatchNorm2d(filter_size[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck1 = nn.Conv2d(filter_size[3], filter_size[4], kernel_size=kernel_size, padding=padding)
        self.bn_bottleneck1 = nn.BatchNorm2d(filter_size[4])
        self.bottleneck2 = nn.Conv2d(filter_size[4], filter_size[4], kernel_size=kernel_size, padding=padding)
        self.bn_bottleneck2 = nn.BatchNorm2d(filter_size[4])
        
        # Decoder (Layer 1)
        self.up_pool1 = nn.ConvTranspose2d(filter_size[4], filter_size[3], kernel_size=2, stride=stride)
        self.up_conv11 = nn.Conv2d(filter_size[4], filter_size[3], kernel_size=kernel_size, padding=padding)
        self.up_conv1 = nn.Conv2d(filter_size[3], filter_size[3], kernel_size=kernel_size, padding=padding)
        
        # Decoder (Layer 2)
        self.up_pool2 = nn.ConvTranspose2d(filter_size[3], filter_size[2], kernel_size=2, stride=stride)
        self.up_conv21 = nn.Conv2d(filter_size[3], filter_size[2], kernel_size=kernel_size, padding=padding)
        self.up_conv2 = nn.Conv2d(filter_size[2], filter_size[2], kernel_size=kernel_size, padding=padding)
        
        # Decoder (Layer 3)
        self.up_pool3 = nn.ConvTranspose2d(filter_size[2], filter_size[1], kernel_size=2, stride=stride)
        self.up_conv31 = nn.Conv2d(filter_size[2], filter_size[1], kernel_size=kernel_size, padding=padding)
        self.up_conv3 = nn.Conv2d(filter_size[1], filter_size[1], kernel_size=kernel_size, padding=padding)
        
        # Decoder (Layer 4)
        self.up_pool4 = nn.ConvTranspose2d(filter_size[1], filter_size[0], kernel_size=2, stride=stride)
        self.up_conv41 = nn.Conv2d(filter_size[1], filter_size[0], kernel_size=kernel_size, padding=padding)
        self.up_conv4 = nn.Conv2d(filter_size[0], filter_size[0], kernel_size=kernel_size, padding=padding)
        
        # output layer
        self.classifier = nn.Conv2d(filter_size[0], self.n_class, kernel_size=1)

        # BatchNorm2d layers
        self.bn_up1 = nn.BatchNorm2d(filter_size[3])
        self.bn_up2 = nn.BatchNorm2d(filter_size[2])
        self.bn_up3 = nn.BatchNorm2d(filter_size[1])
        self.bn_up4 = nn.BatchNorm2d(filter_size[0])
  

    
    def forward(self, x):
        # Encoder
        xconv11 = relu(self.bn11(self.conv11(x)))
        xconv1 = relu(self.bn1(self.conv1(xconv11)))
        xpool1 = self.pool1(xconv1)
        
        xconv21 = relu(self.bn21(self.conv21(xpool1)))
        xconv2 = relu(self.bn2(self.conv2(xconv21)))
        xpool2 = self.pool2(xconv2)
        
        xconv31 = relu(self.bn31(self.conv31(xpool2)))
        xconv3 = relu(self.bn3(self.conv3(xconv31)))
        xpool3 = self.pool3(xconv3)
        
        xconv41 = relu(self.bn41(self.conv41(xpool3)))
        xconv4 = relu(self.bn4(self.conv4(xconv41)))
        xpool4 = self.pool4(xconv4)
        
        # Bottleneck
        xbottleneck1 = relu(self.bn_bottleneck1(self.bottleneck1(xpool4)))
        xbottleneck2 = relu(self.bn_bottleneck2(self.bottleneck2(xbottleneck1)))
        
        # Decoder
        xup_pool1 = self.up_pool1(xbottleneck2)
        xup1 = torch.cat([xup_pool1, xconv4], dim=1)
        xup_conv11 = relu(self.up_conv11(xup1))
        xup_conv1 = relu(self.up_conv1(xup_conv11))
        xup_conv1 = relu(self.bn_up1(xup_conv1))  # Apply BatchNorm2d
        
        xup_pool2 = self.up_pool2(xup_conv1)
        xup2 = torch.cat([xup_pool2, xconv3], dim=1)
        xup_conv21 = relu(self.up_conv21(xup2))
        xup_conv2 = relu(self.up_conv2(xup_conv21))
        xup_conv2 = relu(self.bn_up2(xup_conv2))  # Apply BatchNorm2d
        
        xup_pool3 = self.up_pool3(xup_conv2)
        xup3 = torch.cat([xup_pool3, xconv2], dim=1)
        xup_conv31 = relu(self.up_conv31(xup3))
        xup_conv3 = relu(self.up_conv3(xup_conv31))
        xup_conv3 = relu(self.bn_up3(xup_conv3))  # Apply BatchNorm2d
        
        xup_pool4 = self.up_pool4(xup_conv3)
        xup4 = torch.cat([xup_pool4, xconv1], dim=1)
        xup_conv41 = relu(self.up_conv41(xup4))
        xup_conv4 = relu(self.up_conv4(xup_conv41))
        xup_conv4 = relu(self.bn_up4(xup_conv4))  # Apply BatchNorm2d
        
        # Output layer
        output = self.classifier(xup_conv4)

        return output  # size=(N, n_class, H, W)
