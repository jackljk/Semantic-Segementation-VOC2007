import torch.nn as nn
import torch

class UNET(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        input_shape = (224, 224, 3)
        filter_size = [64, 128, 256, 512, 1024]
        kernel_size = 3
        padding = 1
        stride = 1
        self.n_class = n_class
        self.cropb = True

        # Activation function
        self.relu = nn.ReLU(inplace=True)

        # Encoder (Layer 1)
        self.conv11 = nn.Conv2d(input_shape[2], filter_size[0], kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv1 = nn.Conv2d(filter_size[0], filter_size[0], kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(filter_size[0])

        # Encoder (Layer 2)
        self.conv21 = nn.Conv2d(filter_size[0], filter_size[1], kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv2 = nn.Conv2d(filter_size[1], filter_size[1], kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(filter_size[1])

        # Encoder (Layer 3)
        self.conv31 = nn.Conv2d(filter_size[1], filter_size[2], kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv3 = nn.Conv2d(filter_size[2], filter_size[2], kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(filter_size[2])

        # Encoder (Layer 4)
        self.conv41 = nn.Conv2d(filter_size[2], filter_size[3], kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv4 = nn.Conv2d(filter_size[3], filter_size[3], kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(filter_size[3])

        # Bottleneck
        self.bottleneck1 = nn.Conv2d(filter_size[3], filter_size[4], kernel_size=kernel_size, padding=padding, stride=stride)
        self.bottleneck2 = nn.Conv2d(filter_size[4], filter_size[4], kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn_bottleneck1 = nn.BatchNorm2d(filter_size[4])

        # Decoder (Layer 1)
        self.up_pool1 = nn.ConvTranspose2d(filter_size[4], filter_size[3], kernel_size=2,stride=2)
        self.up_conv11 = nn.Conv2d(filter_size[4], filter_size[3], kernel_size=kernel_size, padding=padding, stride=stride)
        self.up_conv1 = nn.Conv2d(filter_size[3], filter_size[3], kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn_up_conv1 = nn.BatchNorm2d(filter_size[3])

        # Decoder (Layer 2)
        self.up_pool2 = nn.ConvTranspose2d(filter_size[3], filter_size[2], kernel_size=2, stride=2)
        self.up_conv21 = nn.Conv2d(filter_size[3], filter_size[2], kernel_size=kernel_size, padding=padding, stride=stride)
        self.up_conv2 = nn.Conv2d(filter_size[2], filter_size[2], kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn_up_conv2 = nn.BatchNorm2d(filter_size[2])

        # Decoder (Layer 3)
        self.up_pool3 = nn.ConvTranspose2d(filter_size[2], filter_size[1], kernel_size=2, stride=2)
        self.up_conv31 = nn.Conv2d(filter_size[2], filter_size[1], kernel_size=kernel_size, padding=padding, stride=stride)
        self.up_conv3 = nn.Conv2d(filter_size[1], filter_size[1], kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn_up_conv3 = nn.BatchNorm2d(filter_size[1])

        # Decoder (Layer 4)
        self.up_pool4 = nn.ConvTranspose2d(filter_size[1], filter_size[0], kernel_size=2, stride=2)
        self.up_conv41 = nn.Conv2d(filter_size[1], filter_size[0], kernel_size=kernel_size, padding=padding, stride=stride)
        self.up_conv4 = nn.Conv2d(filter_size[0], filter_size[0], kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn_up_conv4 = nn.BatchNorm2d(filter_size[0])

        # output layer
        self.classifier = nn.Conv2d(filter_size[0], self.n_class, kernel_size=1)


    def crop(self, source, target):
        target_size = target.size()[2:]
        source_size = source.size()[2:]

        delta = [(s - t) // 2 for s, t in zip(source_size, target_size)]
        return source[:, :, delta[0]:source_size[0] - delta[0], delta[1]:source_size[1] - delta[1]]


    def forward(self, x):
        # Encoder
        conv11 = self.relu(self.bn1(self.conv11(x)))
        conv1 = self.relu(self.bn1(self.conv1(conv11)))
        pool1 = self.pool1(conv1)

        conv21 = self.relu(self.bn2(self.conv21(pool1)))
        conv2 = self.relu(self.bn2(self.conv2(conv21)))
        pool2 = self.pool2(conv2)

        conv31 = self.relu(self.bn3(self.conv31(pool2)))
        conv3 = self.relu(self.bn3(self.conv3(conv31)))
        pool3 = self.pool3(conv3)

        conv41 = self.relu(self.bn4(self.conv41(pool3)))
        conv4 = self.relu(self.bn4(self.conv4(conv41)))
        pool4 = self.pool4(conv4)

        # Bottleneck 1
        bottleneck = self.bottleneck1(pool4)
        bottleneck = self.bn_bottleneck1(bottleneck)
        bottleneck = self.relu(bottleneck)

        # Bottleneck 2
        bottleneck = self.bottleneck2(bottleneck)
        bottleneck = self.bn_bottleneck1(bottleneck)
        bottleneck = self.relu(bottleneck)

        # Decoder
        up_pool1 = self.up_pool1(bottleneck)
        if self.cropb:
            up_pool1_cropped = self.crop(conv4, up_pool1)  
            concat1 = torch.cat((up_pool1, up_pool1_cropped), 1)   

        up_conv11 = self.relu(self.bn_up_conv1(self.up_conv11(concat1)))
        up_conv1 = self.relu(self.bn_up_conv1(self.up_conv1(up_conv11)))

        up_pool2 = self.up_pool2(up_conv1)
        if self.cropb:
            up_pool2_cropped = self.crop(conv3, up_pool2)  
            concat2 = torch.cat((up_pool2, up_pool2_cropped), 1)

        up_conv21 = self.relu(self.bn_up_conv2(self.up_conv21(concat2)))
        up_conv2 = self.relu(self.bn_up_conv2(self.up_conv2(up_conv21)))

        up_pool3 = self.up_pool3(up_conv2)
        if self.cropb:
            up_pool3_cropped = self.crop(conv2, up_pool3)  
            concat3 = torch.cat((up_pool3, up_pool3_cropped), 1)
            

        up_conv31 = self.relu(self.bn_up_conv3(self.up_conv31(concat3)))
        up_conv3 = self.relu(self.bn_up_conv3(self.up_conv3(up_conv31)))

        up_pool4 = self.up_pool4(up_conv3)
        if self.cropb:
            up_pool4_cropped = self.crop(conv1, up_pool4)  
            concat4 = torch.cat((up_pool4, up_pool4_cropped), 1)

        up_conv41 = self.relu(self.bn_up_conv4(self.up_conv41(concat4)))
        up_conv4 = self.relu(self.bn_up_conv4(self.up_conv4(up_conv41)))

        output = self.classifier(up_conv4)


        return output  # size=(N, n_class, H, W)