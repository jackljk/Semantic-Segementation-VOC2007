
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedUNet(nn.Module):
    def __init__(self, n_classes):
        super(SimplifiedUNet, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # Decoder
        self.dec_upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec_upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dec_upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Final Classifier
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = F.relu(self.enc_conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.enc_conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.enc_conv3(x))
        x = F.max_pool2d(x, 2)

        # Bottleneck
        x = F.relu(self.bottleneck_conv(x))

        # Decoder
        x = F.relu(self.dec_upconv1(x))
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_upconv2(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_upconv3(x))
        x = F.relu(self.dec_conv3(x))

        # Final Classifier
        x = self.final_conv(x)
        return x