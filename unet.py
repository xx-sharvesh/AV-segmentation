import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=64):
        """
        U-Net architecture for segmentation tasks.

        Parameters:
        - in_channels: Number of input channels (e.g., 3 for RGB images, 1 for grayscale).
        - out_channels: Number of output channels (e.g., 1 for binary segmentation).
        - base_filters: Number of filters in the first layer.
        """
        super(UNet, self).__init__()

        # Contracting path
        self.enc1 = self.contract_block(in_channels, base_filters)
        self.enc2 = self.contract_block(base_filters, base_filters * 2)
        self.enc3 = self.contract_block(base_filters * 2, base_filters * 4)
        self.enc4 = self.contract_block(base_filters * 4, base_filters * 8)

        # Bottleneck
        self.bottleneck = self.contract_block(base_filters * 8, base_filters * 16)

        # Expanding path
        self.dec4 = self.expand_block(base_filters * 16, base_filters * 8)
        self.dec3 = self.expand_block(base_filters * 8, base_filters * 4)
        self.dec2 = self.expand_block(base_filters * 4, base_filters * 2)
        self.dec1 = self.expand_block(base_filters * 2, base_filters)

        # Final layer
        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def contract_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        """Contraction block with two convolutional layers and max pooling."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def expand_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        """Expansion block with transposed convolution and two convolutional layers."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder
        dec4 = self.dec4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)  # Skip connection
        dec3 = self.dec3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec2 = self.dec2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec1 = self.dec1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)

        # Final layer
        output = self.final(dec1)
        return output
