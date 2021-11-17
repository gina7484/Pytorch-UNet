""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)


        self.up1 = Up(512, 512 // factor, bilinear)
        self.up2 = Up(256, 256 // factor, bilinear)
        self.up3 = Up(128, 128 // factor, bilinear)
        self.up4 = Up(64, 64, bilinear)

        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        #self.up3 = Up(256, 128 // factor, bilinear)
        #self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        print("x5 x4 shape")
        print(x5.shape, x4.shape)
        
        x = self.up1(x5, x4)

        print("x x3 shape")
        print(x.shape, x3.shape)

        x = self.up2(x, x3)

        print("x x2 shape")
        print(x.shape, x2.shape)

        x = self.up3(x, x2)

        print("x x1 shape")
        print(x.shape, x1.shape)

        x = self.up4(x, x1)

        print("x shape")
        print(x.shape)

        logits = self.outc(x)

        print("logits shape")
        print(logits.shape)

        return logits
