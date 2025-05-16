import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, features):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for feature in features:
            self.encoders.append(DoubleConv(in_channels, feature))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature
            
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        prev_channels = features[-1] * 2
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(prev_channels, feature, kernel_size=2, stride=2)
            )
            self.decoders.append(DoubleConv(feature*2, feature))
            prev_channels = feature
            
        self.final_conv = nn.Conv2d(prev_channels, num_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip = skip_connections[idx]
            # if x.shape != skip.shape:
            #     x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.decoders[idx](x)
        return self.final_conv(x)


def build_unet(cfg):
    return UNet(
        in_channels=cfg.get('in_channels', 5),
        num_classes=cfg.get('num_classes', 1),
        features=[64,128,256,512]
    )

