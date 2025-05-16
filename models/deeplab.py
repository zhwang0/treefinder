# File: models/deeplab.py

import types
import torch
from torch import nn
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights
)
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models import resnet50, ResNet50_Weights


def create_deeplabv3plus(in_channels, num_classes, backbone, pretrained):
    """
    Create a DeepLabV3+ model with custom input channels and number of classes.
    Args:
      in_channels: number of input bands (e.g. 5 for RGB+NIR+NDVI)
      num_classes: number of output segmentation classes
      backbone:    name of the ResNet backbone (only "resnet50" supported currently)
      pretrained:  whether to load ImageNet-pretrained weights for the backbone
    """
    
    backbone = backbone.lower()
    if backbone != "resnet50":
        raise ValueError(f"Unsupported backbone: {backbone}")

    # Load base model
    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet50(weights=weights, progress=True)

    # Replace first conv to accept in_channels
    old_conv = model.backbone.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    )
    if pretrained:
        with torch.no_grad():
            # copy the RGB weights
            new_conv.weight[:, :3, :, :].copy_(old_conv.weight)
            # for extra channels, replicate the mean of RGB weights
            if in_channels > 3:
                mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, 3:, :, :].copy_(
                    mean_weight.repeat(1, in_channels - 3, 1, 1)
                )
    model.backbone.conv1 = new_conv

    # Replace main classifier head
    old_cls = model.classifier[-1]
    model.classifier[-1] = nn.Conv2d(
        old_cls.in_channels,
        num_classes,
        kernel_size=old_cls.kernel_size,
        stride=1
    )

    # # Replace auxiliary classifier if present
    # if model.aux_classifier is not None:
    #     old_aux = model.aux_classifier[-1]
    #     model.aux_classifier[-1] = nn.Conv2d(
    #         old_aux.in_channels,
    #         num_classes,
    #         kernel_size=old_aux.kernel_size,
    #         stride=1
    #     )
    model.aux_classifier = None  # Disable auxiliary classifier
    
    # 5) Monkey‐patch forward: call original then grab 'out'
    orig_forward = model.forward
    def forward_out(self, x):
        return orig_forward(x)['out']
    model.forward = types.MethodType(forward_out, model)

    return model


def build_deeplabv3(cfg: dict) -> nn.Module:
    in_ch       = int(cfg.get('in_channels', 5))
    num_classes = int(cfg.get("num_classes", 1))
    backbone    = cfg['deeplab_pretrained'].get("backbone", "resnet50")
    pretrained  = bool(cfg['deeplab_pretrained'].get("pretrained", True))
    return create_deeplabv3plus(
        in_channels=in_ch,
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained
    )
    

# tf version 
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, dilations=(6, 12, 18)):
        super().__init__()
        # 1×1 projection of global pooled feature
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 1×1 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 3×3 dilated convs
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 
                          padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for d in dilations
        ])
        # fuse
        self.project = nn.Sequential(
            nn.Conv2d((2 + len(dilations)) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode='bilinear', align_corners=False)

        feats = [gp, self.conv1(x)] + [conv(x) for conv in self.convs]
        x = torch.cat(feats, dim=1)
        return self.project(x)
class DeepLabV3PlusTF(nn.Module):
    """
    PyTorch port of your TF DeepLabV3+:
      - ResNet50 backbone (no pretrained weights)
      - ASPP with rates 6,12,18 + image pooling
      - Skip from layer2 (512→48)
      - Two 3×3 convs, upsample to full image
    """
    def __init__(self, img_channels=3, nclasses=2, backbone_pretrained=False):
        super().__init__()
        # 1) backbone
        weights = ResNet50_Weights.IMAGENET1K_V1 if backbone_pretrained else None
        resnet = resnet50(weights=weights, replace_stride_with_dilation=[False, True, True])
        # adjust first conv if needed
        if img_channels != 3:
            old = resnet.conv1
            resnet.conv1 = nn.Conv2d(img_channels, old.out_channels,
                                     kernel_size=old.kernel_size,
                                     stride=old.stride,
                                     padding=old.padding,
                                     bias=(old.bias is not None))
        # keep layers up through layer3
        self.stage0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.stage1 = resnet.layer1   # low-level features (stride=4)
        self.stage2 = resnet.layer2
        self.stage3 = resnet.layer3   # high-level features for ASPP (stride=16)

        # 2) ASPP on stage3 output
        self.aspp = ASPP(in_channels=1024, out_channels=256, dilations=(6,12,18))

        # 3) low-level projection (from stage1)
        self.low_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 4) decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256   , 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 5) final classifier
        self.classifier = nn.Conv2d(256, nclasses, 1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        # backbone forward
        x0 = self.stage0(x)
        low_feat = self.stage1(x0)
        x2 = self.stage2(low_feat)
        high_feat = self.stage3(x2)

        # ASPP + upsample to low-level spatial size
        x_aspp = self.aspp(high_feat)
        x_aspp = F.interpolate(x_aspp, size=low_feat.shape[2:], 
                               mode='bilinear', align_corners=False)

        # project low-level features
        x_low = self.low_proj(low_feat)

        # concat & decode
        x = torch.cat([x_aspp, x_low], dim=1)
        x = self.decoder(x)

        # final upsample to input resolution
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return self.classifier(x)


def build_deeplab_plus_tf(cfg: dict) -> nn.Module:
    return DeepLabV3PlusTF(
        img_channels   = int(cfg.get('in_channels', 5)),
        nclasses       = int(cfg.get('num_classes', 1))
    )
