import torch
from torch import nn
from transformers import Mask2FormerForUniversalSegmentation, AutoConfig


class Mask2FormerWrapper(nn.Module):
  def __init__(self, num_classes, image_size, num_channels, pretrained_name):
    super().__init__()
    self.image_size = image_size

    # Load and override config
    config = AutoConfig.from_pretrained(pretrained_name)
    config.num_classes = num_classes
    config.image_size = image_size
    config.num_channels = num_channels  # manually handle custom channel inputs

    # Load pretrained model with updated config
    self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
      pretrained_name,
      config=config,
      ignore_mismatched_sizes=True
    )

    # Replace classifier head to match new number of classes
    in_ch = self.mask2former.class_predictor.in_features
    self.mask2former.class_predictor = nn.Linear(in_ch, num_classes + 1)  # +1 for "no object" class

  def forward(self, x):
    # Use only the first 3 channels for compatibility with pretrained weights
    outputs = self.mask2former(pixel_values=x[:, :3])

    # Extract class logits and mask logits
    class_logits = outputs.class_queries_logits  # [B, Q, num_classes+1]
    mask_logits = outputs.masks_queries_logits   # [B, Q, H, W]

    # Remove "no object" class from class logits
    class_logits = class_logits[:, :, :-1]  # [B, Q, num_classes]

    # Softmax over classes, sigmoid over masks
    class_probs = class_logits.softmax(dim=-1)     # [B, Q, num_classes]
    # mask_probs = mask_logits.sigmoid()             # [B, Q, H, W]

    # Multiply and sum over queries → [B, num_classes, H, W]
    segmentation_map = torch.einsum("bqc,bqhw->bchw", class_probs, mask_logits)

    # Resize to target image size
    return nn.functional.interpolate(
        segmentation_map,
        size=(self.image_size, self.image_size),
        mode='bilinear',
        align_corners=False
    )
    

def build_mask2former(cfg: dict):
  num_classes = int(cfg.get('num_classes', 1))
  image_size = int(cfg.get('image_size', 224))
  num_channels = 3
  pretrained_name = cfg['mask2former_pretrained'].get(
    'mask2former_weights', 'facebook/mask2former-swin-tiny-ade-semantic'
  )

  return Mask2FormerWrapper(
    num_classes=num_classes,
    image_size=image_size,
    num_channels=num_channels,
    pretrained_name=pretrained_name
  )
