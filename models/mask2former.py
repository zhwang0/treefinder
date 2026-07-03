import torch
from torch import nn
from transformers import Mask2FormerForUniversalSegmentation, AutoConfig


def _find_patch_embed(m2f):
  """Locate the Swin patch-embedding module (the one exposing a `.projection`
  Conv2d stem), tolerating transformers-version differences in the module path."""
  plm = m2f.model.pixel_level_module
  candidates = [
    lambda: plm.encoder.swin.embeddings.patch_embeddings,  # transformers wraps Swin under .swin
    lambda: plm.encoder.embeddings.patch_embeddings,       # backbone directly under .encoder
  ]
  for get in candidates:
    try:
      pe = get()
    except AttributeError:
      continue
    if isinstance(getattr(pe, 'projection', None), nn.Conv2d):
      return pe
  # Fallback: first submodule exposing a 3-channel Conv2d `projection` stem
  for mod in plm.modules():
    proj = getattr(mod, 'projection', None)
    if isinstance(proj, nn.Conv2d) and proj.in_channels == 3:
      return mod
  raise RuntimeError("Could not locate Mask2Former patch-embedding projection conv")


class Mask2FormerWrapper(nn.Module):
  def __init__(self, num_classes, image_size, num_channels, pretrained_name):
    super().__init__()
    self.image_size = image_size
    self.num_channels = num_channels

    # Load and override config
    config = AutoConfig.from_pretrained(pretrained_name)
    config.num_classes = num_classes
    config.image_size = image_size

    # Load pretrained model with updated config
    self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
      pretrained_name,
      config=config,
      ignore_mismatched_sizes=True
    )

    # Expand the Swin patch-embedding stem for >3-channel input (RGB+NIR+NDVI):
    # copy the pretrained RGB weights and mean-fill extra channels (as in deeplab.py)
    if num_channels != 3:
      patch_embed = _find_patch_embed(self.mask2former)
      old_proj = patch_embed.projection
      new_proj = nn.Conv2d(
        num_channels, old_proj.out_channels,
        kernel_size=old_proj.kernel_size, stride=old_proj.stride,
        padding=old_proj.padding, bias=(old_proj.bias is not None)
      )
      with torch.no_grad():
        new_proj.weight[:, :3, :, :].copy_(old_proj.weight)
        mean_weight = old_proj.weight.mean(dim=1, keepdim=True)
        new_proj.weight[:, 3:, :, :].copy_(mean_weight.repeat(1, num_channels - 3, 1, 1))
        if old_proj.bias is not None:
          new_proj.bias.copy_(old_proj.bias)
      patch_embed.projection = new_proj

    # Replace classifier head to match new number of classes
    in_ch = self.mask2former.class_predictor.in_features
    self.mask2former.class_predictor = nn.Linear(in_ch, num_classes + 1)  # +1 for "no object" class

  def forward(self, x):
    # Use the configured number of input bands (RGB+NIR+NDVI)
    outputs = self.mask2former(pixel_values=x[:, :self.num_channels])

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
  num_channels = int(cfg.get('in_channels', 5))
  pretrained_name = cfg['mask2former_pretrained'].get(
    'mask2former_weights', 'facebook/mask2former-swin-tiny-ade-semantic'
  )

  return Mask2FormerWrapper(
    num_classes=num_classes,
    image_size=image_size,
    num_channels=num_channels,
    pretrained_name=pretrained_name
  )
