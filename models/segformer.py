import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class SegFormer(nn.Module):
    def __init__(self, num_classes, image_size, num_channels, pretrained_name):
        super().__init__()
        
        self.image_size = image_size

        # Initialize SegFormer with the custom configuration
        config = SegformerConfig(
            image_size=image_size,
            num_classes=num_classes,
            num_channels=num_channels
        )
        self.segformer = SegformerForSemanticSegmentation(config=config)

        # Load pretrained model weights
        pretrained_model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_name,
            ignore_mismatched_sizes=True
        )

        # Update patch_embedding with dup first convolutional layers for N-channel input
        with torch.no_grad():
            pretrained_model.segformer.encoder.patch_embeddings[0].proj.weight = nn.Parameter(
                torch.cat([pretrained_model.segformer.encoder.patch_embeddings[0].proj.weight] * 3, dim=1)[:, :num_channels, :, :]
            )
        
        # Update the model's state_dict with the modified pretrained weights
        model_dict = self.segformer.state_dict()
        pretrained_dict = pretrained_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        self.segformer.load_state_dict(model_dict, strict=False)


        # Modify classifier head to match the number of classes
        # HF uses decode_head or segmentation_head
        head_module = (
            self.segformer.decode_head
            if hasattr(self.segformer, 'decode_head')
            else self.segformer.segmentation_head
        )
        in_ch = head_module.classifier.in_channels
        self.segformer.decode_head.classifier = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    
    
    def forward(self, x):
        out = self.segformer(x).logits
        out = torch.nn.functional.interpolate(
            out, 
            size=(self.image_size, self.image_size), 
            mode='bilinear', 
            align_corners=False)
        return out


def build_segformer(cfg: dict):
    num_classes    = int(cfg.get('num_classes', 1))
    image_size     = int(cfg.get('image_size', 224))
    num_channels   = int(cfg.get('in_channels', 5))
    pretrained_name = cfg['segformer_pretrained'].get('segformer_weights', 'nvidia/segformer-b0-finetuned-ade-512-512')

    return SegFormer(
        num_classes=num_classes,
        image_size=image_size,
        num_channels=num_channels,
        pretrained_name=pretrained_name
    )





class SegFormer3(nn.Module):
    def __init__(self, num_classes, image_size, num_channels, pretrained_name):
        super().__init__()
        
        self.image_size = image_size

        # Initialize SegFormer with the custom configuration
        config = SegformerConfig(
            image_size=image_size,
            num_classes=num_classes,
            num_channels=num_channels
        )
        self.segformer = SegformerForSemanticSegmentation(config=config)

        # Load pretrained model weights
        pretrained_model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_name,
            ignore_mismatched_sizes=True
        )

        # # Update patch_embedding with dup first convolutional layers for N-channel input
        # with torch.no_grad():
        #     pretrained_model.segformer.encoder.patch_embeddings[0].proj.weight = nn.Parameter(
        #         torch.cat([pretrained_model.segformer.encoder.patch_embeddings[0].proj.weight] * 3, dim=1)[:, :num_channels, :, :]
        #     )
        
        # Update the model's state_dict with the modified pretrained weights
        model_dict = self.segformer.state_dict()
        pretrained_dict = pretrained_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        self.segformer.load_state_dict(model_dict, strict=False)


        # Modify classifier head to match the number of classes
        # HF uses decode_head or segmentation_head
        head_module = (
            self.segformer.decode_head
            if hasattr(self.segformer, 'decode_head')
            else self.segformer.segmentation_head
        )
        in_ch = head_module.classifier.in_channels
        self.segformer.decode_head.classifier = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    
    
    def forward(self, x):
        out = self.segformer(x[:,:3]).logits
        out = torch.nn.functional.interpolate(
            out, 
            size=(self.image_size, self.image_size), 
            mode='bilinear', 
            align_corners=False)
        return out


def build_segformer3(cfg: dict):
    num_classes    = int(cfg.get('num_classes', 1))
    image_size     = int(cfg.get('image_size', 224))
    num_channels   = int(cfg.get('in_channels', 3))
    pretrained_name = cfg['segformer_pretrained'].get('segformer_weights', 'nvidia/segformer-b0-finetuned-ade-512-512')

    return SegFormer3(
        num_classes=num_classes,
        image_size=image_size,
        num_channels=3,
        pretrained_name=pretrained_name
    )




from transformers import AutoConfig
class SegFormer_large(nn.Module):
  def __init__(self, num_classes, image_size, num_channels, pretrained_name):
    super().__init__()
    self.image_size = image_size

    # Load config from pretrained model and override input/output settings
    config = AutoConfig.from_pretrained(pretrained_name)
    config.num_classes = num_classes
    config.num_channels = num_channels  # this overrides the default 3-channel input
    config.image_size = image_size

    # Instantiate model with modified config
    self.segformer = SegformerForSemanticSegmentation.from_pretrained(
      pretrained_name,
      config=config,
      ignore_mismatched_sizes=True
    )

    # If num_channels != 3, update the first conv layer manually
    if num_channels != 3:
      with torch.no_grad():
        old_proj = self.segformer.segformer.encoder.patch_embeddings[0].proj
        new_proj = nn.Conv2d(
          in_channels=num_channels,
          out_channels=old_proj.out_channels,
          kernel_size=old_proj.kernel_size,
          stride=old_proj.stride,
          padding=old_proj.padding
        )
        # Copy weights from original 3 channels to new channels if possible
        if num_channels > 3:
          new_proj.weight[:, :3, :, :] = old_proj.weight
          new_proj.weight[:, 3:, :, :] = old_proj.weight[:, :1, :, :].repeat(1, num_channels - 3, 1, 1)
        else:
          new_proj.weight = nn.Parameter(old_proj.weight[:, :num_channels, :, :])
        new_proj.bias = old_proj.bias
        self.segformer.segformer.encoder.patch_embeddings[0].proj = new_proj

    # Replace the classifier to match the required output channels (num_classes)
    in_ch = self.segformer.decode_head.classifier.in_channels
    self.segformer.decode_head.classifier = nn.Conv2d(in_ch, num_classes, kernel_size=1)

  def forward(self, x):
    out = self.segformer(x[:,:3]).logits
    return nn.functional.interpolate(out, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)

def build_segformer_large3(cfg: dict):
    num_classes    = int(cfg.get('num_classes', 1))
    image_size     = int(cfg.get('image_size', 224))
    num_channels   = int(cfg.get('in_channels', 3))
    pretrained_name = cfg['segformer_pretrained'].get('segformer_weights', 'nvidia/segformer-b1-finetuned-ade-512-512')

    return SegFormer_large(
        num_classes=num_classes,
        image_size=image_size,
        num_channels=3,
        pretrained_name=pretrained_name
    )